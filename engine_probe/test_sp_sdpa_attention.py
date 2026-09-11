"""Minimal CPU repro: under Ulysses SP, verl installs the all-to-all ONLY into
`transformers.integrations.flash_attention._flash_attention_forward`.

Our runs set `+actor_rollout_ref.model.override_config.attn_implementation=sdpa`
(verl_coding/run_coding_practice_qwen3_5_9b_4l20.sh:118), so that hook is never
reached -> every full_attention layer attends only inside its own rank's
contiguous 1/sp slice of the sequence.

Signature of the bug (as opposed to a numerical issue):
  rank 0 output == reference EXACTLY (its shard is causally self-contained)
  rank k>0 output is grossly wrong (cannot see shards 0..k-1)

Run:  torchrun --nproc_per_node=2 test_sp_sdpa_attention.py
"""

import os

import torch
import torch.distributed as dist
from transformers.models.qwen3_5.modeling_qwen3_5 import Qwen3_5TextConfig, Qwen3_5TextModel

from verl.models.transformers.monkey_patch import apply_monkey_patch
from verl.utils import ulysses as U


def install_gloo_all_to_all_shim():
    """gloo has no alltoall; emulate it with all_gather so the fix can be checked on CPU.

    Semantics copied from verl.utils.ulysses.all_to_all_tensor: split `local_input` into
    ws chunks along scatter_dim, rank r keeps chunk r from every rank, concatenated along
    gather_dim in rank order.
    """
    from verl.utils import ulysses as _U

    def shim(local_input, scatter_dim, gather_dim, group=None, async_op=False):
        assert not async_op
        group = _U.get_ulysses_sequence_parallel_group() if group is None else group
        ws = dist.get_world_size(group)
        me = dist.get_rank(group)
        buf = [torch.empty_like(local_input) for _ in range(ws)]
        dist.all_gather(buf, local_input.contiguous(), group=group)
        mine = [torch.tensor_split(t, ws, scatter_dim)[me].contiguous() for t in buf]
        return torch.cat(mine, dim=gather_dim).contiguous()

    _U.all_to_all_tensor = shim
    import verl.utils.ulysses as u2

    u2.all_to_all_tensor = shim
    # SeqAllToAll.forward resolved the name at def time only inside the function body,
    # so rebinding the module global is enough.


def main():
    dist.init_process_group("gloo")
    install_gloo_all_to_all_shim()
    rank, ws = dist.get_rank(), dist.get_world_size()
    group = dist.group.WORLD

    def log(*a):
        if rank == 0:
            print("[sdpa-sp]", *a, flush=True)

    impl = os.environ.get("ATTN_IMPL", "sdpa")
    T = 64
    # kv=4 -> plain GQA; kv=1 -> forces the kv-repeat path (sp > num_key_value_heads)
    n_kv = int(os.environ.get("N_KV", "4"))
    cfg = Qwen3_5TextConfig(
        vocab_size=256,
        hidden_size=64,
        intermediate_size=128,
        num_hidden_layers=2,
        num_attention_heads=4,
        num_key_value_heads=n_kv,
        head_dim=16,
        layer_types=["full_attention"] * 2,  # no GDN -> no triton, runs on CPU
        max_position_embeddings=T,
        use_cache=False,
    )
    cfg._attn_implementation = impl

    torch.manual_seed(0)
    model = Qwen3_5TextModel(cfg).eval()
    # Qwen3_5TextConfig.model_type is "qwen3_5_text"; force the qwen3_5 branch of
    # apply_monkey_patch so we get exactly the patches the real 9B run gets.
    log(f"model_type before override: {cfg.model_type!r}")
    cfg.model_type = "qwen3_5"
    apply_monkey_patch(model, ulysses_sp_size=ws, use_remove_padding=True, use_fused_kernels=False)

    # count whether the ulysses hook is ever reached
    import transformers.integrations.flash_attention as fa

    calls = {"n": 0}
    hooked = fa._flash_attention_forward

    def counting(*a, **kw):
        calls["n"] += 1
        return hooked(*a, **kw)

    fa._flash_attention_forward = counting

    ids = torch.randint(0, 256, (1, T))
    # packed batch, exactly like the engine's rmpad path: position ids restart per sample
    # and `cu_seqlens` carries the (global) boundaries. HF turns the restarts into a
    # block-causal mask, so the sp=1 reference below is block-diagonal too.
    lens = [int(x) for x in os.environ.get("SEQ_LENS", "40,24").split(",")]
    assert sum(lens) == T, f"SEQ_LENS must sum to {T}"
    pos = torch.cat([torch.arange(n) for n in lens]).unsqueeze(0)
    cu = torch.tensor([0, *torch.tensor(lens).cumsum(0).tolist()], dtype=torch.long)
    # verl treats qwen3_5 as a VLM: the engine only pads, the TextModel wrapper
    # slices `inputs_embeds`. Mirror that -> feed embeddings, not ids.
    with torch.no_grad():
        emb = model.embed_tokens(ids)
    fwd = dict(
        input_ids=None,
        inputs_embeds=emb,
        position_ids=pos,
        attention_mask=None,
        use_cache=False,
        cu_seqlens=cu,
        cu_seqlens_cpu=cu,
    )

    # ---- reference: sp off -------------------------------------------------
    U.set_ulysses_sequence_parallel_group(None)
    assert U.get_ulysses_sequence_parallel_world_size() == 1
    with torch.no_grad():
        ref = model(**fwd)[0]

    # ---- sp on: TextModel slices inputs_embeds internally -------------------
    U.set_ulysses_sequence_parallel_group(group)
    assert U.get_ulysses_sequence_parallel_world_size() == ws
    with torch.no_grad():
        loc = model(**fwd)[0]

    assert loc.shape[1] == T // ws, f"expected local slice {T // ws}, got {loc.shape[1]}"
    buf = [torch.empty_like(loc) for _ in range(ws)]
    dist.all_gather(buf, loc.contiguous(), group=group)
    got = torch.cat(buf, dim=1)

    log(f"attn_impl={impl} world={ws} seq={T} packed_lens={lens}")
    log(f"_flash_attention_forward called {calls['n']} times  <-- 0 means the ulysses a2a never ran")
    log(f"global rel_l2 = {(got - ref).norm().item() / ref.norm().item():.4e}")
    for k in range(ws):
        sl = slice(k * (T // ws), (k + 1) * (T // ws))
        r = (got[:, sl] - ref[:, sl]).norm().item() / ref[:, sl].norm().item()
        log(f"  shard {k}: rel_l2 = {r:.4e}")

    dist.destroy_process_group()


if __name__ == "__main__":
    main()
