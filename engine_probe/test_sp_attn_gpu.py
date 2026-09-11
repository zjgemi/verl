"""GPU/NCCL check of the Qwen3.5 Ulysses SP attention paths, one backend per run.

The CPU sibling (`test_sp_sdpa_attention.py`) can only exercise sdpa: flash attention
needs a GPU and half precision. This one runs the real all-to-all over NCCL and can
therefore also exercise `flash_attention_2`, whose all-to-all lives in a different place
(`_ulysses_flash_attention_forward`, installed on
`transformers.integrations.flash_attention._flash_attention_forward`) and which nobody
has ever run for qwen3_5 -- flash_attn used to be uninstallable in our image.

Reference is the same model, same dtype, same backend with SP off, so the only thing
that differs between the two numbers is the sequence parallelism. In bf16 a correct
implementation lands at bf16 noise on *every* shard; the failure mode we are guarding
against is unmistakable instead -- shard 0 exactly 0.0 (rank 0's slice is causally
self-contained) and the later shards grossly wrong.

Run:  PYTHONPATH=/personal/verl ATTN_IMPL=flash_attention_2 \
        torchrun --nproc_per_node=2 test_sp_attn_gpu.py
"""

import os

import torch
import torch.distributed as dist
from transformers.models.qwen3_5.modeling_qwen3_5 import Qwen3_5TextConfig, Qwen3_5TextModel

from verl.models.transformers.monkey_patch import apply_monkey_patch
from verl.utils import ulysses as U


def main():
    local_rank = int(os.environ.get("LOCAL_RANK", 0))
    torch.cuda.set_device(local_rank)
    dist.init_process_group("nccl")
    rank, ws = dist.get_rank(), dist.get_world_size()
    group = dist.group.WORLD

    def log(*a):
        if rank == 0:
            print("[sp-gpu]", *a, flush=True)

    impl = os.environ.get("ATTN_IMPL", "sdpa")
    dtype = torch.bfloat16
    T = int(os.environ.get("SEQ_LEN", "512"))
    n_kv = int(os.environ.get("N_KV", "4"))
    cfg = Qwen3_5TextConfig(
        vocab_size=256,
        hidden_size=256,
        intermediate_size=512,
        num_hidden_layers=2,
        num_attention_heads=8,
        num_key_value_heads=n_kv,
        head_dim=32,
        layer_types=["full_attention"] * 2,  # isolate the attention path from GDN
        max_position_embeddings=T,
        use_cache=False,
    )
    cfg._attn_implementation = impl

    torch.manual_seed(0)
    model = Qwen3_5TextModel(cfg).to(device="cuda", dtype=dtype).eval()
    cfg.model_type = "qwen3_5"  # the config says `qwen3_5_text`; the patch keys off `qwen3_5`
    apply_monkey_patch(model, ulysses_sp_size=ws, use_remove_padding=True, use_fused_kernels=False)

    import transformers.integrations.flash_attention as fa

    calls = {"n": 0}
    hooked = fa._flash_attention_forward

    def counting(*a, **kw):
        calls["n"] += 1
        return hooked(*a, **kw)

    fa._flash_attention_forward = counting

    lens = [int(x) for x in os.environ.get("SEQ_LENS", str(T)).split(",")]
    assert sum(lens) == T, f"SEQ_LENS must sum to {T}"
    ids = torch.randint(0, 256, (1, T), device="cuda")
    pos = torch.cat([torch.arange(n) for n in lens]).unsqueeze(0).cuda()
    cu = torch.tensor([0, *torch.tensor(lens).cumsum(0).tolist()], dtype=torch.long)
    with torch.no_grad():
        emb = model.embed_tokens(ids)
    fwd = dict(
        input_ids=None,
        inputs_embeds=emb,
        position_ids=pos,
        attention_mask=None,
        use_cache=False,
        cu_seqlens=cu.cuda(),
        cu_seqlens_cpu=cu,
    )

    U.set_ulysses_sequence_parallel_group(None)
    with torch.no_grad():
        ref = model(**fwd)[0]
    n_ref = calls["n"]

    U.set_ulysses_sequence_parallel_group(group)
    with torch.no_grad():
        loc = model(**fwd)[0]
    n_sp = calls["n"] - n_ref

    assert loc.shape[1] == T // ws, f"expected local slice {T // ws}, got {loc.shape[1]}"
    buf = [torch.empty_like(loc) for _ in range(ws)]
    dist.all_gather(buf, loc.contiguous(), group=group)
    got = torch.cat(buf, dim=1)

    ref, got = ref.float(), got.float()
    log(f"attn_impl={impl} dtype={dtype} world={ws} seq={T} packed_lens={lens} n_kv={n_kv}")
    log(f"_flash_attention_forward calls: sp_off={n_ref} sp_on={n_sp}")
    log(f"global rel_l2 = {(got - ref).norm().item() / ref.norm().item():.4e}")
    per_shard = []
    for k in range(ws):
        sl = slice(k * (T // ws), (k + 1) * (T // ws))
        per_shard.append((got[:, sl] - ref[:, sl]).norm().item() / ref[:, sl].norm().item())
        log(f"  shard {k}: rel_l2 = {per_shard[-1]:.4e}")
    # The bug is not "nonzero error" -- a correct run is either bit-exact or at dtype noise on
    # every shard. It is the *asymmetry*: rank 0's slice is causally self-contained, so it stays
    # exact while the ranks that should have attended across the shard boundary blow up.
    if per_shard[0] == 0.0 and max(per_shard[1:]) > 1e-2:
        log("  <-- shard 0 exact while later shards are wrong: the all-to-all did not run")

    dist.destroy_process_group()


if __name__ == "__main__":
    main()
