"""Attribute the Ulysses-SP-induced forward drift to individual layers / sublayers.

Runs the *same* weights on the *same* tokens twice inside one torchrun job:

  A. reference   : ulysses group unset  -> sp_size == 1, no cp_context, no all-to-all
  B. cumulative  : ulysses group = WORLD -> sp_size == N, real SP forward
  C. isolated    : replay each decoder layer under SP but feed it the *reference*
                   input, so the measured error is injected by that layer alone
  D. gdn split   : inside linear_attention layers, separate the conv1d cross-rank
                   prefix from the chunk_gated_delta_rule CP prefix scan

No vLLM, no FSDP, no sampling noise -> every number is a pure kernel/algorithm delta.

Usage (single node, N GPUs):
    torchrun --nproc_per_node=8 sp_layer_attrib.py --model /trisol/input/model --seqlen 8192
"""

import argparse
import json
import os

import torch
import torch.distributed as dist

from verl.models.transformers.monkey_patch import apply_monkey_patch
from verl.utils import ulysses as U

# ---------------------------------------------------------------- recorders

PHASE = {"name": "none"}
REC = {}  # phase -> layer_idx -> dict


def _rec(layer_idx):
    return REC.setdefault(PHASE["name"], {}).setdefault(layer_idx, {})


def install_gdn_recorder():
    """Wrap _packed_chunk_gated_delta_rule so we can see post-conv q/k/v and the CP context."""
    import verl.models.transformers.qwen3_5 as q35

    orig = q35._packed_chunk_gated_delta_rule

    def wrapped(self, query, key, value, g, beta, cu_seqlens, cu_seqlens_cpu, cp_context=None):
        out = orig(self, query, key, value, g, beta, cu_seqlens, cu_seqlens_cpu, cp_context)
        r = _rec(self.layer_idx)
        r["q"] = query.detach()
        r["k"] = key.detach()
        r["v"] = value.detach()
        r["g"] = g.detach()
        r["beta"] = beta.detach()
        r["cu_seqlens"] = cu_seqlens
        r["cu_seqlens_cpu"] = cu_seqlens_cpu
        r["cp_context"] = cp_context
        r["module"] = self
        r["out"] = out[0].detach()
        return out

    q35._packed_chunk_gated_delta_rule = wrapped
    return orig


# ---------------------------------------------------------------- helpers


def gather_seq(x, group, dim=1):
    """Inverse of slice_input_tensor: concat rank-ordered shards along `dim`."""
    ws = dist.get_world_size(group)
    buf = [torch.empty_like(x) for _ in range(ws)]
    dist.all_gather(buf, x.contiguous(), group=group)
    return torch.cat(buf, dim=dim)


def slice_seq(x, group, dim=1):
    ws = dist.get_world_size(group)
    rk = dist.get_rank(group)
    return x.chunk(ws, dim=dim)[rk].contiguous()


def err(a, b):
    """relative L2 + max abs, computed in fp32."""
    a = a.float()
    b = b.float()
    d = (a - b).abs()
    denom = b.norm().item()
    return {
        "rel_l2": (d.norm().item() / denom) if denom > 0 else float("nan"),
        "max_abs": d.max().item(),
        "ref_rms": (b.pow(2).mean().sqrt().item()),
    }


def layer_types(model):
    cfg = model.config
    inner = getattr(cfg, "text_config", cfg)
    lt = getattr(inner, "layer_types", None)
    if lt is None:
        lt = getattr(cfg, "layer_types", None)
    return lt


def get_layers(model):
    m = model.model
    if hasattr(m, "language_model"):
        m = m.language_model
    return m.layers


# ---------------------------------------------------------------- main


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", default="/trisol/input/model")
    ap.add_argument("--seqlen", type=int, default=8192)
    ap.add_argument("--out", default="/trisol/output/sp_layer_attrib.json")
    ap.add_argument("--logit-window", type=int, default=2048)
    args = ap.parse_args()

    local_rank = int(os.environ.get("LOCAL_RANK", 0))
    torch.cuda.set_device(local_rank)
    dist.init_process_group("nccl")
    group = dist.group.WORLD
    rank = dist.get_rank()
    ws = dist.get_world_size()
    dev = torch.cuda.current_device()

    def log(*a):
        if rank == 0:
            print("[probe]", *a, flush=True)

    assert args.seqlen % ws == 0, "seqlen must divide evenly across ranks"

    # ---- model ------------------------------------------------------------
    from transformers import AutoModelForCausalLM, AutoTokenizer

    attn_impl = "flash_attention_2"
    try:
        model = AutoModelForCausalLM.from_pretrained(
            args.model, dtype=torch.bfloat16, attn_implementation=attn_impl, trust_remote_code=True
        )
    except Exception as e:  # noqa: BLE001
        log(f"flash_attention_2 unavailable ({type(e).__name__}: {e}); falling back to sdpa")
        attn_impl = "sdpa"
        model = AutoModelForCausalLM.from_pretrained(
            args.model, dtype=torch.bfloat16, attn_implementation=attn_impl, trust_remote_code=True
        )
    model = model.to(dev).eval()
    apply_monkey_patch(model, ulysses_sp_size=ws, use_remove_padding=True, use_fused_kernels=False)
    orig_gdr = install_gdn_recorder()

    layers = get_layers(model)
    ltypes = layer_types(model)
    if ltypes is None:
        ltypes = [getattr(lyr, "layer_type", getattr(lyr, "block_type", "?")) for lyr in layers]
    log(f"attn_impl={attn_impl} n_layers={len(layers)} world={ws}")
    log("layer_types:", json.dumps({t: ltypes.count(t) for t in set(ltypes)}))

    # ---- inputs -----------------------------------------------------------
    tok = AutoTokenizer.from_pretrained(args.model, trust_remote_code=True)
    seed_text = (
        "We solve the coupled ODE system numerically with scipy.integrate.solve_ivp, "
        "using an adaptive Radau stepper because the Jacobian is stiff near the boundary. "
        "import numpy as np\nfrom scipy.integrate import solve_ivp\n\n"
        "def rhs(t, y, k1, k2):\n    a, b = y\n    return [-k1*a + k2*b, k1*a - k2*b]\n\n"
        "The equilibrium constant is K = k1/k2 and the relaxation time is 1/(k1+k2). "
    )
    ids = tok(seed_text * 200, return_tensors="pt").input_ids[0]
    assert ids.numel() >= args.seqlen, f"seed text too short: {ids.numel()}"
    input_ids = ids[: args.seqlen].unsqueeze(0).to(dev)
    T = args.seqlen
    position_ids = torch.arange(T, device=dev).unsqueeze(0)
    cu_seqlens = torch.tensor([0, T], device=dev, dtype=torch.long)
    cu_seqlens_cpu = cu_seqlens.cpu()

    # ---- capture hooks on decoder layers ---------------------------------
    cap = {}

    def pre_hook(idx):
        def f(mod, a, kw):
            cap.setdefault(PHASE["name"], {}).setdefault(idx, {})["in"] = (a, dict(kw))

        return f

    def post_hook(idx):
        def f(mod, a, kw, out):
            h = out[0] if isinstance(out, tuple) else out
            cap.setdefault(PHASE["name"], {}).setdefault(idx, {})["out"] = h.detach()

        return f

    handles = []
    for i, lyr in enumerate(layers):
        handles.append(lyr.register_forward_pre_hook(pre_hook(i), with_kwargs=True))
        handles.append(lyr.register_forward_hook(post_hook(i), with_kwargs=True))

    common = dict(attention_mask=None, cu_seqlens=cu_seqlens, cu_seqlens_cpu=cu_seqlens_cpu, use_cache=False)

    # ---- phase A: reference (sp off) --------------------------------------
    PHASE["name"] = "ref"
    U.set_ulysses_sequence_parallel_group(None)
    assert U.get_ulysses_sequence_parallel_world_size() == 1
    with torch.no_grad():
        ref_out = model.model(input_ids=input_ids, position_ids=position_ids, **common)
    ref_h = ref_out[0].detach()
    log("phase A done (reference)")

    # ---- phase B: cumulative SP forward -----------------------------------
    PHASE["name"] = "sp"
    U.set_ulysses_sequence_parallel_group(group)
    assert U.get_ulysses_sequence_parallel_world_size() == ws
    with torch.no_grad():
        sp_out = model.model(
            input_ids=slice_seq(input_ids, group), position_ids=slice_seq(position_ids, group), **common
        )
    sp_h = gather_seq(sp_out[0].detach(), group)
    log("phase B done (cumulative SP)")

    cumulative = []
    for i in range(len(layers)):
        r = cap["ref"][i]["out"]
        s = gather_seq(cap["sp"][i]["out"], group)
        cumulative.append({"layer": i, "type": ltypes[i], **err(s, r)})

    final = {"hidden": err(sp_h, ref_h)}

    # ---- probability-space delta (same formula as rollout_probs_diff) ------
    with torch.no_grad():
        w = min(args.logit_window, T - 1)
        tgt = input_ids[0, T - w :].clone()  # next-token ids for positions T-w-1 .. T-2
        pos = slice(T - w - 1, T - 1)
        lg_r = model.lm_head(ref_h[:, pos, :]).float()
        lg_s = model.lm_head(sp_h[:, pos, :]).float()
        p_r = torch.log_softmax(lg_r, -1).gather(-1, tgt.view(1, -1, 1)).exp()
        p_s = torch.log_softmax(lg_s, -1).gather(-1, tgt.view(1, -1, 1)).exp()
        final["probs_diff_mean"] = (p_s - p_r).abs().mean().item()
        final["probs_pearson"] = torch.corrcoef(torch.stack([p_r.flatten(), p_s.flatten()]))[0, 1].item()
        final["logit_window"] = w
        del lg_r, lg_s
    log("final:", json.dumps(final))

    # ---- phase C: isolated per-layer replay --------------------------------
    PHASE["name"] = "iso"
    isolated = []
    for i, lyr in enumerate(layers):
        a, kw = cap["ref"][i]["in"]
        a = list(a)
        kw = dict(kw)
        # slice every sequence-dim tensor; cu_seqlens stays global (that is the SP contract)
        hs = kw.pop("hidden_states", None)
        if hs is None and a:
            hs = a[0]
            a[0] = slice_seq(hs, group)
        else:
            kw["hidden_states"] = slice_seq(hs, group)
        pe = kw.get("position_embeddings")
        if pe is not None:
            kw["position_embeddings"] = tuple(slice_seq(t, group) for t in pe)
        if kw.get("position_ids") is not None:
            kw["position_ids"] = slice_seq(kw["position_ids"], group)
        with torch.no_grad():
            o = lyr(*a, **kw)
        o = o[0] if isinstance(o, tuple) else o
        g = gather_seq(o.detach(), group)
        isolated.append({"layer": i, "type": ltypes[i], **err(g, cap["ref"][i]["out"])})
    log("phase C done (isolated)")

    # ---- phase D: split conv vs gated-delta-rule inside linear layers -------
    gdn = []
    for i in range(len(layers)):
        if ltypes[i] != "linear_attention":
            continue
        r = REC["ref"].get(i)
        s = REC["iso"].get(i)
        if r is None or s is None:
            continue
        row = {"layer": i}
        # (a) conv error: q/k/v are the post-conv projections, fed identical hidden states
        for nm in ("q", "k", "v"):
            row[f"conv_{nm}"] = err(gather_seq(s[nm], group), r[nm])["rel_l2"]
        # (b) end-to-end of this sublayer (conv + CP prefix scan together)
        row["gdn_out"] = err(gather_seq(s["out"], group), r["out"])["rel_l2"]
        # (c) CP prefix scan alone: feed the *reference* post-conv q/k/v through the CP path
        cp = s.get("cp_context")
        if cp is not None:
            with torch.no_grad():
                o, _ = orig_gdr(
                    s["module"],
                    slice_seq(r["q"], group),
                    slice_seq(r["k"], group),
                    slice_seq(r["v"], group),
                    slice_seq(r["g"], group),
                    slice_seq(r["beta"], group),
                    s["cu_seqlens"],
                    s["cu_seqlens_cpu"],
                    cp,
                )
            row["cp_scan_only"] = err(gather_seq(o.detach(), group), r["out"])["rel_l2"]
        gdn.append(row)
    log("phase D done (gdn split)")

    for h in handles:
        h.remove()

    if rank == 0:
        res = {
            "config": {
                "model": args.model,
                "world_size": ws,
                "seqlen": T,
                "attn_impl": attn_impl,
                "layer_types": ltypes,
            },
            "final": final,
            "cumulative": cumulative,
            "isolated": isolated,
            "gdn_split": gdn,
        }
        os.makedirs(os.path.dirname(args.out), exist_ok=True)
        with open(args.out, "w") as f:
            json.dump(res, f, indent=2)

        def summarize(rows, key="rel_l2"):
            byt = {}
            for r in rows:
                byt.setdefault(r["type"], []).append(r[key])
            return {t: {"n": len(v), "mean": sum(v) / len(v), "max": max(v)} for t, v in byt.items()}

        print("=" * 72, flush=True)
        print("RESULT_JSON_BEGIN", flush=True)
        print(
            json.dumps(
                {
                    "final": final,
                    "isolated_by_type": summarize(isolated),
                    "cumulative_last": cumulative[-1],
                    "gdn_split_mean": {
                        k: sum(r[k] for r in gdn if k in r) / max(1, sum(1 for r in gdn if k in r))
                        for k in ("conv_q", "conv_k", "conv_v", "gdn_out", "cp_scan_only")
                    }
                    if gdn
                    else {},
                }
            ),
            flush=True,
        )
        print("RESULT_JSON_END", flush=True)
        print("=" * 72, flush=True)
        for r in isolated:
            print(f"  iso L{r['layer']:02d} {r['type']:<17} rel_l2={r['rel_l2']:.3e} max={r['max_abs']:.3e}", flush=True)
        for r in cumulative:
            print(f"  cum L{r['layer']:02d} {r['type']:<17} rel_l2={r['rel_l2']:.3e}", flush=True)
        for r in gdn:
            print(f"  gdn L{r['layer']:02d} " + " ".join(f"{k}={v:.3e}" for k, v in r.items() if k != "layer"), flush=True)

    dist.barrier()
    dist.destroy_process_group()


if __name__ == "__main__":
    main()
