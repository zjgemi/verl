#!/usr/bin/env python3
"""Measure the CP-vs-non-CP numerical gap of fla's gated delta rule.

Why this exists
---------------
The SP=8 vs SP=1 probe jobs showed the FSDP<->vLLM divergence is dominated by
Ulysses SP (probs_diff 0.0239 vs 0.0032, 140x on rollout_corr/kl). fla PR #1180
attributes that to the context-parallel affine chain running through
``tl.dot(f32, f32)``, which Triton defaults to tf32 -- 10 mantissa bits -- while
the non-CP path propagates the state with f32 fma on CUDA cores.

This runs the *same* op two ways on the same inputs:
  reference : chunk_gated_delta_rule(..., cu_seqlens=...)   -- what SP=1 does
  test      : chunk_gated_delta_rule(..., cp_context=...)   -- what SP>1 does
and reports the gap. Mathematically the two are identical, so the whole gap is
numerical error introduced by the CP prefix scan.

Run it once per precision (FLA_CP_PRECISION is read at import time):

    torchrun --nproc_per_node=2 test_cp_precision.py                  # tf32 (stock)
    FLA_CP_PRECISION=ieee torchrun --nproc_per_node=2 test_cp_precision.py

Requires the installed fla to be patched by patch_fla_cp_precision.py; without
the patch the env var is ignored and both runs are the stock tf32 arm.
"""

import argparse
import json
import os

import torch
import torch.distributed as dist

from fla.ops.cp import build_cp_context
from fla.ops.gated_delta_rule import chunk_gated_delta_rule


def make_inputs(total_len: int, hv: int, head_k: int, head_v: int, device, dtype, g_scale: float):
    """Deterministic inputs, identical on every rank (same seed, same device type).

    Shapes mirror verl's qwen3_5.py: query/key are already repeat_interleave'd to
    num_v_heads, g is fp32, beta is the activation dtype.

    ``g_scale`` is the knob that decides whether CP matters at all: g is the
    per-token log decay, so the state a rank inherits from its predecessors is
    attenuated by exp(sum(g)) over the slice. At g_scale=0.05 a 1024-token slice
    attenuates by exp(-40) and the CP affine chain is numerically irrelevant --
    reference and CP agree bit-for-bit. Real GatedDeltaNet decay has to be far
    slower than that for the recurrence to carry information at all.
    """
    gen = torch.Generator(device=device).manual_seed(1234)
    q = torch.randn(1, total_len, hv, head_k, device=device, dtype=torch.float32, generator=gen).to(dtype)
    k = torch.randn(1, total_len, hv, head_k, device=device, dtype=torch.float32, generator=gen).to(dtype)
    v = torch.randn(1, total_len, hv, head_v, device=device, dtype=torch.float32, generator=gen).to(dtype)
    # g = -A_log.exp() * softplus(a + dt_bias): always negative, small magnitude.
    a = torch.randn(1, total_len, hv, device=device, dtype=torch.float32, generator=gen)
    g = -torch.nn.functional.softplus(a) * g_scale
    beta = torch.rand(1, total_len, hv, device=device, dtype=torch.float32, generator=gen).to(dtype)
    return q, k, v, g, beta


def gap(a: torch.Tensor, b: torch.Tensor) -> dict:
    d = (a.float() - b.float()).abs()
    scale = b.float().abs().mean().clamp_min(1e-12)
    return {
        "max_abs": d.max().item(),
        "mean_abs": d.mean().item(),
        "rel_mean": (d.mean() / scale).item(),
    }


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--lengths", type=int, nargs="+", default=[2048, 8192, 32768])
    ap.add_argument("--g-scales", type=float, nargs="+", default=[0.001],
                    help="per-token log-decay scale; small = state survives across ranks")
    ap.add_argument("--hv", type=int, default=32, help="num_v_heads")
    ap.add_argument("--head-k", type=int, default=128)
    ap.add_argument("--head-v", type=int, default=128)
    ap.add_argument("--dtype", default="bfloat16")
    args = ap.parse_args()

    dist.init_process_group("nccl")
    rank, world = dist.get_rank(), dist.get_world_size()
    torch.cuda.set_device(rank)
    device = torch.device("cuda", rank)
    dtype = getattr(torch, args.dtype)

    import fla.ops.cp.chunk_delta_h as cdh
    import fla.ops.gated_delta_rule.chunk as gdr_chunk

    prec = getattr(cdh, "FLA_CP_PRECISION_STR", "<unpatched>")

    # Instrumentation: a zero gap is only meaningful if the CP prefix scan
    # actually ran and actually produced a non-zero inherited state. Both have
    # bitten this test once already (fast decay made CP a mathematical no-op).
    probe = {"calls": 0, "state_absmax": None, "state": None}
    _orig_pre = gdr_chunk.chunk_gated_delta_rule_fwd_h_pre_process

    def _counting_pre(*a, **kw):
        out = _orig_pre(*a, **kw)
        probe["calls"] += 1
        probe["state_absmax"] = None if out is None else out.abs().max().item()
        # Keep the fp32 state itself: comparing bf16 *outputs* is too blunt --
        # affine-chain error below one bf16 ulp disappears entirely, which is
        # how an earlier version of this test read 0.0 for every precision.
        probe["state"] = None if out is None else out.detach().clone()
        return out

    gdr_chunk.chunk_gated_delta_rule_fwd_h_pre_process = _counting_pre
    if rank == 0:
        print(f"# world_size={world} FLA_CP_PRECISION={prec} dtype={args.dtype} "
              f"hv={args.hv} head_k={args.head_k} head_v={args.head_v}", flush=True)
        print(f"# gpu={torch.cuda.get_device_name(0)}", flush=True)

    kw = dict(initial_state=None, output_final_state=False, use_qk_l2norm_in_kernel=True)

    for total_len in args.lengths:
        assert total_len % (world * 64) == 0, f"{total_len} must be a multiple of world*64"
        for g_scale in args.g_scales:
            q, k, v, g, beta = make_inputs(total_len, args.hv, args.head_k, args.head_v, device, dtype, g_scale)
            cu = torch.tensor([0, total_len], device=device, dtype=torch.int32)
            cu_cpu = cu.cpu()

            # --- reference: the non-CP path, i.e. exactly what SP=1 computes ---
            o_ref, _ = chunk_gated_delta_rule(q, k, v, g=g, beta=beta, cu_seqlens=cu, **kw)

            # --- test: the CP path on this rank's contiguous slice -------------
            part = total_len // world
            sl = slice(rank * part, (rank + 1) * part)
            ctx = build_cp_context(cu, group=dist.group.WORLD, cu_seqlens_cpu=cu_cpu)
            o_loc, _ = chunk_gated_delta_rule(
                q[:, sl], k[:, sl], v[:, sl], g=g[:, sl], beta=beta[:, sl], cp_context=ctx, **kw
            )

            # --- the sensitive test: the fp32 state rank r inherited from the
            # affine chain, vs the exact state the sequential path reaches at
            # the same boundary. This IS the quantity fla PR #1180 is about.
            if rank > 0:
                _, ref_state = chunk_gated_delta_rule(
                    q[:, : rank * part], k[:, : rank * part], v[:, : rank * part],
                    g=g[:, : rank * part], beta=beta[:, : rank * part],
                    cu_seqlens=torch.tensor([0, rank * part], device=device, dtype=torch.int32),
                    initial_state=None, output_final_state=True, use_qk_l2norm_in_kernel=True,
                )
                sg = gap(probe["state"].reshape(ref_state.shape), ref_state)
                state_err = torch.tensor([sg["max_abs"], sg["mean_abs"], sg["rel_mean"]], device=device)
            else:
                state_err = torch.zeros(3, device=device)
            err_all = torch.empty(world, 3, device=device)
            dist.all_gather_into_tensor(err_all, state_err)

            # The inherited state lives on the later ranks; rank 0 starts from
            # zero by construction, so only gathering rank 0 tells us nothing.
            st = torch.tensor([probe["state_absmax"] or 0.0], device=device)
            st_all = torch.empty(world, 1, device=device)
            dist.all_gather_into_tensor(st_all, st)

            gathered = torch.empty(world, *o_loc.shape, device=device, dtype=o_loc.dtype)
            dist.all_gather_into_tensor(gathered, o_loc.contiguous())
            o_cp = torch.cat([gathered[i] for i in range(world)], dim=1)

            if rank == 0:
                # Surviving state mass a later rank inherits: if this underflows,
                # CP is mathematically a no-op and the test measures nothing.
                slice_decay = g[:, :part].sum(dim=1).mean().exp().item()
                rec = {
                    "len": total_len, "world": world, "prec": prec, "g_scale": g_scale,
                    "slice_decay": slice_decay,
                    "is_last_rank": ctx.is_last_rank, "pre_ranks": ctx.pre_num_ranks,
                    "pre_calls": probe["calls"],
                    "inherited_state_by_rank": [round(x, 6) for x in st_all.flatten().tolist()],
                    "state_rel_err_by_rank": [round(x, 9) for x in err_all[:, 2].tolist()],
                    "state_max_abs_err_by_rank": [round(x, 9) for x in err_all[:, 0].tolist()],
                    **gap(o_cp, o_ref),
                }
                # Per-rank breakdown: error should grow with rank (longer prefix chain).
                per_rank = [gap(gathered[i], o_ref[:, i * part:(i + 1) * part])["mean_abs"] for i in range(world)]
                rec["mean_abs_by_rank"] = [round(x, 9) for x in per_rank]
                print(json.dumps(rec), flush=True)
            dist.barrier()

    dist.destroy_process_group()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
