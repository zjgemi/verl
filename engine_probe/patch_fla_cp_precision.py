#!/usr/bin/env python3
"""Raise the precision of FLA's context-parallel affine chain.

Background
----------
Under Ulysses SP the Qwen3.5 GatedDeltaNet runs through ``fla.ops.cp``, which
turns the recurrence into a 3-kernel prefix scan: each rank summarises its slice
into affine parameters (M, H), the summaries are all-gathered, and every rank
composes the prefix. Both the per-chunk composition of M and the cross-rank
transform run through ``tl.dot(f32, f32)``, and Triton's default for fp32 inputs
is tf32 -- 10 mantissa bits. The non-CP path never does this: it propagates the
state with f32 fma on CUDA cores. So SP>1 loses precision that SP=1 does not,
and the loss accumulates with sequence length.

Upstream fixed this in fla PR #1180 ("[CP] use tf32x3 affine chain in kcp",
merged 2026-08-27), but that landed after 0.5.2 and is in no release yet.

This patch adds the same knob to an installed fla, following the convention
fla already uses for ``FLA_TRIL_PRECISION`` (ops/utils/solve_tril.py:18):

    FLA_CP_PRECISION = ieee | tf32x3 | tf32      (default: tf32)

The default reproduces current behaviour bit-for-bit, so a patched install is a
valid control arm.

Usage
-----
    python3 patch_fla_cp_precision.py           # apply
    python3 patch_fla_cp_precision.py --check   # report status, change nothing
"""

import argparse
import os
import re
import shutil
import sys

MARKER = "FLA_CP_PRECISION"

HEADER = '''import os

# --- verl engine-bias patch: context-parallel affine chain precision ---------
# Triton defaults fp32 tl.dot to tf32 (10 mantissa bits). The CP prefix scan
# composes transition matrices across a whole rank slice, so that error
# accumulates with sequence length -- the non-CP path does not pay it.
# See fla PR #1180. Default 'tf32' == upstream 0.5.1 behaviour.
FLA_CP_PRECISION_STR = os.environ.get('FLA_CP_PRECISION', 'tf32')
assert FLA_CP_PRECISION_STR in ['ieee', 'tf32', 'tf32x3'], \\
    f"FLA_CP_PRECISION must be one of 'ieee', 'tf32', or 'tf32x3', but got {FLA_CP_PRECISION_STR}"
# A @jit'ed kernel cannot read a plain module global -- Triton raises
# NameError("Cannot access global variable ... from within @jit'ed function")
# unless it is instantiated as tl.constexpr. (fla's own solve_tril.py takes the
# other route and threads DOT_PRECISION through the autotune configs.)
FLA_CP_PRECISION = tl.constexpr(FLA_CP_PRECISION_STR)
# -----------------------------------------------------------------------------
'''

# The six affine-chain matmuls, verified to live in exactly the three kernels
# fla PR #1180 patches: pre_process_fwd_kernel_merged,
# pre_process_bwd_kernel_merged and merge_fwd_bwd_kernel.
SITES = [
    "b_m = tl.dot(b_m_i.to(tl.float32), b_m.to(tl.float32))",
    "b_h = tl.dot(b_h.to(tl.float32), tl.trans(b_m)) + tl.trans(b_he)",
    "b_h = tl.dot(b_m.to(tl.float32), b_h.to(tl.float32)) + b_he.to(tl.float32)",
    "b_h = tl.dot(b_h.to(tl.float32), tl.trans(b_ag_m).to(tl.float32)) + tl.trans(b_ag_h).to(tl.float32)",
    "b_h = tl.dot(b_ag_m.to(tl.float32), b_h.to(tl.float32)) + b_ag_h.to(tl.float32)",
]

EXPECTED_HITS = 6  # site[0] appears twice (fwd + bwd kernel); the rest once each


def target_path() -> str:
    import fla.ops.cp.chunk_delta_h as m

    return m.__file__


def add_precision(call: str) -> str:
    """Append input_precision=FLA_CP_PRECISION to the tl.dot(...) in `call`."""
    start = call.index("tl.dot(")
    i = start + len("tl.dot(")
    depth = 1
    while depth:
        if call[i] == "(":
            depth += 1
        elif call[i] == ")":
            depth -= 1
        i += 1
    close = i - 1
    return f"{call[:close]}, input_precision=FLA_CP_PRECISION{call[close:]}"


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--check", action="store_true")
    args = ap.parse_args()

    path = target_path()
    src = open(path).read()

    if MARKER in src:
        n = src.count("input_precision=FLA_CP_PRECISION")
        print(f"already patched: {path} ({n} call sites)")
        return 0 if n == EXPECTED_HITS else 1
    if args.check:
        print(f"NOT patched: {path}")
        return 1

    total = 0
    for site in SITES:
        n = src.count(site)
        if n == 0:
            print(f"ERROR: pattern not found, refusing to patch:\n  {site}", file=sys.stderr)
            return 1
        src = src.replace(site, add_precision(site))
        total += n

    if total != EXPECTED_HITS:
        print(f"ERROR: expected {EXPECTED_HITS} call sites, matched {total}", file=sys.stderr)
        return 1

    # Insert after the whole import block: the header needs `tl` in scope, and
    # the module does not import os itself.
    anchor = "from fla.utils import USE_CUDA_GRAPH, autotune_cache_kwargs, check_shared_mem\n"
    if src.count(anchor) != 1:
        print(f"ERROR: expected exactly one '{anchor.strip()}' anchor", file=sys.stderr)
        return 1
    src = src.replace(anchor, anchor + "\n" + HEADER, 1)

    shutil.copy2(path, path + ".orig")
    with open(path, "w") as f:
        f.write(src)

    print(f"patched {path} ({total} call sites); backup at {os.path.basename(path)}.orig")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
