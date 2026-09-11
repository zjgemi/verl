#!/usr/bin/env python3
"""Does triton accept input_precision={ieee,tf32x3,tf32} on this GPU, and does it matter?

This is the one thing patch_fla_cp_precision.py cannot check without a GPU: the
patched tl.dot calls have to actually JIT-compile. Ampere has no TMA, so the
tf32x3 path is the interesting one -- fla's own solve_tril.py only autotunes
over 'ieee' when TMA is absent, which hints tf32x3 may be TMA-gated.

Reference is fp64 on CUDA cores.
"""

import torch
import triton
import triton.language as tl


@triton.jit
def _dot_kernel(A, B, C, M: tl.constexpr, N: tl.constexpr, K: tl.constexpr, PREC: tl.constexpr):
    off_m = tl.arange(0, M)
    off_n = tl.arange(0, N)
    off_k = tl.arange(0, K)
    a = tl.load(A + off_m[:, None] * K + off_k[None, :])
    b = tl.load(B + off_k[:, None] * N + off_n[None, :])
    c = tl.dot(a, b, input_precision=PREC)
    tl.store(C + off_m[:, None] * N + off_n[None, :], c)


def run(prec: str, a: torch.Tensor, b: torch.Tensor) -> torch.Tensor:
    M, K = a.shape
    _, N = b.shape
    c = torch.empty((M, N), device=a.device, dtype=torch.float32)
    _dot_kernel[(1,)](a, b, c, M=M, N=N, K=K, PREC=prec)
    return c


def main() -> int:
    torch.manual_seed(0)
    dev = "cuda"
    M = N = K = 64
    a = torch.randn(M, K, device=dev, dtype=torch.float32)
    b = torch.randn(K, N, device=dev, dtype=torch.float32)
    ref = (a.double() @ b.double())

    print(f"gpu={torch.cuda.get_device_name(0)}  cc={torch.cuda.get_device_capability(0)}")
    print(f"triton={triton.__version__}  torch={torch.__version__}")
    print()

    rc = 0
    for prec in ["tf32", "tf32x3", "ieee"]:
        try:
            c = run(prec, a, b)
        except Exception as e:  # compile or launch failure
            print(f"{prec:8s} FAILED: {type(e).__name__}: {str(e)[:300]}")
            if prec == "ieee":
                rc = 1
            continue
        err = (c.double() - ref).abs().max().item()
        rel = err / ref.abs().max().item()
        print(f"{prec:8s} ok   max_abs_err={err:.3e}  rel={rel:.3e}")
    return rc


if __name__ == "__main__":
    raise SystemExit(main())
