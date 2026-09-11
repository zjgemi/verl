"""Stage 1: vLLM 生成固定 token 序列 + 记录 per-token logprob。
输出 /tmp/probe/seqs.pt，供 stage 2 的 HF 前向逐条比对。
独立进程运行，避免 vLLM 显存释放不干净影响 HF。
"""
import argparse, os, torch
from transformers import AutoTokenizer
from vllm import LLM, SamplingParams

def main():
    P = argparse.ArgumentParser()
    P.add_argument("--model", required=True)
    P.add_argument("--n-seq", type=int, default=8)
    P.add_argument("--max-tokens", type=int, default=20000)
    P.add_argument("--gpu-mem", type=float, default=0.85)
    P.add_argument("--out", default="/tmp/probe/seqs.pt")
    a = P.parse_args()

    PROMPTS = [
        "Implement a full conjugate-gradient solver for sparse SPD systems in pure Python, "
        "including preconditioning, convergence diagnostics, and an extensive test suite. "
        "Explain every design decision in detail as you go.",
        "Write a complete molecular dynamics engine: neighbour lists, Verlet integration, "
        "Lennard-Jones and Coulomb interactions, thermostats. Derive the equations first.",
        "Design and implement a from-scratch automatic differentiation library with reverse mode, "
        "broadcasting, and a small neural network trained on it. Reason carefully about each step.",
        "Build a finite element solver for 2D Poisson problems: mesh generation, assembly, "
        "boundary conditions, sparse solve, and error analysis against an analytic solution.",
        "Implement a complete SAT solver with DPLL, unit propagation, clause learning, and VSIDS. "
        "Walk through the algorithm design in depth before writing code.",
        "Write a numerical library for spherical harmonics: recurrences, normalisation, rotation, "
        "and validation against known identities. Be exhaustive.",
        "Implement a lattice Boltzmann fluid simulation with D2Q9, bounce-back walls, and "
        "visualisation. Derive the collision operator step by step.",
        "Build a symbolic algebra system: expression trees, simplification, differentiation, "
        "and polynomial factorisation. Discuss the representation trade-offs at length.",
    ]
    PROMPTS = (PROMPTS * ((a.n_seq // len(PROMPTS)) + 1))[: a.n_seq]

    tok = AutoTokenizer.from_pretrained(a.model, trust_remote_code=True)
    prompts = [
        tok.apply_chat_template([{"role": "user", "content": p}], tokenize=False, add_generation_prompt=True)
        for p in PROMPTS
    ]

    llm = LLM(
        model=a.model, trust_remote_code=True, dtype="bfloat16",
        gpu_memory_utilization=a.gpu_mem,
        max_model_len=a.max_tokens + 2048,
        enforce_eager=False,
    )
    # 与 rollout 完全一致的采样设置 (rollout.yaml: temperature=1.0, top_p=1, top_k=-1)
    sp = SamplingParams(
        temperature=1.0, top_p=1.0, top_k=-1,
        max_tokens=a.max_tokens, ignore_eos=True,   # 强制跑满长度, 这是自变量
        logprobs=0, seed=1234,
    )
    outs = llm.generate(prompts, sp)

    recs = []
    for o in outs:
        c = o.outputs[0]
        lps = [lp[t].logprob for t, lp in zip(c.token_ids, c.logprobs, strict=True)]
        recs.append({
            "prompt_ids": torch.tensor(o.prompt_token_ids, dtype=torch.long),
            "resp_ids": torch.tensor(c.token_ids, dtype=torch.long),
            "vllm_logprobs": torch.tensor(lps, dtype=torch.float64),
        })
        print(f"seq: prompt={len(o.prompt_token_ids)} resp={len(c.token_ids)} "
              f"mean_lp={sum(lps)/len(lps):.4f}", flush=True)

    os.makedirs(os.path.dirname(a.out), exist_ok=True)
    torch.save({"model": a.model, "recs": recs}, a.out)
    print(f"WROTE {a.out}  n={len(recs)}")


if __name__ == "__main__":
    main()
