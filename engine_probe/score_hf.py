"""Stage 2: 用 HF/torch 路径在 stage 1 固定下来的 token 上重算 logprob, 与 vLLM 比对。

复刻 verl/utils/debug/metrics.py 的口径:
    probs_diff = mean| exp(lp_hf) - exp(lp_vllm) |     <- 概率空间
    pearson    = corrcoef(exp(lp_hf), exp(lp_vllm))
额外给出按 token 位置分箱的曲线 (训练指标里没有, 用来判断误差是否沿 recurrent state 累积)。
"""
import argparse, json, gc, torch, torch.nn.functional as F
from transformers import AutoModelForCausalLM

P = argparse.ArgumentParser()
P.add_argument("--seqs", default="/tmp/probe/seqs.pt")
P.add_argument("--out", default="/tmp/probe/result.json")
P.add_argument("--bin", type=int, default=2000)
P.add_argument("--attn", default="sdpa,flash_attention_2")
a = P.parse_args()

blob = torch.load(a.seqs, weights_only=False)
MODEL, RECS = blob["model"], blob["recs"]
print(f"model={MODEL}  n_seq={len(RECS)}", flush=True)

try:
    from causal_conv1d import causal_conv1d_fn as DAO_CONV
    print("causal_conv1d (Dao CUDA ext): AVAILABLE", flush=True)
except Exception as e:
    DAO_CONV = None
    print(f"causal_conv1d (Dao CUDA ext): MISSING ({type(e).__name__})", flush=True)


def load_model(attn):
    """Qwen3.5 是 VL 架构 (Qwen3_5ForConditionalGeneration), 权重前缀 model.language_model.*。
    按序尝试几种 loader, 并显式检查权重是否真的加载上了 —— 静默的 key mismatch 会让
    整个探针测出一个假的巨大偏差。"""
    from transformers import AutoConfig
    import transformers as tf
    cands = []
    for name in ("Qwen3_5ForConditionalGeneration", "Qwen3_5ForCausalLM"):
        if hasattr(tf, name):
            cands.append((name, getattr(tf, name)))
    cands.append(("AutoModelForCausalLM", AutoModelForCausalLM))
    for name, cls in cands:
        try:
            m = cls.from_pretrained(MODEL, dtype=torch.bfloat16,
                                    attn_implementation=attn, trust_remote_code=True)
        except Exception as e:
            print(f"  loader {name}: {type(e).__name__}: {str(e)[:160]}", flush=True)
            continue
        # 校验: 随便取一个 linear-attention 层的 conv1d 权重, 全零/未初始化就是没加载上
        bad = [n for n, p in m.named_parameters()
               if p.numel() and not torch.isfinite(p).all()]
        convw = [p for n, p in m.named_parameters() if n.endswith("conv1d.weight")]
        if bad:
            print(f"  loader {name}: {len(bad)} 个参数含 nan/inf, 跳过", flush=True); continue
        if not convw:
            print(f"  loader {name}: 找不到 conv1d.weight, 跳过", flush=True); continue
        if convw[0].abs().max().item() == 0.0:
            print(f"  loader {name}: conv1d.weight 全零 => 权重没加载上, 跳过", flush=True); continue
        print(f"  loader {name}: OK  (conv1d.weight absmax={convw[0].abs().max().item():.4f})", flush=True)
        return m.cuda().eval()
    print(f"!! attn={attn}: 所有 loader 都失败", flush=True)
    return None


def conv_modules(model):
    return [m for m in model.modules() if hasattr(m, "causal_conv1d_fn")]


@torch.no_grad()
def logprobs_for(model, rec):
    """整段一次前向 (recurrent state 不能切), lm_head 分块以省显存。"""
    ids = torch.cat([rec["prompt_ids"], rec["resp_ids"]]).unsqueeze(0).cuda()
    n_resp = rec["resp_ids"].numel()
    dec = model.get_decoder()
    h = dec(input_ids=ids).last_hidden_state[0]          # (L, H)
    head = model.get_output_embeddings()
    # 预测第 t 个 response token 的位置是 t-1
    start = ids.shape[1] - n_resp - 1
    h = h[start : start + n_resp]
    tgt = rec["resp_ids"].cuda()
    out = torch.empty(n_resp, dtype=torch.float64, device="cuda")
    CH = 2048
    for i in range(0, n_resp, CH):
        lg = head(h[i : i + CH]).float()
        out[i : i + CH] = torch.log_softmax(lg, -1).gather(
            1, tgt[i : i + CH].unsqueeze(1)).squeeze(1).double()
        del lg
    return out.cpu()


def metrics(hf_lp, vl_lp, bin_size):
    p_hf, p_vl = hf_lp.exp(), vl_lp.exp()
    d = (p_hf - p_vl).abs()
    corr = torch.corrcoef(torch.stack([p_hf, p_vl]))[0, 1].item()
    bins = {}
    for lo in range(0, d.numel(), bin_size):
        seg_d = d[lo : lo + bin_size]
        seg_l = (hf_lp - vl_lp)[lo : lo + bin_size]
        bins[f"{lo // 1000}k-{min(lo + bin_size, d.numel()) // 1000}k"] = {
            "probs_diff": round(seg_d.mean().item(), 6),
            "logp_diff": round(seg_l.mean().item(), 6),
            "n": seg_d.numel(),
        }
    return {
        "probs_diff_mean": d.mean().item(), "probs_diff_max": d.max().item(),
        "probs_diff_std": d.std().item(), "pearson": corr,
        "hf_log_ppl": -hf_lp.mean().item(), "vllm_log_ppl": -vl_lp.mean().item(),
        "logp_diff_mean": (hf_lp - vl_lp).mean().item(), "bins": bins,
    }


results = {}
for attn in [x for x in a.attn.split(",") if x]:
    model = load_model(attn)
    if model is None:
        continue

    cms = conv_modules(model)
    gdn = type(getattr(cms[0], "chunk_gated_delta_rule", None)).__name__ if cms else "?"
    print(f"\n### attn={attn}  conv_modules={len(cms)}  gdn_fn={gdn}", flush=True)

    variants = [("torch_fallback", None)]
    if DAO_CONV is not None:
        variants.append(("dao_causal_conv1d", DAO_CONV))

    for vname, fn in variants:
        for m in cms:
            m.causal_conv1d_fn = fn
        agg_hf, agg_vl = [], []
        per_bin = {}
        for i, rec in enumerate(RECS):
            hf = logprobs_for(model, rec)
            vl = rec["vllm_logprobs"]
            agg_hf.append(hf); agg_vl.append(vl)
            print(f"  [{attn}/{vname}] seq{i}: hf_lp={hf.mean():.4f} vllm_lp={vl.mean():.4f}", flush=True)
        M = metrics(torch.cat(agg_hf), torch.cat(agg_vl), a.bin)
        # 分箱要按序列内位置对齐, 上面 cat 会串位置 -> 重算
        bb = {}
        for rec, hf in zip(RECS, agg_hf, strict=True):
            for lo in range(0, hf.numel(), a.bin):
                k = f"{lo // 1000}k"
                dd = (hf[lo:lo+a.bin].exp() - rec["vllm_logprobs"][lo:lo+a.bin].exp()).abs()
                bb.setdefault(k, []).append(dd)
        M["bins"] = {k: {"probs_diff": round(torch.cat(v).mean().item(), 6),
                         "n": int(torch.cat(v).numel())} for k, v in sorted(bb.items(), key=lambda x: int(x[0][:-1]))}
        key = f"{attn}/{vname}"
        results[key] = M
        print(f"  ==> {key}: probs_diff_mean={M['probs_diff_mean']:.5f} "
              f"pearson={M['pearson']:.4f} hf_ppl={M['hf_log_ppl']:.4f} vllm_ppl={M['vllm_log_ppl']:.4f}", flush=True)
        print(f"      bins: {json.dumps(M['bins'])}", flush=True)

    del model; gc.collect(); torch.cuda.empty_cache()

json.dump({"model": MODEL, "results": results}, open(a.out, "w"), indent=2)
print(f"\nWROTE {a.out}")
print(json.dumps({k: {kk: vv for kk, vv in v.items() if kk != 'bins'} for k, v in results.items()}, indent=2))
