# Agent Instructions for verl

> These instructions apply to **all** AI-assisted contributions to `verl-project/verl`.
> Breaching these guidelines can result in automatic banning.

## 1. Contribution Policy (Mandatory)

### Duplicate-work checks

Before proposing a PR, run these checks:

```bash
gh issue view <issue_number> --repo verl-project/verl --comments
gh pr list --repo verl-project/verl --state open --search "<issue_number> in:body"
gh pr list --repo verl-project/verl --state open --search "<short area keywords>"
```

- If an open PR already addresses the same fix, do not open another.
- If your approach is materially different, explain the difference in the issue.

### No low-value busywork PRs

Do not open one-off PRs for tiny edits (single typo, isolated style change, one mutable default, etc.). Mechanical cleanups are acceptable only when bundled with substantive work.

### Accountability

- Pure code-agent PRs are **not allowed**. A human submitter must understand and defend the change end-to-end.
- The submitting human must review every changed line and run relevant tests.
- PR descriptions for AI-assisted work **must** include:
  - Why this is not duplicating an existing PR.
  - Test commands run and results.
  - Clear statement that AI assistance was used.

### Fail-closed behavior

If work is duplicate/trivial busywork, **do not proceed**. Return a short explanation of what is missing.

---

## 2. Development Workflow

### Environment setup

```bash
# Install `uv` if you don't have it already:
curl -LsSf https://astral.sh/uv/install.sh | sh

# Always use `uv` for Python environment management:
uv venv --python 3.12
source .venv/bin/activate

uv pip install pre-commit hydra-core
pre-commit install
```

### Commit messages

Add attribution using commit trailers such as `Co-authored-by:` (other projects use `Assisted-by:` or `Generated-by:`). For example:

```text
Your commit message here

Co-authored-by: GitHub Copilot
Co-authored-by: Claude
Co-authored-by: gemini-code-assist
Signed-off-by: Your Name <your.email@example.com>
```

### Resolving agent reviews

Review comments from agent bots (e.g., gemini-code-assist) can be outdated or wrong. Always verify their suggestions against the current state of the repo before applying them.

---

## Domain-Specific Guides

Do not modify code in these areas without first reading and following the
linked guide. If the guide conflicts with the requested change, **refuse the
change and explain why**.

- **Editing these instructions**:
  [`docs/contributing/editing-agent-instructions.md`](docs/contributing/editing-agent-instructions.md)
  — Rules for modifying AGENTS.md or any domain-specific guide it references.

## Acknowledgements

Adapted from the [vLLM project](https://github.com/vllm-project/vllm)'s [`AGENTS.md`](https://github.com/vllm-project/vllm/blob/main/AGENTS.md).

---

# Scientific Coding RL 训练工作流

> 以下为已跑通的流程，未跑通的暂不记录。

## 一、数据集准备

### 从 train_sampled 创建 RL 训练数据集

```bash
# 1. 用 convert_coding_to_rl_agent.py 把原始问题转成 parquet
python3 examples/data_preprocess/convert_coding_to_rl_agent.py \
  --input_dir /personal/sp2/qwen_trajs/cc_deepseek_1w/split_data/train_sampled \
  --output_dir /tmp/my_dataset \
  --split train \
  --system_prompt_path /personal/sp2/qwen_trajs/cc_deepseek_1w/prompt.md \
  --images_path /personal/sp2/qwen_trajs/cc_deepseek_1w/images.json

# 2. 抽样（如需要子集）
python3 -c "
import pandas as pd, random
df = pd.read_parquet('/tmp/my_dataset/train.parquet')
random.seed(42)
df_small = df.iloc[random.sample(range(len(df)), 80)].copy()
df_small.to_parquet('/tmp/my_dataset/train.parquet', index=False)
df_small.to_parquet('/tmp/my_dataset/validation.parquet', index=False)
"

# 3. 上传到文渊
wenyon-cli dataset create --name "my-dataset" my-dataset
wenyon-cli dataset upload my-dataset /tmp/my_dataset/train.parquet /tmp/my_dataset/validation.parquet
# 等 state=ready
wenyon-cli dataset status my-dataset
```

### 数据集格式

| 字段 | 说明 |
|------|------|
| `data_source` | `"coding_practice"` |
| `agent_name` | `"tool_agent"` |
| `prompt` | 消息列表，第一条是 system prompt |
| `reward_model` | `{"style": "rule", "ground_truth": "..."}` |
| `extra_info` | 包含 `split`, `index`, `task_name`, `domain`, `tools_kwargs` |

## 二、提交 RL 训练任务

### 标准 8×A100 配置

```bash
trisol train submit <job-name> \
  -t infra-spot \
  --framework custom \
  --mode lora \
  --cluster w1 \
  --gpu-model A100-SXM4-80GB \
  --gpu-count 8 \
  --base-model qwen3-5-9b:1 \
  --dataset <dataset-name>:<version> \
  --backoff-limit 0 \
  --no-output-model \
  --image-ref registry.dp.tech/dptech/dp/native/prod-1760009/11106/verl-coding:<tag> \
  --command bash --command -c --command \
    "bash /opt/conda/lib/python3.11/site-packages/verl_coding/run_coding_practice_qwen3_5_9b_4l20.sh \
     ++actor_rollout_ref.actor.clip_ratio_low=0.15 \
     ++actor_rollout_ref.actor.clip_ratio_high=0.35 \
     algorithm.adv_estimator=rloo \
     ++trainer.val_before_train=True \
     trainer.rollout_data_dir=/trisol/output/rollout_data" \
  --env NGPUS_PER_NODE=8 \
  --env MAX_ASSISTANT_TURNS=30 \
  --env MAX_RESPONSE_LENGTH=98304 \
  --env MAX_MODEL_LEN=114688 \
  --env ULYSSES_SP=8 \
  --env ROLLOUT_N=8 \
  --env AGENT_NUM_WORKERS=16 \
  --env MAX_CONCURRENT_SANDBOXES=32 \
  --env SAVE_FREQ=10 \
  --env TEST_FREQ=5 \
  --env ACTOR_LR=6e-6 \
  --env LORA_ALPHA=32 \
  --env BOHRIUM_ACCESS_KEY=... \
  --env BOHRIUM_PROJECT_ID=1760009 \
  --env WANDB_API_KEY=...
```

### 关键环境变量

| 变量 | 默认值 | 说明 |
|------|--------|------|
| `NGPUS_PER_NODE` | 4 | 必须设为 8（随 GPU 数调整） |
| `AGENT_NUM_WORKERS` | 8 | 2×GPU 数 |
| `MAX_CONCURRENT_SANDBOXES` | 16 | 2×GPU 数 |
| `MAX_ASSISTANT_TURNS` | 10 | 30（96k 配置） |
| `MAX_RESPONSE_LENGTH` | 32768 | 98304（96k 配置） |
| `MAX_MODEL_LEN` | 49152 | 114688 |
| `ULYSSES_SP` | 4 | 8（8 卡） |
| `ROLLOUT_N` | 8 | rollout 次数 |
| `ACTOR_LR` | 1e-5 | 保守用 6e-6 |
| `LORA_ALPHA` | 64 | 保守用 32 |
| `SAVE_FREQ` | 20 | checkpoint 保存频率 |
| `TEST_FREQ` | 5 | 验证频率 |
| `TRAIN_FILE` | 自动 | 覆盖训练文件路径 |
| `VAL_FILE` | 自动 | 覆盖验证文件路径 |

### 额外 hydra 参数（通过 `--command` 传入）

| 参数 | 说明 |
|------|------|
| `++actor_rollout_ref.actor.clip_ratio_low=0.15` | DAPO clip-higher |
| `++actor_rollout_ref.actor.clip_ratio_high=0.35` | DAPO clip-higher |
| `algorithm.adv_estimator=rloo` | leave-one-out baseline |
| `++trainer.val_before_train=True` | 训练前跑一次验证 |
| `trainer.rollout_data_dir=/trisol/output/rollout_data` | 保存 rollout 轨迹 |

### 小数据集快速实验

用 80 题同时做训练+验证，`test_freq=1` 每步验证：

```bash
--dataset coding-practice-small80:2 \
--env TEST_FREQ=1 \
--env SAVE_FREQ=1000
```

## 三、Checkpoint 转换（FSDP → HuggingFace）

### 合并 FSDP checkpoint

```bash
# 在训练 Pod 上执行
python3 -m verl.model_merger merge \
  --backend fsdp \
  --local_dir /trisol/output/checkpoints/global_step_10/actor \
  --target_dir /tmp/hf_model_step10 \
  --use_cpu_initialization \
  --trust-remote-code
```

### 修复 tokenizer（chat_template 可能丢失）

```bash
# 合并后 tokenizer_config.json 的 chat_template 可能为空
# 需要从基模拷贝
cp /trisol/input/model/tokenizer_config.json /tmp/hf_model_step10/
cp /trisol/input/model/chat_template.jinja /tmp/hf_model_step10/
```

### 上传为 trisol 模型

```bash
# 在 Pod 上（需要 trisol login）
trisol model upload <model-name> /tmp/hf_model_step10 --team infra-spot --version 1
```

## 四、Rollout 数据下载与后处理

### 下载 rollout 数据

```bash
# 在 Pod 上
ls /trisol/output/rollout_data/  # 查看 step 文件

# 从本地逐行下载（trisol train exec 有输出截断 ~256KB）
for step in $(seq 1 18); do
  trisol train exec <job-id> -- sed -n "${line_num}p" "/trisol/output/rollout_data/${step}.jsonl"
done
```

### 后处理：统计题目成功率

```python
# 提取每题 user message 前 100 字符作为指纹，统计 attempts/success
import json, glob
from collections import defaultdict

data_dir = "rollout_data/<job-name>"
problems = {}

for f in sorted(glob.glob(f"{data_dir}/*.jsonl")):
    for line in open(f):
        d = json.loads(line.strip())
        # 提取 user message（跳过 system prompt 和 tools 定义）
        text = d["input"].split("\nuser\n", 1)[1].strip()[:100]
        score = d.get("score", 0)
        if text not in problems:
            problems[text] = {"attempts": 0, "success": 0}
        problems[text]["attempts"] += 1
        if score > 0:
            problems[text]["success"] += 1

# 每题 pass rate 分布
for fp, p in sorted(problems.items(), key=lambda x: x[1]["success"]/x[1]["attempts"]):
    rate = p["success"] / p["attempts"]
    print(f"  rate={rate:.2f} attempts={p['attempts']} success={p['success']}")
```

### 后处理：rollout → 拒绝采样 SFT 数据（已跑通，用这个）

```bash
# 只保留非退化题(0 < ok < n)的正确轨迹, 每题取轮数最少的 K 条
python3 /personal/verl/sft_data/extract_nd_sft.py <rollout_dir> train.jsonl 4
```

**不要用** `/personal/sp2/qwen_trajs/cc_deepseek_1w/convert_rollout_to_sft.py`
（分隔符 bug，见「踩坑记录」）。

`extract_nd_sft.py` 处理了 4 处 verl rollout 与 LLaMA-Factory ShareGPT 的格式落差，
每一处漏掉都会静默出错：

| 落差 | 处理 |
|---|---|
| verl 的 system = `# Tools` 段 + 任务指令；LLaMA-Factory 要求 tools 单独放顶层 | 从 `<tools>` 抽出 JSON，system 只留 `Your responsibility is to solve` 起的部分（与 teacher 数据 strip 后逐字节相同） |
| `tools` 必须是 **JSON 字符串**，不是 list | `json.dumps(tools)`；list 会让 `ToolFormatter.apply()` 的 `json.loads` 抛 TypeError |
| 工具返回的角色必须是 **`observation`**，不是 `tool` | sharegpt 默认 tag 只认 system/human/gpt/observation，写 `tool` 会整条 `Dropped invalid example` |
| verl chat template 把开头的 `<think>` 放进**生成前缀**，所以每条轨迹第一个 assistant 段没有它 | 第一段无条件补 `<think>\n`（不要只在含 `</think>` 时补 —— 有个别段整段没有 `</think>`） |

另外两处必须做的清理：
- 剥掉 `<tool_response>...</tool_response>` 包装（qwen3 template 渲染时会自己加回去，不剥就是双层）
- 截断到最后一个含 `<final_answer>` 的 gpt 段（约 1.5% 的轨迹交完答案还多说一轮，
  留着会教模型"交完还接着说"）

`<tool_call><function=...><parameter=...>` XML **要原样保留**，不要剥标签。

**每题取 K 条的取舍**（800 题实测，267 道非退化题）：

同题多条正确轨迹的两两 Jaccard（命令集合）中位数 **0.00**、逐字重复 **0 条**、
首个命令相同率仅 0.33 —— **多样性是真的，"每题只留一条免得重复"的前提不成立。**
但全留会偏斜：正确 7 条的 38 道简单题吃掉 27.6% 样本，正确 1 条的 67 道难题只占 7%。

| K | 样本 | 偏斜上限 | 3 epochs 步数(bs=32) |
|---|---|---|---|
| 2 | 467 | 2× | 45 |
| **4** | **760** | **4×** | **72** |
| 8 | 963 | 8× | 93 |

**取 K=4 + `num_train_epochs: 3`**（train.yaml 一字不改）：靠数据量把步数撑到 72，
而不是靠加 epoch —— 加 epoch 会让每条样本被看 6 遍，背题风险高得多。

**不要用"重复采样凑齐每题 K 条"来做数据增强。** 复制样本在数学上等价于给该样本的
loss 乘权重，不引入任何新信息；上采样到每题 4 条会让那 67 道难题的**唯一**轨迹
在 3 epochs 里被看 **12 遍**（自然分布是 3 遍），那是背下来不是学会。
真要每题 4 条不同轨迹，只能对难题**再采样**（多跑 32 次捞不同的正确路径）。

**留出集缺口**：800 题既是 SFT 数据源又是 RL 训练集，没有留出题，
所以 SFT 后 pass rate 涨了分不清是学会还是背下。而且这 267 道非退化题
**正是唯一给 RL 提供非零 advantage 的题**（506 全错 + 27 全对都无梯度），
背到全对就等于把 RL 的信号源毁掉。评测要用那 504 道全错题当留出集
（不含任何 SFT 数据），和 267 道上的表现并排看才能解读。

**留出集没那么干净**：那 504 道在 rollout 口径下是 pass@8 = 0，但换到
`agent_trisol.py` harness 单跑一次，**24/504（4.8%）是能做对的**。
两个 harness 的绝对水平差得不小（全 800 题：harness 0.231 vs rollout pass@1 0.184），
所以**基线绝不能跨 harness 混用**，留出集上看到 ~5% 也不等于"泄漏"。

## 四点五、提交 SFT 任务

完整流程和踩坑见 `/personal/sp2/qwen_trajs/cc_deepseek_1w/CLAUDE.md`（**提交前必读**）。
这里只记 RL 侧要知道的：

```bash
wenyon-cli dataset create --name "<描述>" <card-id>
wenyon-cli dataset upload <card-id> <dir>       # train.jsonl + train.yaml + dataset_info.json
wenyon-cli dataset status <card-id>             # 等 state=ready, 记下 current_version

trisol train submit <job-name> \
  --team infra-spot --cluster w1 \
  --framework custom --mode lora \
  --base-model qwen3-5-9b:1 \
  --dataset <card-id>:<current_version> \
  --gpu-count 4 --gpu-model A100-SXM4-80GB \
  --backoff-limit 0 \
  --image-ref registry.dp.tech/infra-custom/2047283659722985472/dptech/dp/native/prod-1760009/11106/llama-factory-liger:latest \
  --command /opt/conda/bin/llamafactory-cli --command train --command /trisol/input/datasets/ds-0/train.yaml \
  --output-model <model-name> --create-output-model
```

`train.yaml` / `dataset_info.json` **直接从上一个成功的数据集下载来复用**，不要手写：

```bash
wenyon-cli dataset download rescue-sft-data --paths train.yaml,dataset_info.json -o /tmp/dl_ref
```

上传后核对 sha256 与参考一致（`wenyon-cli dataset files <card-id>`），能挡住手滑改动。

**`warm_state=cold` 是稳态，不用等它变 warm** —— 参考数据集被成功训练时也是 cold，
预热是任务 lease 时按需触发。「等 15 分钟」这条经验的实质就是等 15 分钟。

### 提交后必查（不要等跑完）

```bash
trisol train logs <job> --team infra-spot --tail 5000 > /tmp/j.log
grep -n "Using tool format" /tmp/j.log                                  # 期望 qwen3_5
grep -c "For each function call, return a json object" /tmp/j.log       # 期望 0（hermes，错的）
grep -c "Function calls MUST follow the specified format" /tmp/j.log    # 期望 >0（qwen3_5 XML，对的）
grep -c "Dropped invalid example" /tmp/j.log                            # 期望 0
grep "Num examples\|Total optimization steps" /tmp/j.log                # 与 train.jsonl 行数对上
```

再从 dump 的样例里确认 assistant 段是 `<think>` 开头 + `<tool_call><function=...>` XML。

**`--tail 100000` 会返回空**（CLI 的问题），用 `--tail 5000`。
**tqdm 进度条不流式输出**，日志里会一直停在 `0%| | 0/N`，只有 `{'loss': ...}` 会按
`logging_steps` 出现 —— 看不到进度条推进**不代表卡住**。9B / 4×A100 / 长轨迹数据实测
约 **4.6 分钟/步**，`logging_steps=10` 意味着开训后约 45 分钟才有第一个打点。

## 五、踩坑记录

### convert_rollout_to_sft.py 的分隔符 bug（会静默丢掉题目）

`/personal/sp2/qwen_trajs/cc_deepseek_1w/convert_rollout_to_sft.py` 按
`input.split("\n\nuser\n", 1)` 切 system / user，但 verl 写出的 `rollout_data`
里分隔符是**单换行** `"\nuser\n"`。切分失败后它**不报错**，直接跳过 system+题目，
产出的 ShareGPT 样本第一段就是 assistant 输出 —— **题目整个丢了**，
拿去 SFT 等于训"看不见题目就写代码"。

已有的 `trajectories_sft_*.jsonl`（teacher 系列）都是别的管线产的，全部带 system，
**未受污染**；这个 bug 只在拿该脚本处理 verl rollout_data 时触发。

修好的版本在 `/personal/verl/sft_data/extract_nd_sft.py`（用 `partition("\nuser\n")`
并对 system 前缀 assert，切分失败直接炸而不是静默丢数据）。
**转完必须验一遍角色序列**：
```python
Counter(tuple(y["from"] for y in x["conversations"][:3]) for x in recs)
# 期望 100% ('system','human','gpt')
```

### 跨管线用题面文本做 join 必须先 NFC 归一化

`rollout_data` 里的题面和 `trajectories_*/`（`agent_trisol.py` 产出）里的题面存在
**Unicode 码位差异**：同一个 `Å` 在 rollout 侧是 `U+00C5`（LATIN CAPITAL LETTER A
WITH RING ABOVE），在轨迹 JSON 侧是 `U+212B`（ANGSTROM SIGN）。字形相同、
字符串长度相同、`difflib.unified_diff` 打出来的两行**逐字看不出区别**。

800 题里有 3 道命中（2 个 `Å`、1 个 `â†` 的组合标记），直接哈希只能匹配 797/800。

```python
import unicodedata as ud, hashlib
def key(t): return hashlib.md5(ud.normalize('NFC', t).strip().encode()).hexdigest()[:12]
```

加了 NFC 之后 800/800。另外 rollout 的 `input` 在题面后面还接着**生成前缀**
`"\nassistant\n<think>\n"`，取题面要 `.split("\nuser\n",1)[-1].rsplit("\nassistant\n",1)[0]`，
只 `strip()` 是去不掉的。

### 日志输出

- `coding_sandbox_tool.py` 用 `logging.getLogger(__file__)` 在 Ray 接管日志后不输出
- 改用 `print()` 直接写 stdout

### 数据集 lease

- 文渊数据集刚上传完（state=ready, warm_state=cold）立即提交训练会报 `lease failed`
- 等 15 分钟预热后再提交

### DeepSpeed + LLaMA-Factory

- 27B 模型 SFT 需要 ZeRO-3 + CPU offload
- 数据集只读，DeepSpeed 配置文件通过 `/tmp` 注入：
  ```bash
  echo '<base64-config>' | base64 -d > /tmp/ds_zero3.json
  cp /trisol/input/datasets/ds-0/train.yaml /tmp/train.yaml
  echo "deepspeed: /tmp/ds_zero3.json" >> /tmp/train.yaml
  /opt/conda/bin/llamafactory-cli train /tmp/train.yaml
  ```
- 8×A100 + 27B + ZeRO-3 + cutoff_len=60K 才能跑通（64K OOM）

### 自采样数据的 SFT loss 比 teacher 数据**高**，这是正常的（2026-09-07）

拿基模自己的正确 rollout 做 SFT（STaR / 拒绝采样第一轮）时，**不要**指望"自己的输出所以 loss 低"。
实测（`coding-rl-...-800v27`，即产出该批数据的那次 run 本身）：

| 数据 | 基模 per-token NLL |
|---|---|
| 基模自己的 rollout（全部，含失败） | **0.343 – 0.392** |
| 同上，vLLM 采样路径 | 0.258 – 0.293 |
| teacher（Qwen3.7-Plus）数据 | **≈ 0.18** |

**自采样数据的 loss 约是 teacher 数据的 2 倍。** 原因是定义决定的，不是异常：
rollout 用 `temperature=1.0, top_p=1, top_k=-1`（`verl/trainer/config/rollout/rollout.yaml:17,20,23`）
纯采样，所以自采样的期望 NLL **恒等于模型自己的 per-token 熵**。同一份日志里
`actor/entropy = 0.299–0.332`，与 `rollout_log_ppl` 0.258–0.293 对得上（15% 以内）——
恒等式在数据里成立。而模型自由探索时的熵没有理由低于它读套路化 teacher 文本时的交叉熵。

**verl 里直接测这个量的指标（以前没用上，以后要用）：**
- `rollout_corr/rollout_log_ppl` —— 采样(vLLM)策略在自己采出 token 上的 per-token NLL
- `rollout_corr/training_log_ppl` —— 同样 token，FSDP 训练路径算的 NLL（与 SFT loss 同路径，直接可比）
- `actor/entropy` —— 策略自己的 per-token 熵，应与上面两个同量级

**对比 loss 时先查 warmup。** 两个任务 step 10 的 loss 不可直接比：
自采样任务 warmup 只 7 步（step10 `lr=4.997e-05`，已实质训练），
teacher 任务 warmup 72 步（step10 `lr=6.164e-06`，基本等于未训练）。
后者的 step10 loss ≈ 基模初始 loss，前者不是。

### SFT loss 平**不能**推断"没学到"——teacher SFT 也是平的（2026-09-07）

**不要**用"loss 曲线平"去判断 SFT 有没有效果。反例是同架构、同 LoRA 配置、
已知有效的 teacher SFT（`qwen-9b-sft-rescue-qwen35-64k-r32`）：

| | 自采样 760 条 | teacher rescue 2496 条 |
|---|---|---|
| 步数 / epoch | 72 / 3 | 234 / 3 |
| loss | 0.1980 → 0.1875（**−5.3%**） | 0.1906 → 0.1630（**−14.5%**） |
| grad_norm | 0.050 → 0.042 | 0.064 → 0.045 |
| `ΔW/W` 中位数 | **0.256%** | **0.547%**（2.14×） |

**两边的 loss 都基本是平的，grad_norm 也在同一个区间**，而 teacher 那个 adapter
是确实改变了行为的（合并后用于 RL）。所以 loss 平 + grad_norm 小既不能证明学到了、
也不能证明没学到 —— 原因是 loss 是 **token 平均**量，长轨迹里绝大多数 token 是高度
可预测的续写，少数决策点 token 才决定行为，决策点变了平均 NLL 也可以不动。
**判据只能是评测。**

**要判断"权重到底动了没有"，直接量 LoRA 的 `ΔW/W`，几分钟就能做完，不用 GPU：**

```bash
trisol model download <model-id>:<ver> -o /tmp/lora            # adapter 仅 ~330MB
trisol model download <base-id>:1 model.safetensors-00001-of-00004.safetensors -o /tmp/base_shard
```

```python
# ||BA||_F 不要展开成 (out,in) 大矩阵, 用迹恒等式:
#   ||BA||_F^2 = tr( (B^T B) (A A^T) )        A:(r,in)  B:(out,r)
fro = torch.trace((B.T@B) @ (A@A.T)).clamp(min=0).sqrt().item() * (lora_alpha / r)
ratio = fro / W.norm().item()                  # W 从 base shard 里 safe_open 取
```

注意 adapter key 是 `base_model.model.model.language_model.layers.N....lora_A.weight`，
去掉 `base_model.model.` 前缀、把 `.lora_A.weight` 换成 `.weight` 才能对上 base 的 key；
一个 shard 只能覆盖部分层（12 个模块就够 —— 实测同类模块跨层高度一致，
teacher/自采样 的比值在 12 个模块上是 2.09×–2.35×）。

参照量级：`ΔW/W` 在 **0.25%–0.55%** 都是"真的训过"，两者差 2.14× 与步数差
3.25×（72 vs 234）大致相符，不需要额外解释。

### 自采样拒绝采样 SFT 一轮：**阴性**（2026-09-07 评测结案）

`qwen-9b-sft-nd800-k4-64k-r32`（自己的 800 题 rollout，非退化题正确轨迹每题取 K=4，
760 条 / 72 步 / 3 epochs）在**同一批 267 道非退化题**上评测，harness 和基线完全一致
（`batch_run_trisol.py -c 16`，`agent_trisol.py`，单跑一次，`eval.correct` 为准）：

| | 正确 | 率 |
|---|---|---|
| SFT (nd800-k4) | 148/266 | **0.5564** |
| 基模同题 | 138/266 | **0.5188** |

`Δ = +0.0376`（+10 道）。配对：都对 96 / 只 SFT 对 52 / 只基模对 42 / 都错 76。
**McNemar 双侧 p = 0.35**，配对 Δ 的 95%CI = **[−0.034, +0.109]** —— 跨 0，
**没有明显效果**（评测前写死的判据是 ≥0.58）。

行为侧三个量**几乎完全不变**：平均轮数 19.2 vs 19.2、交出 `<final_answer>` 的
0.692 vs 0.669、撞 30 轮 82 vs 91。

**真正有信息量的不是净增 10 道，是 94/266（35.3%）的题结果翻了。**
行为大幅改变、方向没学对，正负几乎对冲。这和 `ΔW/W = 0.256%`、loss 只降 5.3%
是同一幅图像的三个侧面。回头看也印证了上一节：**loss 平不能推断没学到，
但权重动了同样不能推断学对了 —— 判据只能是评测。**

这一轮**不能**推翻"自采样 SFT 这条路可行"，因为量级本来就小（72 步，
teacher 那个有效的 adapter 是 234 步 / `ΔW/W` 0.547%）。区分"量不够"和
"路不对"需要下一个实验（K=8 → 963 条，或对难题再采样），不要用这一轮的
p=0.35 去否定整条路线。

### 评测跑到中途**绝不能**读数（同一次评测踩到，2026-09-07）

同一次评测，进度 17% 时和跑完时：

| 进度 | SFT | 基模同题 |
|---|---|---|
| 46/267（17%） | **0.630** | 0.500 |
| 266/267（100%） | **0.5564** | 0.5188 |

**因为 `batch_run_trisol.py` 是并发池，简单题先跑完** —— 早期样本对难度的偏斜是
系统性的，不是随机的。17% 那个读数偏高了 7.4 个点，足以把"没效果"读成"有效果"。
（配对比较能控住题目难度，但控不住"早期完成的题本身就简单"这件事：
基模在那 46 道上是 0.500，全 266 道上是 0.5188，看起来接近，
所以**不要**用"基线也差不多"来自我说服早期读数可信。）

耗时参考：267 道 / c=16 / 4×L20 ≈ **5 小时**（0.84 题/分钟）。要中途看就只看
"有没有系统性故障"（`model` 字段对不对、有没有 `total_turns==0`），不看正确率。

**★ 提交顺序已改成随机打乱（2026-09-16 起默认开）。** `discover_problems()` 原本是
`sorted(base_dir.glob(...))` ⇒ 按**类目字母序**提交，所以中途读数有**两层**偏斜叠加：
(a) 并发池里简单题先跑完，(b) 部分结果只覆盖字母序靠前的那批类目。
**(b) 是可以修掉的**：`batch_run_trisol.py` 新增 `--shuffle-seed`（默认 **42**）与
`--no-shuffle`（复现旧口径）。shuffle 放在 `discover_problems()` **之后** ⇒
已有输出照常 skip，补跑缺题的 symlink 手法不受影响。
实测前 290 道覆盖的类目数 **203 → 251**（全集 563 个类目）。

**(a) 修不掉** —— 池子是 work-conserving 的，快的一定先回来。
所以 **"中途不读正确率"这条规矩不因为 shuffle 而放宽**，shuffle 只是让
"跑完的那批题"在**题目构成**上无偏，难度偏斜仍在。

### 待修：value_check 缺绝对容差（下次顺手改）

`verl/utils/reward_score/coding_reward.py` 的 `value_check` 只有相对容差：

```python
return abs(a - b) <= max(abs(a), abs(b)) * tol
```

当 ground truth 为 0 或接近 0 时，右边趋近 0，**任何浮点机器误差都判错**。实测 good27 的
1280 条 rollout 中有 16 条属于此类假阴性（占 1.25%），典型样本：

```
pred = [200, 2.220446049250313e-16, True, 4.989831893063952e-15]
gts  = [200, 0.0,                   True, 0.0]     -> 判 0
```

修法：加绝对容差下限

```python
return abs(a - b) <= atol + rtol * max(abs(a), abs(b))   # atol=1e-6, rtol=0.01
```

### 已排除的猜测（不要再重复排查）

- **沙盒创建失败回退本地 subprocess**：全量 Loki 日志确认整个 10 步训练只发生 **1 次**
  （`trisol train logs <job> --history --grep "Sandbox execution failed"`），不是问题来源。
  该处 `logger.warning` 在 RewardLoopWorker 下**能**正常输出，阴性结果可信。
- **reward 稀疏 / 退化组**：good27 上退化组（组内 8 条全对或全错）只占 13.8%，
  86% 的组都有非零 advantage，不是瓶颈。

### 30 轮上限是有意设计（理由已在 2026-09-04 更正）

撞 `MAX_ASSISTANT_TURNS=30` 的轨迹 pass_rate 只有 0.036（未撞的是 0.76），占全部 rollout
41%。**不要调大轮数**，但当初写下的理由（"大概率在打转"）**已被证伪**，正确理由是：

教师轨迹（Qwen3.7-Plus，7809 条）判对的部分**轮数中位数只有 9**，p90=23，
21% 在 5 轮内解完。**30 轮对"会做"的轨迹是 3 倍余量，根本用不到。**
教师判错的部分则 median/p75/p90/p99 **全是 30** —— 做不出来就一路撞到顶，
和我们模型撞顶的形态完全同构。

### 撞顶轨迹的真相：不是打转，是从没上路（2026-09-04 定位）

在 lr3e-5（7 步）和 lr1.5e-5+ent（10 步）两个运行上独立复现，四条证据：

| 测量 | 撞顶轨迹 | 未撞顶轨迹 |
|---|---|---|
| 出现 `<final_answer>` | **6.2%** | 96.3% |
| 通过率 | 0.036 | 0.763 |
| 命令重复率 | 0.035 | 0.051 |
| 报错率（轮 0-5 → 25-30） | 33.6% → 32.7%（**平的**） | 36.7% → 13.2%（**在收敛**） |
| 工具输出长度（轮 0-10 → 20-30） | 1005 → 827 | 989 → 504 |

- 撞顶 = **压根没交答案**，不是交了错答案
- 撞顶轨迹的命令重复率**比正常轨迹还低** ⇒ 不是死循环、不是打转
- 报错率从第 5 轮起就**卡住不动**，工具输出长度不收窄 ⇒ **从一开始就没有进展**，
  再给 30 轮也一样
- `corr(每题撞顶率, 该题未撞顶通过率)` = −0.48（t=−2.57）⇒ 撞顶率是**难度的代理变量**

**结论：撞顶是"没找到解题路径"的表象，不是独立的行为缺陷。**
惩罚长轮数（overlong soft punish 之类）是治标且有害——模型的逃生路径会变成
"提前交没把握的答案"，把"没交"（0 分）变成"交错"（还是 0 分），
代价是毁掉未撞顶那 0.76 的准确率。**不要做轮数惩罚。**

### 撞顶分析到此为止（2026-09-06 结案，不要再做）

上面两节的结论已经收敛成一句话：**撞顶 = 没找到解题路径 = 做错，没有独立信息量。**

**以后分析 rollout 只看 `score`，不要再拆 `cap%` / `pr_cap` / `pr_un`，
不要再做撞顶率的反事实分解。** 这套拆分在定位"训练量不足 vs 能力问题"时用过一轮，
已经用完了；继续拆只会把"做错"重新包装成一个假的独立维度，浪费时间。

同理，筛数据集、算退化率、算 pass@k **一律用原始口径**：
`ok = score > 0`，撞顶的轨迹就是 `ok=0`，照常进组。
不要再引入"未撞顶口径"（只统计 `tc<29` 的样本）—— 它的分母和原始口径不一致，
两个数并排出现必然造成混淆。2026-09-06 就踩过：同一批 800 题报了
"非退化 267"和"非退化 198"两个数，198 其实是 267 的子集，
差的 69 道里 65 道是"跑完的全对、错的全是撞顶"，那恰恰是**真信号**不该排除。

（下面"题目数量不是当前瓶颈"一节里"496 题能榨出约 123 道真正可学的"是同一个
错误口径的产物，按原始口径重算才作数。）

### 学习率的实测边界（2026-09-03/04）

| `ACTOR_LR` | `pr_un` 趋势 | 结论 |
|---|---|---|
| 6e-6 | — | 推不动，10 步 entropy 无趋势 |
| ~~**1.5e-5** + `ENTROPY_COEFF=0.001`~~ | ~~−0.0011/step，t=−0.26~~ | ~~**能力不掉，可用**~~ **已被 30 步实验推翻，见下节** |
| 3e-5 | −0.0201/step，**t=−3.12** | 能力真的掉，overshot |

3e-5 下 entropy 以 −0.033/step 直线下降，7 步从 0.35 掉到 0.20；
1.5e-5+entropy_coeff 降到 −0.0091/step。

**⚠️ 上面这张表的 1.5e-5 那行是 10 步窗口得出的，已作废。10 步不足以判断 lr 安全 ——
`coding-rl-nd267` 用同样配置跑到 30 步，前 16 步确实平稳（score slope t=−0.45），
第 17 步之后腰斩。判 lr 安全的最小窗口至少要跨过第一个 epoch 边界。**

### RL 在第二个 epoch 腰斩：不是过拟合、不是变懒、不是 lr（2026-09-09 结案）

`coding-rl-nd267`（267 题 / lr=1.5e-5 / `ENTROPY_COEFF=0.001` / rloo / clip 0.15-0.35 /
rollout_n=8 / batch=16 / LoRA r32 α64），30 步后手动停掉。epoch 边界在 **step 17**
（16 步 × 16 = 256 ≈ 267 题）。

| | epoch 0 (1–16) | epoch 1 (17–30) | Δ | Welch t |
|---|---|---|---|---|
| `critic/score/mean` | **0.4014** ± .080 | **0.1908** ± .068 | −0.211 | **−7.75** |
| `actor/entropy` | 0.2896 ± .029 | 0.1726 ± .051 | −0.117 | −7.61 |
| `response_length/mean` | 23539 | 21854 | −1686 | −0.92（不显著） |

**两个 epoch 是同一批 267 题，题目难度被控住了，所以这不是 batch 抽样噪声。**

**排除掉的解释（都查过，别再重复排查）：**

- **不是"变懒/提前收手"**：跨 epoch 取 `global_seqlen` 相差 <5% 的配对 36 组，
  **36/36 全为负**，Δscore 均值 −0.148。更直接：step 25–30 的 `resp_len` 均值 25990
  **比 epoch 0 的 23539 还长**，score 只有 0.204 vs 0.401。
- **不是过拟合/背题**：方向反了。在已见过的题上过拟合应让**训练 reward 上升**，实测腰斩。
- **不是基础设施**：全生命周期 `Sandbox execution failed` 1 条、`Traceback` 0 条。
- **⚠️ ~~不是 token-mean 的长度不对称~~ —— 这条"排除"已被推翻，它恰恰是根因，
  见下面「隐式轮数惩罚」一节。** 排错的方法本身是错的，记下来别再犯：
  `loss_agg_mode` 默认 token-mean，`critic/advantages/mean` 是
  `masked_select(advantages, response_mask)` 的 token 级均值（`metric_utils.py:490`），
  RLOO 组内序列级 advantage 恒和为零 ⇒ token-mean 非零 ⟺ 长度不对称。
  我按这个恒等式从 batch 级 `adv_mean` 反解 `L_fail/L_success`，得到 **0.67–1.20 还会翻符号**，
  就判"无系统性不对称"。**错在 batch 级反解把不同长度的组混在一起相互抵消了** ——
  直接量成败轨迹长度是 `L_fail/L_ok` = **1.33（早期）/ 1.93（后期）**，
  按轮次位置分解更是单调到 −0.93。**不对称必须按轮次位置分解才看得见。**
- **配置没写错**：`loss = pg_loss − 0.001 × entropy_loss` 逐位对得上
  （st26：0.0018387 − 0.001×0.16331 = 0.0016754 = 实测 `actor/loss`），entropy bonus 方向正确；
  `kl_coef=0` 无梯度；mini-batch 4×8=32 / 128 ⇒ 每步 4 次内层更新，`ppo_epochs=1`，标准。

**真正有信息量的是失败模式反转：**

| | corr(score, turns) | corr(score, entropy) |
|---|---|---|
| epoch 0 | **−0.771** (t=−4.54) | +0.108 (t=+0.41) |
| epoch 1 | **+0.769** (t=+4.17) | +0.155 (t=+0.54) |

epoch 0 是健康形态（轮数越多分越低 —— 会做的几步做完，不会做的才磨，与「撞顶=没找到路径」一致）。
epoch 1 **符号翻转**：失败模式从"磨到顶还是做错"变成"**早早放弃**"。step 20–24 的坑：
resplen 25784→12809→13598（谷底 st22 score 0.070，turns 39.4→17.2），
然后 st25 长度回到 28264、**能力没回来**（score 0.180）。

**entropy 坍缩是同源症状，不是已证实的原因**：全程 −0.0077/step、t=−14.8，
epoch 1 加速 2.5 倍（−0.0045 → −0.0112），但**在每个 epoch 内部 entropy 与 score 相关性≈0**
（t=0.41 / 0.54）。全程那个 +0.72 是两者都在跌的伪相关。**不要写"entropy 坍缩导致能力下降"。**

**机制已用 rollout_data 定案，见下一节。**（当初写的假设"60–80% rollout 拿负 advantage、
RLOO 主导信号是压低"**是错的**，RLOO 组内序列级 advantage 恒和为零，正负质量结构性相等；
非退化组内正:负条数早期就是 1:1.11。别再沿用那个说法。）

### RLOO + token-mean = 没人配置的隐式轮数惩罚（2026-09-09，rollout_data 实测）

nd267 的 3840 条轨迹（30 步 × 128）。行为侧先看三个量：

| | 早期 1–10 | 崩溃 20–24 | 后期 25–30 |
|---|---|---|---|
| 交 `<final_answer>` 率 | 0.597 | 0.245 | 0.456 |
| `P(对 \| 交了答案)` | **0.702** | 0.516 | **0.449** |
| 工具调用次数 | 20.1 | 10.7 | 14.2 |
| 输出字符数 | 70293 | 44448 | **77914（比早期还多）** |

`score = 交答案率 × P(对|交了答案)`（`P(对|没交)` 四组全是 **0.000**）。两个因子都掉，
且条件正确率**单调掉到底**（0.702→0.449）—— 不只是"不交答案"，交上来的也更错。

**失败模式的性质变了**（没交答案的轨迹里）：

| | 撞顶（≥28 次工具调用） | 中途自己停（≤15 次） |
|---|---|---|
| 早期 1–10 | **0.959** | 0.033 |
| 崩溃 20–24 | 0.041 | **0.793** |
| 后期 25–30 | 0.158 | 0.545 |

agent loop 的终止条件是"assistant 消息里没有 tool_call"，所以"中途停"字面意思就是
**模型说了段话、不调工具了、也不交答案**。后期输出字符更多而工具调用少 29%、
报错率高 36% ⇒ 概率质量从 `<tool_call>` 转移到了散文上。

**根因：把 RLOO 的 advantage 按轮次位置展开。** 每个非退化组里，第 t 轮还在调工具的
那些轨迹的 `A_i` 之和（`A_success=(n−k)/(n−1)`，`A_fail=−k/(n−1)`）：

| 轮次 | 早期 1–10 | 后期 25–30 |
|---|---|---|
| 0 | **+0.092** | +0.034 |
| 8 | 0.000 | −0.274 |
| 20 | −0.536 | −0.179 |
| 28 | **−0.926** | −0.153 |

**早期前 8 轮净正、第 9 轮变号、之后单调负到 −0.93；后期变号点移到第 1–2 轮。**
因为成功轨迹短（14.0 轮 / 59036 字符）失败轨迹长（24.4 轮 / 78403 字符，
`L_fail/L_ok` 早期 1.33、后期 1.93），**超过成功轨迹典型长度的 token 几乎只出现在负样本里**。

**这就是「30 轮上限是有意设计」那一节明令禁止的轮数惩罚 —— 没人配置它，
它是 RLOO + token-mean 的结构性副产物。** 那节预言的后果逐字应验（"逃生路径变成提前交
没把握的答案"）。

**缓解手段，按成本排：**

0. **`actor.loss_agg_mode=seq-mean-token-mean`（一行，零成本，先做这个）**。每 token 权重
   变成 `1/L_i`，长失败轨迹被自身长度稀释。实测第 28 轮净梯度 **−0.926 → −0.275（0.30×）**，
   变号点 9→10 轮。形状还是单调负，不治本。
   （**注意**：只看 batch 级 `adv_mean` 反解长度不对称会得到 0.67–1.20 "无不对称"的错误结论，
   必须按轮次位置分解才看得见。）
1. **turn-level advantage 用 MC 估 value（VinePPO 类）** —— 最对症，直接消除位置偏置。
   需要沙盒能在轮边界 fork（现在是一条轨迹复用一个沙盒，`610186de`），rollout 成本 ×k。
2. **学 critic 走 GAE** —— verl 原生支持但高风险：267 题的数据训不出能从 100k 中途状态
   预测成败的 critic，不准时 token 级 advantage 就是噪声，比 RLOO 无偏的序列级估计更糟。
3. **启发式 turn-level shaping（奖励不报错）—— 不要做**。报错率区分成败的效应量早期只有
   **d≈0.30**（成功 0.181 vs 失败 0.232），而且直接可 hack（跑 30 次 `echo hi`）。

**OPD 在这个坐标系里的位置**：k1 是逐 token 的 `KL(student‖teacher)`，**没有轮次位置偏置，
也从不压低任何行为**，且不需要发明 reward source —— 是"token 级信号"里唯一免费的那条路。

**两条硬教训：**
1. **`TEST_FREQ=10` / `SAVE_FREQ=10` 太稀**：st10 val=0.333 看着好、st20 已经 0.111。
   而且 27 题验证集二项 sd≈9%，读不出信号 —— **验证集必须扩大，否则看不见崩溃。**
2. **跑满 1 个 epoch 之前不要下"这个 lr 能用"的结论**，见上一节的作废记录。

### Trisol 日志：active job 的 `--history` 默认只回看 1 小时

`trisol train logs <job> --history` 的 `--since` 默认值对**运行中**的任务是
`[max(now-1h, created_at), now]`，跑了两天的任务只能拿到最后 1 小时。取全生命周期必须显式给：

```bash
trisol train logs <job> --team infra-spot --history \
  --since 2026-09-08T05:00:00Z \
  --grep "training/global_step" --tail 5000
```

### ★★ `--direction forward` 会静默丢掉最新的日志（2026-09-16，我照抄自己的配方踩的）

上面这条命令**以前写的是 `--direction forward`，那会漏步数**。同一个任务、同一个
`--since`（`created_at`）、同一个 `--tail 5000`、同一个 `--grep`，只差方向：

| | 返回 |
|---|---|
| `--direction forward` | step **1–18**（18 行） |
| 不给 / 默认（backward） | step **1–24**（24 行，全部） |

forward 给的是从 `--since` 起**最早**的一批并在远小于 `--tail` 的地方截断（机制不明，
只记现象）。我据此报了"step 18"，用户当场纠正实际是 23–24 步。
⇒ **取当前进度一律不给 `--direction`**；forward 只在明确要看"开头那几步"时用。

**报进度前必做时间自洽性检查**，它能独立抓住这类漏数据：

```
步数 × mean(timing_s/step) + 启动开销 ≈ now − created_at
```

本次：24 × 6465s = 43.1h + 1.6h = 44.7h，`created_at` 09-14 04:40 + 44.7h = 09-16 01:22，
与当前时刻吻合；而 18 步只推到 09-15 14:10，**比当时时间早 11 小时** —— 一算就露馅。
（`--tail`、`--since`、`--direction` 任一处出问题都会表现为"步数偏少"，
所以这个检查是对**整条取数链路**的验证，不是只针对方向。）

（`--since` 用 `trisol train get` 里的 `created_at`。`--grep` 是**纯子串**不是正则。
另外 `--tail` 对已终止的多节点任务在**非** `--history` 模式下会报
`Error: pods "..." not found (code=500)` —— 活体 pod 流已经没了，只能走 `--history`。）

**`--tail` 必须在 1..5000 之间**：`--history --tail 8000` 不会截断到上限，而是**只返回一行
400 报错**，看起来像"日志是空的"。（和 SFT 那节记的 `--tail 100000` 返回空是同一个原因。）

**从 metrics 行提取数值不要用 `tr`。** `tr -d ' ' `之类顺手写的清洗里只要带 `-`，
负值就被吃掉，`pg_loss:-0.00047` 会变成 `0.00047`，符号翻了还看不出来。用正则取：

```python
re.search(re.escape(key) + r':(-?\d+\.?\d*(?:[eE][+-]?\d+)?)', line).group(1)
```

（`actor/pg_loss`、`distillation/loss` 这些**带符号**的量是判据本身，符号错了结论就反了。）

**⚠️ 之前这里写的是 `r':(-?[\d.eE+]+)'`，它会吃掉科学计数法的负指数**：字符类里有 `e` 和 `+`
却没有 `-`，`3.52e-05` 被截成 `"3.51999099166278e"`，指数整个丢掉。2026-09-11 读自蒸馏
`distillation/loss` 时踩到 —— 真值 3.52e-05，截断后看起来像个 3.5 量级的数，
**判据差了五个数量级还不报错**（`float()` 那一步才会炸，只打印就完全看不出来）。
用上面的写法：尾数和指数分开匹配，指数的正负号显式允许。

**`trisol train get -o json` 没有顶层 `envs` 字段** —— 提交时的环境变量在 `custom_config` 里，
要核对某次任务实际带了什么 env（比如确认 treatment 真的生效）得去那里找。

### 平台注入的 `MASTER_PORT` 落在临时端口范围内，会被随机抢占（2026-09-11）

`coding-rl-nd267-spfix-0911` 第一次提交在 `actor_rollout_init_model()` 就挂了，
**与镜像、SP 修复、flash 全无关系**：

```
verl/workers/engine_workers.py:88  initialize_global_process_group_ray(timeout_second=None)
verl/utils/distributed.py:92       torch.distributed.init_process_group(init_method=None)
DistNetworkError: server socket has failed to listen ... port: 36655, code: -98, EADDRINUSE
```

`init_method=None` ⇒ torch 走 `env://` rendezvous，读平台注入的 `MASTER_ADDR`/`MASTER_PORT`。
**平台挑的 36655 落在 Linux 临时端口范围 32768–60999 之内**，于是 Ray / vLLM 开的任何
出站 socket 都可能先把它抢走。是随机竞态，同配置的 nd267 当初没撞上。

修法是把 rendezvous 端口钉到临时端口范围下界之外，在 command 最前面 export（`--env`
会不会被平台注入覆盖不确定，export 在脚本里执行则一定赢）：

```python
command[2] = "export MASTER_PORT=29517; " + command[2]
```

**这一项不破坏单变量前提** —— 它只是 TCPStore 的地址，没有任何数值依赖它。
`--backoff-limit 0` 意味着撞上就直接终止，所以值得预防性地钉住而不是靠重试。

### 训练量不足（2026-09-02 定位，仍然成立但不是唯一的墙）

`coding-rl-a100-lora-9b-96k-30t-n8-8gpu-good27` 十步 metrics：

```
grad_norm ~0.01-0.05 无趋势 | pg_clipfrac ~0.05 | ppo_kl ~0.009 | entropy 0.31-0.39 无趋势
val acc: 0.444 0.481 0.444 0.333 0.481 0.370 0.444 0.481 0.593 0.481
```

- `lr=6e-6` + Adam ⇒ LoRA 的 B（零初始化）10 步累计只走 ~6e-5，`alpha/rank=1`，
  `ΔW/W` 在 0.2% 量级 —— **第 10 步的模型和基模基本是同一个**
- val 波动：27 题、p≈0.46 的二项标准差 9.6%，观测极值仅 +1.44σ / −1.27σ，纯采样噪声
- `pg_clipfrac=0.05`、`ppo_kl=0.009` 都远低于健康区间（clipfrac 0.1~0.2），
  说明**离信任域边界很远，lr 有大量上调空间**

`ACTOR_LR=6e-6` + `LORA_ALPHA=32` 是脚本默认（`lr=1e-5` / `alpha=64`）的 0.3 倍有效学习率。
当初为"保守"压的这两个数正是推不动的直接原因。

**判断训练是否真的在动，先看这三个数，不要看 val acc：**
`actor/grad_norm`、`actor/pg_clipfrac`、`actor/entropy` 的**趋势**。

**但训练量只是第一层墙。** 把 lr 调到 1.5e-5 让模型真的动起来之后，`pr_un` 依然纹丝不动
地停在 0.75 —— 说明**真正的墙是解题路径缺失**，RL 只能放大模型已有的行为，
路径不在分布里，采样 8 次也采不出来。**先 SFT 把路径灌进去，再 RL。**

### 题目数量不是当前瓶颈（2026-09-04）

496 题实测：非退化（0 < pass < 8）只有 **171 题（34.5%）**，63.1% 是全错。
而 good27 是专门挑出来的 100% 非退化 —— **27 题的信号质量反而更高**。

更关键的是 `train_batch_size=16` 固定，**每步只用 16 道题算梯度，题库是 27 还是 800，
每步的信号量完全一样**。题目多只改变跨步多样性，不改变每步信号强度。

扩题库解决的是"训练久了过拟合"，不是"现在不涨点"。**应该放在 SFT 之后做**，
那时非退化率会自己涨。按未撞顶口径，496 题里能榨出约 123 道真正可学的。

### Trisol：已结束任务的 rollout 数据取回

`trisol train exec` **只对 running pod 有效**，任务一 canceled 就进不去；
`train output ls` 只能列目录读不到内容。已结束任务用 checkpoint 通道：

```bash
trisol train checkpoint rescan <job> --checkpoint-prefix rollout_data --checkpoint-atomic
trisol train checkpoint list <job>          # 轮询到 status=ready，约几十分钟
trisol train checkpoint download <job> <checkpoint-id> -o <dir>
```

`rescan` 要求终态（running 时返回 409）。注意 `--no-output-model` 的 PFS 副本
在任务终止 **7 天后永久删除**，rescan 要赶在这之前。

### ★ 取 checkpoint 光改 prefix 不够，还要 `--checkpoint-scan-path`（2026-09-13）

verl 写的 checkpoint 目录名是 `global_step_*`，而平台默认发现规则的 prefix 是
`checkpoint-`，所以**默认情况下 checkpoint 不会被归档**。但**只把 prefix 改成
`global_step_` 一样什么都扫不到** —— 发现规则是在**输出根**（`/trisol/output`）下扫
**直接子目录**，而 verl 的 `trainer.default_local_dir=/trisol/output/checkpoints`
⇒ 输出根下只有 `.trisol/` 和 `checkpoints/`，`global_step_*` 在**下面一层**。

只改 prefix 的 rescan **不报错**，`checkpoint list` 就一直只有旧条目，
看起来像"归档很慢"。我等了 50 分钟才想起来去 `train output ls` 看真实目录结构。

```bash
trisol train checkpoint rescan <job> --team infra-spot \
  --checkpoint-scan-path checkpoints \
  --checkpoint-prefix global_step_ --checkpoint-atomic
```

（`--checkpoint-scan-depth 2` 是另一条路，但它会用 scan-root 相对路径给 checkpoint 命名，
不如 scan-path 干净。）

**先 `trisol train output ls <job> --team infra-spot` 看清目录层级再定规则**，
不要凭记忆猜。`output ls` 只有 `ls`、**没有下载子命令**，所以取文件只能走 checkpoint 通道。

### 长跑任务的 `--checkpoint-prefix` 该给 `global_step_`，不是 `rollout_data`

`opd-teacher27b-prod-0911` 提交时我写的是 `--checkpoint-prefix rollout_data --checkpoint-atomic`
（从探针任务继承来的）。后果：平台在开跑 1.5h 时把 `rollout_data/` 归档了一份
**3.2M 的早期快照**就再没动过，而真正想要的 `global_step_*` 全程没被归档，
跑完还得补一次 rescan + 重新归档 155 GB。

**提交长跑时 prefix 直接给 `global_step_`（配 scan-path），rollout 数据留到事后 rescan。**
~~rollout 数据是纯追加的，事后取没有损失~~ —— **这句是错的，见下节。归档是一次性的，
错过的窗口永久取不回来。** 正确的做法是**同时**配好两条规则（checkpoint 用 scan-path
+ `global_step_` prefix，rollout 另起一条），不要指望"事后补"。

单个 step 的体积参考（9B / LoRA / world_size=8）：`model_world_size_8_rank_*.pt`
**2.38 GB × 8 = 19 GB**，加 optim 0.4 GB ⇒ **约 19.4 GB/step**。
`SAVE_FREQ=5` 跑 40 步 = 8 个 checkpoint = **155 GB**，归档不是一会儿的事。
注意这是**全量 FSDP shard（含基模权重）**，不是只有 LoRA adapter。

---

# FSDP↔vLLM 引擎偏差（`rollout_probs_diff_mean`）

> 起因：自蒸馏 step 0 量出 `abs_loss≈0.2` 的地板（见 OPD 一节），
> 想知道这 0.2 能不能靠对齐 kernel 消掉。**结论：不能，单卡 kernel 差异只占 ≤11%。**

## 这个量怎么读

`verl/utils/debug/metrics.py:98-120`：`mean|exp(actor_old_log_probs) − exp(rollout_old_log_probs)|`，
概率空间不是 logprob 空间，只在 `response_mask` 上取。
`docs/faq/faq.rst:180-207` 给的健康线是 **< 0.005**，> 0.01 算推理引擎精度问题。
由 `rollout.calculate_log_probs`（yaml 默认 True）开启，`ray_trainer.py:1574` 见到
`rollout_log_probs` 才算。

**必须在固定 step 读。** nd267 全 30 步是 0.0413 → 0.0093 的**单调下降**，不是步间噪声：

| | mean | sd |
|---|---|---|
| 全 30 步 | 0.0305 | 0.0105 |
| **step 1–5** | **0.0412** | **0.0062** |

**step 1 可以排除 LoRA**：B 零初始化 ⇒ ΔW=0，FSDP 侧和 vLLM 侧是同一份基模权重，
所以 0.041 既不来自 LoRA 数值也不来自权重同步。

## ★ 这个量被 entropy 污染，跨配置比较必须改用 pearson（2026-09-10）

上面"单调下降所以要固定 step 读"这条**理由是错的**（结论碰巧对）。它下降主要不是数值对齐变好，
**86% 是 entropy 从 0.3007 坍缩到 0.0914 带来的**（nd267 30 步，`pd ~ entropy` 拟合
`b=+0.1317`，预测 Δ=−0.0276 vs 实测 Δ=−0.0319）。

机制是恒等式级的：`|Δp| ≈ p(1−p)·ε`。模型自信时 p≈1，**同样的 logit 误差 ε 产生的
`|Δp|` 趋近 0**。所以 `probs_diff` 一半是置信度计，不是纯数值对齐计。

nd267 30 步实测，**几乎所有偏差度量都被 entropy 污染，只有 pearson 不受影响**：

| 量 | 空间 | `r` vs entropy | t | CV |
|---|---|---|---|---|
| `rollout_probs_diff_mean` | 概率 | +0.901 | +10.97 | 0.342 |
| `rollout_corr/kl` (k1) | log | +0.786 | +6.73 | 0.348 |
| `rollout_corr/k3_kl` | log | +0.772 | +6.42 | 0.353 |
| `rollout_corr/log_ppl_abs_diff` | log | +0.873 | +9.47 | 0.242 |
| **`rollout_actor_probs_pearson_corr`** | 概率 | **−0.150** | **−0.80** | **0.039** |

**⚠️ "log 空间的量（`abs_loss` 一族）因为 `|Δlogp|≈ε` 与 p 无关所以抗 entropy"——
这个理论预测被数据否掉了**，log 空间和概率空间一样被污染（r=0.77–0.87）。
**⇒ OPD 的 `abs_loss≈0.2` 地板是在 v4 那一步的 entropy 下量的，
换 entropy 就会漂，不能当固定常数用。** 偏相关：控制 turns 后 `pd~entropy` r=+0.800(t=7.06)，
控制 entropy 后 `pd~turns` 只剩 +0.407(t=2.35)。

**跨配置一律用 `rollout_actor_probs_pearson_corr`：**

| 配置 | 轮数 | resp_len | pearson |
|---|---|---|---|
| 单卡探针（1 次前向，无 SP、无多轮） | 1 | 20000 | **0.9997** |
| 8 轮 / SP=8（`probe-bias-sp8-0910`） | 8 | 8192 | **0.9333** |
| nd267 st1 / SP=8 | 30 | 98304 | 0.8439（30 步全距 0.72–0.90） |
| v4 自蒸馏 / SP=8 | 30 | 98304 | 0.8239 |

驱动 pearson 的是**轮数/序列长度**，不是 entropy：nd267 内 `corr(pearson, num_turns)=��0.348(t=−1.96)`、
`corr(pearson, global_seqlen)=−0.358(t=−2.03)`、`corr(pearson, entropy)=−0.150(t=−0.80, ns)`。
**⇒ 光把 30 轮/98k 缩到 8 轮/8k 就吃掉 nd267→单卡缺口的 57%**，
剩给 SP + FSDP + 余下 8 轮的不到 43%。

**注意 `response_length` 单独看是骗人的**（`corr(pd, resp_len)=+0.131, t=0.70` 不显著）：
st5 resp_len 28242 → pd 0.0375，st25 resp_len 28264 → pd 0.0187，**长度几乎相同、pd 差 2 倍**。
那是 entropy 在动。要看长度效应必须先换到 pearson 口径。

## 单卡探针：conv1d 假设**已证伪**，且没有沿序列累积（2026-09-10）

在真训练镜像（`verl-coding:202608292148`）里、L20 单卡、8 条序列 × 20000 token，
用固定 token 两次前向对比（vLLM 生成时记 logprob → HF 重打分同一批 token）：

| | 2B torch conv | 2B Dao conv | **9B torch conv** | **9B Dao conv** | 训练 step 1 |
|---|---|---|---|---|---|
| `probs_diff_mean` | 0.00326 | 0.00326 | **0.00258** | **0.00257** | **0.0413** |
| pearson | 0.99976 | 0.99976 | 0.9997 | 0.9997 | — |
| hf/vllm log_ppl | .7773/.7769 | .7773/.7769 | .3910/.3906 | .3909/.3906 | .3997/.2461 |

三条独立证伪：

1. **conv1d 不是原因。** 训练镜像里 `causal_conv1d` **缺失** ⇒ HF 走
   `F.silu(self.conv1d(...))` torch fallback，vLLM 走自带 fused Triton 口；
   Qwen3.5-9B 有 24/32 层是 `linear_attention`(GatedDeltaNet)，占 75%，看着很像元凶。
   装上 Dao 的 `causal_conv1d` 后：2B **+0.1%**、9B **−0.4%**（预注册判据是 ≥50%）。
   开关确实生效（`probs_diff_max` 0.0906→0.1151），不是没换上。
   （我早先单算一次 op 得到 `max|fused−torch|=0.0625` —— 那是随机输入上的最坏值，
   **不能当端到端效应**，别再拿这种数当证据。）
2. **不沿序列累积。** 按位置分箱是平到略降：0.0039@0k → 0.0023@18k。
   "recurrent state 逐步放大偏差"这个机制也是错的。
3. **不是模型规模。** 9B(0.0026) **低于** 2B(0.0033)。

`fla` 0.5.1 在镜像里**存在** ⇒ GDN chunk kernel 与 vLLM vendored 的 fla 同源，本来就对齐。

## ⇒ 环境对齐的天花板是 ≤11%，不要为它打镜像

上表的 torch-conv 那一列里，**attention kernel 也是不对齐的**（HF `sdpa` vs vLLM
FlashAttention，镜像里 `flash_attn` 缺失），conv 也不对齐，**两者全不对齐的总量只有
0.0033 / 0.0305 = 11%**。所以把 `causal_conv1d` + `flash_attn` 都装上，最多消掉一成。

~~而且 `flash_attn` **装不上**：`2.8.3.post1+cu12torch2.8` 的 wheel 在 torch 2.11 下
`ImportError: undefined symbol: _ZN3c104cuda29c10_cuda_check_implementationEiPKcS2_ib`，
没有 torch-2.11 的 wheel，源码编译 1–2 小时。装了一个坏的 flash_attn 比不装更糟，已卸掉。~~
**⚠️ 2026-09-11 更正：torch-2.11 的 wheel 是有的，装得上，而且没用。见下面「flash_attn 阴性」一节。**

剩下的嫌疑（16× 的缺口在这里面）：**Ulysses SP / FSDP 本身 / 多轮+prefix caching / A100-vs-L20**。

## 归因测试的设计约束（踩过的两个）

- **`max_token_len = ppo_max_token_len_per_gpu × sp_size`**（`workers/engine/utils.py:80`），
  切片后每卡 token 数**与 SP 无关**恒为 `ppo_max_token_len_per_gpu` ⇒
  **改 SP 不改单卡激活峰值，SP=1 不会 OOM。**
- 但有一道硬 assert：`seqlen_balancing.py:384` `assert max_token_len >= max_seq_len` ——
  单条序列不能超过 micro-batch 预算。**SP=1 时 `max_token_len` 就等于 budget**，
  所以要么缩短序列要么抬 budget（抬 budget 会同时翻倍激活，多一个变量，不要）。
- **`cp_context` 的门是 `sp_size > 1`，不是"样本有没有跨卡"**
  （`verl/models/transformers/qwen3_5.py:214-223`）。`slice_input_tensor` 确实是在
  packed 扁平序列上切连续块（`ulysses.py:121-134`），所以 budget ≥ 单样本长度时
  样本不会被切开 —— 但只要 SP>1，每个 rank 的 `seq_len = total/sp`，`cp_context` 必然被建，
  GatedDeltaNet 就换成 `_prepend_cp_conv_prefix` + CP gated delta rule。
  **⇒ "样本没跨卡所以 SP 退化了"这个担心不成立。**
  PR `6a6242f3` 自测也是这个方向：many-short（不跨卡）output err **5.58e-4** >
  sequence-cut（跨卡）**3.61e-4**。

## ★ 归因结果：Ulysses SP 是主因（2026-09-10，两个探针任务）

同一镜像、同一配置（8 轮 / resp 8192 / n=4 / batch 8 / 1 步），只改 `ULYSSES_SP`：

| | SP=1 | SP=8 |
|---|---|---|
| `rollout_probs_diff_mean` | **0.0032** | **0.0239**（7.5×） |
| `rollout_actor_probs_pearson_corr` | 0.9996 | 0.9333 |
| `rollout_corr/kl` | 0.00045 | 0.0683（**152×**） |
| `rollout_corr/log_ppl_abs_diff` | 0.00078 | 0.0679（87×） |

**SP=1 的 0.0032 与单卡探针的 0.0033 完全吻合** ⇒ 上一节"环境对齐天花板 ≤11%"那句话
在 SP=1 下是"100%"，在 SP=8 下才是 11%；**缺口全部由 SP 贡献，不是 FSDP、不是多轮、
不是 A100-vs-L20。**（那三个嫌疑到此排除，别再排查。）

嫌疑落在 `fla.ops.cp` 的 CP 前缀扫描：affine 参数 (M, H) 的复合走 `tl.dot(f32, f32)`，
**Triton 对 fp32 输入默认 tf32（10 位尾数）**，而非 CP 路径用 CUDA core 的 f32 fma。
上游 fla PR **#1180**（"[CP] use tf32x3 affine chain in kcp"，2026-08-27 合入）就是修这个，
**但在 0.5.2 之后、任何 release 都还没带上**，镜像里是 0.5.1。

A100 实测三档 `tl.dot` 精度（64³，fp64 参考）：

| | max_abs_err | rel |
|---|---|---|
| tf32 | 2.376e-02 | 7.178e-04 |
| tf32x3 | 1.823e-05 | 5.506e-07 |
| ieee | 7.565e-06 | 2.285e-07 |

**tf32 比 ieee 差约 3100×。三档在 A100(cc8.0) 上都能编译**（担心 tf32x3 被 TMA gate，没有）。

回填工具：`engine_probe/patch_fla_cp_precision.py`（给 6 处 `tl.dot` 加
`input_precision=FLA_CP_PRECISION`，env `ieee|tf32x3|tf32`，**默认 tf32 与原版逐位相同**，
所以打了补丁的镜像同时是 control 臂）。镜像
`verl-coding-flacp-20260910:latest`（base `202608292148`）。

**⚠️ 这条线索已被双臂实验证伪，见下面「tf32 不是原因」一节。补丁本身仍是好的（可复用），
但不要再把 tf32 当作 SP 偏差的解释。**

## CP 数值实验的四个坑（2026-09-10，全是我自己踩的）

想在小规模上直接量 tf32→ieee 的效果，四次都测出 0 或假信号：

1. **Triton `@jit` 读不了普通模块全局量** —— `NameError: Cannot access global variable ...
   Triton kernels can only access global variables that are instantiated as constexpr`。
   必须 `FLA_CP_PRECISION = tl.constexpr(os.environ.get(...))`，且 header 要插在
   `import triton.language as tl` **之后**。（fla 自己的 `solve_tril.py` 走的是另一条路：
   把 `DOT_PRECISION` 塞进 `triton.Config` autotune 参数。）
   **这个坑没有 GPU 就发现不了 —— 补丁差点带着 NameError 出厂。**
2. **world_size=2 结构上测不出这个 bug。** `chunk_gated_delta_rule_fwd_h_pre_process` 被
   `if not context.is_last_rank:` 门住；world=2 时 rank0 的入态恒为零、rank1 是 last rank，
   **没有任何 rank 会施加非平凡的 M**。`b_h = tl.dot(b_m, b_h) + b_he` 那几个复合点需要
   `pre_num_ranks ≥ 2`。实测 stock/tf32/tf32x3/ieee 四臂
   `state_rel_err_by_rank = [0.0, 0.0]` 逐位相同，而 `inherited_state_by_rank=[0.0, 3.50]`
   证明态确实跨了卡。**要复现必须 ≥4 卡（训练是 SP=8，rank7 要串 7 个 M）。**
3. **脏 triton cache 会造出假的非零差。** 换 `FLA_CP_PRECISION` 不改 kernel 源码哈希，
   autotune/编译缓存命中旧产物。我先测出的 1.5e-05 … 7.5e-05 那批数
   **`rm -rf /root/.triton/cache` 后全部不复现，是缓存伪影，已作废。**
   **换 precision 必须清 cache。**
4. **比 bf16 输出太钝。** op 返回 bf16，affine chain 的误差低于 1 ulp 就整个消失
   （观测到的 max_abs 量子是 2^-11 / 2^-9）。敏感量是 `pre_process` 交给每个 rank 的
   **fp32 recurrent state**，要拿 `output_final_state=True` 在同一边界上跑精确前缀来比。

另外**输入设计能让 CP 变成数学上的 no-op**：g 是逐 token 对数衰减，rank 继承的态被
`exp(Σg)` 衰减。`g_scale=0.05` 时 1024 token 衰减 exp(-40)，CP 与非 CP 逐位相同 ——
测出 0.0 不是"没 bug"，是"这个输入下 CP 不起作用"。必须报告 `slice_decay` 才知道测了什么。

## ★ tf32 affine chain **不是** SP 偏差的原因（2026-09-10，双臂实验证伪）

同一镜像（`verl-coding-flacp-20260910`）、同一配置、SP=8，只改 `FLA_CP_PRECISION`：

| | SP=1 参照 | **tf32**（control） | **ieee**（treatment） |
|---|---|---|---|
| `rollout_probs_diff_mean` | 0.00323 | 0.02655 | **0.02908** |
| `rollout_actor_probs_pearson_corr` | 0.99956 | 0.9220 | **0.9178** |
| `rollout_corr/kl` | 0.00045 | 0.0775 | **0.0785** |
| `k3_kl` | 0.00049 | 0.0736 | 0.0953 |
| `log_ppl_abs_diff` | 0.00078 | 0.0659 | 0.0688 |
| `actor/entropy` | 0.3534 | 0.3621 | 0.3692 |

**ieee 完全没有往 SP=1 方向移动，甚至略差**（差异都在批间噪声内）。
预注册判据（拉向 0.0032 / 0.9996）**完全没达到**。

顺带确认了 control 臂复现：打了补丁但设 tf32，与 stock 镜像的上一轮 SP=8
（0.02392 / 0.9333 / 0.0683）落在同一档，**镜像不是混淆因子**，
"默认 tf32 逐位等同原版"在端到端也成立。

**阴性结果可信，因为 treatment 确实生效了，两条独立证据：**

1. job spec 里 `{"name": "FLA_CP_PRECISION", "value": "ieee"}`。
2. **计时签名 —— 只有走 fla CP kernel 的 FSDP 路径变慢，vLLM 生成路径没变：**

| | tf32 | ieee | 倍数 |
|---|---|---|---|
| `timing_s/old_log_prob` | 63.5 | **297.9** | **4.7×** |
| `timing_s/update_actor` | 143.1 | **357.0** | **2.5×** |
| `timing_s/ref` | 10.0 | 14.4 | 1.4× |
| `timing_s/gen`（vLLM，不走 fla CP） | 791.9 | 822.3 | 1.04×（噪声） |

kernel 确实以 ieee 重编译、确实慢了 2.5–4.7 倍，**而偏差纹丝不动。**

**⇒ fla PR #1180 修的是真问题（单次 matmul tf32 vs ieee 差 3100× 是 A100 实测），
但它在我们这个规模下不是主导项。不要再拿 tf32 解释 SP 偏差。**
（补丁本身留着，`engine_probe/patch_fla_cp_precision.py`，以后要复现或做别的精度实验可复用。）

**当时列的剩余嫌疑里，第三条已经坐实（见下一节）。另两条仍未区分，但优先级降了：**
- **CP 把递归拆开这件事本身** —— 即使全 ieee，前缀扫描的求和顺序也和非 CP 路径不同，
  这是结构性的，不是精度问题。
- `_prepend_cp_conv_prefix` 的跨 rank conv 边界处理。
- ~~那 8/32 层 full-attention 的 Ulysses all-to-all。~~ ← **是它，而且是正确性 bug，见下节。**

## ★★ 根因：sdpa 下 Qwen3.5 的 Ulysses all-to-all 压根没跑（2026-09-11）

**这不是精度问题，是静默的正确性 bug。** 两个事实复合而成：

1. `verl/models/transformers/monkey_patch.py:541-549` —— qwen3_5 分支只把 Ulysses 装进
   `transformers.integrations.flash_attention._flash_attention_forward` 这**一个** hook。
2. `verl_coding/run_coding_practice_qwen3_5_9b_4l20.sh:118` 强制
   `+actor_rollout_ref.model.override_config.attn_implementation=sdpa`
   （镜像里 flash_attn 装不上，transformers 5.14.1 遇到 `flash_attention_2` 缺库直接 ImportError，
   不静默回退，所以当初才加的这个 override）。

HF 的 `Qwen3_5Attention.forward` 经 `ALL_ATTENTION_FUNCTIONS.get_interface("sdpa", ...)` 派发，
**永远到不了那个 hook**。于是 SP=8 下那 8/32 层 full_attention
**每个 rank 只在自己那 1/8 连续片段内 attend**，看不见前面 rank 的 token。
verl 没有任何 guard —— `apply_monkey_patch` 只 assert head 整除。

llama / qwen2 不受影响：它们有自己的 `*_attn_forward` 补丁，在 `ALL_ATTENTION_FUNCTIONS`
**外面**做 a2a。qwen3_5 原本没有。

**判据：shard 0 恰好为 0。** CPU / gloo 2 卡复现（`engine_probe/test_sp_sdpa_attention.py`）：

```
_flash_attention_forward called 0 times   <-- a2a 从没执行
global rel_l2 = 2.6303e-01
  shard 0: rel_l2 = 0.0000e+00      <-- rank0 的片段因果自洽，看不见前面也不影响
  shard 1: rel_l2 = 3.7200e-01
```

rank0 **逐位精确**、rank k>0 巨错 —— 这个签名把它和数值误差彻底分开，
**精度问题不可能给出 exact 0**。这也解释了 tf32/ieee 双臂为什么纹丝不动：修错地方了。

**影响面比 metric 本身重要**：rollout 来自 vLLM 是对的，但**训练侧前向一直是错的** ——
`old_log_prob`、`ref`、`update_actor` 的梯度，所有 `ULYSSES_SP=8` 的 run 都受影响
（nd267、OPD 那几次、SP 归因的两轮探针全在内）。

**修复**（已在仓库）：`verl/models/transformers/qwen3_5.py` 加 `qwen3_5_attn_forward`，
照 `qwen2.py` 的写法在 `ALL_ATTENTION_FUNCTIONS` 外面做 gather-seq / scatter-heads；
`monkey_patch.py` 在 `ulysses_sp_size > 1` 时挂到 `Qwen3_5Attention` / `Qwen3_5MoeAttention`。
两个不显然的点：

- **SP>1 必须重算 `num_key_value_groups`**。`repeat_kv(k, max(sp//kv_heads,1))` + head scatter
  之后，sdpa/eager 内部还会按 `module.num_key_value_groups` 再 `repeat_kv` 一次，
  用 config 里的旧比值就错。改成 `q.size(1)//k.size(1)`，调用前后临时换掉再还原。
- `attention_mask is not None` ~~直接 raise~~ —— 本地 shard 的 mask 在 a2a 之后没有意义。
  **第一版就是这么写的，实跑直接炸，见下节。**

验证：world ∈ {2,4} × `num_key_value_heads` ∈ {4,1}，**每个 shard rel_l2 全 0.0000e+00**，
与 SP=1 逐位相同。

### ★ 修复的后半段：sdpa 会物化一个**本地 shard 的** block-causal mask（2026-09-11）

第一版（只做 a2a + 对 mask raise）提交后在 `actor_rollout_compute_log_prob` 就挂了：

```
NotImplementedError: Qwen3.5 Ulysses SP expects attention_mask=None ...
got a mask of shape (1, 1, 15789, 15789) built for the local shard.
```

来路是 HF 自己：verl 传给模型的 `attention_mask` **是 None**，但
`Qwen3_5TextModel.forward` 会调 `create_causal_mask(...)`，而
`masking_utils.find_packed_sequence_indices` 看到 packed `position_ids`（每条样本从 0 重启）
就物化一个 4-D block-causal mask。`15789 = 126312/8` 是**本地 shard** 的长度 ⇒
a2a 之后 q/k/v 是全长 126312，这个 mask 的行列全对不上。

**不能改成"建一个全局 mask"**：126312² 个 bool ≈ **16 GB**，物理上不可行。

**正解是 packed-varlen：按全局 `cu_seqlens` 分段，每段用 `attention_mask=None` 调
attention interface。** `sdpa_attention_forward` 在 mask 为 None 时自己走
`is_causal = q_len>1 and attention_mask is None and module.is_causal` ⇒
拿到 flash/mem-efficient kernel，显存 O(N)，而"段内 causal + 段间不可见"**正是**
block-causal 的定义，不是近似。

**拦路的是一行 `pop`**：`qwen3_5_decoder_layer_forward` 开头
`kwargs.pop("cu_seqlens")` / `pop("cu_seqlens_cpu")`，只喂给 `self.linear_attn(...)`，
`self_attn(...)` 拿到的是 pop 之后的 `**kwargs` ⇒ **注意力层根本看不到 cu_seqlens**。
修法是 SP>1 时把这两个键塞回 `self_attn` 的 kwargs（`attn_kwargs`）。

`cu_seqlens` 确实是**全局未切片**的，且 SP 的 pad 已经作为**末尾多出来的一段**追加好了
（`transformer_impl.py:1121-1132`）；`pass_packed_cu_seqlens` 在 `_build_module()`
**之后**探签名（`transformer_impl.py:571`），那时 forward 已经被换成
`forward_with_normal_backend`（签名里有 `cu_seqlens`）⇒ 恒为 True，不用担心拿不到。
读边界**优先用 `cu_seqlens_cpu`**，读 cuda 那份会每层同步一次。

另外两个必须做的：
- **flash backend 要跳过**（`"flash" not in attn_impl`），否则 `_flash_attention_forward`
  的 hook 会**再做一次 a2a**，两次叠加就错了。
- `cu_seqlens` 缺失**且** mask 非 None 时仍然 raise —— 否则分段退化成"整条 causal"，
  样本 i 能看到样本 i-1，是静默的跨样本泄漏，比报错糟得多。

验证扩成 world ∈ {2,4} × kv ∈ {4,1} × 分包 ∈ {单条 64、40+24、16×4} 共 12 组：
分段边界与 shard 边界对齐时**逐位 0.0000e+00**，不对齐（40+24）时 **5.4e-08** ——
那是 sliced-causal 与 masked-full 的求和顺序差，fp32 噪声量级，不是正确性问题。

### ★★★ 结案：修完 SP 偏差**全部消失**，GDN/CP 路径不贡献任何可测偏差（2026-09-11）

`probe-spattn-fix2-0911`（2098301506187960320，SP=8、8 轮、resp 8192、n=4、batch 8、1 步，
与前两轮探针**同配置**）：

| | SP=1 参照 | SP=8 stock（有 bug） | **SP=8 修复后** |
|---|---|---|---|
| `rollout_probs_diff_mean` | 0.00323 | 0.02392 | **0.00328** |
| `rollout_actor_probs_pearson_corr` | 0.99956 | 0.9333 | **0.99953** |
| `rollout_corr/kl` | 0.00045 | 0.0683 | **0.00024** |
| `rollout_corr/k3_kl` | 0.00049 | — | 0.00045 |
| `rollout_corr/log_ppl_abs_diff` | 0.00078 | 0.0679 | **0.00046** |
| `actor/entropy_loss` | 0.3534 | — | 0.3516 |

**四个量全部回到 SP=1 档，kl / log_ppl 甚至更低。** entropy 0.3516 vs 0.3534，
所以不是 entropy 污染造成的假回归（判据见上面 pearson 那节）。

**⇒ 两条结论，都要记住：**

1. **`ULYSSES_SP` 不再是偏差来源。** 之前写的"SP 是主因"成立，但"主因"的全部内容就是这个
   sdpa 下 a2a 没跑的正确性 bug，不是任何精度/结构性问题。
2. **那 24/32 层 GatedDeltaNet 的 fla CP 路径不贡献任何可测偏差。**
   之前列的两条残余嫌疑（"CP 拆递归的求和顺序"、`_prepend_cp_conv_prefix` 的跨 rank conv
   边界）到此**一并排除，不要再查**。tf32 阴性结果的真正解释就是修错了地方。

剩下的 0.0033 与**单卡探针的 0.0033 完全吻合** ⇒ 现在偏差 100% 来自单卡 kernel 不对齐
（HF sdpa vs vLLM FA、torch conv vs fused conv）。**这一成也已经逐项测过并且全是阴性
（conv −0.4%、attention −1.0%），0.0033 就是地板，不要再投入**，见下节。

**必须回头重测的东西**：OPD 自蒸馏地板 `abs_loss ≈ 0.2` 是在带 bug 的 SP=8 下量的
（[[opd-engine-bias-floor]]），`rollout_corr/kl` 从 0.157 → 现在同口径 0.0002 量级，
**那条 0.2 的地板几乎肯定大幅下移**，27B 教师的判据要重建。
另外所有 `ULYSSES_SP=8` 的历史 run（nd267 的 epoch2 腰斩分析、OPD 两轮）都是在错误的
训练侧前向下跑的，结论要重新审视。

**CPU 复现脚本的四个必须项**（少一个就测不出来）：
- **gloo 没有 `alltoall`** —— 得把 `verl.utils.ulysses.all_to_all_tensor` 换成 all_gather 模拟。
- **`Qwen3_5TextConfig.model_type` 是 `qwen3_5_text` 不是 `qwen3_5`** ⇒
  不手动改掉，`apply_monkey_patch` 整个 qwen3_5 分支都不会进，等于什么都没打。
- **必须喂 `inputs_embeds` 不是 `input_ids`** —— verl 把 qwen3_5 当 VLM，engine 只 pad，
  是 `patch_vlm_for_ulysses_input_slicing` 在 TextModel 内部切 `inputs_embeds`；
  喂 ids 会看到 "expected local slice 32, got 64"。
- **`torchrun` 的 `sys.path[0]` 是脚本所在目录不是 cwd** ⇒ 必须 `PYTHONPATH=/personal/verl`，
  否则 import 到的是已安装的 verl，改了源码也"看不到效果"。

打镜像用 `engine_probe/patch_qwen3_5_sp_attn.py`（幂等、base64 自包含、`--check` 带运行时证明）
+ `engine_probe/Dockerfile.spattn`，产物
**`verl-coding-spattn-20260911c:latest`**（base `202608292148`，含上面的 packed-varlen 后半段；
`...b` 是只有 a2a 的第一版，**会在 mask 上 raise，不要再用**）。

**改完补丁必须验"产物与仓库逐字节相同"**，不要靠眼看：

```bash
mkdir -p /tmp/fakepkg/verl/models/transformers
git show HEAD:verl/models/transformers/qwen3_5.py > /tmp/fakepkg/.../qwen3_5.py   # 干净 HEAD
# 把 patcher 的 verl_dir() 指到 /tmp/fakepkg/verl 跑一遍, 再 diff 仓库的改动版
```

踩过：`FN_SRC` 从仓库抽出来时 `rstrip('\n')` 了，插入时只给 `"\n\n"` ⇒ 产物比仓库**少一个空行**
（`diff` 报 `493a494 >`），要 `"\n\n\n"`。这种一行空白的差别在镜像里看不出来，
但会让"补丁 == 仓库代码"这个前提悄悄失效。

## ★ flash_attn 装得上，但**不降低**偏差（2026-09-11，阴性，别再试）

`/personal/flash_attn-2.8.3+cu12torch2.11cxx11abiTRUE-cp311-cp311-linux_x86_64.whl`
与训练镜像逐项匹配（镜像里 torch 2.11.0+cu128 / `_GLIBCXX_USE_CXX11_ABI=True` / py3.11.14），
`pip install --no-deps` 直接成功，`is_flash_attn_2_available()=True`，L20 上 `flash_attn_func`
真跑得出结果。**上面划掉的"装不上"那句作废** —— 之前失败只是因为拿的是 torch2.8 的 wheel。

单卡 L20 / Qwen3.5-9B（8/32 full_attention，与训练同款）/ 8 条 × 20000 token，
vLLM 生成时记 logprob → HF 在**同一批 token** 上重打分，两臂只差 `attn_implementation`
（conv 两臂都固定为 torch fallback）：

| | `sdpa` | `flash_attention_2` |
|---|---|---|
| `probs_diff_mean` | 0.0027472 | **0.0027186（−1.04%）** |
| `pearson` | 0.9996885 | 0.9996966 |
| `probs_diff_max` | 0.1819 | 0.1528 |
| hf/vllm `log_ppl` | .42953/.42913 | .42950/.42913 |

**预注册判据是 ≥50%（和 conv 那次同一条），实测 −1.0%。** treatment 确实生效：
日志有 `### attn=flash_attention_2`，且 `probs_diff_max` / `logp_diff_mean` 都变了
（不是同一次前向的复制）。按位置分箱两臂逐档同形（0k 0.00425 vs 0.00416 … 18k 0.00237 vs 0.00234）。

**⇒ 单卡 kernel 不对齐的两项到此全部测完且全阴：conv −0.4%、attention −1.0%。
剩下的 0.0027–0.0033 不是哪个 kernel 能换掉的，是 bf16 下 HF 整段前向与 vLLM
chunked-prefill 的 GEMM 规约顺序差**（pearson 0.9997 = 已经贴着 bf16 的分辨率）。
**不要再为"对齐引擎"做任何事。**

装 flash_attn 剩下的唯一理由是**性能/少维护一份补丁**（可以不再 `override attn_implementation=sdpa`，
走 verl 原生 `_flash_attention_forward` 的 a2a）。那条路每层要 all_gather 一次 `position_ids`
（`monkey_patch.py:87-147`），我们的 sdpa 补丁按全局 `cu_seqlens` 分段**零额外通信**，
所以**换过去不保证更快**，要拿实测 `timing_s/old_log_prob` 说话。

### 原生 flash 的 SP 路径 preflight：通过（2026-09-11，2×L20 真 NCCL）

"qwen3_5 + SP 从来没人真跑过原生 flash a2a"这个风险**已经排掉**。
`engine_probe/test_sp_attn_gpu.py`（bf16 / 2 层纯 full_attention / world=2 / n_kv=4，
参照是同模型同 dtype 同 backend 的 SP=1）：

| | 单条 512 | packed 320+192（**段边界与 shard 边界不对齐**） |
|---|---|---|
| `sdpa`（我们的补丁） | **0.0** 逐位 | 6.19e-03（两 shard 同量级，bf16 噪声） |
| `flash_attention_2`（verl 原生 hook） | **0.0** 逐位 | **0.0** 逐位 |

`_flash_attention_forward` 计数 `sp_off=2 sp_on=2`（每层一次）⇒ hook 确实在跑；
`position_ids` 经 `Qwen3_5Attention.forward` 的 `**kwargs` 一路传到
`_ulysses_flash_attention_forward`，a2a 的门（`position_ids is not None`）正常触发。

**判据要看的是 shard 之间的不对称，不是"误差非零"。** 修复前的 bug 签名是
shard0 **恰好 0** 而 shard1 **0.372**；这里 sdpa 那 6.19e-03 两个 shard 齐平
（我的补丁按段切 causal，与 masked-full 的求和顺序不同，fp32 下是 5.4e-08，bf16 下放大到
6e-03 —— bf16 的相对分辨率本来就是 2^-8≈3.9e-03）。flash 那边因为 varlen 边界是从
`position_ids` 重算的，与分片无关，所以两种分包都逐位相同。

⇒ **原生 flash 路可以用**，`run_coding_practice_qwen3_5_9b_4l20.sh` 的 `ATTN_IMPL`
默认就是 `flash_attention_2`（`40569084`），装不上 flash_attn 的镜像上传 `ATTN_IMPL=sdpa` 逃生。

### 镜像谱系（用哪个）

| tag | base | 内容 |
|---|---|---|
| `verl-coding-spattn-20260911c` | `202608292148` | 只有 SP 补丁。**旧，别用** |
| `verl-coding-spattn-fa:20260911` | 同上 | + flash_attn 2.8.3。**但 run 脚本是 base 那份，硬编码 `attn_implementation=sdpa`，切 backend 必须尾部 `++...` 覆盖** |
| **`verl-coding-spattn-fa:20260911b`** | 同上 | + 仓库 HEAD 的 run 脚本 ⇒ `ATTN_IMPL` env 真的存在、默认 flash。**用这个** |

（全在 `registry.dp.tech/dptech/dp/native/prod-1760009/11106/` 下。）

**`:20260911b` 里的 `verl_coding` 仍停在 `c03260cb` 之前** —— 那个提交把沙盒
`timeout=18000` 改成 `sandbox_timeout` 默认 **7200**（寿命 5h→2h），是实质行为改变，
故意没打进去，免得混进"只修 SP bug"的对照里。要上它得单独打一版。

**换 site-packages 里的文件前先用 md5 gate 住 base 的状态**：
镜像里那份 run 脚本的 md5 必须等于仓库 `6c19df7b~1` 的 `0dd3bd88...`，
不等就说明 base 不是你以为的那个 commit，直接 `cp` HEAD 会静默带进/丢掉别的改动。

**CPU 沙盒上不要用 `is_flash_attn_2_available()` 验 flash_attn** ——
它的门是 `is_available and (is_torch_cuda_available() or is_torch_mlu_available())`，
无 GPU 时**结构上恒为 False**，看起来像"装坏了"。CPU 侧能验的判据是
`import flash_attn` + **`import flash_attn_2_cuda`**（后者是编译好的 CUDA 扩展，
ABI 不匹配会在这里炸）；`is_flash_attn_2_available()` 留到 GPU 沙盒再看。

## lbg 沙盒/镜像的实测约束（2026-09-10）

- `lbg sdbx exec`：**所有 flag 必须在 `sandbox_id` 之前**，用 `--cwd` 指定目录，
  命令要作为**单个 shell-quoted 参数**传（`-- bash -lc '...'` 不行，会在 `/root` 跑）。
- `lbg sdbx template rm` 需要 `--force`。
- **`lbg image commit --name` 必须带显式非 latest tag**（`--name verl-coding-spattn-fa:20260911`）,
  否则 400 `TAG_REQUIRED`。而 `lbg sdbx template create` 反过来**拒绝** `:latest`,
  `lbg sdbx create --image` 则两者都收 —— 三条路径的 tag 规则各不相同。
- **集群镜像缓存预热会卡住 create**（400 `image preparation is still running` / `code: 10`），
  新 commit 出来的镜像尤其容易卡几十分钟。**base 镜像 `verl-coding:202608292148` 一直是热的**,
  所以要临时验证什么, 从 base 起沙盒 + `--mount-user-storage` 拿 `/personal` 里的补丁/wheel,
  比等新镜像预热快得多（补丁脚本本来就是幂等自包含的）。
- **`lbg image build` 只接受 ≤64 KiB 的 Dockerfile，没有 build context** ⇒
  要带进镜像的文件只能 base64 内联进 `RUN`。
  把 `--check` 和一次 `import` 放进同一个 `RUN`，补丁没打上就构建失败，等于自带验收。
- **构建失败要看 `lbg image build-log <id>`，不要看 `lbg image get` 的 `errorMsg`** ——
  后者只回显被截断的 base64（`BUILDKIT_STEP_FAILED ... failedStage="RUN echo \"IyEv..."`），
  完全看不出为什么。`build-log` 给的是真实 buildkit stdout，一眼就是那行 AssertionError。
  （2026-09-11 踩到：`--check` 的 smoke-test 模型 4 个 head 却传 `ulysses_sp_size=8`，
  被 verl 自己的整除 assert 拦下 —— 与磁盘、镜像大小无关。构建本身 **43 秒**就完成。）
- **`/personal` 在沙盒里挂得上**：`lbg sdbx create <template> --mount-user-storage`，
  于是本地 NAS 上的 wheel / 模型（`/personal/Qwen/Qwen3.5-9B`）可以直接用，
  不用为传文件发愁（`lbg sdbx files write` 只能写内联文本，传不了 200MB 的 wheel）。
  `pip install` 必须给**完整的 wheel 文件名**，`cp x.whl /tmp/fa.whl` 再装会被
  `Invalid wheel filename (wrong number of parts)` 拒掉。
- **多卡 GPU 沙盒基本要不到**：`ali-openkruise-a100-prod` 上根本没有多卡 A100 SKU；
  默认 provider 的 8×A100 `Unschedulable`、8×L20 `NoStock`、4×L20 502，**只有 2×L20 起得来**。
  **调 CP 数值这种需要 ≥4 卡的事，别指望沙盒，直接提 Trisol。**
- 创建沙盒收到 **502 先查 `lbg sdbx list --query failed --json` 再重试**，盲重试会攒孤儿沙盒。

---

# OPD（On-Policy Distillation）落地记录

> 目标：vanilla agentic OPD，Qwen3.5-27B 当 teacher。**不follow MAD-OPD/ReOPD/SEED 那些算法。**

## verl 原生 OPD 是支持 agentic 的（已核实，不用改代码）

`_compute_teacher_logprobs` 是在多轮 agent loop **跑完之后**、对整条 prompt+response 调用的，
工具返回的 token 由 `response_mask` 在 loss 里屏蔽。两条路径都接好了：
`verl/experimental/agent_loop/agent_loop.py:821`，以及我们实际用的 TQ 路径
`verl/trainer/ppo/v1/agent_loop_tq.py:144`。`TaskCreateLoopOutput.as_dict()` 会把
`teacher_ids` / `teacher_logprobs` 从 `extra_fields` 提到顶层。

## 纯 Thinking-Machines OPD 的开关组合

```
distillation.distillation_loss.loss_mode=k1
distillation.distillation_loss.use_policy_gradient=True
distillation.distillation_loss.use_task_rewards=False
```

`losses.py:209-226` 的语义：`use_task_rewards=False` 会先把 `policy_loss = 0.0`，
再 `policy_loss += distill_loss * 1.0`（系数被强制成 1.0，不走 `distillation_loss_coef`）。
`use_policy_gradient=True` 时 advantage = `-distillation_losses.detach()`。
**⇒ 无任务奖励、无 rloo，纯蒸馏。**

`use_topk=False`（k1 就是）时 `_validate_topk_logprobs` 直接 return
（`verl/workers/config/distillation.py:188-191`），所以 vLLM teacher **不需要**
额外配 `max_logprobs`，那道 engine 启动门根本不触发。

## ★ 教师 vLLM 的 `prompt_logprobs` fp32 log_softmax OOM（大词表必踩）

`gpu_memory_utilization` 的显存预算是启动时 dummy forward 量出来的，**那时不知道会有
`prompt_logprobs`**，所以下面这块 buffer **完全不在预算内**：

```
gpu_model_runner.py:3794 _bookkeeping_sync -> _get_prompt_logprobs_dict
gpu_model_runner.py:5644 -> sampler.compute_logprobs(logits)
sampler.py:306          -> logits.log_softmax(dim=-1, dtype=torch.float32)
```

普通生成只对最后一个位置算 logits；`prompt_logprobs` 要对**整个 prefill chunk** 算，
再 upcast 到 fp32，于是每 chunk 需要 `[max_num_batched_tokens, vocab] fp32`。

**Qwen3.5 词表 248320（约是常见 128k 的两倍）**，教师 yaml 默认
`max_num_batched_tokens=8192` ⇒ `8192 × 248320 × 4 B ≈ 8.1 GB`。
实测报 `Tried to allocate 7.33 GiB`（÷4÷248320 = 7923 tokens，对得上 chunk 大小），
`gpu_memory_utilization=0.8` 下只剩 7.28 GiB，差 0.05 GiB 炸掉。

**修法：调小 `distillation.teacher_models.teacher_model.inference.max_num_batched_tokens`（8192→2048，
buffer 7.3GB→1.9GB）+ `inference.gpu_memory_utilization` 0.8→0.6。**
调大 TP **没用** —— vLLM 的 logits 是 all-gather 之后再算 softmax 的，chunk 大小是唯一杠杆。
代价是 114k 序列从 14 个 chunk 变 56 个，但教师不是瓶颈（单条 rollout 就 219s）。

教师 engine 一死会级联出 `AssertionError: number of items:[0] < k_partitions:[1]`
（`balance_batch` 拿到空 batch），**那是级联不是根因**，别去查 balance_batch。

## 教师池是**额外**占卡的

`distillation.n_gpus_per_node × nnodes` 必须等于 `Σ(num_replicas × per_replica_world_size)`，
否则 `distillation.py:281-286` 直接 ValueError。单教师时 `num_replicas` 和 `key` 自动推导
（`pool_size // per_replica`、`"default"`）。总卡数 = trainer + teacher，**不是共用**。

教师的长度参数从 `actor_rollout_ref.rollout` 用 `oc.select` 插值继承，然后
`validate_and_prepare_for_distillation` 会**原地改写**成
`prompt_length += response_length; response_length = 1`，所以
`max_model_len >= prompt + response + 1`（我们是 114689）。

## 镜像里只有 vLLM，没有 sglang

`registry.dp.tech/dptech/dp/native/prod-1760009/11106/verl-coding:202608292148`：
vLLM 0.26.0 可用，`import sglang` → `ModuleNotFoundError`。
**不要为了 sglang 退回 0821 那个镜像** —— 会丢掉 0821→0829 之间所有改动，
而且 k1 路径下 vLLM teacher 本来就够用。

## Trisol 多节点跑 verl 的契约（平台不起 Ray，要自己 bootstrap）

平台注入 `NODE_RANK` / `MASTER_ADDR` / `MASTER_PORT` / `NNODES` / `NPROC_PER_NODE` /
`WORLD_SIZE` / `TRISOL_POD_IP` / `NCCL_NET=IB` / `BASH_ENV=/tmp/trisol-bash-env`，
但**不会起 Ray**。要点：

- **`BASH_ENV=/tmp/trisol-bash-env` 注入 `set -e`** —— 自定义命令里任何非零 rc 都会中止任务，
  启动脚本第一行必须 `set +e`。
- **平台注入的 `NNODES=2` 会被 run 脚本当成 `trainer.nnodes`** —— trainer 只该占 1 个节点时
  必须在尾部追加 `trainer.nnodes=1`（hydra last-wins）。
- **rank1 不能用 `ray start --block`** —— 任务要等**所有** pod 退出才终止，worker 得自己
  轮询 head 的 6379 端口，探测不到再退。
- `ray.init()` 不传 `address` 时会认 `RAY_ADDRESS` 环境变量，
  `ppo_trainer.yaml` 的 `ray_kwargs.ray_init` 只有 `num_cpus: null`，**不需要改 verl**。
- 提交用 `--nodes 2 --gpus-per-node 8`（**没有** `--node-count` 这个 flag），
  多节点还要 `--transport rdma --rdma-custom-image-ack`。

### ★★ 单节点任务**一个都不注入**，多节点 launcher 直接复用会死（2026-09-17）

上面那句"平台注入 `NODE_RANK` / `MASTER_ADDR` / `NNODES` / `WORLD_SIZE`"**只对多节点成立**。
`--gpu-count 8`（单节点）实测这四个全是**空**的：

```
+ TOTAL=16                                  <- ${WORLD_SIZE:-16} 落到硬编码默认
+ HEAD=:6379                                <- ${MASTER_ADDR} 为空
+ export RAY_ADDRESS=:6379
+ echo '=== rank= host=... nnodes= world=16 ==='    <- NODE_RANK / NNODES 全空
ValueError: Malformed host:                 <- ray.init() 读了 RAY_ADDRESS=":6379"
```

`opd-flashnext-7809-bs128n1-0916` 就这么在 **47 秒**挂掉（`--backoff-limit 0` ⇒ 直接终态）。
**`ray start --head` 其实成功了**（日志里有 `Ray runtime started`），死在后面那个
`ray.init(address="auto")` 的就绪探针上 —— 所以看到 "Ray 起来了" 不代表 bootstrap 通了。
而且 `TOTAL=16` 还会让就绪探针等 16 张卡等到超时，是第二个独立的死法。

修法两行，双向兼容（`NODE_RANK` 空走 `${NODE_RANK:-0}` 恰好是 head 分支，不用改）：

```bash
TOTAL=${WORLD_SIZE:-$NG}
[ -n "${MASTER_ADDR}" ] || MASTER_ADDR="${TRISOL_POD_IP:-127.0.0.1}"
HEAD="${MASTER_ADDR}:6379"
```

⇒ **多节点 launcher 改单节点跑之前，先把所有 `${平台变量:-硬编码}` 的默认值过一遍**，
那些默认值是按 2 节点写的，单节点下会静默生效。

## ★ mktemp 的工具配置在多节点下 FileNotFoundError

`run_coding_practice_qwen3_5_9b_4l20.sh:92-96` 用 `mktemp /tmp/coding_tools_XXXXXX.yaml`
生成工具配置，**只在 head 上生成**；而 agent loop worker 是 CPU Ray actor，
Ray 会把它们**摊到两个节点**，rank1 上的 `TaskCreateLoopWorkerTQ.__init__()` 直接
`FileNotFoundError: /tmp/coding_tools_XXXXXX.yaml`。单节点永远复现不出来。

修法：启动脚本在**每个节点**都写一份确定路径的副本，再用尾部 override 指过去：

```bash
TOOL_SRC=$(python3 -c "import verl_coding, os; print(os.path.join(os.path.dirname(verl_coding.__file__), 'coding_tools.yaml'))")
sed "s/max_concurrent_sandboxes: [0-9]*/max_concurrent_sandboxes: ${MAX_CONCURRENT_SANDBOXES}/" \
    "$TOOL_SRC" > /tmp/coding_tools_shared.yaml
# 尾部追加: actor_rollout_ref.rollout.multi_turn.tool_config_path=/tmp/coding_tools_shared.yaml
```

## step 0 是自蒸馏诊断，不是训练

teacher 用和 student **完全相同**的权重（`model_path=/trisol/input/model`）：
k1 的 student logprob 来自 FSDP，teacher logprob 来自 vLLM，权重相同 ⇒
**`distillation/abs_loss` 的残差就是 FSDP↔vLLM 的引擎偏差地板**，
顺带把长期存疑的 `training/rollout_actor_probs_pearson_corr ≈ 0.81–0.89` 量化了。
这个地板不测出来，后面 27B 教师的 loss 数值没有参照。

### 自蒸馏地板实测值（2026-09-10，`opd-step0-selfteacher-2node-v4` step 1）

9B self-teacher，`batch=8 × n=4 = 32` 条轨迹，2 节点 16 卡（trainer 8 + teacher 8，TP=2 × 4 replica）。

**开关组合端到端验证成立**：`actor/loss` 与 `actor/distillation/loss` **逐位相同**（−0.06612963229417801），
`actor/pg_loss=−0.00047` 算了但没进 loss，`actor/kl_coef=0` ⇒
`use_task_rewards=False` 的语义（先 `policy_loss=0` 再 `+= distill_loss×1.0`）确认。
`critic/score/mean=0.15625` 仅记录，不进梯度。

**引擎偏差地板 —— 三个独立量测同一件事，全落在 0.15–0.20 nats/token：**

| 量 | 值 |
|---|---|
| `actor/distillation/abs_loss` | **0.2007** |
| `rollout_corr/kl` / `k3_kl` | 0.1574 / 0.1461 |
| `rollout_corr/log_ppl_abs_diff` | 0.1536（FSDP 0.3997 vs vLLM 0.2461） |
| `training/rollout_actor_probs_pearson_corr` | **0.8239**（历史 0.81–0.89 区间内） |

**⇒ 后续真教师（27B）跑出的 `abs_loss` 低于 ~0.2 的部分与引擎偏差不可区分。**
另外 `rollout_corr/training_log_ppl=0.3997` 落在「自采样数据 SFT loss 比 teacher 高」那节
记的 0.343–0.392 区间内（FSDP 路径，与 SFT loss 同口径），两处对上了。

**⚠️ 需要警惕（观察，不是结论）**：self-teacher 下正确梯度**恒等于零**，但

```
actor/grad_norm            : 0.455    ← RL 那几次是 0.01–0.05
actor/distillation/loss_min: -22.71
actor/distillation/loss_max:   7.73   (loss_max_clamp=10.0 未触顶, **仅此自蒸馏成立**)
```

**⚠️ "未触顶"只是自蒸馏的性质，不要当成通例。** 真教师跑的
`opd-27b-7809-bs128n1-0914` 里 `loss_max` **每一步都是 14.7–26.7**，
默认 `loss_max_clamp=10.0` **一直在起作用**。clamp 发生在 range 指标记录**之后**
（`losses.py:263-265`），所以 `loss_min`/`loss_max` 报的是**未截断**的原始 `d_t`，
看到它超过 10 不代表 clamp 没配，恰恰相反。

这 0.455 **全部由引擎偏差驱动，且是系统性偏差不是零均值噪声** —— 方向上把 FSDP 的 student
往 vLLM 的数值行为上拽；单 token 尾巴到 −22.7 说明偏差分布重尾。
真教师那步要看真实信号能否压过这个量级。

**吞吐参考**：`timing_s/step=2259s`（37.7 min），其中 `gen=1800s` 占 80%，
`update_actor=274s`、`old_log_prob=112s`、`ref=33s`；教师 logprob 没有单独的 timing 项。
`num_turns/mean=52.6`（assistant+tool 交替计数，对应 30 个 assistant 轮上限），
`response_length/mean=27986`、`max=68807`，`perf/mfu/actor=0.104`。

### ★★★ SP bug 修完后地板塌掉 1900 倍，上面那张表整体作废（2026-09-11）

同配置（自蒸馏、2 节点、教师 TP=2×4 replica、`batch=8 × n=4`）在
`verl-coding-spattn-fa:20260911` 上重跑两臂，`opd-step1-self-flash-0911` /
`opd-step1-self-sdpa-0911`：

| | v4（带 SP bug） | **flash** | **sdpa** |
|---|---|---|---|
| `actor/distillation/loss`（带符号） | **−6.61e-02** | **+3.52e-05** | **+5.66e-05** |
| `actor/distillation/abs_loss` | 0.2007 | **0.00602** | **0.00570** |
| `rollout_corr/kl` | 0.1574 | 3.65e-04 | 3.09e-04 |
| `rollout_corr/k3_kl` | 0.1461 | 3.32e-04 | 3.21e-04 |
| `rollout_corr/log_ppl_abs_diff` | 0.1536 | 4.13e-04 | 3.94e-04 |
| `rollout_actor_probs_pearson_corr` | 0.8239 | **0.99953** | **0.99950** |
| `actor/grad_norm` | 0.455 | **0.01256** | **0.01280** |
| `loss_min` / `loss_max` | −22.71 / 7.73 | −1.23 / 0.82 | −1.14 / 1.35 |

**带符号的 loss 绝对值降到 1/1900，`abs_loss` 降到 1/33。** 三条独立佐证说明这不是偶然：

- `grad_norm` 0.455 → 0.0126，**回到 RL 那几次的 0.01–0.05 区间**。上一节写的
  "0.455 全部由引擎偏差驱动"得到了直接确认 —— bug 修掉，它就没了。
- `loss_min` 的重尾 −22.7 → −1.2，偏差分布不再重尾。
- `pearson` 0.9995 与单卡探针（0.9996）、`probe-spattn-fix2-0911`（0.99953）**三处吻合**。

**⇒ 上一节"真教师 `abs_loss` 低于 ~0.2 的部分不可区分"这条判据作废，新地板是
`abs_loss ≈ 0.006` / 带符号 loss ≈ **3–6e-05**。** 27B 教师那次的 `+0.0402` 是在旧地板下量的，
相对新地板是 **1100 倍**，所以"27B 有真实信号"这个结论**更强了**而不是被推翻 ——
但那次的 `abs_loss=0.2571` 里约 0.2 是引擎偏差，**要在修复后的镜像上重跑才能读出纯教学信号的量级**。

**两个 backend 的地板是同一个**（`abs_loss` 0.00602 vs 0.00570，差 5.5%），
符合「flash_attn 阴性」那节的结论：单卡 kernel 不对齐只有 ~1% 的效应。

**timing 分不出 backend，不要用它选**：

| | flash | sdpa | |
|---|---|---|---|
| `timing_s/gen`（**vLLM，不经过任何 attention 补丁**） | 2214.7 | 2363.2 | flash 快 **6.3%** |
| `timing_s/old_log_prob` | 113.6 | 119.5 | flash 快 4.9% |
| `timing_s/update_actor` | 207.9 | 180.7 | sdpa 快 13.1% |
| `timing_s/step` | 2614 | 2740 | flash 快 4.6% |

**`gen` 是噪声标尺** —— 它完全走 vLLM，两臂在这一项上结构相同，实测却差 6.3%。
`old_log_prob` 的 4.9% 比噪声还小，`update_actor` 的 13% 与噪声同量级且方向相反。
**单步单次的 timing 差异全部落在噪声里，选 backend 只能按"上游维护哪条路"来定。**

**踩过的两个提交坑（都是我自己的）**：
- `WANDB_API_KEY` 构造 submit 命令时被截断（86→60 字符），任务在 `trainer.fit()`
  第一行 `Tracking(...)` 崩 `wandb.errors.CommError: returned error 401`。
  **崩在 step 0 之前，前面所有基础设施（两节点 Ray / 教师 engine / 工具配置）其实都跑通了**，
  别被"OPD 失败"误导去查蒸馏侧。
- `--dataset` 必须写 `NAME:VERSION_CODE`；`trisol train get` 返回的 `version_code=0`
  是平台归一化后的显示值，**不能照抄**，要用 `wenyon-cli dataset status` 的 `current_version`。

### ★ 27B 真教师 vs 自蒸馏：判据要用**带符号**的 loss，不是 `abs_loss`（2026-09-10）

同配置换成 Qwen3.5-27B 教师，与 v4 自蒸馏对比 step 1：

| | v4 自蒸馏（9B self-teacher） | 27B 真教师 | |
|---|---|---|---|
| `actor/distillation/abs_loss` | 0.2007 | **0.2571** | +28% |
| `actor/distillation/loss`（带符号） | **−0.0661** | **+0.0402** | **符号翻转** |
| `actor/distillation/loss_max` | 7.73 | 19.38 | 2.5× |

**预注册判据是 `abs_loss ≥ 0.35`（地板 0.2 的 1.75 倍），实测 0.2571，没达到。**
但**不要因此判"27B 教师没信号"** —— `abs_loss` 是个坏判据：它把引擎偏差那条重尾
（自蒸馏时 `loss_min=−22.7`）和真实教学信号加在一起，两者同号相加，分不开。

**有信息量的是带符号的 `actor/distillation/loss`：自蒸馏 −0.0661 → 真教师 +0.0402。**
自蒸馏时正确梯度恒等于零，那个 −0.0661 纯粹是引擎偏差的系统性方向；换真教师后
**整体符号翻正**，说明有一个与引擎偏差方向相反、且量级相当的真实信号。
`use_task_rewards=False` 时 `actor/loss` 与它逐位相同，所以这就是实际进梯度的那个数。

**⇒ 以后 OPD 一律看 `actor/distillation/loss`（带符号）+ 它与自蒸馏基线的差，
不要看 `abs_loss` 的绝对值。** 相关：[[opd-engine-bias-floor]]。

**⚠️ 这一节的两个数都是在带 SP bug 的镜像上量的，基线已经换了**（见上一节）：
自蒸馏基线从 −0.0661 变成 **+3.5e-05**，所以 27B 的 `+0.0402` 现在是地板的
**1100 倍**——"27B 有真实信号"这个结论更强了。但 `abs_loss=0.2571` 里约 0.2 是引擎偏差，
**27B 那步必须在修复后的镜像上重跑**，否则读不出纯教学信号的量级。

### 三个同名的 `distillation/*` 量是**不同的东西**，不是同一个量的统计（2026-09-15 读代码）

| 指标 | 是什么 | 出处 |
|---|---|---|
| `distillation/loss_min` / `loss_max` / `mean_loss` | 逐 token 的 `d_t = logπ_student(y_t) − logπ_teacher(y_t)` 的极值/均值，**clamp 之前** | `compute_distillation_loss_range()` |
| `distillation/abs_loss` | 同一个 `d_t` 的 **token-mean 绝对值** | `..._reverse_kl_estimator` |
| `actor/distillation/loss` | **PPO surrogate 标量**（token-mean），才是进梯度的那个 | `distillation_ppo_loss` |

因为 `y_t ~ student`，`E[d_t] = KL(student‖teacher)`（**reverse** KL），单位 **nats/token**。
符号：`d_t > 0` = student 比 teacher 更自信 ⇒ 压下去；`d_t < 0` = teacher 更喜欢 ⇒ 抬上来。

`use_policy_gradient=True` 时 `advantages = -distillation_losses.detach()` 送进 vanilla PPO，
**ratio≈1 时 surrogate 数值上退化成 `mean(d_t)`** —— 实测 `distillation/ppo_kl=4.2e-05`、
`pg_clipfrac=3.3e-04` ⇒ ratio 确实钉在 1，所以 `actor/distillation/loss` 可以直接当平均
reverse KL 读。`use_task_rewards=False` 时它与 `actor/loss` **逐位相同**（实测 0.054208063）。

### ★★ 评测结果以 `results.md` 为准，目录名不能靠猜（2026-09-15，我自己踩的）

**`/personal/sp2/qwen_trajs/cc_deepseek_1w/results.md` 是评测结果的权威表**，
里面声明了每一行对应哪个 `trajectories_*/` 目录。按目录名望文生义会取错数据：

我要评 27B 的 teacher SFT，`ls` 看到 `trajectories_qwen27b_sft_rc_60k_train/`（名字最像），
算出 0.5032、`Δ=−0.0089 / p=0.597`，据此下了"**同一份数据在 27B 上收益归零**"的结论。
**全错。** 正确目录是 `trajectories_qwen27b_teacher_v2/`（results.md 里写着），
重算 `27B base 0.5095 → 0.5613，Δ=+0.0518，McNemar p=5.81e-4`，**显著为正**。

**最刺眼的是判据在我手里没用**：我把 `models=Counter(...)` 打印出来了，
那个目录的 `model` 字段是 **`qwen3-5-27b`** —— 正是 results.md 末尾注明的
"2026-09-03 之前的 SFT 评测数字作废（LoRA 未生效，实际测的是基模）"那个坑。
`model` 回显基模名而不是 LoRA key，说明请求根本没打到 adapter，难怪 ≈ base。

⇒ **两条硬规矩**：
1. 评测前先 `cat results.md` 拿目录映射，不要 `ls` 之后按名字猜。
2. `model` 字段不是"顺手打印一下"，**它是判据**：值必须等于那个 LoRA key，
   等于基模名就直接作废，不要继续算下去。

**噪声基线（results.md 记录）**：同一个模型重跑一次，总体准确率漂 **±1pp**，
但**逐题判定有 14% 翻转** ⇒ 模型间差异小于约 2pp 一律不可信，
**必须配对检验，不能比原始准确率之差**。

### SFT 收益随基模变强而递减，但**不归零**（2026-09-15，修正后的全图）

全部同 harness（`batch_run_trisol.py -c 16`、`agent_trisol.py`、单跑一次、
`eval.correct is True` 为正确、`None` 计 FAIL），train_sampled 800 题：

| 对比 | Δ | McNemar p |
|---|---|---|
| 9B base 0.2313 → 9B merged 0.3550 | **+0.1237** | 1.67e-14 |
| 27B base 0.5095 → 27B teacher-v2 0.5613 | **+0.0518** | 5.81e-4 |
| 9B base → 27B base | +0.2785 | 1.55e-52 |
| 9B merged 0.3577 → 27B teacher-v2 0.5630 | +0.2053 | 2.28e-31 |

同一份 qwen3.7-plus 数据，9B 上 +12.4 点、27B 上 +5.2 点。**看着像减半，
但按"吃掉 teacher−base 缺口的比例"算是一样的**：teacher(3.7-plus) = 0.688，
9B 缺口 45.7 点吃掉 27.1%，27B 缺口 17.8 点吃掉 29.1%。
⇒ **SFT 的效率与基模尺寸无关，变的只是缺口本身。**
不要用"大模型学不动"解释 27B 的小 Δ。

### ★★ `critic/score/mean` 在 `ROLLOUT_N=1` 下是**留出集读数**，但噪声地板 0.036（2026-09-16）

`opd-27b-7809-bs128n1-0914`（bs128 × n=1 / 7808 题 / `data.seed=42`）。两条都要记住：

**1. 它是免费的留出集。** `RandomSampler` 不放回 + `n=1` ⇒ 每步的 128 道题在本 epoch 内
**从没训过**（27 步用掉 3456/7808 = 44.3%，无重复）。不像 RL 那样被"同一批题反复看"污染，
**每步是一次无偏泛化估计**。（`use_task_rewards=False` 下它不进梯度，纯记录。）

**2. 噪声地板正好是二项 sd，别拿单步差值说事。** 每步 128 道**不同**的题、每题 1 条 rollout
⇒ `sd = sqrt(p(1−p)/128) = 0.0357`（题目难度异质**不会**让它变大：每题只抽一次，
`Var = E[p(1−p)]/n + Var(p)/n = p̄(1−p̄)/n` 精确相消）。实测 27 步 sd = **0.0400**
⇒ 超出部分只有 0.018，**几乎全部步间波动都是"这批抽到什么题"**。

**⇒ 选窗口比大小必须做置换检验。** 实测：肉眼选出的 `21-26 vs 11-20` 是 Δ=+0.0508 /
Welch **p=0.005**，看着很硬；但把"所有连续切分里最大的 Δ"当统计量、打乱步序重算 20000 次，
观测值 +0.0610（最优切法是 18-20 vs 21-27，比肉眼选的还极端）对应 **p=0.135**。
**诚实的单一判据是全程线性趋势：+0.00176/step，t=+1.86，p=0.075 —— 还没显著。**

功效换算（斜率维持的话）：t 越过 2 在 ~28 步、2.5 在 ~33 步、3.0 在 ~37 步；
按 1.78 h/step 算再跑 10 步（约 18 h）就能到 t=3。**别在 27 步上硬读。**

交叉验证成立：训练侧 1-10（0.193）vs 11-20（0.195）持平，与评测侧
step20 vs step10 `Δ=−0.0034 / p=1` 吻合 —— 两个口径互相印证。

### ★★★ OPD 不涨点的直接测量：位移小 + 增量方向近正交（2026-09-16）

不要靠 loss 曲线猜"模型动没动"，**直接量 LoRA 的 `ΔW/W`，几分钟、不用 GPU**
（方法见「SFT loss 平不能推断没学到」那节的迹恒等式）。三个 checkpoint 实测
（12 个模块中位，base shard1 覆盖）：

| | `ΔW/W` | 训练量 |
|---|---|---|
| OPD step10（bs128×n1） | **0.1029%** | 1280 轨迹 / 80 次更新 |
| OPD step20（bs128×n1） | **0.1391%** | 2560 / 160 |
| OPD step30（bs128×n1） | **0.1725%** | 3840 / 240 |
| OPD step40（bs128×n1） | **0.2012%** | 5120 / 320 |
| OPD step40（bs8×n4，另一个 run） | 0.107% | 1280 / 80 |
| 自采样 SFT（**阴性**） | 0.256% | 72 步 |
| teacher SFT（有效） | 0.547% | 234 步 |

**⇒ 跑到 step40、4 倍训练量之后，位移仍低于那个 p=0.35 的阴性自采样 SFT（0.256%），
只有有效 teacher SFT 的 37%。**（跨方法可比：都是同基模 + LoRA r32 α64。）

### √t 标度一路成立到 step40（2026-09-17 四点实测）

| 段 | 增量范数 `ΔW/W` | `cos(增量, 之前的累积)` |
|---|---|---|
| 0→10 | 0.1029% | — |
| 10→20 | 0.0806% | **0.1703** |
| 20→30 | 0.0858% | **0.1667** |
| 30→40 | 0.0847% | **0.1427** |

**每段走的距离几乎恒定（0.081–0.086%），方向始终近正交（cos 0.14–0.17，略降）。**
`st40/st10 = 0.2012/0.1029 = ` **1.955 ≈ √4** ——
4× 训练量（1280→5120 轨迹、80→320 次更新）恰好换来 2× 位移，
**随机游走标度到 step40 都没有变成定向行走。**

这与评测侧四点持平（step10/20/30/40，见上面那节）、训练侧
`critic/score/mean` 21-30 与 31-37 持平，是同一幅图像的三个侧面。

**★ "cos(增量, 累积)"的参照向量有歧义，两个数差 4 倍，必须写清楚用的哪个：**

| 定义 | 10→20 | 20→30 | 30→40 |
|---|---|---|---|
| **增量 vs 段首的累积**（本文件一贯口径） | 0.1703 | 0.1667 | 0.1427 |
| 增量 vs 段末的累积 | 0.52 | 0.59 | 0.68 |

后者看起来像"越来越定向"，**那是定义造成的假象** —— 段末累积里本来就含这一段增量，
两者必然越来越像。**一律用前者。**

**★ 同一 run 内 `lora_A` 会漂，不是常数**（2026-09-17 纠正本文件早先的说法）。
我之前写的"`cos(A10,A20)=1.0000`，verl 下 A 在训练中不变"是**四位小数的舍入假象**，
实测 `cos(A10,At)` = 0.99998766 / 0.99990… / **0.99962360**，相对变化 2.7e-03 … 2.6e-02。

⇒ **范数分解不受影响**（每个 step 用自己的 A，见下面配方 1 的拼接写法），
但**任何建立在"同一子空间"上的论证都作废** —— 比如"Δ10 和 Δ20 在同一 rank-32 行空间里
所以余弦可以直接读成梯度方向的重合度"。余弦仍是良定义的 Frobenius 夹角，
但它同时含了 A 漂移的贡献，**不能纯粹解释成"梯度方向互相抵消"。**

**★ 跨 run 的方向余弦在 LoRA 上结构性不可做。** 我先算出"两个独立 OPD run 的
ΔW 方向 `cos=0.0034`，完全正交"，差点当成"OPD 学到的全是噪声"的铁证。**错的**：
两个 run 的 `lora_A` 是不同随机初始化（`cos(A10,A40)=0.0016`，行空间主角 cos 0.077
≈ 随机基线 `sqrt(r/d)=0.088`），`ΔW = B@A` 被约束在 A 张成的 rank-32 行空间里，
**子空间不同 ⇒ 即使真实梯度方向相同 cos 也必然≈0**。做跨 run 方向比较前
必须先验 `cos(A_run1, A_run2)`。

**算增量和夹角的三个配方**（都保持在 `(r,r)` / `(2r,2r)` 规模，不物化 `(out,in)`）：

```python
# 1) 增量范数: ΔW20 − ΔW10 = [B20 | −B10] @ [A20 ; A10]  再套迹恒等式
#    这个拼接写法对 A20 != A10 也成立, 所以上面那条 "A 会漂" 不影响它
B = torch.cat([B20, -B10], dim=1); A = torch.cat([A20, A10], dim=0)
fro = torch.trace((B.T@B) @ (A@A.T)).clamp(min=0).sqrt() * (alpha / r)
# 2) 内积: <ΔW10, ΔW20>_F = tr((B10.T@B20) @ (A20@A10.T)) * (alpha/r)**2
# 3) 夹角: cos(增量, 累积) = (r20**2 - r10**2 - ri**2) / (2*r10*ri)   余弦定理
```

**别忘了 `scale = alpha/r`**（这里是 2）。漏掉它 `ΔW/W` 小一半（0.051% vs 0.103%），
而**比值和余弦都是 scale 不变的** ⇒ 只有绝对值错，其余一切自洽，最容易蒙混过关。
自检：两条不同路径算出的增量范数必须对上。

**lr 的证据：有上调空间，但这个判据在 OPD 下要打折。** 实测全 27 步
`lr` 恒定 **1.5e-5（无 warmup、无 decay）**、`grad_norm` 0.06–0.09 **稳定不衰减**、
`distillation/ppo_kl` **2–4e-05**、`pg_clipfrac` **3e-04** —— 后两个比健康区间
（clipfrac 0.1–0.2）低 2–3 个数量级，表面签名与 2026-09-02「RL 推不动是 lr 太小」完全相同。
**但不能照搬**：OPD 下 `use_policy_gradient=True` + `ppo_epochs=1` 使 ratio 结构性钉在 1
（surrogate 退化成 `mean(d_t)`，见「三个同名 distillation/* 」那节），clipfrac 低有一部分
是定义决定的。`ppo_kl=2e-5` 那条仍然干净 —— 每次更新后策略变化确实极小。

**未知，不要在这里填推测**：为什么同一 run 内增量方向近正交。一个结构性候选
（读代码得出，**未验证**）是 on-policy 的分布漂移 —— student 每步重新采样，
梯度方向本来就不固定；`distillation/loss` 从 step3 起 27 步无下降趋势（0.046–0.055）
与此一致，但同样区分不了"靠近太慢"和"漂移吃掉收益"。要分开需要一个 off-policy
固定数据集的蒸馏对照跑同样的量。

### ★★ bs128/n=1 的 step10 评测：等预算对照下 **`ROLLOUT_N=4→1` 无差别**（2026-09-16）

`opd-27b-7809-bs128n1-0914` 的 step10 checkpoint（`TRAIN_BATCH_SIZE=128`、`ROLLOUT_N=1`、
`PPO_MINI_BATCH_SIZE=16`、7809 题库、`++data.seed=42`）vs `opd-teacher27b-prod-0911` 的 step40
（bs8 × n=4），**四方公共集 n=776**（同 harness、`eval.correct is True` 为正确、`None` 计 FAIL）：

| | 正确 | 率 |
|---|---|---|
| SFT-merged | 280/776 | 0.3608 |
| OPD step40（bs8×n4） | 229/776 | 0.2951 |
| **OPD step10（bs128×n1）** | **213/776** | **0.2745** |
| 基模 | 183/776 | 0.2358 |

| 对比 | Δ | 95%CI | 配对(both/only_b/only_a/neither) | p |
|---|---|---|---|---|
| step10 vs 基模 | **+0.0387** | [+0.010,+0.067] | 134/79/49/514 | **0.0101** |
| **step10 vs step40** | **−0.0206** | [−0.049,+0.008] | 159/54/70/493 | **0.178** |
| step10 vs SFT-merged | −0.0863 | [−0.116,−0.057] | 175/38/105/458 | 1.89e-08 |

**这是一个刻意配成等预算的干净对照** —— 三项训练量全同，唯一变量是题目多样性 4×：

| | 轨迹数 | 题目数 | 优化更新 |
|---|---|---|---|
| step40（bs8×n4，mini4） | 1280 | **320** | 40×2 = 80 次 × 16 条 |
| step10（bs128×n1，mini16） | 1280 | **1280** | 10×8 = 80 次 × 16 条 |

⇒ **`p=0.178`、CI 跨 0：把 `ROLLOUT_N` 从 4 降到 1、题目多样性翻 4 倍，结果不变。**
这确认了「OPD 里 `rollout.n` 是惰性参数」那节的推论在**结果侧**成立：
n=1 不吃亏（也没白赚），省下的预算换成题是安全的。相关 [[opd-rollout-n-inert]]。

**自校验先做再读新数**：同一脚本在 776 集上重算 `base → merged` 得 `+0.1250 / p=2.81e-14`，
与本文件记的 789 集 `+0.1242 / p=2.63e-14` 吻合。

**15 题 `rc=1` 的死因与 step40 那次逐字相同**：`--max-model-len 262144` 撞顶 ⇒
`maximum context length is 262144 tokens ... prompt contains at least 262145 input tokens` ⇒
重试 3–4 次全败（日志里 40 次 400）⇒ `agent_trisol.py` **优雅退出 `sys.exit(1)`、0 个 traceback**，
不写输出文件。上下界 0.2675–0.2863 都夹在基模与 merged 之间，结论不变。
（其中 1 条报的是 `prompt contains 988560208 characters` —— 约 9.9 亿字符，
某次工具返回炸了，这个还没查。）

**`batch_run_trisol.py` 的 `subprocess.run(..., text=True)` 不捕获 stdout/stderr**
⇒ 子进程输出全部继承到父进程的重定向文件里（本次 **1.1 GB**）。找单题死因不能靠
`grep Traceback`（沙盒里模型自己写的代码报错有 545 个 traceback、521 个 AttributeError，
全是噪声），要 grep **API 错误原文**或 `File ".../agent_trisol.py"`（后者为 0 即优雅退出）。

**进度条卡住 ≠ 死了**：我看到 785/800 十小时没动就判定 runner 死了，**错的**。
`ps aux | grep batch_run_trisol` 当时只匹配到**上一个 session 残留的 shell wrapper**
（它的命令行里含有那个字符串），真正的 python runner 早已不在。而实际情况是评测**已经跑完**
（最慢一条 7571s，尾部长轨迹在收尾）。⇒ 判死活要看 `grep -E '^ *\[[0-9]+/800\]'` 的进度行
和日志末尾的 `Total:` 汇总，**不要用 `ps | grep <脚本名>`**，wrapper 会假阳性。

### ★★★ OPD step40 评测：**阳性**，第一个统计显著的正结果（2026-09-13）

`opd-27b-step40`（27B 教师 / k1 / `use_task_rewards=False` / SP 修复后的镜像
`verl-coding-spattn-fa:20260911b` / 40 步 / LoRA r32 α64）在**与 SFT 那轮完全相同**的
266 道非退化题上评测，harness 一致（`batch_run_trisol.py -c 16`、`agent_trisol.py`、
单跑一次、`eval.correct is True` 为正确、`None` 计 FAIL）：

| | 正确 | 率 |
|---|---|---|
| **OPD step40** | **165/266** | **0.6203** |
| SFT nd800-k4 | 148/266 | 0.5564 |
| 基模 | 138/266 | 0.5188 |

- **vs 基模**：Δ=**+0.1015**，配对 98 / 只 OPD 67 / 只基模 40 / 都错 61，
  **McNemar 精确双侧 p=0.0116**，95%CI **[+0.025, +0.178]** —— 不跨 0。
- **vs SFT**：Δ=+0.0639，p=0.1038，CI [−0.009, +0.137] —— **跨 0，说不上比 SFT 好。**
- 翻转率 **40.4%**（107/265），比 SFT 的 35.5% 还高 ⇒ 行为改变幅度同样大，
  但这次净方向是正的（对比 SFT 那轮 94/266 翻转、净增 10 道、p=0.35）。

**口径必须自校验，不要从记忆抄历史数字。** 同一脚本重算 SFT vs 基模得到
Δ=+0.0377 / p=0.3533，与本文件早先记的 +0.0376 / p=0.35 逐位吻合 —— 这一步是在读
OPD 结果**之前**做的，它才是"新数可信"的依据。基线目录是
`trajectories_qwen9b_base_with_rc`（**不是** `_base_full`，后者同子集给 137/266）。

**三条限制，引用这个 0.6203 时必须一起说：**
1. **没有留出集** —— ~~OPD 训的是全部 800 题，这 266 道全见过~~
   **这句是我写错的，见下面「OPD 只跑了 0.4 epoch」一节：实际只有期望 40% 见过，
   且每题最多见一次。** 但因为 `data.seed=null`，**无法识别是哪 320 题**，
   所以仍然**不能声称为泛化** —— 理由从"全见过"换成"分不清哪些见过"。
2. 只评了 step 40 单点（loss 在 ~step 10 后就饱和，但那是推断不是测量）。
3. 每条轨迹只跑一次，没有重复采样的方差估计。

**必查 `model` 字段**：265/265 全为 `opd-step40`、endpoint 全为那个 LoRA 服务
⇒ 排除「请求 `--served-model-name` 打到基模」的历史坑。`eval.correct` 分布
True 165 / **None 72** / False 28 —— `None`（撞轮数上限没交答案）占 27%，
**按 FAIL 计**，用 `ok/(ok+bad)` 会虚高约 28 个点。

**掉的那 1 题不是评测偏差，是硬失败**：`LAMMPS_Simulating_peridynamic_..._2`
撞 `--max-model-len 262144` 上限 ⇒ API 400 ⇒ 重试 3 次全败 ⇒ 子进程 rc=1、**不写输出文件**。
所以文件数是 799/800 而进度条报了 266 个 OK。计入分母（266 口径）与剔除（165/265=0.6226）
结论相同。**文件数与进度条不一致时要去 `grep "rc=1"` 找死因，不要当成"少写了一个"。**

### RL `-r2` 35 步：崩溃点推后到 step 30，且 entropy 方向与 nd267 **相反**（2026-09-13）

`coding-rl-nd267-spfix-0911-r2`（SP 修复后的镜像，其余同 nd267），epoch 边界仍在 step 17：

| | step 1–16 | 17–29 | **30–35** |
|---|---|---|---|
| `critic/score/mean` | 0.519 | **0.549** | **0.277** |
| `actor/entropy` | 0.271 | 0.383 | **0.618** |
| `actor/grad_norm` | 0.0156 | 0.0166 | 0.0157 |

- **跨过 epoch 边界后 13 步没崩**（17–29 的 0.549 还略高于 epoch0 的 0.519），
  nd267 是 step 17 立刻腰斩 ⇒ **修 SP bug 确实推后了崩溃，但没有消除它。**
- **entropy 是单调上涨的（0.27 → 0.76），nd267 崩溃时是坍缩（0.29 → 0.17）。**
  两次崩溃的 entropy 方向相反 ⇒ 本文件早先写的"entropy 坍缩是同源症状"**只对 nd267 成立**，
  不是崩溃的必要伴随。机制未查，**不要在这里填推测**。
- `grad_norm` 全程 0.013–0.020，无趋势，崩溃期也没变 ⇒ 不是梯度爆炸/消失。

### ★★ Trisol checkpoint 归档是**一次性**的，错过就永久取不回来（2026-09-13，代价 40 分钟 + 34 个 step 的 rollout）

`coding-rl-nd267-spfix-0911-r2` 提交时我又写成了 `--checkpoint-prefix rollout_data`
（**上一节已经写明不要这么干，我照着自己的记录又犯了一次**）。后果比上次严重：
35 步的 rollout jsonl 和 7 个 `global_step_*` 全在 PFS 上，但平台只在开跑约 2h 时
归档了一份 **8.1M 的 `1.jsonl`**（只有 step 1），**其余 34 步永久丢失**。

试了 3 次 rescan + 1 次 delete 都救不回来，机制是这样的（证据链，不是猜）：

1. **`rescan` 只 requeue *failed* 的归档** —— `rescan --help` 原文：
   "A rescan also requeues a terminal **failed** archive whose source checkpoint is still complete"。
   同名 checkpoint 已经是 `ready` 就**永远不会**重新归档，rescan 静默成功、列表纹丝不动。
2. **`--checkpoint-atomic` 只匹配目录，不匹配文件** —— help 原文是
   "every visible prefix-matching **directory**"。所以
   `--checkpoint-scan-path rollout_data --checkpoint-prefix 1 --checkpoint-atomic`
   压根扫不到 `1.jsonl`…`35.jsonl` 这 35 个**文件**，返回 0 个新条目也不报错。
3. **`delete` 是软删除，字节还在，重建的记录会直接指回那份旧归档。**
   `checkpoint delete <id> --yes`（**不加 `--yes` 会拒绝执行**）之后再 rescan，
   确实出现了新 id，但 `list -o json` 里：
   `file_count: 1`、`total_size_bytes: 8542558`（**恰好就是那个 1.jsonl**）、
   `storage_path` 与被删的那条**完全相同**，而且
   **`archived_at − created_at ≈ 0.09 s`** —— 320 MB 的拷贝不可能在 90 毫秒内完成，
   ⇒ 没有发生任何新的归档动作。之后 `download --include '*'` 返回
   "No files matched the selection"。

**判据记住这个：新建的 checkpoint 记录如果 `archived_at` 与 `created_at` 相差不到 1 秒，
就说明它复用了旧归档，没有真的去读 PFS。** 不要等、不要重试、不要再 delete。

**⇒ 唯一可靠的做法是在提交时就把规则配对，长跑任务不要留"事后补"的余地。**

### ★★ `expired` 的归档记录**能救回来**，但必须先 delete（2026-09-17，与上节互补）

上节说的是"`ready` 的记录 rescan 不动、delete 之后会复用旧字节"。
**`expired` 是另一种状态，救法相反、而且真能救。**

`opd-27b-7809-bs128n1-0914` 停任务时 `global_step_40` 正在归档，被打断后记录停在：

```
id 2100454166219460608  status expired  file_count 0  total_size_bytes 0
created_at 2026-09-17T05:17:52Z  archived_at None
```

- **`rescan` 对它无效** —— help 只承诺 requeue 终态 **failed** 的归档，`expired` 不在其列，
  rescan 静默成功、列表纹丝不动（与上节 `ready` 的表现相同，都是"什么都没发生"）。
- **delete 在这里是安全的**，判据就是那三个零：`file_count=0 / total_size_bytes=0 /
  archived_at=None` ⇒ **没有任何旧字节可供复用**，不存在上节那个"复用旧归档"的陷阱。
  （`ready` 且 `file_count>0` 的记录**不要** delete，那才是上节的坑。）
- delete 之后 rescan，新记录 `2100463920228601856`：
  `created 05:56:37 → archived 05:58:37`，**耗时 120 秒**、32 文件、19,471,457,084 B
  ⇒ 通过上节的"`archived_at − created_at ≫ 1 s`"判据，是真归档。

**前提是 PFS 上的字节还在**（`--no-output-model` 的副本在任务终止 7 天后才删）。
delete 之前先 `trisol train output ls <job> --team infra-spot` 逐文件核对。

### ★ 完整性自检必须包含**顶层** `data.pt`（2026-09-17，我自己的脚本 bug）

verl LoRA checkpoint（9B / world_size=8）的完整签名是 **32 文件 / 19,471,457,084 B**：

| 文件 | 数量 × 字节 |
|---|---|
| `actor/model_world_size_8_rank_*.pt` | 8 × 2,379,519,256 |
| `actor/optim_*.pt` | 8 × 51,896,487 |
| `actor/extra_state_*.pt` | 8 × (15205 / 15141) |
| `actor/fsdp_config.json` / `lora_train_meta.json` | 46 / 67 |
| `actor/huggingface/*` | 5 个（tokenizer.json 19,989,325 最大） |
| **`data.pt`（顶层，不在 `actor/` 下）** | **7,316** |

我的校验脚本 glob 根在 `actor/`，于是算出 19,471,449,768 ≠ 19,471,457,084 报
`INTEGRITY FAIL`，**差额恰好 7316 = `data.pt`**。下载其实是完好的。
⇒ **对不上先看差额是不是等于某个已知文件**，不要直接判下载失败重下 19.5 GB。

### ★ `trisol model upload` 之前必须先 `trisol model create`（2026-09-17）

upload 的 help 原文是"upload a new version of **the given model**" —— 模型不存在时：

```
Error: resolve model "opd-bs128-step40": no model matching "opd-bs128-step40" (code=E_ERROR)
```

看起来像权限/命名问题，其实只是没建。顺序是：

```bash
trisol model create opd-bs128-step40 --team infra-spot -o json     # 返回**模型 id**
trisol model upload opd-bs128-step40 <dir> --team infra-spot --version 1 --progress plain
#   末尾打印的裸数字是**版本 id**, 不要填进 adapters[].model_id (见上面那节)
```

### ★ inference 的 `endpoint` 字段是 `None`，真地址在 `public_endpoint_url`（2026-09-17）

服务 `running 4/4` 时 `trisol inference get <id> -o json` 的 `endpoint` **仍然是 null**，
容易误判成"还没就绪"。要用的字段：

| 字段 | 值 |
|---|---|
| `public_endpoint_url` | `https://<alias>.w1.inference.trisol.dp.tech` ← **评测用这个** |
| `internal_endpoint_url` | 集群内地址 |
| `domain_alias` | 那个 alias |

### r2 的验证曲线：27 题验证集读不出 acc 信号，但 `num_turns` 读得出（2026-09-13）

rollout 数据没了之后，唯一零成本的替代是训练日志里的 `val-core` 曲线（`TEST_FREQ=5`）。
**验证集就是那 27 题**（所有 acc 都是 `k/27` 的整数商，`num_turns/mean=30.222=816/27`）：

| step | val acc | 题数 | `num_turns/mean` | min | max |
|---|---|---|---|---|---|
| 0（未训练） | 0.3333 | 9 | 30.2 | 2 | 60 |
| 5 | 0.4444 | 12 | 26.6 | 2 | 60 |
| 10 | 0.4815 | 13 | 21.7 | 2 | 60 |
| 15 | 0.6296 | 17 | 20.9 | 2 | 60 |
| 20 | 0.4074 | 11 | 18.8 | 2 | 60 |
| 25 | 0.5185 | 14 | 16.2 | 2 | 60 |
| 30 | 0.3333 | 9 | 22.7 | 2 | 60 |
| 35 | 0.2963 | 8 | 16.5 | 2 | 60 |

- **acc 这一列读不出任何东西**：n=27、p≈0.45 的二项 sd = **0.096（2.6 题）**，
  step0→25 斜率 +0.0055/step **t=+1.18**、全程 −0.0022/step **t=−0.61**，都不显著。
  相邻两次 0.6296→0.4074 差 6 题 = 2.3σ，**是噪声而不是"step 15 到 20 之间崩了"**。
  （这正是本文件早先那条"27 题验证集二项 sd≈9%，读不出信号 —— 验证集必须扩大"的实例，
  这次是在事后被迫拿它当唯一数据源才吃到全部代价。）
- **`num_turns/mean` 是这份日志里唯一有信号的量**：30.2 → 16.2，斜率 **−0.32/step、t=−3.36**。
  验证集是固定的同一批 27 题，**难度被完全控住**，所以这个下降是真的行为改变。
  `corr(acc, turns) = −0.194`（t=−0.48）⇒ 在 27 题的分辨率下，轮数减半**没有**换来 acc 提升。
- `num_turns/max` **每一步都是 60**（=30 个 assistant 轮的上限 ×2）⇒ 始终有题撞顶；
  `min` 恒为 2。所以均值下降不是"上限被削掉"，是分布整体左移。

**不要**把"轮数减半 + acc 没涨"直接读成 nd267 那个"中途自己停"的失败模式 ——
那个判据需要 rollout 里的 `<final_answer>` 交出率和工具调用次数，
**这两个量在只有验证 metrics 的情况下量不出来，r2 的 rollout 已经丢了。**

### ★★ OPD 只跑了 **0.4 epoch**，每题最多见一次（2026-09-13，纠正我自己的错误记录）

`opd-teacher27b-prod-0911`（2098353556137451520）的实际配置，从
`trisol train get ... -o json` 的 `custom_config` 和命令尾部逐项读出来的：

| 项 | 值 | 来源 |
|---|---|---|
| 数据集 | `coding-practice-800-train-27-val` | `datasets[0].wenyon_id` |
| `TRAIN_BATCH_SIZE` | **8** | `custom_config.env` |
| `ROLLOUT_N` | 4 | 同上 |
| `++trainer.total_training_steps` | **40** | 命令尾部 override |

⇒ `steps_per_epoch = 800 / 8 = ` **100**，跑的 40 步 = **0.4 epoch**。
题目槽位 40×8 = **320 / 800（40%）**，轨迹 320×4 = **1280 条**。

**`RandomSampler` 是不放回的**（`verl/trainer/ppo/utils.py:154,163`，
torchdata 的 stateful `RandomSampler`，`replacement` 默认 False）⇒ 一个 epoch 内是 800 的一个排列，
前 320 个就是训练过的，**剩下 ≥480 题一次都没见过**，见过的那 320 题**每题恰好一次**。

**⇒ 本文件早先写的"OPD 训的是全部 800 题，这 266 道全见过"是错的。**
期望只有 266×0.4 ≈ 106 道见过、约 160 道没见过。这让 `+0.1015 / p=0.0116` 那个阳性结果
**更难用"背题"解释**（要靠 106 道单次见过的题撑起全部 +27 道净增）。

**但识别不出是哪 320 题，所以补不成留出集分析：**

- `data.seed` 默认 `null`（`verl/trainer/config/data/legacy_data.yaml:71`），
  `run_coding_practice_qwen3_5_9b_4l20.sh` **也没有设**（grep `seed` 零命中）⇒
  `create_rl_sampler` 走 `torch.Generator()` **不 manual_seed**，采样顺序**不可复现**。
- `rollout_data` 归档只有 `1.jsonl`（3.3M，32 行 = 8 题 × 4）⇒ 只能认出 8 题。
  （又是 `--checkpoint-prefix rollout_data` 那个坑，见「Trisol checkpoint 归档是一次性的」。）

**以后提交 RL/OPD 一律显式给 `data.seed`**（例如 `++data.seed=42`），
成本是一个字符串，收益是事后随时能重建"训练过哪些题"从而做留出集分析。
想测泛化的正路仍然是 `split_data/test` 那 977 题（与 `train_sampled` 交集为 0，已实测）。

**顺带对比 RL 的训练量口径**（两者不在同一个 regime，不要混着读）：

| | 题库 | batch | steps/epoch | 跑的步数 | epoch 数 |
|---|---|---|---|---|---|
| `coding-rl-nd267-spfix-0911-r2` | 267 | 16 | 16.7 | 35 | **2.1** |
| `opd-teacher27b-prod-0911` | 800 | 8 | 100 | 40 | **0.4** |

RL 是"同一批题反复看到崩"，OPD 是"连一遍都没看完就停了" ——
所以 OPD 那条 loss 在 ~step 10 饱和的**推断**不能当成"训练量够了"，
step 40 之后还有 60 步才到第一个 epoch 边界。

### ★★ OPD 里 `rollout.n` 是**惰性参数**，多 rollout 换不到任何东西（2026-09-14，读代码定案）

`opd-teacher27b-prod-0911` 用的是 `TRAIN_BATCH_SIZE=8 × ROLLOUT_N=4`。**那 4× 是白花的。**

`verl/trainer/distillation/losses.py` 在 `use_policy_gradient=True` 时
`advantages = -distillation_losses.detach()` —— **逐 token，只由 teacher/student 的 logprob 差决定，
不含任何组内统计量**。而 `ppo_loss` 那一支（真正消费上游 `adv_estimator` 算出的 advantage 的地方）
被 `use_task_rewards=False` 先置零再加回 `distill_loss × 1.0`。
⇒ RLOO/GRPO 那套"同题多条互为 baseline 降方差"的机制在 OPD 里**结构上不存在**。

顺带：`algorithm.adv_estimator=grpo` 来自 run 脚本 `:107`，从来没被覆盖过，
所以它**算了但被丢掉**。看到日志里有 grpo 不要以为在起作用。

`n` 剩下的两个真实影响，改 n 时必须一起调否则悄悄变了两个变量：

1. **每步 token 数** = `batch × n`。要保持步时长和沙盒并发不变，就得让 `batch × n` 不变。
2. **`ppo_mini_batch_size` 会被乘上 `rollout.n`**（`ray_trainer.py:1316,1346`；
   `trainer_base.py:1605,1628`）⇒ 每步内层更新次数 = `batch×n / (mini×n)`。
   prod 是 `mini=4 × n=4 = 16` 条/mini、2 次内层更新；换 n=1 要写 `PPO_MINI_BATCH_SIZE=16` 才等价。

**`n=1` 对 GRPO 路径是安全的**（哪怕 advantage 会被丢弃，它仍会被算一遍）：
`core_algos.py:315-317` 显式处理组大小 1（`id2mean=0.0`、`id2std=1.0`），不会除零。

⇒ **OPD 一律 `ROLLOUT_N=1`**，把预算全换成不同的题。同样的 32 条轨迹/步，
n=4 是 8 道题、n=1 是 32 道题，**每步题目多样性 4×**。

### `TEST_FREQ` 设很大**不能**关验证，必须 ≤ 0（2026-09-14）

`ray_trainer.py:1690-1691`：

```python
if test_freq > 0 and (is_last_step or global_steps % test_freq == 0):
```

`is_last_step` 是 **OR** 进去的 ⇒ `TEST_FREQ=100000` 仍然会在最后一步跑一次**完整**验证
（976 题验证集按单条 ~1 小时的 gen 量级，是实打实的代价）。
只有 `test_freq <= 0` 才整个短路。同理 `val_before_train` 是独立的门（`:1401`），
要一起关：`++trainer.val_before_train=False`。

### `trisol train submit` 只能配**一条** checkpoint 发现规则（2026-09-14，纠正本文件早先的说法）

上面「长跑任务的 `--checkpoint-prefix`」那节写的"**同时**配好两条规则（checkpoint 用 scan-path
+ `global_step_` prefix，rollout 另起一条）"——**做不到**。
`--checkpoint-prefix` / `--checkpoint-scan-path` / `--checkpoint-atomic` 三个 flag 都是**单值**的，
没有可重复的语法。而且 `--checkpoint-atomic` 的 help 原文是
"every visible prefix-matching **directory**" ⇒ 它匹配不了 `rollout_data/*.jsonl` 这种**文件**。

⇒ 长跑任务只能二选一。**选 `global_step_`（配 `--checkpoint-scan-path checkpoints`）** ——
权重错过了就真没了；rollout jsonl 虽然也是一次性归档（见「归档是一次性的」），
但可以用 `++data.seed=<N>` 本地重建"训练过哪些题"来补偿大部分需求。

### `filter_overlong_prompts=True` 会静默丢题，seed 复现必须在**过滤后**的帧上做（2026-09-14）

`coding-practice-rl:7` 的 train 是 **7809** 行，但日志里是：

```
dataset len: 7809
filter dataset len: 7808
```

`rl_dataset.py:197+` 的 `maybe_filter_out_long_prompts` 是保序的 `dataframe.filter`，
用**和 rollout 同一份工具 schema** 渲染 chat template 后按 `max_prompt_length=16384` 截。
本地用 `/personal/Qwen/Qwen3.5-9B` 的 tokenizer 重算（**不含**工具 schema 前缀）：
max **25714** / p99 3515 / median 1522，超 16384 的**恰好 1 条**：

```
row 530  len 25714  BWA_Indexing_a_reference_genome_with_bwa_index_Build_the_FM-
row 390  len 14964  <- 次长, 加上工具前缀仍在线下(平台只丢了 1 条, 与此一致)
```

⇒ **`RandomSampler` 跑在 7808 行上，不是 7809。** 直接对 7809 行重放会整体错位。
正确的复盘配方（已存 `/personal/verl/opd_7809_seed42_plan.json`，含前 3 步的 row/task/题面哈希）：

```python
df = pd.read_parquet('train.parquet').drop(index=530).reset_index(drop=True)   # 7808
g = torch.Generator(); g.manual_seed(42)
order = list(iter(RandomSampler(data_source=range(len(df)), generator=g)))
step_k_rows = order[(k-1)*batch : k*batch]        # k 从 1 开始
```

（`drop_last=True`，`ray_trainer.py:405-409` ⇒ `steps/epoch = floor(7808/32) = 244`。
验证 loader 是 `drop_last=False`，`:418-423`。）

### 题库盘点：`coding-practice-rl:7` 就是那个 7809 题的库（2026-09-14）

不要再新建/上传题库。`wenyon-cli dataset list` **没有搜索/过滤 flag**（只有 `--include-hidden`），
找版本号走 `trisol dataset get coding-practice-rl`（submit 的 help 就是这么建议的）。
`trisol train list -o json` 会返回**全 team** 最近 50 个任务（含别人的），不适合定位数据集。

| split | 行数 |
|---|---|
| train | **7809** |
| validation | 976 |
| test | 977 |

已实测：三个 split 按 NFC 归一化题面哈希**两两无交集**；800 题那个卡是 train 的**真子集**；
system prompt 与 800 卡**逐字节相同**；`extra_info` / `tools_kwargs` 格式一致；
**`images.json` 存在**（注意 `coding-practice-rl-public` 那个变体**没有** images）。

⇒ 想测泛化用 `validation`(976) 或 `test`(977)，它们与训练集结构上不相交。

### ★★ 沙盒配额是**按轨迹**持有的，batch 必须远大于配额数才填得满尾巴（2026-09-14 实测 2.05×）

`verl_coding/coding_sandbox_tool.py:229` 的 `while not await quota.try_acquire.remote(instance_id): await asyncio.sleep(5)`
在**沙盒 create 时**拿许可，只在 `calc_reward`/`cleanup` 里释放 —— docstring 明说每次工具执行后的
`release()` 是 no-op。⇒ **一条轨迹全程占一个许可，不是一次工具调用占一个。**

TaskCreate loop 自己**没有**并发上限：`AgentLoopWorkerTQ.generate_sequences` 对每个样本
`asyncio.create_task` 后 fire-and-forget，`prompts.chunk(len(self.agent_loop_workers))` 是静态
round-robin、那一层没有 work stealing —— **沙盒许可是唯一的闸，而它是 work-conserving 的。**

⇒ `MAX_CONCURRENT_SANDBOXES=32` 配 `train_batch_size=32` 时**零排队**，
`timing_s/gen` 就退化成"最慢那一条轨迹的墙上时间"。prod 实测
`response_length max/mean = 71880/24239 = 2.97` ⇒ 有近 3 倍的尾巴在空转。

把 batch 抬到 128（4× 工作量）实测：

| | bs32 | **bs128** |
|---|---|---|
| `timing_s/step` | — | 2.04× |
| **每条轨迹墙上时间** | 105.0 s | **51.1 s（2.05× 吞吐）** |
| 61 步预计 | 9.5 d | **4.62 d** |

**保持优化过程不变的关键是 `PPO_MINI_BATCH_SIZE` 不动**（`ray_trainer.py:1316,1346` 会把它
乘上 `rollout.n`）：bs32/mini16 = 244 步 × 2 = **488 次 16 条轨迹的更新**；
bs128/mini16 = 61 × 8 = **488 次 16 条轨迹的更新**，逐项相同，只是步内 staleness 更大。
（bs128/mini64 只有 122 次更新，**不要**这么配。）

OPD 下步内 staleness **只是分布漂移，不是梯度偏差**：`losses.py` 的有效目标是逐 token
`KL(student‖teacher)`，**没有 importance ratio**（`use_policy_gradient=True` 时 `ppo_loss` 算了
但被 `policy_loss = 0.0` 丢掉）。

**TQ 路径不产出 `agent_loop/*` 指标** —— `_performance_metrics`（`agent_loop.py:1236-1268`）
里的 `agent_loop/generate_sequences/{min,max,mean}` 只在非 TQ 路径可达。
枚举了全部 91 个 metric key，没有任何 `agent_loop/*`，**单条轨迹墙上时间在我们的日志里量不到**，
只能用 `timing_s/gen ÷ batch` 反推。

### `--since` 必须从 `created_at` 原样抄 UTC，写成本地时间会静默返回空日志（2026-09-14）

我把本地时间当 UTC 填了 `--history --since 2026-09-14T11:30:00Z`，而 pod 的真实
`created_at` 是 `2026-09-14T04:40:23.051Z` ⇒ `--since` 落在**未来**，
`trisol train logs` **返回空且不报错**，我据此以为 trainer 卡在初始化，白等了 10 分钟。

⇒ `--since` 永远从 `trisol train get <job> -o json` 的 `created_at` 逐字复制，不要自己算。

### trisol 子命令的两个名字坑

- 推理服务是 **`trisol inference`**，没有 `trisol serve`（`unknown command "serve"`，
  再 pipe 给 `python -m json.tool` 会叠一个 JSON decode traceback，看起来像别的问题）。
- **`trisol train cancel` 是 `train stop` 的废弃别名，没有 `--yes`**。
  非交互要用 `trisol train stop <job> --team <t> --no-input`。

### checkpoint download 不续传，`.partial` 会从头重下（2026-09-15）

19.5 GB 的 checkpoint 下到 558 MB 时进程被回收（上个 session 的后台 bash 随 session 一起死），
重跑 `trisol train checkpoint download` 时 4 个 `model_world_size_8_rank_*.pt.partial`
**从 0 重新开始**，之前的字节作废。`extra_state_*` / `huggingface/` 那些小文件会被正确
`skipped (already complete)`，但大分片不行。

⇒ **长下载一律 `nohup ... &` 起，不要用工具的 `run_in_background`**（后者绑 session）。
实测速率 ~11.8 MB/s（4 分片并发），19.5 GB 约 **26 分钟**；完成时 stdout 有
`Downloaded 17, skipped 15 (already complete), failed 0`。

### ★★★ OPD 训练量曲线读完了：step10/20/30/40 **四点全部持平**（2026-09-16）

`opd-27b-7809-bs128n1-0914`（bs128 × n=1 / 7809 题 / `++data.seed=42`）的 step20、step30
checkpoint 各起一个评测服务跑完 800 题，与已有的 step10、`opd-teacher27b-prod-0911` 的 step40
并到 **789 道六方公共集**（同 harness：`batch_run_trisol.py -c 16`、`agent_trisol.py`、
单跑一次、截断默认开 8000/3000/5000、`eval.correct is True` 为正确、`None` 计 FAIL）：

| | 正确 | 率 | 轨迹 / 优化更新 |
|---|---|---|---|
| SFT-merged | 283/789 | **0.3587** | — |
| OPD step40（bs8×n4） | 233/789 | 0.2953 | 1280 / 80 |
| **OPD step30（bs128×n1）** | **226/789** | **0.2864** | 3840 / 240 |
| OPD step10（bs128×n1） | 216/789 | 0.2738 | 1280 / 80 |
| OPD step20（bs128×n1） | 209/789 | 0.2649 | 2560 / 160 |
| 基模 | 185/789 | 0.2345 | — |

| 对比 | Δ | 95%CI | p |
|---|---|---|---|
| step30 vs 基模 | **+0.0520** | [+0.021,+0.084] | **0.0016** |
| step20 vs 基模 | +0.0304 | [+0.002,+0.059] | 0.048 |
| **step30 vs step20** | **+0.0215** | [−0.007,+0.050] | **0.159** |
| **step30 vs step10** | **+0.0127** | [−0.015,+0.040] | **0.419** |
| **step20 vs step10** | **−0.0089** | [−0.036,+0.019] | **0.589** |
| step30 vs step40 | −0.0089 | [−0.038,+0.020] | 0.611 |
| step30 vs merged | −0.0722 | [−0.102,−0.043] | 2.5e-06 |

**⇒ 训练量从 1280 涨到 3840 轨迹（3×）、更新次数 80→240，评测纹丝不动。**
三点 0.2738 / 0.2649 / 0.2864 **非单调**，摆幅全在 results.md 记的噪声地板
（同模型重跑漂 ±1pp、逐题 14% 翻转）附近。`step30 vs step40` p=0.611 ⇒
**3 倍训练量仍追不上等预算的 step40。**

**这与两个独立口径吻合，是同一幅图像的三个侧面：**
1. **训练侧 `critic/score/mean`**（`ROLLOUT_N=1` 不放回 ⇒ 每步一次免费留出集读数）：
   1-10 = 0.1930、11-20 = 0.1953（持平，与 `step20 vs step10` 的 p=0.589 对上）、
   21-30 = 0.2328、**31-37 = 0.2333（抬头停在 21-30 那一档）**。
   全程线性趋势 +0.00156/step **t=+2.89（n=37）** 已越过 t=2.5，但**下游读不出**
   （`step30 vs step20` p=0.159）⇒ 训练侧这条趋势线目前**没有任何下游证据支持它对应能力提升**。
   斜率随步数缓慢变小（27 步 +0.00176 → 30 步 +0.00167 → 37 步 +0.00156）⇒
   涨的是统计功效不是效应量。
2. **`ΔW/W`**：0.103%(st10) → 0.139%(st20) → 只 1.348×，`cos(增量, 累积)=0.136` 近正交
   ⇒ 净位移按 √t 增长。见 [[opd-delta-w-near-orthogonal]]。

**⇒ 不要再靠"多跑步数"救 OPD。** 该动的是教师（27B → qwen3.8 系列，
外部 teacher 通道已实现并实测通过）或算法，不是训练量。

**口径自校验必须在读新数之前做**（这次也做了）：同一脚本在 789 集上重算
`merged vs base = +0.1242 / p=2.633e-14`、`step40 vs base = +0.0608 / p=1.098e-4`、
`step10 vs base = +0.0393 / p=0.008008`，与本文件历史记录**逐位吻合**。

`model` 字段全对（800/800 `opd-bs128-step20`、800/800 `opd-bs128-step30`）。
step30 的 `eval.correct`：None 490（61%）/ True 226 / False 83；800 题 0 硬失败，耗时 37672 s。

**★ 两个查完整性的假信号（都在这轮踩到）：**
- **`grep -c 'rc=1'` 假阳性** —— 命中的是模型输出里的科学计数法
  （`rc=1.000000e+00`、`rc=1.573882e-247`）。因为 `batch_run_trisol.py` 的
  `subprocess.run(..., text=True)` 不捕获子进程输出，整个 agent transcript 落进父进程的
  重定向文件（本次 GB 量级）。找硬失败要 `grep -aoE '\] (FAIL|OK).{0,120}'` 再筛，
  或直接比"输出文件数 == 800"。
- **`ls *.json | grep -v summary` 会误伤题目** —— 有道题名里带 `summary`
  （`squidpy_..._extracting_summary_texture_..._10.json`），于是数出 799 让我以为死了 1 题。
  数输出文件要用 `p.name.startswith('_') or p.name == 'summary_trisol.json'` 精确排除，
  并与 `base.glob("*/raw/*/*/generator_output.json")` 算出的题名集合做差集定位缺题。

### ★ `trisol model upload` 打印的是**版本 id**，不是模型 id（2026-09-16）

upload 末尾那个裸数字是 `versions[].id`。把它填进推理服务 spec 的 `adapters[].model_id`，
部署会在 model preparation 阶段失败：

```
status_message = Model preparation failed after 3 attempts: ...
  resolve adapter "<key>" prefix: model adapter: version v1 for model <版本id> not found
```

平台按 `model_id + version_code` 解析。**模型本身是好的**（`trisol model get` 里
`status: ready`、文件数/大小都对），不要重传，改 spec 重建服务即可。

```bash
trisol model get <name> --team infra-spot -o json
# 顶层 id        = 模型 id      <- adapters[].model_id 填这个
# versions[].id  = 版本 id      <- upload 打印的，不要填
```

判据：spec 里的 `model_id` 必须出现在 `trisol model list` 的 id 列里。
（`trisol inference delete` 需要 `--yes`，`-y` 不是它的 shorthand。）

### 起评测服务照抄老 spec：`inference revision` + `--from-file`（2026-09-15）

`trisol inference get` **不返回** runtime spec（只有状态和 endpoint），要拿完整启动参数得用
**`trisol inference revision <service_id> <revision>`**，`spec.runtime.args` / `spec.adapters` 都在里面。

克隆一个只换 adapter 的评测服务：

```python
spec = json.load(open('rev40.json'))['spec']
i = spec['runtime']['args'].index('--lora-modules')
spec['runtime']['args'][i+1] = f'{KEY}=/mnt/adapters/{KEY}'
spec['adapters'] = [{'model_id': NEW_MODEL_ID, 'version_code': 1, 'key': KEY}]
for k in ['cpu','memory','shm_size','ephemeral_storage']:
    spec['resources'].pop(k)          # --from-file: GPU 服务不得设置平台托管字段
```

```bash
trisol inference create --name <svc> --team infra-spot --cluster w1 \
  --model qwen3-5-9b:1 --from-file spec.json -o json --no-input
```

**改完必须 diff 一遍**（归一化掉那 4 个 pop 的字段后应当只差 adapter 那 3 行），
不要靠眼看。起来约 **5 分钟**（`preparing` 预热 → `launching` → `running 4/4`）。

**冒烟测试只看 `model` 字段回显，不要看 `content`**：带 `--reasoning-parser qwen3` 时
正文进 `reasoning_content`，`max_tokens=20` 下 `content` 是 `None`，那不是失败。
要验的是响应里 `"model": "<你的 lora key>"` —— 这一条挡住"请求打到基模"那个历史坑。

### 评测产物在 `/personal/sp2/qwen_trajs/cc_deepseek_1w/`，不在 `/personal/verl`

`trajectories_*/` 全在前者。我在 `/personal/verl` 下搜不到就以为评测没在跑，
差点把还在被依赖的推理服务停掉（用户纠正）。**拆共享基础设施前先定位依赖它的活。**

`batch_run_trisol.py` 的题目发现是 `base.glob("*/raw/*/*/generator_output.json")`，
**`pathlib.glob` 跟随符号链接**，而 `train_sampled` 的 800 个叶子全是指向 `train/` 的 symlink
⇒ `find` **不带 `-L` 会数成 0**，看起来像"题目没了"。数题用 `python3 -c` 跑同一个 glob。

### ★ 800 题三方评测：OPD 显著赢基模，但显著输给 teacher SFT（2026-09-14）

789 道公共题（三方都有输出），harness 一致（`batch_run_trisol.py -c 16`、`agent_trisol.py`、
单跑一次、`eval.correct is True` 为正确、`None` 计 FAIL）：

| | 正确 | 率 |
|---|---|---|
| SFT-merged `qwen-9b-sft-merged-qwen35-64k-r32` | 283/789 | **0.3587** |
| OPD step40 | 233/789 | **0.2953** |
| 基模 `qwen3-5-9b` | 185/789 | **0.2345** |

- **OPD vs 基模**：Δ=**+0.0608**，配对 134/99/51/505，McNemar 精确双侧 **p=1.098e-4**，
  CI [+0.0304,+0.0913]，翻转 19.0% ⇒ 比之前 266 子集那个 `+0.1015 / p=0.0116` **证据更强**。
- **OPD vs SFT-merged**：Δ=**−0.0634**，189/44/94/462，**p=2.504e-5**，CI [−0.0926,−0.0342]
  ⇒ **显著输给 teacher SFT**（+0.061 vs +0.124，差 2×）。
- **自校验**：同一脚本重算 SFT-merged vs 基模得 Δ=+0.1242 / p=2.633e-14，与本文件早先记的
  +0.1237 吻合 ⇒ 打分管线可信。**这一步要在读新结果之前做。**
- OPD 缺的那 11 题（`--max-model-len 262144` 撞顶 → API 400 → 重试 3 次 → rc=1 不写文件）：
  基模 0/11、merged 1/11。OPD/800 的上下界 0.2913–0.3050 都夹在基模 0.2313 与 merged 0.3550
  之间 ⇒ 结论不变。

**推断（不是实测）**：OPD 教师是 Qwen3.5-27B（train_sampled 上 50.7%/55.9%），
merged-SFT 的教师是 qwen3.7-plus（67.1%）。**"方法弱"还是"教师弱"要靠换教师实验才能分开**，
不要用这一轮去否定 OPD。

---

# 换 teacher：qwen3.8 系列核实 + 外部 teacher 通道（2026-09-15）

> 动机：800 题三方评测显示 OPD 输给 teacher SFT，而两者教师不同（27B vs qwen3.7-plus），
> 「方法弱 vs 教师弱」要靠换教师区分。候选是 qwen3.8-27B / qwen3.8-flash-next。

## 模型清单（实测，都是真权重不是 API）

| trisol 名 | model_id | 体积 | `architectures` / `model_type` |
|---|---|---|---|
| `qwen3-8-27b` | 2088292879372922880 | 51.8 GiB | `Qwen3_5ForConditionalGeneration` / `qwen3_5` |
| `qwen3-8-flash-next` | 2092684765470666752 | **335.3 GiB** | `Qwen4ExpForConditionalGeneration` / **`qwen4_exp`**（176B 总 / 6B 激活，262K ctx） |

## ★ tokenizer gate 通过 —— k1 换 teacher 的前置条件成立（实测）

k1 把 student 的 `sequence_ids` **原样**喂给 teacher（`teacher_manager.py` 末尾
`assert teacher_ids.shape[0] == teacher_logprobs.shape[0] == len(sequence_ids)`），
所以 teacher 和 student **必须同 tokenizer**。逐项核对 vs Qwen3.5-9B：

- `vocab.json`、`merges.txt` **md5 完全相同**
- `tokenizer.json` 只多 7 个 audio/tts special token（248070–248076），全在 `vocab_size=248320` 内
- 三个模型 `len(tokenizer)` 都是 **248077**
- 22 条真实文本（20 条数据集题面 + `Å` + tool-call XML + `<final_answer>`）编码
  **逐字节相同，0 处不一致**

⇒ 换 3.8 系列当 teacher **不需要**重新对齐 tokenizer。

## ★★ 现有训练镜像装不进 3.8：flash-next 硬失败，27B 是**静默数值错误**（实测）

`verl-coding*` 系列镜像里 vLLM 0.26 / transformers 5.14.1：

- **flash-next**：`qwen4_exp` 未注册 ⇒ **直接崩**，一眼能看出来。
- **qwen3.8-27B**：`model_type` 仍是 `qwen3_5` ⇒ **能加载**，但 config 里新增的
  `output_gate_type: "swish"` **两边都不认**，硬编码走 sigmoid：
  - `transformers/models/qwen3_5/modeling_qwen3_5.py:718` `attn_output = attn_output * torch.sigmoid(gate)`
  - `vllm/model_executor/models/qwen3_next.py:398` 同一行

  **不报错、不警告，只是算错。** 而且两个 repo 里都没有 remote-code `.py`，
  `trust_remote_code` 救不了。

⇒ **"架构名相同所以能加载"是错的推理**。换模型必须 diff config，逐个新字段确认代码认不认。

## ★★ verl 为什么不原生支持训推分离（读代码定案）

**verl 里有"分离"，但那个词不是你以为的意思。** `RolloutMode` 三档
（`verl/workers/rollout/replica.py:54-67`），STANDALONE 的 docstring 逐字写着
"separate GPU resource, **disaggregated architecture**"，但 `init_standalone()`（`:189`）做的是

```python
resource_pool_spec = {f"rollout_pool_teacher_{rank}{suffix}": [gpus_per_replica_node] * nnodes}
```

⇒ **新开一个 Ray resource pool**。分离的是 GPU，不是进程 / 集群 / 生命周期。

**结构性原因是 student rollout 每步要同步权重**（`verl/workers/engine_workers.py:683`
`async def update_weights(...)`，前 `sleep` 后 `wake_up`，配 `release_kv_cache`/`resume_kv_cache`）。
这要求 verl 持有引擎进程句柄、能与它建 NCCL/IPC ⇒ **student 的 rollout 引擎结构上不能是外部服务。**

**teacher 是搭便车被圈进来的。** `teacher_model.py` 的 `_initialize_llm_servers()` 直接复用
`get_rollout_replica_class(...)` + `init_colocated(pool)` + `LLMServerClient` + 负载均衡器，
`is_teacher_model=True` 只是同一个类上的一个 flag。于是 teacher **继承了 rollout 的全部约束，
而它一条都不需要** —— 权重冻结，永远不 `update_weights`/`sleep`/`wake_up`。
**这是复用的副作用，不是设计取舍。**

次要原因（推断，有代码支撑）：verl 要 token-in/token-out（`prompt_ids` 直接进、
`prompt_logprobs` 直接出，不走 tokenizer 往返），而它要同时支持 vLLM/sglang/trtllm 三家
（`replica.py:321-383`），三家 HTTP 层对"token id 数组当 prompt"+`prompt_logprobs` 的支持不统一；
另外 `LLMServerClient` 还做 least-in-flight 负载均衡 + sticky session（为 prefix caching），
需要引擎注册表，外部服务只给 URL 拿不到。

**⇒ 接外部 teacher 的注入点只有一个**：`MultiTeacherModelManager.get_client()` 返回的那个 dict。
换成 HTTP 版 client，只需实现 `generate(request_id, prompt_ids, sampling_params)`
一个方法、返回带 `extra_fields["prompt_logprobs"]` 的 `TokenOutput`。
teacher 接口窄得出奇（`teacher_manager.py:34-141`）：
`sampling_params = {"max_tokens": 1, "temperature": 1.0, "prompt_logprobs": num_logprobs}`，
k1 路径**不用** `teacher_ids`（只有 topk 路径用）。

**真正的收益不是省卡，是把 teacher 的 vLLM 版本与 student rollout 的 vLLM 版本解开** ——
现在它们被迫共用一个镜像，这正是 qwen3.8 装不进来的根因。

## 外部 teacher 通道实测可行

vLLM `/v1/completions` 接受 **token id 数组**当 prompt
（`vllm/entrypoints/openai/completion/protocol.py:94` `prompt_logprobs: int | None = None`；
注意 vLLM 0.26 起 `vllm.entrypoints.openai.protocol` 这个模块**已不存在**，挪到了
`completion/protocol.py`）。对两个在跑的 trisol 服务实测 POST
`{"prompt": [ids...], "max_tokens": 1, "prompt_logprobs": 0}`：

返回的 `prompt_logprobs` 与输入位置 **1:1 对齐**（首元素 null，每个 dict 以该位置的 token id 为键）
—— 正是 verl 消费的语义。吞吐（4000 token）：27B **4265 tok/s**、flash-next **8551 tok/s**。
折算 prod（128 traj × ~24239 tok ≈ 3.1M tok/step）⇒ 27B 四副本约 **3 min/step**，
远低于 `timing_s/gen` 的 2000 秒量级 ⇒ **teacher 不会成为瓶颈。**

**坑**：`prompt_token_ids` 有时返回 `null`，拿它和输入比会得到假的 `ids_match=False`
（200 token 时直接 `TypeError: object of type 'NoneType' has no len()`）。
k1 不用 `teacher_ids`，且我们本来就知道发出去的 ids，**不必比**。

## ★★ `prompt_logprobs` 的显存公式：0.947 MiB/token，且只由 chunk 决定

`_get_prompt_logprobs_dict` → `sampler.compute_logprobs(logits)` →
`logits.log_softmax(dim=-1, dtype=torch.float32)`，**按每个被调度的 prefill chunk** 执行：

```
峰值 = max_num_batched_tokens × vocab × 4 B
```

Qwen 词表 248320 ⇒ **0.9473 MiB / token**。与历史 OOM 记录**逐位吻合**：
报 `Tried to allocate 7.33 GiB`，7.33 GiB ÷ 0.9473 MiB = **7923 token** = 当时的 chunk 大小。

⇒ **10 万 token 的 prompt 不比 4 千 token 的更费显存**，只是 chunk 数多。
**长轨迹本身不是障碍，chunk 才是唯一杠杆**（调 TP 没用，vLLM 是 all-gather 完 logits 再 softmax）。

| `max_num_batched_tokens` | fp32 buffer |
|---|---|
| 8192（vLLM 默认量级） | **7.58 GiB** |
| 4096 | 3.79 GiB |
| **2048** | **1.89 GiB** |
| 1024 | 0.95 GiB |

（`log_softmax` 的输入 bf16 logits 同形状还要 0.47 MiB/token；历史 OOM 报的是 fp32 那次分配，
上表按 fp32 口径，真实峰值可能再高 50%，**这一项没实测**。）

**8 卡 flash-next 评测服务实测占用 77037/81920 MiB ⇒ 只剩 4.77 GiB**，
而它的启动参数里**没有** `--max-num-batched-tokens`（走默认）⇒ 默认 chunk 装不下。
KV cache 完全不是约束（`/metrics` 实测 `kv_cache_size_tokens=2040781`、
`kv_cache_max_concurrency=7.78`、使用率 **1.68%**，一条 10 万 token 轨迹只占 4.9%）。

**不要在正在评测的服务上试** ——`prompt_logprobs` 触发 OOM 在 vLLM V1 里会打死 engine core，
整个服务挂掉，评测全废。要跑 teacher 就另起专用服务：
`--max-num-batched-tokens 2048` + `--gpu-memory-utilization 0.85`。

## ★ 起 flash-next 服务：镜像 5 分钟 + 权重 25 分钟，**但早期 ETA 会骗你 14 倍**（实测）

`qwen38-flashnext-teacher`（2099737790336995328，TP=8 / A100×8 / vLLM 0.29.0）：

```
Pulled image ... in 5m7.205s   Image size: 8674227544 bytes
Filesystem type for checkpoints: NFS. Checkpoint size: 335.28 GiB. Available RAM: 1493.90 GiB.
Loading safetensors checkpoint shards:   3% | 4/131 [14:18<6:19:53, 179.47s/it]   <- 报 6 小时
Loading safetensors checkpoint shards: 100% | 131/131 [25:35<00:00,  1.57it/s]    <- 实际 25m35s
```

**⚠️ tqdm 的 ETA 在前几个 shard 上高估了 14 倍。** 机制：各 worker 先
`Prefetching checkpoint files into page cache`（`weight_utils.py:825`，8 线程后台），
**头几个 shard 要等 NFS 冷读**（288s、365s、199s、179s/it），prefetch 一旦追上就变成
**1.57 it/s**。⇒ **不要用早期 ETA 排期**，它测的是冷缓存不是稳态。
从 pod Created 到 `status=running` 端到端 **约 31 分钟**（含镜像 5m7s）。

期间 `status` 一直停在 `deploying | Waiting for pods to become ready`、
`/health` 的 startup probe 一直 refused —— **看起来像卡住，其实在正常加载**。
判据去 `trisol inference logs <id> --tail 400 | grep "Loading safetensors"` 看
**shard 计数在不在动**，不要看 ETA、也不要凭 status 判死活。

## ★★★ 专用 teacher 服务实测：`prompt_logprobs` 在 114688 token 上通过，**长度不是障碍**

`qwen38-flashnext-teacher`（2099737790336995328）= 从评测服务 revision 克隆 + 5 处改动：
`--max-num-batched-tokens 2048`（新增）、`--gpu-memory-utilization 0.9→0.85`、
`--max-model-len 262144→131072`、去掉 4 个 parser flag（teacher 不生成内容）、换
`--served-model-name`（冒烟时靠 `model` 字段回显挡住"打到基模"那个历史坑）。
**改完用归一化 diff 核对**（pop 掉 4 个平台托管字段后应当只差这 5 处），不要眼看。

**单条递增长度**（`max_tokens=1, prompt_logprobs=0`，token-id 数组当 prompt）：

| T | 耗时 | tok/s | `len(prompt_logprobs)==T` |
|---|---|---|---|
| 2048 | 2.5 s | 830（首请求含 warmup） | ✓ |
| 8192 | 0.8 s | 9851 | ✓ |
| 32768 | 3.4 s | 9534 | ✓ |
| **114688** | **12.7 s** | **9056** | **✓** |

**⇒「峰值只由 chunk 决定、与 prompt 长度无关」这个推断被证实。** T=114688 是 T=2048 的
56 倍长度，chunk 相同 ⇒ 不但没 OOM，吞吐只降 8%。**OPD 的长轨迹对 teacher 不构成显存障碍。**

**并发压测**（每条 114688 token，首 token 加 salt 破坏 prefix caching）：

| 并发 | 成功 | 墙上时间 | 总吞吐 | 单条延迟 min/med/max |
|---|---|---|---|---|
| 4 | 4/4 | 44.3 s | 10363 tok/s | 13.0 / 33.7 / 44.2 s |
| 8 | 8/8 | 86.2 s | 10650 tok/s | 12.9 / 54.6 / 86.0 s |
| 16 | 16/16 | 170.1 s | 10788 tok/s | 12.8 / 96.2 / 169.7 s |

**0 失败。吞吐饱和在 ~10700 tok/s，与并发数无关；延迟随并发线性增长**
⇒ 已达算力上限，多余请求在 vLLM scheduler 里按 KV 排队（**不是** OOM 风险）。
实测 `kv_cache_size_tokens=1706658`、`kv_cache_max_concurrency=13.02`，
一条 114688 的轨迹占 KV 的 6.7% ⇒ 16 条并发（1.83M token）超出容量，排队是预期且安全的。

显存在压测前后**都是 73029/81920 MiB**（util 0.85 比 0.9 正好少占 4096 MiB，对得上）,
余量 8.68 GiB ≫ 2048-chunk 所需的 1.89 GiB。

**折算 prod**（bs128 × ~24239 tok ≈ 3.1M tok/step）：teacher 侧 ≈ **4.8 min/step**，
而 bs128 的 `timing_s/step` 是 4600 s 量级 ⇒ **teacher 占约 6%，不是瓶颈**（与 27B 的估计一致）。

## ★★ 外部 teacher client 已实现并实测通过（2026-09-15）

代码已在仓库，**`AsyncTeacherLLMServerManager` 一行未改** —— 新 client 同时复刻
`LLMServerClient.generate` 的签名和 `extract_prompt_logprobs` 的输出形状，消费方分辨不出两条路。

| 文件 | 改动 |
|---|---|
| `verl/experimental/teacher_loop/external_teacher_client.py` | **新增**：`shape_prompt_logprobs()` + `ExternalTeacherClient` + `ExternalTeacherRequestError` |
| `verl/workers/config/distillation.py` | 新增 `DistillationExternalTeacherConfig`、`all_teachers_external()`、`is_external`、`world_size` 外部时返回 **0** |
| `verl/trainer/config/distillation/distillation.yaml` | `teacher_models.teacher_model.external` 段 |
| `verl/experimental/teacher_loop/teacher_model.py` | 外部 teacher 早退（不起 engine、不建 load balancer）、只对本地 teacher 切池、`get_client()` 分派 |
| `verl/trainer/ppo/v1/trainer_base.py` | `all_teachers_external` 时跳过 `teacher_pool` 与 `Role.TeacherModel` 映射；取池处改成 `None` |
| `verl/trainer/main_ppo_v0.py` | 同上（legacy v0 路径的两处门） |

开启方式（`nnodes=0`，teacher **0 卡**）：

```bash
distillation.enabled=true distillation.nnodes=0 \
distillation.distillation_loss.loss_mode=k1 \
distillation.distillation_loss.use_policy_gradient=True \
distillation.distillation_loss.use_task_rewards=False \
distillation.teacher_models.teacher_model.external.base_url=https://xxx.inference.trisol.dp.tech \
distillation.teacher_models.teacher_model.external.model_name=<served-model-name>
```

`api_key` 走 `external.api_key_env`（默认 `TEACHER_API_KEY`）读环境变量，不要写进 config。

### ★ 零 GPU 是靠 `world_size` 返回 0 表达的，不要去改那个求和判据

`DistillationConfig.__post_init__` 的 `Σ(num_replicas × per_replica_world_size) == n_gpus_per_node × nnodes`
**一个字都不用改**：外部 teacher 的 `world_size` 恒为 0，`nnodes=0` 时右边也是 0。
反过来，外部 teacher 配 `nnodes>0` 会被显式拦下（要 GPU 却不用 GPU 是配置错误）。

另一处必须动的是 `ResourcePoolManager.get_resource_pool(Role.TeacherModel)` ——
它是 `resource_pool_dict[self.mapping[role]]`，**角色没映射就 KeyError**，
所以调用点要先查 `Role.TeacherModel in ...mapping`。

### ★★ k1 也必须返回 `prompt_ids`，尽管它不用

我原本打算 `extra_fields` 只放 `prompt_logprobs`。**错了**：
`teacher_manager.py:138-140` 无条件 `torch.tensor(extra_fields["prompt_ids"])` 并 assert
`teacher_ids.shape[0] == teacher_logprobs.shape[0] == len(sequence_ids)`。
k1 不**使用** ids 的语义，但张量照建、形状照断言 ⇒ 两个数组都得给，长度都是 S。

形状规则（照 `vllm_rollout/utils.py:484-517` 逐条复刻）：跳过 index 0（首 token 无条件 logprob）、
`k==0` 取字典里唯一那项、`k>0` 按 `rank-1` 落位且丢掉 `rank>k`、最后**补一行 dummy**
（`[0]*max(k,1)` / `[0.0]*max(k,1)`）。⇒ 位置 `i` 存的是 token `i+1` 的 logprob。

**离线等价性已逐位验证**：随机构造 vLLM 对象版与 JSON 版同源输入，
`(S,k) ∈ {(1,0),(2,0),(17,0),(1,32),(5,32),(9,1)}` 六组全部 `match=True`（含 S=1 只有 dummy 行的边界）。

### 实测：对 `qwen38-flashnext-v029`（老评测服务）短 prompt 全通过

127 token，`Authorization: Bearer` 走环境变量：

| k | 结果 |
|---|---|
| **0**（k1 路径） | shape 127×1，**next-token 对齐 True**（`pid[i][0] == ids[i+1]` 全中），per-token NLL **0.8679** |
| **20** | shape 127×20，按 rank 单调 True，无 `None` 落位，next token 落在 top-20 的 124/126 |
| 32 | **HTTP 400**：`Requested prompt logprobs of 32, which is greater than max allowed: 20` |

**⇒ 老评测服务没带 `--max-logprobs`，默认上限 20。** 这是服务端启动参数，不是 client 问题；
但 topk 类 loss（`forward_kl_topk` 默认 topk=32/128）走外部 teacher 时**必须起服务时就配 `--max-logprobs`**，
否则每个请求 400。k1 用 `prompt_logprobs=0`，不受这条限制。

并发 8 条 / `max_concurrency=4` 的信号量：形状全对，wall 0.2s。

**`qwen38-27b-eval-tp2`(2099319421876056064) 已 `stopped`** ⇒ 打它的 endpoint 返回 **404 空 body**，
不是 client bug。测外部 teacher 前先 `trisol inference get <id> -o json` 看 `status`。

### 4xx 不重试的判据要用异常字段，不要 substring 匹配

第一版写的是 `if isinstance(e, RuntimeError) and "HTTP 4" in str(e): raise` ——
而那条消息的尾部**嵌着服务器返回的 body**，body 里恰好出现 `HTTP 4` 的 5xx 就会被当成 4xx 跳过重试。
改成 `ExternalTeacherRequestError` 带 `status` 字段，判 `400 <= e.status < 500`。

实测：4xx **0.03s** 抛出（`max_retries=3, retry_backoff_s=5.0` 下若重试会是 35s+）；
传输错误（不可路由 host）3 次尝试、backoff 0.1→0.2、总 0.30s 后抛 —— 两条路径都按设计走。

三处防御性报错是故意的：
- **返回的 `prompt_logprobs` 条数 ≠ `len(prompt_ids)`** ⇒ 直接炸。这是 tokenizer 不一致 /
  服务端截断的唯一信号，放过去就是**静默污染蒸馏目标**。
- 缺 `prompt_logprobs` 字段 ⇒ 服务端不支持（非 vLLM 类）。
- 多模态输入 ⇒ 这条通道不支持，要用就 colocate。

## trisol CLI 补充坑（本窗新踩）

- `trisol model download <name>:<ver> a.json b.json -o <dir>` ⇒ `accepts between 1 and 3 arg(s)`。
  单文件形态**只收一个** repo path，多文件只能循环。
- 公共模型 `--model`/`download` 的**名字解析只在自己的模型里找** ⇒ `no model matching "..."`。
  用数字 id + 废弃的三参形态：`trisol model download <id> <ver> <path> --output -`。
  列表要 `trisol model list --public --search <kw> --all`。
- `trisol model files <id>` ⇒ `accepts 2 arg(s)`，必须给 version：`trisol model files <id|name> <ver>`。
  它的 `-o json` 是**裸 list**，不是 `{"items": [...]}`。
- `trisol inference` 的诊断三件套：`pod-events`（K8s 事件，能看到镜像拉取耗时）、
  `logs --tail N`（活体 pod 快照，`--history` 走 Loki）、`status`。
  **`trisol inference get` 不返回 runtime spec**，要 `trisol inference revision <svc_id> <rev>`。
- **heredoc `python3 - <<'EOF'` 的 stdout 再 pipe 给 `grep -v` 会吞掉退出码**，
  静默失败两次（`Exit code 1` 且无输出）。用不同的定界符（`PYEOF`）并把 stderr
  过 `grep -viE` 而不是 `2>/dev/null`。

---

# 工具输出长度与截断（2026-09-15，实测 + 已落地）

## 结论先行：上下文超限只占评测失败的 **1.9%**，不是主因

`trajectories_opd_bs128_step10`（800 题）全量统计，峰值上下文按**发出过的请求**算
（= 每条 assistant 消息**之前**的累计上下文），换算用实测 chars/token：

| 失败类型 | 数量 | 峰值上下文中位（tokens，估算） |
|---|---|---|
| 硬失败 `rc=1`（API 400 上下文超限） | **15/800** | > 262,144 |
| `None`（撞 30 轮没交答案） | 490/785 | **28,942** = 上限的 **11%** |
| `False`（交了但错） | 81/785 | 33,325 |
| `True` | 214/785 | 21,328 |

全体峰值中位 **27,492 tokens = 上限的 10.5%**，p95 110,666，只有 **5/785（0.6%）**
超过上限的 80%。**主导失败模式（62% 的 `None`）离上限差 9 倍，与长度无关。**

**★ 算峰值必须只算"真的发出去过"的请求。** 我第一版把"下一次请求"（撞 30 轮上限后
根本没发）也累进去，于是有 3 条轨迹看起来超限——而它们明明有输出文件。
**有输出文件 = 没超限，这是个免费的自检，算出矛盾就是口径错了。**

## chars/token 必须实测，两类文本差 35%

用 `/personal/Qwen/Qwen3.5-9B` 的 tokenizer 在 25 条轨迹上实测：

| 文本 | chars/token |
|---|---|
| **工具返回**（密集/结构化） | **2.65** |
| assistant | 3.58 |
| sys + user | ~3.55 |

上下文构成：工具 **49.1%** / assistant 46.1% / sys+user 4.7%。

## 工具返回长度分布（n=18,904，47.0M 字符）

p50 **462** / p75 1450 / p90 **5214** / p95 10803 / p99 25828 / p99.9 163522 /
max **883,678** / mean 2488。⇒ **典型返回很短，长尾极重。**

## ★ 长返回的 97% 是沙盒把命令**原样回显**，纯重复

沙盒失败时返回 `[error] 执行命令失败: <逐字回显的命令> ... 返回码: N 标准输出: ... 标准错误: ...`。
那段回显**已经在上下文里**（就是 assistant 那次 `tool_call` 的 arguments）⇒ 纯浪费。

- 含 `返回码:` 的 3,862 条返回里，**标记之前**占全部工具字符的 **34.7%**，
  标记之后只占 **6.8%**。
- 其中长返回（>8000）：总长中位 13,811，`返回码:` **之后**的有用载荷中位只有 **460 字符**，
  回显占比中位 **96.8%**。
- 与前一条 assistant 命令**精确逐字匹配**的回显：2,185 条、7,883,187 字符 =
  **全部工具字符的 16.8%**。

## ★ 必须 head+tail，不能只留 head

长的**非错误**返回（`ls -la` 列表、构建日志、`cat` 源码）答案在**尾部**。
实例：一条 170,055 字符的 salmon index 日志，最后一行才是最终答案数组
`[18946, 18902, 1.0024, 5.7409, 5.7408, True, False]`。只留 head 会把答案剪掉。

## 阈值选 8000（head 3000 + tail 5000）

| 阈值 T | 命中的返回 | 砍掉的工具字符 | 峰值 >80% 上限的轨迹 |
|---|---|---|---|
| 不截断 | — | — | 5 |
| 32,000 | 0.71% | 18.4% | 1 |
| 16,000 | 2.79% | 27.3% | 0 |
| **8,000** | **7.02%** | **41.9%** | **0** |
| 4,000 | 11.9% | 56.6% | 0 |

p90=5,214 ⇒ **93% 的返回一个字不动**；砍掉 41.9% 工具字符 ≈ 20.6% 总上下文；
把工具侧贡献封顶在 `30 × 8000 / 2.65 ≈ 90k` tokens。
tail 给 5000 是因为 `返回码` / stderr / 最终答案都在尾部。

## 相关≠因果：长返回轨迹的正确率低是难度混淆

| | 有该长度以上的返回 | 没有 |
|---|---|---|
| >16,000 字符 | **14.3%** | **33.6%** |
| >32,000 | 14.8% | 29.2% |
| >64,000 | 14.9% | 28.0% |
| >100,000 | 16.1% | 27.7% |

**但控制轮数后就没了**：只看每条轨迹**前 10 条**工具返回，三个结果组的长度中位
分别是 438 / 484 / 441，**几乎相同** ⇒ 是难题跑得久、返回更多更长，不是长返回导致做错。
**不要拿上面那张表说"截断能涨点"。**

## 落地：`agent_trisol.py` 的截断**默认开**（2026-09-15 起）

`/personal/sp2/qwen_trajs/cc_deepseek_1w/agent_trisol.py` 原本**完全没有**工具输出截断。
新增 `truncate_tool_output()` + 三个环境变量，**默认
`TOOL_OUTPUT_MAX_CHARS=8000 / HEAD=3000 / TAIL=5000`，以后所有评测都开**。
要复现历史（不截断）口径就显式设 `TOOL_OUTPUT_MAX_CHARS=0`。

理由：开截断的目的**不是涨点，是消除"API 400 上下文超限 ⇒ rc=1 ⇒ 不写输出文件"这个
硬失败模式**，让每道题至少能跑完。截断口径与不截断口径的结果**直接对比即可，
把哪些题是哪个口径记清楚就行**（用户决策）。

**记录约定**：目录里放一个标记文件说明混合口径，例如
`trajectories_opd_bs128_step10/_TRUNCATED_RERUN.json`（列出补跑的 15 题）、
`trajectories_opd_bs128_step20/_NO_TRUNCATION_PREFIX.json`（列出切换默认之前跑完的 61 题）。
打分脚本要把这些 `_*.json` 和 `summary_trisol.json` 一起排除。

**`_execute_tool` 有两个调用点，必须都改。** 打补丁时 `assert s.count(old) == 2` 再替换，
漏一个就是一半的工具返回不截断，而且不报错。

## ★★ 15 题 A/B：截断把硬失败全救活了，但**救不回正确率**（完美配对）

同一个 step10 adapter、同一批 15 道**在不截断时 `rc=1` 全死**的题，只开截断重跑：

| | 不截断 | **开截断（8000/3000/5000）** |
|---|---|---|
| 产出输出文件 | **0/15** | **15/15** |
| `eval.correct is True` | — | **3/15（0.20）** |
| `None`（撞 30 轮没交答案） | — | 12/15 |
| 峰值上下文 max | > 262,144 | **94,027（上限的 35.9%）** |
| 峰值上下文中位 | — | 23,027 |
| 被截断的工具返回 | — | 36/397 = **9.1%**（预测 7.02%） |

**⇒ 截断消除的是"死于上下文超限"这个失败模式，不是"做不对"。**
3/15 = 0.20 落在 step10 全集 0.2745 的下方，与"这 15 道本来就是最难最长的题"一致。
（预测 7.02% vs 实测 9.1%：这 15 题是长尾里最极端的，命中率高于全体，符合预期。）

**这 15 条已按用户决策合并回 `trajectories_opd_bs128_step10/`（口径记在
`_TRUNCATED_RERUN.json`），step10 从此是完整的 800 题。** 全量重算（789 四方公共集，
自校验 `merged vs base = +0.1242 / p=2.633e-14`、`step40 vs base = +0.0608 / p=1.098e-4`
与本文件历史记录逐位吻合）：

| | 正确 | 率 |
|---|---|---|
| SFT-merged | 283/789 | 0.3587 |
| OPD step40（bs8×n4） | 233/789 | 0.2953 |
| **OPD step10（bs128×n1）** | **216/789** | **0.2738** |
| 基模 | 185/789 | 0.2345 |

| 对比 | Δ | 95%CI | p |
|---|---|---|---|
| step10 vs 基模 | **+0.0393** | [+0.011,+0.067] | **0.0080** |
| **step10 vs step40** | **−0.0215** | [−0.049,+0.006] | **0.152** |
| step10 vs SFT-merged | −0.0849 | [−0.114,−0.056] | 2.40e-08 |

补齐 15 题后 step10 从 776 集的 0.2745 变成 789 集的 0.2738，**三条结论一个都没变**
（尤其「等预算下 `ROLLOUT_N` 4→1 无差别」仍然是 p=0.152）。
之前报的上下界 0.2675–0.2863 也被实测值坐实。

~~**训练侧同样值得做**：`verl_coding/coding_sandbox_tool.py` 也没有截断，
那 16.8% 的纯重复既吃 rollout 上下文也吃吞吐。（未做。）~~
**⚠️ 这句作废** —— 工具本身确实不截，但截断在 agent loop 里，**训练侧一直在截，而且比评测严 4 倍**。见下节。

## ★★ 训练侧早就在截，上限 2048/middle —— 不一致的方向与直觉相反（2026-09-16 实测）

`run_coding_practice_qwen3_5_9b_4l20.sh:59` `max_tool_response_length=${MAX_TOOL_RESPONSE_LENGTH:-2048}`
（→ `:160` 的 `actor_rollout_ref.rollout.multi_turn.max_tool_response_length`），
`tool_response_truncate_side` 取默认 **`middle`**（`verl/workers/config/rollout.py:58`）⇒
`tool_agent_loop.py:548-555` 做 `head 1024 + "...(truncated)..."(17) + tail 1024` = **2065 字符硬顶**。

`rollout_data/nd267`（3840 轨迹 / 64170 条工具返回）实测，**p90/p95/p99/p999 全部恰好 2065**，
2000–2099 桶里有 16431 条；工具返回只占 output 字符的 **20.9%**。
⇒ **不要在 rollout 数据上统计"工具返回长度分布"当作真实分布**，它是被 2065 削平的。

**⇒ 真正的口径不一致是 train 2048/middle vs eval 8000(3000+5000)，训练比评测严。**
（评测侧 2026-09-15 起默认开 8000，见上节。）

### 未截断的真分布与两个上限的代价（`trajectories_qwen9b_base_with_rc` 799 轨迹 / 19125 条返回，实测）

原始 p50 409 / p75 1380 / p90 5049 / p95 9728 / p99 22270 / max 345761 / mean 2070。

| 上限 | 命中返回 | 保留工具字符 | vs 2048 | 工具占上下文 |
|---|---|---|---|---|
| 不截断 | — | 39.6M | 2.78× | **52.4%** |
| **2048（训练）** | **18.07%** | 14.25M | 1.000× | **28.4%** |
| 3072 | 13.89% | 17.34M | 1.217× | 32.6% |
| 4096 | 11.62% | 19.82M | 1.391× | 35.6% |
| **8000（评测）** | **6.45%** | 26.35M | **1.850×** | **42.3%** |

**2048-middle 丢的到底是什么（实测，按 `返回码:` 拆）：**

| | n | p50 | >2048 | 2048<l≤8000 | >8000 |
|---|---|---|---|---|---|
| 错误返回（含 `返回码:`） | 4645 | 1064 | 39.3% | 21.7% | 17.7% |
| 非错误返回 | 14480 | 245 | 11.2% | 8.4% | 2.9% |

错误返回里 `返回码:` **之后**的有用载荷 p50=**327**、**81.6% ≤1024** ⇒ middle 模式的 tail 1024
**把错误返回的关键信息基本全保住了**（这也是 middle 优于 left/right 的实证理由）。
真正被"腰斩"的是长目录列表 / 构建日志 / `cat` 源码的**中段**；抬上限能新增信息的人群
只有落在 **2048<l≤8000 的那 11.6%** 返回。

### 没有任何测量支持"2048 伤了性能"（两个量，一个阴性一个混淆）

- **重试代理（阴性）**：被截断后下一次 `tool_call` 与本次**逐字相同**的比例 **0.61%**（15700 次机会），
  未截断是 **2.11%**（45518 次）⇒ 截断**没有**引起"重跑同一条命令"。
  （混淆：长输出本来就是不该重复的命令。）
- **每轨迹截断率 vs score（混淆，不可读成因果）**：按 tool 轮数分桶控制后失败组始终更高
  （1–5 轮桶 0.340 vs 0.195；21–30 轮桶 0.242 vs 0.219）—— 与评测侧记的同一个难度混淆。

### 抬上限的代价：换的是 response token 预算，不是"信号"

**工具 token 不进 loss**（`tool_agent_loop.py:457` `response_mask += [0] * len(response_ids)`）⇒
截断**不改变任何被训练的 token**，只改变条件上下文。所以这是吞吐 / 上下文预算 / 训评一致性问题。

**但 masked 的 tool token 照样计入 `response_length`**（`:444` 的门是
`len(agent_data.response_mask) + len(response_ids) >= self.response_length` ⇒ `TERMINATED`）
⇒ **截断直接换的是轮数预算，撞上限的轨迹被砍断 = 几乎必然 score 0。**

按 eval 侧聚合比缩放每条轨迹的工具字符 + 实测 chars/token（工具 2.65 / assistant 3.58）
**估算**（**推断，不是实测**）nd267 的 response token：

| 上限 | mean | p99 | ≥98304 | ≥80% 上限 |
|---|---|---|---|---|
| **2048** | 20630 | 82630 | 0.05% | 51 |
| 4096 | 22758 | 86965 | 0.21% | 70 |
| **8000** | **25250（+22%）** | 91276 | **0.42%（8×）** | 88 |

`timing_s/gen` 占 step 的 80% ⇒ +22% response token 会直接体现在步时长上。

**⇒ 结论：不要把 rollout 的上限抬到 8000。** (a) 新增信息集中在 11.6% 的返回，
而错误返回的关键载荷 2048-middle 已经保住；(b) 代价是 +22% response token、
撞 response 上限的轨迹 ×8、`gen` 同比涨；(c) 没有任何测量支持 2048 伤了性能。
**"训练 2048 / 评测 8000"这个不一致要记着**，它是"训练侧 `critic/score/mean` 与评测侧准确率
不可直接比"的一个候选来源（**未验证**）。

### 小 bug：`Unknown function` 绕过截断

`tool_agent_loop.py:504-506` 的 `Unknown function '<name>'. Available tools: [...]` 是
**早 return**（`return ToolResponse(text=msg), 0.0, {}`），走不到 `:548` 的截断。
模型把 XML 写坏时 `tool_name` 会吞掉整条命令 ⇒ nd267 里 16 条 >3000 字符的返回全是这个，
最长 **15546** 字符。占 64170 条的 **0.025%**，不紧急，但它是"训练侧还有长返回"的唯一来路。

## 复用 `discover_problems()` 的 skip 语义来选题（比加 CLI flag 省事）

`batch_run_trisol.py:discover_problems()` 对 `{output_dir}/{name}.json` 做
`Path.is_file()`，命中就 `Skip problem`。**`is_file()` 跟随符号链接** ⇒
新建一个目录、把已有的 785 个输出 **symlink** 进去，runner 就恰好只跑缺的 15 题，
而新产出的是**真文件**，`os.path.islink()` 一眼能和 symlink 分开。不用改脚本、不用传题目列表。

**题名不要从日志里抄** —— 进度行把名字截断到 60 字符，拿它去 glob 会命中兄弟题目。
要用 `base.glob("*/raw/*/*/generator_output.json")` + `name = 父目录名 + "_" + 目录名` 自己算。
