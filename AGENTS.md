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
  --since 2026-09-08T05:00:00Z --direction forward \
  --grep "training/global_step" --tail 5000
```

（`--since` 用 `trisol train get` 里的 `created_at`。`--grep` 是**纯子串**不是正则。
另外 `--tail` 对已终止的多节点任务在**非** `--history` 模式下会报
`Error: pods "..." not found (code=500)` —— 活体 pod 流已经没了，只能走 `--history`。）

**`--tail` 必须在 1..5000 之间**：`--history --tail 8000` 不会截断到上限，而是**只返回一行
400 报错**，看起来像"日志是空的"。（和 SFT 那节记的 `--tail 100000` 返回空是同一个原因。）

**从 metrics 行提取数值不要用 `tr`。** `tr -d ' ' `之类顺手写的清洗里只要带 `-`，
负值就被吃掉，`pg_loss:-0.00047` 会变成 `0.00047`，符号翻了还看不出来。用正则取：

```python
re.search(re.escape(key) + r':(-?[\d.eE+]+)', line).group(1)
```

（`actor/pg_loss`、`distillation/loss` 这些**带符号**的量是判据本身，符号错了结论就反了。）

**`trisol train get -o json` 没有顶层 `envs` 字段** —— 提交时的环境变量在 `custom_config` 里，
要核对某次任务实际带了什么 env（比如确认 treatment 真的生效）得去那里找。

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

verl 写的 checkpoint 目录名是 `global_step_*`，而平台默认发现规则的 prefix 是
`checkpoint-`，所以**默认情况下 checkpoint 也不会被归档** —— 要用同样的 rescan，
prefix 改成 `global_step_`。

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
走 verl 原生 `_flash_attention_forward` 的 a2a）。但注意那条路**并不更保险**：qwen3_5 + SP 从来
没人真跑过它（flash_attn 一直缺），而且它每层要 all_gather 一次 `position_ids`
（`monkey_patch.py:87-147`），我们的 sdpa 补丁按全局 `cu_seqlens` 分段**零额外通信**。
要换只能拿实测 `timing_s/old_log_prob` 说话，不要因为"flash 听起来更快"就换。

## lbg 沙盒/镜像的实测约束（2026-09-10）

- `lbg sdbx exec`：**所有 flag 必须在 `sandbox_id` 之前**，用 `--cwd` 指定目录，
  命令要作为**单个 shell-quoted 参数**传（`-- bash -lc '...'` 不行，会在 `/root` 跑）。
- `lbg sdbx template rm` 需要 `--force`。
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
actor/distillation/loss_max:   7.73   (loss_max_clamp=10.0 未触顶)
```

这 0.455 **全部由引擎偏差驱动，且是系统性偏差不是零均值噪声** —— 方向上把 FSDP 的 student
往 vLLM 的数值行为上拽；单 token 尾巴到 −22.7 说明偏差分布重尾。
真教师那步要看真实信号能否压过这个量级。

**吞吐参考**：`timing_s/step=2259s`（37.7 min），其中 `gen=1800s` 占 80%，
`update_actor=274s`、`old_log_prob=112s`、`ref=33s`；教师 logprob 没有单独的 timing 项。
`num_turns/mean=52.6`（assistant+tool 交替计数，对应 30 个 assistant 轮上限），
`response_length/mean=27986`、`max=68807`，`perf/mfu/actor=0.104`。

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
