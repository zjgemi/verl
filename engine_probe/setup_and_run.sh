#!/bin/bash
# 在沙盒里跑: 环境勘察 -> 下模型 -> 装 causal_conv1d -> vLLM 生成 -> HF 打分
set +e
cd /tmp/workspace
mkdir -p /tmp/probe
export HF_HUB_DISABLE_PROGRESS_BARS=1

echo "########## 1. 环境勘察 ##########"
nvidia-smi --query-gpu=name,memory.total,driver_version --format=csv
python3 - <<'PY'
import torch, sys
print("torch", torch.__version__, "cuda", torch.version.cuda, "abi", torch._C._GLIBCXX_USE_CXX11_ABI)
print("python", sys.version.split()[0])
print("device", torch.cuda.get_device_name(0), torch.cuda.get_device_capability(0))
import vllm; print("vllm", vllm.__version__)
import transformers; print("transformers", transformers.__version__)
from transformers.utils.import_utils import is_flash_linear_attention_available, is_causal_conv1d_available
print("fla_available       :", is_flash_linear_attention_available())
print("causal_conv1d_avail :", is_causal_conv1d_available())
PY
command -v nvcc && nvcc --version | tail -2

echo "########## 2. 下载 Qwen3.5-2B ##########"
if [ ! -f /tmp/probe/model/config.json ]; then
  pip install -q modelscope 2>&1 | tail -2
  python3 -c "
from modelscope import snapshot_download
p = snapshot_download('Qwen/Qwen3.5-2B', local_dir='/tmp/probe/model')
print('model at', p)
" 2>&1 | tail -5
fi
ls -la /tmp/probe/model | head -8
du -sh /tmp/probe/model

echo "########## 3. 安装 causal_conv1d ##########"
WHL="https://github.com/Dao-AILab/causal-conv1d/releases/download/v1.7.0/causal_conv1d-1.7.0+cu12torch2.10cxx11abiTRUE-cp311-cp311-linux_x86_64.whl"
pip install -q --no-deps "$WHL" 2>&1 | tail -3
python3 -c "
from causal_conv1d import causal_conv1d_fn
import torch
x=torch.randn(2,64,128,device='cuda',dtype=torch.bfloat16)
w=torch.randn(64,4,device='cuda',dtype=torch.bfloat16)
y=causal_conv1d_fn(x,w,None,activation='silu')
print('WHEEL OK, out', tuple(y.shape))
" 2>&1 | tail -4
