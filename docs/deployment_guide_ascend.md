# TerAgent 国产芯片部署指南 — Ascend 910B / 950PR

> **适用版本：** TerAgent v0.1.1+  
> **目标模型：** GLM-5 / GLM-5.2 / DeepSeek V4 / MiniMax M3  
> **硬件平台：** 华为昇腾 Ascend 910B、Ascend 950PR  
> **状态：** 部署指南 + 配置模板（实际硬件验证待后续执行）

---

## 目录

1. [概述](#1-概述)
2. [环境准备](#2-环境准备)
3. [GLM-5 部署 — Ascend 910B](#3-glm-5-部署--ascend-910b)
4. [DeepSeek V4 部署 — Ascend 950PR](#4-deepseek-v4-部署--ascend-950pr)
5. [MiniMax M3 部署 — Ascend 910B](#5-minimax-m3-部署--ascend-910b)
6. [GLM-5.2 部署 — Ascend 910B](#6-glm-52-部署--ascend-910b)
7. [TerAgent 本地部署配置](#7-teragent-本地部署配置)
8. [验证清单](#8-验证清单)
9. [性能调优](#9-性能调优)
10. [常见问题与解决方案](#10-常见问题与解决方案)
11. [附录](#11-附录)

---

## 1. 概述

### 1.1 架构总览

TerAgent 国产化部署架构如下：

```
┌──────────────────────────────────────────────────────────────────────┐
│                      TerAgent 中间件层                                │
│  (agent.local.toml 配置 → 本地推理端点 http://localhost:8xxx)          │
├──────────────┬──────────────┬──────────────┬────────────────────────┤
│  GLM-5       │  GLM-5.2     │  DeepSeek V4 │  MiniMax M3           │
│  :8001/v1    │  :8004/v1    │  :8002/v1    │  :8003/v1              │
├──────────────┼──────────────┼──────────────┼────────────────────────┤
│  MindIE /    │  MindIE /    │  vLLM-Ascend │  MindIE /              │
│  vLLM-Ascend │  vLLM-Ascend │  Ascend PT   │  vLLM-Ascend           │
├──────────────┼──────────────┼──────────────┼────────────────────────┤
│  CANN 8.x    │  CANN 8.x    │  CANN 8.x    │  CANN 8.x              │
│  (910B)      │  (910B x2)   │  (950PR)     │  (910B x2)             │
└──────────────┴──────────────┴──────────────┴────────────────────────┘
```

### 1.2 模型-硬件对应关系

| 模型 | 推荐芯片 | 上下文窗口 | 关键特性 | 推荐推理框架 |
|------|---------|-----------|---------|------------|
| **GLM-5** | Ascend 910B | 200K tokens | 长程任务（8h 自治）、深度推理 | MindIE / vLLM-Ascend |
| **GLM-5.2** | Ascend 910B ×2 | 1M tokens | 长程+1M 上下文、High/Max 双思考、PreservedThinking、5V-Turbo 视觉协调 | MindIE / vLLM-Ascend |
| **DeepSeek V4** | Ascend 950PR | 1M tokens | Flash/Pro 双变体、缓存感知 | vLLM-Ascend / Ascend PyTorch |
| **MiniMax M3** | Ascend 910B | 1M tokens | 原生多模态、桌面操作、MSA | MindIE / vLLM-Ascend |

### 1.3 部署成本优势

| 维度 | 云 API 部署 | 国产芯片本地部署 |
|------|-----------|---------------|
| 推理费用 | ¥0.5-5/万 tokens | **¥0**（电费 + 折旧） |
| 数据安全 | 数据经第三方 | **数据完全本地** |
| 网络延迟 | 50-200ms（公网） | **<5ms（本地）** |
| 可用性 | 依赖云服务 SLA | **自主可控** |
| 月度成本（重度使用） | ¥500-5000 | **¥0 推理成本** |

---

## 2. 环境准备

### 2.1 硬件要求

#### Ascend 910B（适用于 GLM-5 / MiniMax M3）

| 配置项 | 最低要求 | 推荐配置 |
|--------|---------|---------|
| AI 处理器 | Ascend 910B x1 | Ascend 910B x2（M3 多模态推荐） |
| 内存 | 256 GB DDR4 | 512 GB DDR4 |
| 系统盘 | 500 GB NVMe SSD | 1 TB NVMe SSD |
| 模型存储盘 | 1 TB NVMe SSD | 2 TB NVMe SSD（多模型共存） |
| 网络 | 10 GbE | 25 GbE（多卡集群） |
| HCCN | RoCE v2 | RoCE v2 |

#### Ascend 950PR（适用于 DeepSeek V4）

| 配置项 | 最低要求 | 推荐配置 |
|--------|---------|---------|
| AI 处理器 | Ascend 950PR x1 | Ascend 950PR x2（V4 1M 上下文推荐） |
| 内存 | 512 GB DDR5 | 1 TB DDR5 |
| 系统盘 | 500 GB NVMe SSD | 1 TB NVMe SSD |
| 模型存储盘 | 2 TB NVMe SSD | 4 TB NVMe SSD |
| 网络 | 25 GbE | 100 GbE（多卡集群） |

### 2.2 操作系统

| 操作系统 | 版本要求 | 说明 |
|---------|---------|------|
| **openEuler** | 22.03 LTS SP3+ | 推荐首选，华为官方支持 |
| **Kylin V10** | SP2+ | 国产操作系统，兼容性好 |
| **Ubuntu** | 22.04 LTS | 需手动安装 Ascend 驱动 |
| **CentOS** | 7.6 / 8.2 | 企业场景常见 |

> **建议：** 生产环境优先选择 openEuler 22.03 LTS SP3，与 CANN 驱动兼容性最优。

### 2.3 CANN 驱动与固件安装

CANN（Compute Architecture for Neural Networks）是昇腾 AI 处理器的计算框架。

#### 2.3.1 安装步骤

```bash
# 1. 检查 NPU 设备是否识别
npu-smi info
# 预期输出：显示 Ascend 910B / 950PR 设备信息

# 2. 下载 CANN 8.x 驱动和固件
# 访问华为昇腾社区：https://www.hiascend.com/software/cann
# 下载对应版本的：
#   - Ascend-hdk-xxx.npu-driver_x.x.x_linux-x86_64.run
#   - Ascend-hdk-xxx.npu-firmware_x.x.x_linux-x86_64.run

# 3. 安装驱动（需要 root 权限）
chmod +x Ascend-hdk-*.run
./Ascend-hdk-910b.npu-driver_23.0.x_linux-x86_64.run --install
./Ascend-hdk-910b.npu-firmware_7.0.x_linux-x86_64.run --install

# 4. 安装 CANN Toolkit
# 下载 CANN 8.0.RCx toolkit
./Ascend-cann-toolkit_8.0.RC1_linux-x86_64.run --install

# 5. 配置环境变量
source /usr/local/Ascend/ascend-toolkit/set_env.sh
# 建议写入 ~/.bashrc
echo 'source /usr/local/Ascend/ascend-toolkit/set_env.sh' >> ~/.bashrc

# 6. 验证安装
npu-smi info
python3 -c "import acl; print('CANN ACL OK')"
```

#### 2.3.2 CANN 版本兼容性

| CANN 版本 | Ascend 910B | Ascend 950PR | Python |
|-----------|:-----------:|:------------:|--------|
| CANN 8.0.RC1 | ✅ | ✅ | 3.8-3.11 |
| CANN 8.0.RC2 | ✅ | ✅ | 3.8-3.11 |
| CANN 7.0.x | ✅ | ❌ | 3.7-3.10 |

> **重要：** Ascend 950PR 需要 CANN 8.0+ 版本。请在安装前确认 CANN 版本与芯片型号匹配。

### 2.4 Python 环境搭建

#### 2.4.1 Miniconda 安装

```bash
# 下载 Miniconda
wget https://repo.anaconda.com/miniconda/Miniconda3-latest-Linux-x86_64.sh
bash Miniconda3-latest-Linux-x86_64.sh -b -p /opt/miniconda3
export PATH="/opt/miniconda3/bin:$PATH"

# 创建 TerAgent 专用环境
conda create -n teragent python=3.10 -y
conda activate teragent
```

#### 2.4.2 昇腾 PyTorch 安装

```bash
# 方式一：通过昇腾官方源安装（推荐）
pip install torch==2.1.0+ascend -f https://torch-ascend.huawei.com/wheels/

# 方式二：从源码编译（兼容性更好）
git clone https://gitee.com/ascend/pytorch.git -b v2.1.0-ascend
cd pytorch
pip install -r requirements.txt
python setup.py install

# 验证
python3 -c "
import torch
import torch_npu
print(f'PyTorch version: {torch.__version__}')
print(f'NPU available: {torch.npu.is_available()}')
print(f'NPU count: {torch.npu.device_count()}')
"
```

#### 2.4.3 MindSpore 安装（可选，用于 MindIE 推理）

```bash
# 安装 MindSpore 昇腾版
pip install mindspore==2.3.0 -f https://www.mindspore.cn/wheels/ascend/

# 验证
python3 -c "
import mindspore as ms
ms.set_context(device_target='Ascend')
print('MindSpore Ascend OK')
"
```

#### 2.4.4 TerAgent 安装

```bash
# 从源码安装
cd /path/to/TerAgent
pip install -e ".[all]"

# 验证
python3 -c "
import teragent
print(f'TerAgent version: {teragent.__version__}')
config = teragent.load_typed_config()
print('Config loaded OK')
"
```

---

## 3. GLM-5 部署 — Ascend 910B

GLM-5 是智谱 AI 推出的开源大模型，原生支持昇腾 Ascend 910B，是国产化部署的理想选择。GLM-5 的核心优势在于**长程任务模式**（8 小时自治执行）和**深度推理**。

### 3.1 模型权重下载

```bash
# 创建模型存储目录
mkdir -p /data/models/glm-5
cd /data/models/glm-5

# 方式一：从 ModelScope 下载（国内推荐）
pip install modelscope
modelscope download --model ZhipuAI/glm-5 --local_dir /data/models/glm-5

# 方式二：从 HuggingFace 下载（需网络代理）
# git lfs install
# git clone https://huggingface.co/ZhipuAI/glm-5

# 方式三：手动下载
# 访问 https://modelscope.cn/models/ZhipuAI/glm-5
# 下载所有文件到 /data/models/glm-5/
```

**模型文件结构预期：**

```
/data/models/glm-5/
├── config.json
├── tokenizer_config.json
├── tokenizer.model
├── model-00001-of-000xx.safetensors
├── model-00002-of-000xx.safetensors
├── ...
└── modeling_glm_5.py  (如有)
```

### 3.2 推理框架部署

#### 方式一：MindIE 推理（推荐）

MindIE 是华为昇腾官方的推理加速引擎，针对 Ascend NPU 深度优化。

```bash
# 1. 安装 MindIE
pip install mindie

# 2. 启动 GLM-5 推理服务
python -m mindie.server \
    --model_path /data/models/glm-5 \
    --device ascend \
    --port 8001 \
    --max_batch_size 8 \
    --max_seq_len 200000 \
    --kv_cache_size 4096

# 3. 验证服务
curl http://localhost:8001/v1/models
# 预期返回：{"data": [{"id": "glm-5", ...}]}
```

#### 方式二：vLLM-Ascend 推理

vLLM-Ascend 是 vLLM 的昇腾适配版本，API 完全兼容 OpenAI 格式。

```bash
# 1. 安装 vLLM-Ascend
pip install vllm-ascend

# 2. 启动推理服务
python -m vllm.entrypoints.openai.api_server \
    --model /data/models/glm-5 \
    --device npu \
    --port 8001 \
    --max-model-len 200000 \
    --gpu-memory-utilization 0.90 \
    --tensor-parallel-size 1 \
    --trust-remote-code

# 3. 验证服务
curl http://localhost:8001/v1/chat/completions \
    -H "Content-Type: application/json" \
    -d '{
        "model": "glm-5",
        "messages": [{"role": "user", "content": "你好"}],
        "max_tokens": 100
    }'
```

### 3.3 端点配置

GLM-5 推理服务启动后，将在 `http://localhost:8001/v1` 提供 OpenAI 兼容的 API 端点：

| 端点 | 方法 | 说明 |
|------|------|------|
| `/v1/models` | GET | 列出可用模型 |
| `/v1/chat/completions` | POST | Chat 补全接口 |
| `/v1/completions` | POST | Text 补全接口 |
| `/v1/tokenize` | POST | Tokenizer 接口 |
| `/health` | GET | 健康检查 |

### 3.4 GLM-5 长程任务配置

GLM-5 的长程任务模式需要确保推理服务支持长连接和检查点：

```bash
# 启动时增加长上下文支持参数
python -m vllm.entrypoints.openai.api_server \
    --model /data/models/glm-5 \
    --device npu \
    --port 8001 \
    --max-model-len 200000 \
    --enable-prefix-caching \
    --kv-cache-dtype auto \
    --max-num-seqs 4 \
    --trust-remote-code
```

**关键参数说明：**

| 参数 | 值 | 说明 |
|------|-----|------|
| `max-model-len` | 200000 | GLM-5 最大 200K 上下文 |
| `enable-prefix-caching` | - | 启用前缀缓存，长程任务优化 |
| `max-num-seqs` | 4 | 并发序列数，长程任务减少并发以提高单序列吞吐 |
| `gpu-memory-utilization` | 0.90 | NPU 显存利用率 |

---

## 4. DeepSeek V4 部署 — Ascend 950PR

DeepSeek V4 是深度求索推出的开源模型，支持 1M 超长上下文，Flash/Pro 双变体。Ascend 950PR 的大显存和高带宽特别适合 V4 的 1M 上下文场景。

### 4.1 模型权重下载

```bash
# 创建模型存储目录
mkdir -p /data/models/deepseek-v4
cd /data/models/deepseek-v4

# 方式一：从 ModelScope 下载（国内推荐）
pip install modelscope
modelscope download --model deepseek-ai/deepseek-v4 --local_dir /data/models/deepseek-v4

# 方式二：从 HuggingFace 下载
# git lfs install
# git clone https://huggingface.co/deepseek-ai/deepseek-v4

# Flash 变体（轻量版）
# modelscope download --model deepseek-ai/deepseek-v4-flash --local_dir /data/models/deepseek-v4-flash

# Pro 变体（完整版）
# modelscope download --model deepseek-ai/deepseek-v4-pro --local_dir /data/models/deepseek-v4-pro
```

### 4.2 Ascend PyTorch 适配

DeepSeek V4 的推理需要通过 Ascend PyTorch 进行适配：

```bash
# 1. 确认 Ascend PyTorch 环境就绪
python3 -c "
import torch
import torch_npu
print(f'torch_npu version: {torch_npu.__version__}')
print(f'NPU available: {torch.npu.is_available()}')
"

# 2. 安装 DeepSeek V4 推理依赖
pip install transformers==4.45.0
pip install accelerate
pip install sentencepiece
pip install protobuf
```

### 4.3 推理框架部署

#### 方式一：vLLM-Ascend 推理（推荐）

```bash
# DeepSeek V4 Flash 变体
python -m vllm.entrypoints.openai.api_server \
    --model /data/models/deepseek-v4-flash \
    --device npu \
    --port 8002 \
    --max-model-len 1000000 \
    --gpu-memory-utilization 0.92 \
    --tensor-parallel-size 1 \
    --trust-remote-code \
    --enable-prefix-caching \
    --kv-cache-dtype auto
```

#### 方式二：Ascend PyTorch 原生推理

```bash
# 使用 DeepSeek 官方推理脚本 + Ascend PyTorch 适配
export ASCEND_RT_VISIBLE_DEVICES="0"
export PYTHONPATH="/data/models/deepseek-v4:$PYTHONPATH"

python3 /data/models/deepseek-v4/inference.py \
    --model-path /data/models/deepseek-v4 \
    --device npu \
    --port 8002 \
    --max-length 1000000 \
    --tensor-parallel-size 1
```

### 4.4 Flash vs Pro 变体配置

DeepSeek V4 的 Flash 和 Pro 变体共享推理服务端口，通过模型名区分：

```bash
# 同时部署 Flash 和 Pro（不同端口或同端口多模型）
# Flash — 快速响应，极简 prompt
python -m vllm.entrypoints.openai.api_server \
    --model /data/models/deepseek-v4-flash \
    --served-model-name deepseek-v4-flash \
    --device npu \
    --port 8002 \
    --max-model-len 1000000 \
    --trust-remote-code

# Pro — 深度推理，完整 prompt（需要更多资源）
python -m vllm.entrypoints.openai.api_server \
    --model /data/models/deepseek-v4-pro \
    --served-model-name deepseek-v4-pro \
    --device npu \
    --port 8005 \
    --max-model-len 1000000 \
    --trust-remote-code
```

**Flash vs Pro 对比：**

| 特性 | Flash | Pro |
|------|-------|-----|
| 响应速度 | 极快（极简 prompt） | 较慢（完整 prompt + 推理引导） |
| 适用场景 | 简单代码生成、快速对话 | 复杂设计、深度推理 |
| thinking_mode | auto / quick | deep |
| 缓存感知 | 启用 | 启用 |
| 推荐用途 | execute 阶段 | design/plan/review 阶段 |

### 4.5 1M 上下文管理

DeepSeek V4 的 1M 上下文是核心优势，但需要特殊配置以确保稳定性：

```bash
# 1M 上下文的关键配置
--max-model-len 1000000       # 最大序列长度
--enable-prefix-caching        # 前缀缓存，减少重复计算
--kv-cache-dtype auto          # 自动选择 KV cache 数据类型
--gpu-memory-utilization 0.92  # 高显存利用率，为 1M 预留空间
--swap-space 16                # CPU swap 空间 (GB)，防止 OOM
--max-num-seqs 2               # 减少并发序列，为 1M 上下文腾出显存
```

**1M 上下文注意事项：**
- Ascend 950PR 单卡可支持 1M 上下文，但并发数需降低
- 建议开启 prefix caching 以减少重复前缀的计算开销
- 长上下文推理延迟较高，建议配合 TerAgent 的缓存感知压缩策略
- TerAgent 的 DeepSeekV4Compiler 会自动进行缓存前缀冻结和尾部强化

---

## 5. MiniMax M3 部署 — Ascend 910B

MiniMax M3 是 MiniMax 公司推出的开源多模态大模型，原生支持图像、视频和桌面操作。M3 的 MSA（Multi-head Sparse Attention）架构在 1M 上下文下效率极高，是昇腾 910B 上的多模态首选。

### 5.1 模型权重下载

```bash
# 创建模型存储目录
mkdir -p /data/models/minimax-m3
cd /data/models/minimax-m3

# 方式一：从 ModelScope 下载（国内推荐）
pip install modelscope
modelscope download --model MiniMaxAI/minimax-m3 --local_dir /data/models/minimax-m3

# 方式二：从 HuggingFace 下载
# git lfs install
# git clone https://huggingface.co/MiniMaxAI/minimax-m3
```

### 5.2 多模态推理设置

M3 的多模态推理需要额外的视觉编码器配置：

```bash
# 安装多模态推理依赖
pip install pillow
pip install opencv-python-headless
pip install decord  # 视频解码

# 验证视觉编码器
python3 -c "
from PIL import Image
import io
img = Image.new('RGB', (224, 224))
print('Visual encoder dependencies OK')
"
```

### 5.3 推理框架部署

```bash
# MiniMax M3 推理服务
python -m vllm.entrypoints.openai.api_server \
    --model /data/models/minimax-m3 \
    --device npu \
    --port 8003 \
    --max-model-len 1000000 \
    --gpu-memory-utilization 0.90 \
    --tensor-parallel-size 2 \
    --trust-remote-code \
    --enable-prefix-caching \
    --limit-mm-per-prompt image=10,video=2 \
    --kv-cache-dtype auto
```

**M3 关键参数说明：**

| 参数 | 值 | 说明 |
|------|-----|------|
| `max-model-len` | 1000000 | M3 支持 1M 上下文 |
| `tensor-parallel-size` | 2 | 多模态建议双卡并行 |
| `limit-mm-per-prompt` | image=10,video=2 | 每次 prompt 最大多模态内容数 |
| `enable-prefix-caching` | - | MSA 架构下的前缀缓存优化 |

### 5.4 桌面操作 API 服务

M3 的桌面操作能力需要额外启动桌面操作 API 服务：

```bash
# 安装桌面操作依赖
pip install pyautogui
pip install mss  # 屏幕截图
pip install python-xlib

# 启动桌面操作 API 服务（独立端口）
python3 -m teragent.tools.desktop_server \
    --port 8010 \
    --screenshot-interval 0.5 \
    --max-resolution 1920x1080

# 或使用 TerAgent 内置桌面工具
# TerAgent 的 desktop.py 会自动调用桌面截图和交互元素检测
```

### 5.5 视频处理配置

M3 支持视频内容分析，需要配置视频处理管线：

```bash
# 安装视频处理依赖
pip install ffmpeg-python
pip install decord

# 配置视频处理参数（在环境变量中）
export M3_MAX_VIDEO_DURATION=300       # 最大视频时长（秒）
export M3_VIDEO_FRAME_RATE=2           # 视频采样帧率（fps）
export M3_MAX_VIDEO_RESOLUTION=1080p   # 最大视频分辨率
export M3_VIDEO_CODEC_SUPPORT="mp4,avi,mov,mkv,webm"

# 验证视频处理
python3 -c "
import decord
vr = decord.VideoReader('/path/to/test.mp4')
print(f'Video frames: {len(vr)}')
print('Video processing OK')
"
```

---

## 6. GLM-5.2 部署 — Ascend 910B

GLM-5.2 是智谱 AI 推出的旗舰模型，支持 1M 超长上下文、High/Max 双思考模式、PreservedThinking 编码计划保持、5V-Turbo 视觉协调等高级特性。GLM-5.2 在 GLM-5 基础上将上下文窗口从 200K 扩展到 1M，同时引入双思考模式和上下文自动降级机制。

### 6.1 模型权重下载

```bash
# 创建模型存储目录
mkdir -p /data/models/glm-5.2
cd /data/models/glm-5.2

# 方式一：从 ModelScope 下载（国内推荐）
pip install modelscope
modelscope download --model ZhipuAI/glm-5.2 --local_dir /data/models/glm-5.2

# 方式二：从 HuggingFace 下载（需网络代理）
# git lfs install
# git clone https://huggingface.co/ZhipuAI/glm-5.2

# 方式三：手动下载
# 访问 https://modelscope.cn/models/ZhipuAI/glm-5.2
# 下载所有文件到 /data/models/glm-5.2/
```

**模型文件结构预期：**

```
/data/models/glm-5.2/
├── config.json
├── tokenizer_config.json
├── tokenizer.model
├── model-00001-of-000xx.safetensors
├── model-00002-of-000xx.safetensors
├── ...
└── modeling_glm_52.py  (如有)
```

### 6.2 硬件要求

GLM-5.2 的 1M 上下文对硬件要求较高：

| 配置项 | 最低要求 | 推荐配置 |
|--------|---------|---------|
| AI 处理器 | Ascend 910B ×1 | **Ascend 910B ×2**（1M 上下文必需） |
| 内存 | 512 GB DDR4 | 1 TB DDR4 |
| 系统盘 | 500 GB NVMe SSD | 1 TB NVMe SSD |
| 模型存储盘 | 1 TB NVMe SSD | 2 TB NVMe SSD |
| 网络 | 10 GbE | 25 GbE（多卡集群） |
| HCCN | RoCE v2 | RoCE v2 |

> **重要：** GLM-5.2 在 1M 上下文模式下需要 Ascend 910B ×2。单卡可运行 200K 上下文模式（降级模式），但 1M 上下文需要双卡并行以容纳 KV cache。

### 6.3 推理框架部署

#### 方式一：MindIE 推理（推荐）

```bash
# 1. 安装 MindIE
pip install mindie

# 2. 启动 GLM-5.2 推理服务（1M 上下文）
python -m mindie.server \
    --model_path /data/models/glm-5.2 \
    --device ascend \
    --port 8004 \
    --max_batch_size 4 \
    --max_seq_len 1000000 \
    --kv_cache_size 16384 \
    --tensor_parallel_size 2

# 3. 验证服务
curl http://localhost:8004/v1/models
# 预期返回：{"data": [{"id": "glm-5.2", ...}]}
```

#### 方式二：vLLM-Ascend 推理

```bash
# 1. 安装 vLLM-Ascend
pip install vllm-ascend

# 2. 启动推理服务（1M 上下文，双卡并行）
python -m vllm.entrypoints.openai.api_server \
    --model /data/models/glm-5.2 \
    --served-model-name glm-5.2 \
    --device npu \
    --port 8004 \
    --max-model-len 1000000 \
    --gpu-memory-utilization 0.90 \
    --tensor-parallel-size 2 \
    --enable-prefix-caching \
    --kv-cache-dtype auto \
    --swap-space 16 \
    --max-num-seqs 2 \
    --trust-remote-code

# 3. 验证服务
curl http://localhost:8004/v1/chat/completions \
    -H "Content-Type: application/json" \
    -d '{
        "model": "glm-5.2",
        "messages": [{"role": "user", "content": "你好"}],
        "max_tokens": 100
    }'
```

### 6.4 端点配置

GLM-5.2 推理服务启动后，将在 `http://localhost:8004/v1` 提供 OpenAI 兼容的 API 端点：

| 端点 | 方法 | 说明 |
|------|------|------|
| `/v1/models` | GET | 列出可用模型 |
| `/v1/chat/completions` | POST | Chat 补全接口 |
| `/v1/completions` | POST | Text 补全接口 |
| `/v1/tokenize` | POST | Tokenizer 接口 |
| `/health` | GET | 健康检查 |

### 6.5 1M 上下文配置

GLM-5.2 的 1M 上下文需要特殊配置以确保稳定性：

```bash
# 1M 上下文的关键配置
--max-model-len 1000000       # 最大序列长度 1M
--enable-prefix-caching        # 前缀缓存，减少重复计算
--kv-cache-dtype auto          # 自动选择 KV cache 数据类型
--gpu-memory-utilization 0.90  # 高显存利用率，为 1M 预留空间
--swap-space 16                # CPU swap 空间 (GB)，防止 OOM
--max-num-seqs 2               # 减少并发序列，为 1M 上下文腾出显存
--tensor-parallel-size 2       # 双卡并行，分散 KV cache 到两张卡
```

**1M 上下文注意事项：**
- Ascend 910B 双卡可支持 1M 上下文，并发数需降低到 2
- 单卡仅支持 200K 上下文（降级模式），需配置上下文自动降级
- 建议开启 prefix caching 以减少重复前缀的计算开销
- 长上下文推理延迟较高，建议配合 TerAgent 的上下文压缩策略
- TerAgent 的 GLM52Compiler 会自动进行缓存前缀冻结和尾部强化
- KV cache 占用约 80 GB（1M 上下文），需双卡并行

### 6.6 双思考模式配置

GLM-5.2 的 High/Max 双思考模式需要在推理服务端和 TerAgent 两侧配置：

**推理服务端配置：**

```bash
# 启动时配置双思考模式支持
python -m vllm.entrypoints.openai.api_server \
    --model /data/models/glm-5.2 \
    --device npu \
    --port 8004 \
    --max-model-len 1000000 \
    --enable-prefix-caching \
    --max-num-seqs 2 \
    --trust-remote-code
```

**TerAgent 侧配置：**

```toml
[drivers.openai_compatible.glm_52]
base_url = "http://localhost:8004/v1"
api_key_env = "GLM_API_KEY"
model = "glm-5.2"
compiler = "glm_52"
max_context_tokens = 1_000_000
max_output_tokens = 128_000
thinking_mode = "high"                    # 默认 High 思考模式
dual_thinking_enabled = true              # 启用 High/Max 双思考切换
preserved_thinking_enabled = true         # 启用 PreservedThinking
vision_coordination_enabled = true        # 启用 5V-Turbo 视觉协调
context_degradation_enabled = true        # 启用上下文自动降级
long_horizon_enabled = true               # 启用长程任务模式
```

**双思考模式说明：**

| 模式 | 推理深度 | 响应时间 | Token 消耗 | 适用场景 |
|------|---------|---------|-----------|---------|
| **High** | 标准深度推理 | 中等 | ~1.2x | 代码生成、分析、规划 |
| **Max** | 最大推理深度 | 慢（2-5x） | ~3-5x | 架构决策、复杂调试 |

### 6.7 上下文自动降级配置

GLM-5.2 支持在 NPU 内存压力下自动从 1M 降级到 200K 上下文：

```toml
[drivers.openai_compatible.glm_52]
context_degradation_enabled = true
context_degradation_threshold = 0.90     # NPU 内存利用率 > 90% 触发降级
context_degradation_target = 200_000     # 降级到 200K 上下文
context_degradation_recovery_threshold = 0.70  # 内存 < 70% 时恢复到 1M
```

**降级行为：**
1. NPU 内存利用率超过 90% → 触发降级
2. 最大上下文从 1M 降至 200K
3. 现有上下文压缩到 200K 范围内
4. 保留系统提示、最近消息和工具定义
5. NPU 内存回落到 70% 以下 → 恢复 1M 模式
6. 所有降级事件记录日志

### 6.8 5V-Turbo 视觉协调配置

GLM-5.2 可与 GLM-5V-Turbo 协调实现视觉→代码→验证循环：

```toml
[drivers.openai_compatible.glm_5v_turbo]
base_url = "http://localhost:8004/v1"
api_key_env = "GLM_API_KEY"
model = "glm-5v-turbo"
compiler = "glm_5v_turbo"
max_context_tokens = 128_000
```

> **注意：** GLM-5V-Turbo 可与 GLM-5.2 共享同一推理服务端口（8004），通过模型名区分。

---

## 7. TerAgent 本地部署配置

### 7.1 配置模板

TerAgent 提供了专用的本地部署配置模板 `agent.local.toml`，可直接使用：

```bash
# 复制模板到项目根目录
cp examples/agent.local.toml agent.toml

# 或指定配置文件启动
export TERAGENT_CONFIG=/path/to/TerAgent/examples/agent.local.toml
```

完整配置模板见 `examples/agent.local.toml` 文件。

### 7.2 核心配置说明

#### 7.2.1 零成本配置

本地推理的核心优势是**零推理费用**。在 `examples/agent.local.toml` 中：

```toml
[circuit_breaker.budget]
cost_per_million_input = 0.0    # 本地推理无输入成本
cost_per_million_output = 0.0   # 本地推理无输出成本
enable_hard_limit = false        # 无需硬限制（零成本）
```

#### 7.2.2 多模型端点配置

```toml
# GLM-5 本地端点
[drivers.openai_compatible.glm_5]
base_url = "http://localhost:8001/v1"
model = "glm-5"

# GLM-5.2 本地端点（1M 上下文）
[drivers.openai_compatible.glm_52]
base_url = "http://localhost:8004/v1"
model = "glm-5.2"
max_context_tokens = 1_000_000
dual_thinking_enabled = true
preserved_thinking_enabled = true
vision_coordination_enabled = true
context_degradation_enabled = true

# DeepSeek V4 Flash 本地端点
# 注意：deepseek_v4_flash 和 deepseek_v4_pro 不是独立的编译器，
# 而是 DeepSeekV4Compiler 的驱动变体（通过 variant 参数控制）。
[drivers.openai_compatible.deepseek_v4_flash]
base_url = "http://localhost:8002/v1"
model = "deepseek-v4-flash"
compiler = "deepseek_v4"
compiler_variant = "flash"

# DeepSeek V4 Pro 本地端点（深度推理场景）
[drivers.openai_compatible.deepseek_v4_pro]
base_url = "http://localhost:8002/v1"
model = "deepseek-v4-pro"
compiler = "deepseek_v4"
compiler_variant = "pro"

# MiniMax M3 本地端点
[drivers.openai_compatible.minimax_m3]
base_url = "http://localhost:8003/v1"
model = "minimax-m3"
```

> **适配器选择提示：** 以上配置使用 `openai_compatible` 适配器。对于需要原生多模态/桌面操作支持的场景，M3 可使用 `[drivers.minimax_native.minimax_m3]` 配置（`minimax_native` 适配器）；对于需要思考模式（`reasoning_content`）和缓存追踪的场景，GLM-5/5.2 可使用 `[drivers.glm_native.glm_52]` 配置（`glm_native` 适配器）。

#### 7.2.3 本地推理熔断器配置

本地推理的熔断策略与云 API 不同 — 主要关注服务健康而非成本：

```toml
[circuit_breaker.failure_breaker]
max_consecutive = 3             # 本地环境降低失败容忍度（更快发现 NPU 故障）
window_seconds = 60             # 缩短观察窗口

[circuit_breaker.latency_breaker]
warn_latency_ms = 60000         # 本地推理延迟警告阈值（60s，含长上下文场景）
avg_window = 5

[circuit_breaker.progress_detector]
stall_threshold = 8             # 本地环境保持默认停滞检测
```

#### 7.2.4 上下文窗口配置

根据本地硬件能力配置上下文窗口：

```toml
[context_management]
model_token_limit = 200000      # 取所有模型中最小值（GLM-5 = 200K）
reserved_for_output = 16384     # 本地推理可增大输出预留
reserved_for_system = 4096
warn_threshold = 0.75
compact_threshold = 0.85
```

### 7.3 Pipeline 配置

本地部署的 Pipeline 策略与云部署略有不同：

```toml
[execution.pipeline]
# 默认配置：充分利用各模型优势
design_driver = "openai_compatible.deepseek_v4_pro"
plan_driver = "openai_compatible.deepseek_v4_pro"
execute_driver = "openai_compatible.deepseek_v4_flash"
review_driver = "openai_compatible.glm_5"
```

### 7.4 启动与验证

```bash
# 1. 启动所有推理服务
# 终端 1：GLM-5
python -m vllm.entrypoints.openai.api_server --model /data/models/glm-5 --device npu --port 8001 ...

# 终端 2：DeepSeek V4 Flash
python -m vllm.entrypoints.openai.api_server --model /data/models/deepseek-v4-flash --device npu --port 8002 ...

# 终端 3：MiniMax M3
python -m vllm.entrypoints.openai.api_server --model /data/models/minimax-m3 --device npu --port 8003 ...

# 终端 4：GLM-5.2（1M 上下文，双卡并行）
python -m vllm.entrypoints.openai.api_server --model /data/models/glm-5.2 --device npu --port 8004 --max-model-len 1000000 --tensor-parallel-size 2 ...

# 2. 等待所有服务就绪
curl http://localhost:8001/v1/models
curl http://localhost:8002/v1/models
curl http://localhost:8003/v1/models
curl http://localhost:8004/v1/models

# 3. 使用本地配置启动 TerAgent
cd /path/to/TerAgent
cp examples/agent.local.toml agent.toml  # 或设置 TERAGENT_CONFIG
python3 -m teragent

# 4. 验证 TerAgent 连接
python3 -c "
import teragent
config = teragent.load_typed_config()
for name, driver in config.drivers.items():
    print(f'{name}: {driver.base_url} ({driver.model})')
"
```

---

## 8. 验证清单

### 8.1 环境验证

| # | 验证步骤 | 命令 | 预期输出 |
|---|---------|------|---------|
| 1 | NPU 设备识别 | `npu-smi info` | 显示 Ascend 910B/950PR 信息 |
| 2 | CANN 驱动正常 | `python3 -c "import acl"` | 无报错 |
| 3 | PyTorch NPU 可用 | `python3 -c "import torch_npu; print(torch.npu.is_available())"` | `True` |
| 4 | NPU 数量正确 | `python3 -c "import torch_npu; print(torch.npu.device_count())"` | 预期 NPU 数量 |
| 5 | CANN 版本 | `cat /usr/local/Ascend/ascend-toolkit/latest/version.cfg` | 8.0.RCx |
| 6 | 磁盘空间 | `df -h /data/models` | 充足（模型需要数百 GB） |

### 8.2 推理服务验证

| # | 验证步骤 | 命令 | 预期输出 |
|---|---------|------|---------|
| 1 | GLM-5 服务可用 | `curl http://localhost:8001/v1/models` | 返回 glm-5 模型信息 |
| 2 | GLM-5 推理正常 | `curl http://localhost:8001/v1/chat/completions -d '...'` | 返回正常补全结果 |
| 3 | V4 Flash 服务可用 | `curl http://localhost:8002/v1/models` | 返回 deepseek-v4-flash 模型信息 |
| 4 | V4 Flash 推理正常 | `curl http://localhost:8002/v1/chat/completions -d '...'` | 返回正常补全结果 |
| 5 | M3 服务可用 | `curl http://localhost:8003/v1/models` | 返回 minimax-m3 模型信息 |
| 6 | M3 推理正常 | `curl http://localhost:8003/v1/chat/completions -d '...'` | 返回正常补全结果 |
| 7 | M3 多模态正常 | 发送 image_url 请求 | 返回图像分析结果 |
| 8 | GLM-5.2 服务可用 | `curl http://localhost:8004/v1/models` | 返回 glm-5.2 模型信息 |
| 9 | GLM-5.2 推理正常 | `curl http://localhost:8004/v1/chat/completions -d '...'` | 返回正常补全结果 |
| 10 | GLM-5.2 1M 上下文 | 发送长上下文请求（>200K tokens） | 正常处理，无 OOM |
| 11 | GLM-5.2 双思考 | 分别测试 High 和 Max 模式 | Max 模式响应更慢但更深入 |

### 8.3 TerAgent 集成验证

| # | 验证步骤 | 命令/操作 | 预期输出 |
|---|---------|---------|---------|
| 1 | 配置加载 | `teragent.load_typed_config()` | 所有 driver 已加载 |
| 2 | GLM-5 driver | 创建 provider 并发送请求 | 正常响应 |
| 3 | V4 Flash driver | 创建 provider 并发送请求 | 正常响应 |
| 4 | M3 driver | 创建 provider 并发送请求 | 正常响应 |
| 5 | GLM-5.2 driver | 创建 provider 并发送请求 | 正常响应 |
| 6 | Pipeline 执行 | 执行 design→plan→execute→review | 四阶段全部正常 |
| 7 | 路由正确性 | 多模态请求自动路由到 M3 | 路由到 minimax_m3 |
| 8 | 长程任务 | 启用 GLM-5 长程任务模式 | 检查点保存正常 |
| 9 | 熔断器 | 模拟连续失败 | 熔断器触发 |
| 10 | GLM-5.2 双思考 | 切换 High/Max 模式 | 模式切换正常 |
| 11 | GLM-5.2 上下文降级 | 增加上下文至触发降级 | 自动降级到 200K |
| 12 | 5V-Turbo 协调 | 发送视觉+代码请求 | 视觉协调正常工作 |

### 8.4 性能基准

**目标：** 本地推理性能应达到与云 API 相当的水平（考虑网络延迟优势，实际体验可能更好）。

| 指标 | 云 API 基准 | 本地部署目标 | 说明 |
|------|-----------|------------|------|
| 首 token 延迟 | 200-500ms | 100-300ms | 本地无网络开销 |
| 生成速度 | 30-60 tokens/s | 20-50 tokens/s | NPU 推理速度 |
| 1M 上下文加载 | 10-30s | 5-15s | 前缀缓存优化后 |
| 多模态推理 | 2-5s/图 | 1-3s/图 | 本地视觉编码 |
| 并发处理 | 受 API 限速 | 受 NPU 显存限制 | 调整 batch size |
| GLM-5.2 1M 首 token | 500-1000ms | 300-600ms | 1M 上下文更长 |
| GLM-5.2 Max 思考延迟 | 5-15s | 3-10s | 最大推理深度延迟 |

> **注意：** 以上性能指标为预估值，实际性能取决于具体的 NPU 型号、显存大小和推理框架优化程度。建议在部署后使用 `teragent.benchmark` 进行基准测试。

---

## 9. 性能调优

### 9.1 GLM-5 调优

```bash
# 1. KV Cache 优化
--kv-cache-dtype auto           # 自动选择最优数据类型
--enable-prefix-caching         # 前缀缓存（长程任务必需）
--max-num-seqs 4                # 减少并发，增加单序列 KV cache

# 2. 批处理优化
--max-batch-size 8              # Ascend 910B 推荐批大小
--swap-space 8                  # CPU swap 空间 (GB)

# 3. 长程任务专用
--max-model-len 200000          # 精确匹配模型上限
--gpu-memory-utilization 0.90   # 保留 10% 给系统开销
```

### 9.2 DeepSeek V4 调优

```bash
# 1. 1M 上下文优化
--max-model-len 1000000
--enable-prefix-caching         # 缓存前缀冻结（TerAgent V4Compiler 自动利用）
--swap-space 16                 # 大 swap 空间防止 OOM
--max-num-seqs 2                # 1M 上下文下减少并发

# 2. Flash/Pro 切换优化
# Flash 变体：减少推理参数，提高速度
--speculative-decoding          # 推测解码（Flash 模式可选）
# Pro 变体：增加推理深度
--num-spec-tokens 0             # Pro 模式关闭推测解码

# 3. 缓存感知配置
# TerAgent 的 DeepSeekV4Compiler 会自动：
# - 将系统提示和工具定义前置（冻结前缀）
# - 根据 cache_hit_rate 决定压缩策略
# - 生成预热请求初始化缓存
```

### 9.3 MiniMax M3 调优

```bash
# 1. 多模态性能优化
--limit-mm-per-prompt image=10,video=2  # 控制多模态内容数量
--mm-processor-kwargs '{"fps": 2}'      # 视频采样帧率

# 2. MSA 架构优化
--max-model-len 1000000                 # MSA 架构 1M 效率极高
--enable-prefix-caching                 # 配合 MSA 的稀疏注意力

# 3. 双卡并行
--tensor-parallel-size 2                # 多模态推荐双卡
--pipeline-parallel-size 1              # 通常不需要流水线并行
```

### 9.4 GLM-5.2 调优

```bash
# 1. 1M 上下文 KV Cache 优化
--max-model-len 1000000                 # 1M 上下文
--enable-prefix-caching                 # 前缀缓存（1M 场景必需）
--kv-cache-dtype auto                   # 自动选择 KV cache 数据类型
--tensor-parallel-size 2                # 双卡并行分散 KV cache
--swap-space 16                         # 16GB CPU swap 防止 OOM
--max-num-seqs 2                        # 减少并发，为 1M 腾出显存

# 2. 双思考模式优化
# High 模式（默认）：平衡速度和深度
# Max 模式：最大推理深度，建议配合 timeout=300 使用
# TerAgent 的 GLM52Compiler 会根据请求 meta 自动切换思考模式

# 3. 上下文降级配置
# 启用 context_degradation_enabled = true
# 当 NPU 内存 >90% 时自动降级到 200K
# 当 NPU 内存 <70% 时自动恢复到 1M

# 4. PreservedThinking 优化
# 限制保持的推理痕迹数量（默认 10 条）
# 定期清理旧的 PreservedThinking 痕迹
# 仅在编码任务中启用，分析任务可关闭以节省上下文

# 5. 5V-Turbo 视觉协调优化
# 确保 GLM-5V-Turbo 服务可用
# 配置视觉协调超时：multimodal_timeout=600.0
# 设置 5V-Turbo 熔断器：max_consecutive_failures=5
```

**GLM-5.2 调优参数汇总：**

| 参数 | 1M 上下文 | 200K 降级模式 | 说明 |
|------|----------|-------------|------|
| `max-model-len` | 1000000 | 200000 | 根据模式调整 |
| `tensor-parallel-size` | 2 | 1 | 1M 需双卡，200K 单卡 |
| `max-num-seqs` | 2 | 4 | 1M 减少并发 |
| `gpu-memory-utilization` | 0.90 | 0.90 | 保持一致 |
| `swap-space` | 16 | 8 | 1M 需更大 swap |
| `enable-prefix-caching` | ✅ | ✅ | 两种模式都建议开启 |

### 9.5 系统级调优

```bash
# 1. NPU 频率设置（性能模式）
npu-smi set -t freq -i 0 -f high

# 2. 内存锁定（减少页面换出）
ulimit -l unlimited

# 3. CPU 亲和性
export ASCEND_RT_VISIBLE_DEVICES="0,1"  # 指定使用的 NPU
export OMP_NUM_THREADS=8                # 限制 OpenMP 线程数

# 4. CANN 算子优化
export ASCEND_AICPU_PATH=/usr/local/Ascend/ascend-toolkit
export OP_PROTO_LIB=/usr/local/Ascend/ascend-toolkit/latest/op_impl/built-in/op_impl/ascend910b/bin
```

---

## 10. 常见问题与解决方案

### 10.1 CANN 驱动问题

#### Q: `npu-smi info` 无法识别设备

```bash
# 检查驱动是否安装
ls /usr/local/Ascend/
# 重新安装驱动
./Ascend-hdk-910b.npu-driver_23.0.x_linux-x86_64.run --install --force

# 检查内核模块
lsmod | grep drv
# 预期：显示 ascend 相关内核模块
```

#### Q: `import acl` 报错

```bash
# 确认环境变量
source /usr/local/Ascend/ascend-toolkit/set_env.sh
echo $LD_LIBRARY_PATH
# 应包含 Ascend 库路径

# 检查 Python 路径
python3 -c "import sys; print([p for p in sys.path if 'Ascend' in p])"
```

### 10.2 推理服务问题

#### Q: vLLM-Ascend 启动 OOM

```bash
# 减少 max-model-len
--max-model-len 65536  # 先用较小值验证

# 减少 batch size 和并发
--max-num-seqs 1
--gpu-memory-utilization 0.85

# 增加 swap 空间
--swap-space 32  # 32GB CPU swap
```

#### Q: 推理结果乱码或格式错误

```bash
# 检查 tokenizer 配置
ls /data/models/glm-5/tokenizer*
# 确保所有 tokenizer 文件完整

# 重新下载模型权重
modelscope download --model ZhipuAI/glm-5 --local_dir /data/models/glm-5 --force
```

#### Q: 多模态请求报错

```bash
# 检查图片格式和大小
python3 -c "
from PIL import Image
img = Image.open('/path/to/image.png')
print(f'Size: {img.size}, Format: {img.format}')
"

# 确认 vLLM 多模态参数
--limit-mm-per-prompt image=5,video=1  # 降低多模态内容限制
```

### 10.3 TerAgent 集成问题

#### Q: TerAgent 无法连接本地推理服务

```bash
# 检查推理服务是否运行
curl http://localhost:8001/v1/models

# 检查 TerAgent 配置
python3 -c "
import teragent
config = teragent.load_typed_config()
for name, driver in config.drivers.items():
    print(f'{name}: {driver.base_url}')
"

# 检查防火墙
sudo iptables -L -n | grep 8001
```

#### Q: 路由选择错误模型

```python
# TerAgent 的 ModelRouter 会根据以下维度路由：
# 1. 多模态内容 → M3 (desktop_driver / multimodal_driver)
# 2. 长程任务 → GLM-5 (long_horizon_driver)
# 3. 上下文 >200K → V4/M3（排除 GLM-5）
# 检查路由配置
config = teragent.load_typed_config()
```

#### Q: 长程任务检查点丢失

```bash
# 确认 checkpoint 目录权限
ls -la .agent/checkpoints/

# 检查磁盘空间
df -h .agent/

# 确认 GLM-5 推理服务稳定
curl http://localhost:8001/health
```

### 10.4 性能问题

#### Q: 推理速度比云 API 慢

```bash
# 1. 检查 NPU 利用率
npu-smi info -t usages -i 0
# 如果 HBM 利用率 < 50%，可能是 batch 太小

# 2. 增大 batch size
--max-batch-size 16

# 3. 开启 continuous batching
--enable-chunked-prefill       # 分块预填充
--max-num-batched-tokens 4096  # 批量 token 数

# 4. 检查是否为量化模型
# 量化模型推理更快但精度稍低
```

#### Q: 1M 上下文加载缓慢

```bash
# 开启前缀缓存
--enable-prefix-caching

# 减少不必要的上下文
# TerAgent 的上下文压缩会自动处理

# 检查磁盘 IO
iostat -x 1 10
# 模型权重读取速度影响首次加载
```

---

## 11. 附录

### 11.1 参考文档

| 文档 | 链接 |
|------|------|
| 华为昇腾 CANN 开发文档 | https://www.hiascend.com/document |
| CANN 8.0 安装指南 | https://www.hiascend.com/document/detail/zh/CANNCommunityEdition/80RC1 |
| vLLM-Ascend 项目 | https://gitee.com/ascend/vllm |
| MindIE 推理引擎 | https://www.hiascend.com/software/mindie |
| openEuler 操作系统 | https://www.openeuler.org/ |
| GLM-5 模型仓库 | https://modelscope.cn/models/ZhipuAI/glm-5 |
| GLM-5.2 模型仓库 | https://modelscope.cn/models/ZhipuAI/glm-5.2 |
| DeepSeek V4 模型仓库 | https://modelscope.cn/models/deepseek-ai/deepseek-v4 |
| MiniMax M3 模型仓库 | https://modelscope.cn/models/MiniMaxAI/minimax-m3 |

### 11.2 模型规格汇总

| 参数 | GLM-5 | GLM-5.2 | DeepSeek V4 Flash | DeepSeek V4 Pro | MiniMax M3 |
|------|---------|---------|-------------------|-----------------|------------|
| 上下文窗口 | 200K | 1M | 1M | 1M | 1M |
| 最大输出 | 128K | 128K | 384K | 384K | 384K |
| 多模态 | ❌ | ❌（通过 5V） | ❌ | ❌ | ✅ |
| 桌面操作 | ❌ | ❌ | ❌ | ❌ | ✅ |
| 长程任务 | ✅ (8h) | ✅ (8h+) | ❌ | ❌ | ❌ |
| 思考模式 | deep | High/Max | auto/quick | deep | - |
| 缓存感知 | ❌ | ✅ | ✅ | ✅ | ✅ (MSA) |
| PreservedThinking | ❌ | ✅ | ❌ | ❌ | ❌ |
| 5V-Turbo 协调 | ❌ | ✅ | ❌ | ❌ | ❌ |
| 上下文降级 | ❌ | ✅ (1M→200K) | ❌ | ❌ | ❌ |
| 推荐芯片 | Ascend 910B | Ascend 910B ×2 | Ascend 950PR | Ascend 950PR | Ascend 910B |
| 推荐端口 | 8001 | 8004 | 8002 | 8005 | 8003 |

### 11.3 端口分配

| 端口 | 服务 | 说明 |
|------|------|------|
| 8001 | GLM-5 推理 | OpenAI 兼容 API |
| 8002 | DeepSeek V4 Flash 推理 | OpenAI 兼容 API |
| 8003 | MiniMax M3 推理 | OpenAI 兼容 API |
| 8004 | GLM-5.2 推理 | OpenAI 兼容 API（1M 上下文） |
| 8005 | DeepSeek V4 Pro 推理 | OpenAI 兼容 API（可选） |
| 8010 | 桌面操作 API | M3 桌面操作辅助服务 |

### 11.4 硬件验证限制声明

> **重要说明：** 本部署指南基于华为昇腾官方文档和模型开源社区信息编写，旨在为 DevOps 工程师提供完整的部署参考。由于当前环境无实际 Ascend 910B/950PR 硬件，以下内容**未经实际硬件验证**：
>
> - 推理服务的具体启动参数可能需要根据实际硬件和 CANN 版本微调
> - 性能基准数据为预估值，实际性能需要基准测试确认
> - 模型权重下载路径可能随社区更新而变化
> - 部分命令可能需要根据操作系统版本调整
>
> 建议在部署前在测试环境进行完整验证，并参考华为昇腾社区获取最新的兼容性信息。

---

*本文档由 TerAgent 项目维护，最后更新：2025年*
