## 基于条件流匹配的连续遗忘学习项目 (`CFMU-Project`)

### 项目概述

本项目聚焦于**条件流匹配**（Conditional Flow Matching, CFM）生成模型中的**机器遗忘**（Machine Unlearning）问题，目标是在**不从头重训**的前提下，让模型忘记指定类别或指定区域的生成能力。

项目尤其关注**连续遗忘**（Continual / Sequential Unlearning）场景：模型需要按顺序、分阶段忘记多个目标，同时避免后续任务破坏先前遗忘结果，也就是避免已遗忘概念“死灰复燃”。

当前仓库覆盖了：

- `MNIST` 单次遗忘与连续遗忘
- 多个二维合成数据集（如 `Checkerboard`、`Circles`、`Moons`）的区域遗忘
- `CIFAR-10` 上的 CFMU 时间维度连续遗忘实验

### 核心思路与技术

1. **条件流匹配（CFM）**
   以 CFM 作为基础生成模型，通过学习从噪声分布到数据分布的连续向量场完成生成。

2. **能量加权遗忘（Energy-based Reweighted Flow Matching, ERFM）**
   在二维数据集和高维图像连续遗忘中，利用能量函数调节遗忘目标的梯度权重，降低模型继续拟合目标区域/目标类别的能力。

3. **连续遗忘中的退化问题**
   若直接采用朴素连续遗忘，后续阶段的梯度更新会干扰先前已经遗忘的概念，导致历史遗忘目标重新出现。

4. **时间维度护盾：正交约束与动态能量权重**
   在连续遗忘过程中，引入：
   - **正交约束（Orthogonality Loss）**：抑制新阶段更新与历史遗忘方向的有害对齐。
   - **动态能量权重（Energy-guided Weighting）**：依据冻结特征空间中的 retain / forget 结构对样本赋予不同的遗忘敏感度。

### 主要实验文件

- **MNIST 单类别遗忘**：`mnist.ipynb`
- **MNIST 连续多类别遗忘**：`mnist Consecutive forgetting.ipynb`
- **2D 数据集区域遗忘**：`checkboard.ipynb`、`circles.ipynb`、`moons.ipynb`
- **2D 数据集连续区域遗忘**：`checkboard Consecutive forgetting.ipynb`、`checkboard Consecutive forgetting 2.ipynb`
- **CIFAR-10 连续遗忘主脚本**：`CIFAR Consecutive forgetting.py`

其中，`CIFAR Consecutive forgetting.py` 是目前最完整的整合版实验入口，包含：

- CIFAR-10 分类器训练/加载
- CFM 预训练模型继续精炼
- `Naive` 与 `CFMU (Temporal)` 两条连续遗忘实验线
- 固定单样本可视化素材导出
- 论文图与量化表输出

---

## 环境配置

### 已验证环境

本项目当前经过验证的环境如下：

- OS: Linux
- Python: `3.10.20`
- CUDA runtime（PyTorch wheel）: `12.8`
- NVIDIA Driver: `575.57.08`
- GPU: NVIDIA L20

核心依赖版本：

- `torch==2.11.0+cu128`
- `torchvision==0.26.0+cu128`
- `torchcfm==1.0.7`
- `torchdiffeq==0.2.5`
- `torchdyn==1.0.6`
- `torchcde==0.2.5`
- `torchsde==0.2.6`
- `numpy==2.2.6`
- `scipy==1.15.3`
- `scikit-learn==1.7.2`
- `matplotlib==3.10.1`
- `pandas==2.3.3`
- `pillow==12.1.1`
- `jupyterlab==4.3.5`
- `ipykernel==6.29.5`

### 推荐安装方式

建议使用独立的 `conda` 环境，并优先安装与当前项目一致的 CUDA 12.8 版本 PyTorch。

```bash
conda create -n cfmu python=3.10.20 -y
conda activate cfmu

pip install --upgrade pip

pip install torch==2.11.0+cu128 torchvision==0.26.0+cu128 \
  --index-url https://download.pytorch.org/whl/cu128

pip install \
  torchcfm==1.0.7 \
  torchdiffeq==0.2.5 \
  torchdyn==1.0.6 \
  torchcde==0.2.5 \
  torchsde==0.2.6 \
  numpy==2.2.6 \
  scipy==1.15.3 \
  scikit-learn==1.7.2 \
  matplotlib==3.10.1 \
  pandas==2.3.3 \
  pillow==12.1.1 \
  tqdm==4.67.1 \
  pyyaml==6.0.3 \
  jupyterlab==4.3.5 \
  ipykernel==6.29.5
```

如果你只想运行脚本而不使用 notebook，`jupyterlab` 和 `ipykernel` 可以不装。

### 验证安装是否成功

```bash
python - <<'PY'
import torch, torchvision, torchcfm, torchdiffeq
print("torch:", torch.__version__)
print("torchvision:", torchvision.__version__)
print("cuda available:", torch.cuda.is_available())
print("torch cuda:", torch.version.cuda)
print("torchcfm:", torchcfm.__version__)
print("torchdiffeq:", torchdiffeq.__version__)
PY
```

若 `torch.cuda.is_available()` 返回 `False`，请先检查：

```bash
nvidia-smi
```

只要驱动支持 CUDA 12.8+，上述 PyTorch 安装通常即可正常运行。

---

## 快速开始

### 1. 运行 Notebook 实验

本项目最早的大部分实验由 `Jupyter Notebook` 组成。启动方式：

```bash
conda activate cfmu
jupyter lab
```

然后按顺序执行 notebook 中的代码单元即可。

常见使用方式：

- 每个数据集通常先有训练段或预训练 notebook
- 对应的 `Consecutive forgetting.ipynb` 依赖预训练权重，再执行遗忘实验

### 2. 运行 CIFAR-10 连续遗忘主脚本

进入项目目录后直接运行：

```bash
cd /path/to/CFMU-Project
conda activate cfmu
CUDA_VISIBLE_DEVICES=0 PYTHONUNBUFFERED=1 python "CIFAR Consecutive forgetting.py"
```

如需保存日志：

```bash
CUDA_VISIBLE_DEVICES=0 PYTHONUNBUFFERED=1 \
python "CIFAR Consecutive forgetting.py" \
2>&1 | tee "outputs/cifar10_temporal/run.log"
```

### 3. CIFAR-10 脚本默认行为

`CIFAR Consecutive forgetting.py` 默认会：

1. 检查 GPU / 路径 / 数据 /关键超参数
2. 加载或训练 CIFAR-10 分类器
3. 加载已有 `pretrain_model_path`，若存在则继续精炼
4. 精炼过程中保存最优预训练权重
5. 基于同一个精炼后的最优预训练权重，分别运行：
   - `Naive Sequential`
   - `CFMU Temporal`
6. 导出阶段权重、JSON 摘要、固定单样本图、论文图、量化表

---

## 输出目录说明

以 `CIFAR Consecutive forgetting.py` 为例，输出默认写入：

```text
outputs/cifar10_temporal/
```

常见子目录：

- `pretrain/`
  - `figures/`：预训练阶段可视化图
  - `summaries/`：预训练日志与摘要
- `naive/`
  - `checkpoints/`：Naive 各阶段权重
  - `figures/single_samples_by_stage/`：四阶段 x 三类别单样本导出
  - `summaries/`：Naive 历史 JSON
- `temporal/`
  - `checkpoints/`：CFMU 各阶段权重
  - `figures/single_samples_by_stage/`：四阶段 x 三类别单样本导出
  - `summaries/`：Temporal 历史 JSON
- `paper_figures/`
  - 论文示意图、反弹抑制分析图等
- `table3_quantitative/`
  - JSON / CSV / Markdown / PNG 量化表

---

## 复现实验时的建议

1. **优先保证 CUDA 可用**
   CIFAR-10 脚本默认按 GPU 流程设计，建议在有 NVIDIA GPU 的 Linux 服务器上运行。

2. **不要中断预训练精炼**
   若预训练精炼阶段被中断，后续 Naive / Temporal 不会基于完整的精炼最优权重继续执行。

3. **确认预训练权重路径**
   当前脚本优先读取：
   - `./cfm_cifar_pretrain.pt`
   若不存在，会自动回退到：
   - `./outputs/cifar10_temporal/pretrain/cfm_cifar_pretrain.pt`

4. **单样本图已固定 seed**
   当前 CIFAR-10 连续遗忘实验使用固定 seed bank，便于跨阶段、跨方法严格可比。

---

## 说明

- 当前仓库中 notebook 与 Python 脚本并存；早期实验主要在 notebook 中，CIFAR-10 连续遗忘已整理为可直接执行的 Python 脚本。
- 若你希望完全复刻作者当前实验环境，可以参考当前 `conda list` 中的完整包版本；但通常只安装上文列出的核心依赖即可顺利运行项目。
