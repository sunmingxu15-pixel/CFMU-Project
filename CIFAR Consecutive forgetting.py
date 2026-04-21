#!/usr/bin/env python
# coding: utf-8

# In[1]:


# -*- coding: utf-8 -*-
# CIFAR-10 Consecutive Forgetting for CFMU
# Notebook 目标：复现论文 4.3（Temporal Shield）中的 CIFAR-10 连续遗忘。
# 任务顺序：Automobile -> Cat -> Deer。
#
# 单元结构说明（可复盘）：
# 0  封面与导航
# 1  配置与依赖
# 2  通用工具
# 3  CFM 与分类器模型构建
# 4  分类器训练/加载
# 5  特征质心与能量函数
# 6  OT 与主损失
# 7  采样、评估、论文图像输出
# 8  单方法连续遗忘主流程（naive/temporal）
# 9  顶层 orchestrator（整实验入口）
# 10 一键运行
#
# 输出内容：checkpoint、JSON、按阶段单样本目录图、反弹抑制图。


# In[2]:


# 1. 全局配置与依赖
# --------------------------------------------------
# 这是 notebook 的第一段：固定任务定义、默认超参数、路径与开关。
# --------------------------------------------------

import copy
import csv
import json
import random
import re
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import numpy as np
import torch
import torch.nn as nn
import torchdiffeq
import matplotlib.pyplot as plt
from PIL import Image
from torch.optim.lr_scheduler import CosineAnnealingLR
from torch.utils.data import DataLoader
from torchvision import datasets, models, transforms
from torchvision.transforms import ToPILImage
from torchvision.utils import make_grid

from torchcfm.conditional_flow_matching import ConditionalFlowMatcher
from torchcfm.models.unet import UNetModel


# CIFAR10 类别顺序：自动化输出时用中文可读名
CLASS_NAMES = [
    "airplane", "automobile", "bird", "cat", "deer",
    "dog", "frog", "horse", "ship", "truck",
]
FORGET_SEQUENCE = [1, 3, 4]  # automobile -> cat -> deer
FORGET_SEQUENCE_NAMES = [CLASS_NAMES[idx] for idx in FORGET_SEQUENCE]

# 统一配置；如有需要可在运行前微调
CONFIG = {
    "seed": 0,
    "data_root": "./data",
    "output_root": "./outputs/cifar10_temporal",
    "pretrain_model_path": "./cfm_cifar_pretrain.pt",
    "classifier_path": "./outputs/cifar10_temporal/cifar10_resnet18_classifier.pt",
    "centroid_cache_path": "./outputs/cifar10_temporal/cifar10_feature_centroids.pt",
    "batch_size": 128,
    "batch_size_gpu": 96,
    "batch_size_cpu": 64,
    "pretrain_batch_size_gpu": 96,
    "pretrain_batch_size_cpu": 96,
    "num_workers": 2,
    "num_workers_gpu": 6,
    "num_workers_cpu": 2,
    "pin_memory": True,
    "device_preference": "cuda",
    "allow_cpu_fallback": False,
    "classifier_epochs": 15,
    "classifier_lr": 1e-3,
    "epochs_per_stage": 30,
    "lr": 5e-5,
    "weight_decay": 1e-5,
    "beta": 1.0,
    "distill_gamma": 0.0,
    "sigma": 0.0,
    "lambda_max": 8.0,
    "weight_floor": 1e-6,
    "grad_clip_norm": 1.0,
    "n_eval_samples": 4096,
    "sample_batch_size": 256,
    "forget_vis_samples_per_class": 64,
    "forget_vis_nrow": 8,
    "forget_vis_upscale": 3,
    "figure_strip_upscale": 2,
    "paper_fig_dpi": 600,
    "field_vis_grid_size": 17,
    "field_vis_class_samples": 72,
    "field_vis_pairs": 72,
    "field_vis_batch_size": 256,
    "field_vis_margin_ratio": 0.18,
    "field_vis_time_points": (0.0, 1.0 / 3.0, 2.0 / 3.0, 1.0),
    "field_vis_quiver_scale": 10.5,
    "single_vis_seed_offset": 7000,
    "single_vis_upscale": 7,
    "single_vis_gap": 28,
    "single_vis_panel_padding": 12,
    "schematic_grid_size": 25,
    "schematic_quiver_stride": 2,
    "schematic_repulsion_scale": 2.8,
    "schematic_eps": 0.06,
    "schematic_save_dpi": 900,
    "schematic_extent": (-3.0, 3.0, -2.6, 2.8),
    "schematic_centers": {
        "retain": (0.0, -1.0),
        "forget_a": (-1.5, 1.0),
        "forget_b": (0.0, 1.5),
        "forget_c": (1.5, 1.0),
    },
    "ode_atol": 1e-5,
    "ode_rtol": 1e-5,
    "ode_method": "dopri5",
    "vis_ode_method": "rk4",
    "vis_ode_atol": 1e-4,
    "vis_ode_rtol": 1e-4,
    "sample_time_steps_eval": 49,
    "sample_time_steps_vis": 17,
    "num_classes": 10,
    "cfm_use_null_class": True,
    "data_shape": (3, 32, 32),
    "model_num_channels": 192,
    "model_num_res_blocks": 2,
    "model_channel_mult": (1, 2, 2, 2),
    "model_attention_resolutions": "16",
    "model_num_heads": 4,
    "model_dropout": 0.1,
    "train_random_flip_p": 0.5,
    "classifier_use_augmentation": True,
    "align_pretrain_and_unlearning_augmentation": True,
    "run_naive": True,
    "run_temporal": True,
    "run_pretrain_first": True,
    "force_retrain_pretrain": True,
    "pretrain_epochs": 250,
    "pretrain_lr": 5e-4,
    "pretrain_weight_decay": 0.0,
    "pretrain_warmup_ratio": 0.12,
    "pretrain_ema_decay": 0.9995,
    "pretrain_use_amp": True,
    "pretrain_eval_use_ema": True,
    "pretrain_eval_batches": 200,
    "pretrain_grad_accum_steps": 2,
    "pretrain_checkpoint_every": 5,
    "pretrain_label_dropout": 0.1,
    "pretrain_eval_samples_per_class": 32,
    "pretrain_vis_samples_per_class": 10,
    "pretrain_visualize_every": 5,
    "pretrain_target_opt_steps_gpu": 45000,
    "pretrain_target_opt_steps_cpu": 2500,
    "pretrain_max_epochs_gpu": 250,
    "pretrain_max_epochs_cpu": 24,
    "pretrain_early_stop_enabled": False,
    "pretrain_early_stop_patience": 10,
    "pretrain_early_stop_min_epochs": 250,
    "pretrain_early_stop_min_improvement": 1e-4,
    "single_seed_bank_path": "./outputs/cifar10_temporal/temporal/summaries/single_seed_bank.json",
    "single_seed_candidates": 32,
    "single_seed_refresh": True,
    "single_seed_sharpness_weight": 0.035,
    "single_vis_generation_strategy": "eval",
    "single_vis_resample": "lanczos",
    "sequential_use_ema": False,
    "sequential_ema_decay": 0.999,
}


# In[3]:


# 2. 通用辅助函数
# --------------------------------------------------
# 统一管理随机种子、模型冻结、以及 DataParallel 前缀清理。
# --------------------------------------------------


def set_seed(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def freeze_module(module: nn.Module) -> nn.Module:
    """将模型冻结为 eval + no-grad（冻结参数）状态。"""
    module.eval()
    for param in module.parameters():
        param.requires_grad = False
    return module


def strip_module_prefix(state_dict: Dict[str, torch.Tensor]) -> Dict[str, torch.Tensor]:
    """兼容 DataParallel 的 module. 前缀。"""
    if not state_dict:
        return state_dict
    if all(key.startswith("module.") for key in state_dict.keys()):
        return {key[len("module."):]: value for key, value in state_dict.items()}
    return state_dict


def update_ema_model(student_model: nn.Module, ema_target: nn.Module, decay: float) -> None:
    """通用 EMA 更新：同时同步参数与 buffer。"""
    with torch.no_grad():
        for p_ema, p in zip(ema_target.parameters(), student_model.parameters()):
            p_ema.mul_(decay).add_(p.data, alpha=1.0 - decay)
        for b_ema, b in zip(ema_target.buffers(), student_model.buffers()):
            b_ema.copy_(b)


def get_total_condition_classes() -> int:
    """返回条件嵌入可用的标签总数，允许额外的空条件 token。"""
    return int(CONFIG["num_classes"]) + (1 if CONFIG.get("cfm_use_null_class", False) else 0)


def get_null_class_idx() -> Optional[int]:
    """当启用空条件训练时，最后一个标签槽作为 null 条件。"""
    if not CONFIG.get("cfm_use_null_class", False):
        return None
    return int(CONFIG["num_classes"])


def apply_label_condition_dropout(labels: torch.Tensor, drop_prob: float) -> torch.Tensor:
    """把一部分标签替换为 null token，提升条件生成鲁棒性。"""
    null_idx = get_null_class_idx()
    if drop_prob <= 0.0 or null_idx is None:
        return labels
    mask = torch.rand(labels.shape, device=labels.device) < drop_prob
    if not mask.any():
        return labels
    dropped = labels.clone()
    dropped[mask] = null_idx
    return dropped


# In[4]:


# 3. CFM 侧模型定义与加载工具
# --------------------------------------------------
# 包含两部分：
# - CIFAR10ResNet18：用于构建冻结特征网络 Φ
# - UNetModel：CFM 主模型（student/teacher）实例化与权重加载
# --------------------------------------------------


class CIFAR10ResNet18(nn.Module):
    # CIFAR10 输入规格的 ResNet18。
    # forward_features 返回全局平均池化后的特征向量，服务于特征空间 Φ。
    def __init__(self, num_classes: int = 10):
        super().__init__()
        backbone = models.resnet18(weights=None)
        backbone.conv1 = nn.Conv2d(3, 64, kernel_size=3, stride=1, padding=1, bias=False)
        backbone.maxpool = nn.Identity()
        in_features = backbone.fc.in_features
        backbone.fc = nn.Linear(in_features, num_classes)
        self.backbone = backbone

    def forward_features(self, x: torch.Tensor) -> torch.Tensor:
        x = self.backbone.conv1(x)
        x = self.backbone.bn1(x)
        x = self.backbone.relu(x)
        x = self.backbone.maxpool(x)
        x = self.backbone.layer1(x)
        x = self.backbone.layer2(x)
        x = self.backbone.layer3(x)
        x = self.backbone.layer4(x)
        x = self.backbone.avgpool(x)
        x = torch.flatten(x, 1)
        return x

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        feat = self.forward_features(x)
        return self.backbone.fc(feat)


def instantiate_cfm_model(device: torch.device) -> nn.Module:
    # 按 CONFIG 构建 CFM UNet。
    model = UNetModel(
        dim=CONFIG["data_shape"],
        num_channels=CONFIG["model_num_channels"],
        num_res_blocks=CONFIG["model_num_res_blocks"],
        channel_mult=CONFIG.get("model_channel_mult", None),
        attention_resolutions=CONFIG.get("model_attention_resolutions", "16"),
        num_heads=CONFIG.get("model_num_heads", 1),
        dropout=CONFIG.get("model_dropout", 0.0),
        num_classes=get_total_condition_classes(),
        class_cond=True,
    ).to(device)
    return model


def load_cfm_model(checkpoint_path: str, device: torch.device) -> nn.Module:
    # 兼容 checkpoint: 直接 model / dict/model_state_dict / state_dict。
    checkpoint = torch.load(checkpoint_path, map_location=device)

    if isinstance(checkpoint, nn.Module):
        return checkpoint.to(device)

    model = instantiate_cfm_model(device)
    if isinstance(checkpoint, dict) and "model_state_dict" in checkpoint:
        state_dict = checkpoint["model_state_dict"]
    elif isinstance(checkpoint, dict) and "state_dict" in checkpoint:
        state_dict = checkpoint["state_dict"]
    else:
        state_dict = checkpoint

    state_dict = strip_module_prefix(state_dict)
    model.load_state_dict(state_dict)
    return model


# In[5]:


# 4. 分类器（冻结特征网络）构建与加载
# --------------------------------------------------
# 这是 Temporal Shield 中“感知特征空间 Φ”的核心来源。
# 我们使用 CIFAR10 版 ResNet-18：
# - 首层 3x3 stride1，去除 7x7+maxpool 的 ImageNet 偏置
# - 训练/加载一次后缓存，实验期间保持冻结
# --------------------------------------------------


def build_or_load_classifier(
    path: str,
    device: torch.device,
    train_loader: DataLoader,
    test_loader: DataLoader,
    epochs: int,
    lr: float,
) -> nn.Module:
    """有则加载无则训练 ResNet18 分类器并保存。"""
    classifier = CIFAR10ResNet18(num_classes=CONFIG["num_classes"]).to(device)
    path_obj = Path(path)
    path_obj.parent.mkdir(parents=True, exist_ok=True)

    if path_obj.exists():
        state_dict = strip_module_prefix(torch.load(path_obj, map_location=device))
        classifier.load_state_dict(state_dict)
        classifier.eval()
        print(f"[Classifier] 已加载分类器: {path_obj}")
        return classifier

    print("[Classifier] 未找到 CIFAR-10 ResNet-18 分类器，开始训练...")
    criterion = nn.CrossEntropyLoss()
    optimizer = torch.optim.Adam(classifier.parameters(), lr=lr, weight_decay=5e-4)
    scheduler = CosineAnnealingLR(optimizer, T_max=epochs * len(train_loader))

    best_acc = 0.0
    best_state = None
    for epoch in range(epochs):
        classifier.train()
        running_loss = 0.0
        running_total = 0
        running_correct = 0

        for x, y in train_loader:
            x, y = x.to(device), y.to(device)
            optimizer.zero_grad(set_to_none=True)
            logits = classifier(x)
            loss = criterion(logits, y)
            loss.backward()
            optimizer.step()
            scheduler.step()

            running_loss += loss.item() * x.size(0)
            running_total += x.size(0)
            running_correct += (logits.argmax(dim=1) == y).sum().item()

        classifier.eval()
        val_total = 0
        val_correct = 0
        with torch.no_grad():
            for x, y in test_loader:
                x, y = x.to(device), y.to(device)
                pred = classifier(x).argmax(dim=1)
                val_total += y.numel()
                val_correct += (pred == y).sum().item()

        train_loss = running_loss / max(running_total, 1)
        train_acc = 100.0 * running_correct / max(running_total, 1)
        val_acc = 100.0 * val_correct / max(val_total, 1)
        print(
            f"[Classifier] epoch {epoch + 1:02d}/{epochs:02d} | "
            f"train loss {train_loss:.4f} | train acc {train_acc:.2f}% | val acc {val_acc:.2f}%"
        )

        if val_acc >= best_acc:
            best_acc = val_acc
            best_state = copy.deepcopy(classifier.state_dict())

    if best_state is None:
        best_state = classifier.state_dict()
    torch.save(best_state, path_obj)
    classifier.load_state_dict(best_state)
    classifier.eval()
    print(f"[Classifier] 训练完成，最佳验证准确率 {best_acc:.2f}% ，已保存到: {path_obj}")
    return classifier


# In[6]:


# 5. 能量函数相关（Temporal 的“动态敏感度”来源）
# --------------------------------------------------
# 1) 先在训练集上提取每类冻结特征中心（centroid）
# 2) 用 retain/forget 质心构造 E_attr 与 E_rep
# 3) F(x1)=E_attr - E_rep，用于 sigmoid 动态加权
# --------------------------------------------------


@torch.no_grad()
def compute_class_feature_centroids(
    feature_net: nn.Module,
    data_loader: DataLoader,
    device: torch.device,
    cache_path: str,
) -> torch.Tensor:
    """预计算 CIFAR10 类特征中心，后续复用加速每阶段训练。"""
    cache_obj = Path(cache_path)
    if cache_obj.exists():
        centroids = torch.load(cache_obj, map_location=device)
        print(f"[Centroids] 已加载特征中心: {cache_obj}")
        return centroids.to(device)

    sums = None
    counts = torch.zeros(CONFIG["num_classes"], dtype=torch.long, device=device)
    feature_net.eval()

    for x, y in data_loader:
        x, y = x.to(device), y.to(device)
        feats = feature_net.forward_features(x)
        if sums is None:
            sums = torch.zeros(CONFIG["num_classes"], feats.size(1), device=device)
        for cls_idx in range(CONFIG["num_classes"]):
            mask = (y == cls_idx)
            if mask.any():
                sums[cls_idx] += feats[mask].sum(dim=0)
                counts[cls_idx] += mask.sum()

    counts = counts.clamp(min=1).unsqueeze(1)
    centroids = sums / counts

    cache_obj.parent.mkdir(parents=True, exist_ok=True)
    torch.save(centroids.cpu(), cache_obj)
    print(f"[Centroids] 已计算并保存特征中心: {cache_obj}")
    return centroids


@torch.no_grad()
def build_stage_prototypes(centroids: torch.Tensor, forgotten_classes: List[int]) -> Dict[str, torch.Tensor]:
    """根据当前阶段已遗忘类，返回 retain 与 forget 原型子集。"""
    forget_ids = torch.tensor(forgotten_classes, device=centroids.device, dtype=torch.long)
    retain_ids = torch.tensor(
        [idx for idx in range(CONFIG["num_classes"]) if idx not in forgotten_classes],
        device=centroids.device,
        dtype=torch.long,
    )
    return {
        "forget": centroids.index_select(0, forget_ids),
        "retain": centroids.index_select(0, retain_ids),
    }


@torch.no_grad()
def compute_energy(
    x: torch.Tensor,
    feature_net: nn.Module,
    retain_proto: torch.Tensor,
    forget_proto: torch.Tensor,
) -> torch.Tensor:
    """F(x1)=E_attr - E_rep：越偏向 retain 的样本值越大。"""
    feat = feature_net.forward_features(x)
    e_attr = torch.cdist(feat, retain_proto, p=2).pow(2).mean(dim=1)
    e_rep = torch.cdist(feat, forget_proto, p=2).pow(2).mean(dim=1)
    return e_attr - e_rep


# In[7]:


# 6. 损失函数与动态权重
# --------------------------------------------------
# Temporal Shield 的核心数学模块在这里：
# - weighted_cfm_loss：把 per-sample 的 MSE 按 sigmoid 权重聚合
# - compute_orthogonality_loss：历史梯度正交约束
# --------------------------------------------------


def lambda_schedule(step_idx: int, total_steps: int, lambda_max: float) -> float:
    """线性从 0 到 lambda_max 的阶段内调度。"""
    progress = float(step_idx + 1) / float(max(total_steps, 1))
    return lambda_max * progress


def per_sample_mse(vt: torch.Tensor, ut: torch.Tensor) -> torch.Tensor:
    """逐样本 MSE，用于构造能受权重调节的主损失。"""
    return (vt - ut).reshape(vt.size(0), -1).pow(2).mean(dim=1)


def weighted_cfm_loss(vt: torch.Tensor, ut: torch.Tensor, weights: torch.Tensor) -> torch.Tensor:
    """带权重的 CFM 回归损失：sum(w_i * loss_i) / sum(w_i)。"""
    losses = per_sample_mse(vt, ut)
    return (weights * losses).sum() / weights.sum().clamp(min=CONFIG["weight_floor"])


def compute_orthogonality_loss(
    student: nn.Module,
    student_prev: nn.Module,
    t_hist: torch.Tensor,
    xt_hist: torch.Tensor,
    y_hist: torch.Tensor,
) -> torch.Tensor:
    """L_orthogonality = E[ReLU(<Delta v, v_prev>)]。"""
    vt_current = student(t_hist, xt_hist, y_hist)
    with torch.no_grad():
        vt_prev = student_prev(t_hist, xt_hist, y_hist)

    delta = vt_current - vt_prev
    inner = (delta.reshape(delta.size(0), -1) * vt_prev.reshape(vt_prev.size(0), -1)).sum(dim=1)
    return torch.relu(inner).mean()


def compute_retain_distillation_loss(
    student: nn.Module,
    student_prev: nn.Module,
    t_retain: torch.Tensor,
    xt_retain: torch.Tensor,
    y_retain: torch.Tensor,
) -> torch.Tensor:
    """可选的工程增强项：稳定 retain 子空间，默认由 distill_gamma=0 关闭。"""
    vt_current = student(t_retain, xt_retain, y_retain)
    with torch.no_grad():
        vt_prev = student_prev(t_retain, xt_retain, y_retain)
    return torch.mean((vt_current - vt_prev).pow(2))


# In[8]:


# 7. 采样/评估/可视化函数（统一评估入口）
# --------------------------------------------------
# 这些函数用于生成样本、统计遗忘类生成率、输出对比图和结果摘要。
# 每次模型训练完后都回调用于写实验表中的数字。
# --------------------------------------------------


@torch.no_grad()
def build_balanced_eval_bundle(
    samples_per_class: int,
    device: torch.device,
    seed: int,
) -> Dict[str, torch.Tensor]:
    """构造类均衡的固定评估标签与初始噪声。"""
    g = torch.Generator(device=device)
    g.manual_seed(seed)
    labels = torch.arange(CONFIG["num_classes"], device=device).repeat_interleave(samples_per_class)
    z0 = torch.randn(labels.size(0), *CONFIG["data_shape"], generator=g, device=device)
    return {"labels": labels, "z0": z0}


def get_sampling_config(strategy: str) -> Dict[str, float]:
    """按用途返回 ODE 采样配置。"""
    if strategy == "vis":
        return {
            "method": CONFIG.get("vis_ode_method", CONFIG["ode_method"]),
            "atol": float(CONFIG.get("vis_ode_atol", CONFIG["ode_atol"])),
            "rtol": float(CONFIG.get("vis_ode_rtol", CONFIG["ode_rtol"])),
            "time_steps": int(CONFIG.get("sample_time_steps_vis", 17)),
        }
    return {
        "method": CONFIG["ode_method"],
        "atol": float(CONFIG["ode_atol"]),
        "rtol": float(CONFIG["ode_rtol"]),
        "time_steps": int(CONFIG.get("sample_time_steps_eval", 33)),
    }


def integrate_cfm_ode(
    model: nn.Module,
    labels: torch.Tensor,
    z0: torch.Tensor,
    strategy: str = "eval",
) -> torch.Tensor:
    """统一封装 ODE 采样，支持高质量评估与快速可视化两套配置。"""
    sampling_cfg = get_sampling_config(strategy)
    ts = torch.linspace(0.0, 1.0, max(int(sampling_cfg["time_steps"]), 2), device=z0.device)
    ode_kwargs = {"method": sampling_cfg["method"]}
    if sampling_cfg["method"] in {"dopri5", "dopri8", "bosh3", "fehlberg2", "adaptive_heun"}:
        ode_kwargs["atol"] = sampling_cfg["atol"]
        ode_kwargs["rtol"] = sampling_cfg["rtol"]
    traj = torchdiffeq.odeint(
        lambda t, x: model(t, x, labels),
        z0,
        ts,
        **ode_kwargs,
    )
    return traj[-1].clamp(-1.0, 1.0)


@torch.no_grad()
def generate_samples(
    model: nn.Module,
    labels: torch.Tensor,
    device: torch.device,
    batch_size: int,
    z0_all: torch.Tensor = None,
    strategy: str = "eval",
) -> torch.Tensor:
    """给定条件标签，沿 ODE 积分采样生成图像。"""
    samples = []

    for start in range(0, labels.size(0), batch_size):
        batch_labels = labels[start:start + batch_size]
        if z0_all is None:
            z0 = torch.randn(batch_labels.size(0), *CONFIG["data_shape"], device=device)
        else:
            z0 = z0_all[start:start + batch_labels.size(0)]
        samples.append(
            integrate_cfm_ode(
                model=model,
                labels=batch_labels,
                z0=z0,
                strategy=strategy,
            )
        )

    return torch.cat(samples, dim=0)


@torch.no_grad()
def make_class_grid(
    model: nn.Module,
    device: torch.device,
    z0: torch.Tensor = None,
    samples_per_class: int = 10,
    strategy: str = "vis",
) -> torch.Tensor:
    """每类各10张图像拼成网格，用于阶段对比图。"""
    total_samples = CONFIG["num_classes"] * samples_per_class
    if z0 is None:
        z0 = torch.randn(total_samples, *CONFIG["data_shape"], device=device)

    labels = torch.arange(CONFIG["num_classes"], device=device).repeat(samples_per_class)
    x_final = integrate_cfm_ode(model=model, labels=labels, z0=z0, strategy=strategy)
    return make_grid(
        x_final,
        nrow=CONFIG["num_classes"],
        padding=2,
        normalize=True,
        value_range=(-1, 1),
    )


@torch.no_grad()
def make_single_class_grid(
    model: nn.Module,
    class_idx: int,
    device: torch.device,
    n_samples: int,
    nrow: int,
    z0: torch.Tensor = None,
    strategy: str = "vis",
) -> torch.Tensor:
    """只生成某一个类别的样本网格，用于遗忘效果细粒度观察。"""
    labels = torch.full((n_samples,), class_idx, dtype=torch.long, device=device)
    if z0 is None:
        z0 = torch.randn(n_samples, *CONFIG["data_shape"], device=device)

    x_final = integrate_cfm_ode(model=model, labels=labels, z0=z0, strategy=strategy)
    return make_grid(x_final, nrow=nrow, padding=2, normalize=True, value_range=(-1, 1))


@torch.no_grad()
def generate_single_class_image(
    model: nn.Module,
    class_idx: int,
    device: torch.device,
    z0: torch.Tensor,
    strategy: str = "vis",
) -> torch.Tensor:
    """生成单个类别的一张固定样本图，用于论文友好的稀疏可视化。"""
    labels = torch.full((1,), class_idx, dtype=torch.long, device=device)
    x_final = integrate_cfm_ode(model=model, labels=labels, z0=z0, strategy=strategy)
    return x_final[0].detach().cpu()


@torch.no_grad()
def make_single_sample_z0(seed: int, device: torch.device) -> torch.Tensor:
    """由固定整数 seed 生成单样本噪声，确保跨图可复现。"""
    g = torch.Generator(device=device)
    g.manual_seed(int(seed))
    return torch.randn(1, *CONFIG["data_shape"], generator=g, device=device)


def compute_image_sharpness(image: torch.Tensor) -> float:
    """轻量锐度指标：灰度图一阶梯度平均幅值。"""
    x = ((image.clamp(-1.0, 1.0) + 1.0) / 2.0).float()
    gray = x.mean(dim=0, keepdim=True)
    dx = torch.abs(gray[:, :, 1:] - gray[:, :, :-1]).mean()
    dy = torch.abs(gray[:, 1:, :] - gray[:, :-1, :]).mean()
    return float((dx + dy).item())


def score_single_seed(
    model: nn.Module,
    classifier: nn.Module,
    class_idx: int,
    seed: int,
    device: torch.device,
    strategy: str,
    sharpness_weight: float,
) -> Dict[str, float]:
    """评估单个候选 seed 的语义置信度与清晰度综合得分。"""
    z0 = make_single_sample_z0(seed=seed, device=device)
    img = generate_single_class_image(
        model=model,
        class_idx=class_idx,
        device=device,
        z0=z0,
        strategy=strategy,
    )
    with torch.no_grad():
        logits = classifier(img.unsqueeze(0).to(device))
        probs = torch.softmax(logits, dim=1)
        target_conf = float(probs[0, class_idx].item())
    sharpness = compute_image_sharpness(img)
    score = target_conf + float(sharpness_weight) * sharpness
    return {
        "seed": int(seed),
        "target_confidence": target_conf,
        "sharpness": sharpness,
        "score": score,
    }


def resolve_single_seed_bank(
    pretrain_model_path: str,
    classifier: nn.Module,
    device: torch.device,
) -> Dict[int, int]:
    """构建或加载每类固定单样本 seed（best-of-K 后冻结）。"""
    bank_path = Path(CONFIG.get("single_seed_bank_path", "./outputs/cifar10_temporal/temporal/summaries/single_seed_bank.json"))
    refresh = bool(CONFIG.get("single_seed_refresh", True))
    candidate_k = max(1, int(CONFIG.get("single_seed_candidates", 32)))
    sharpness_weight = float(CONFIG.get("single_seed_sharpness_weight", 0.035))
    strategy = str(CONFIG.get("single_vis_generation_strategy", "eval"))
    class_indices = list(FORGET_SEQUENCE)

    if bank_path.exists() and not refresh:
        payload = json.loads(bank_path.read_text(encoding="utf-8"))
        seed_map = {int(k): int(v["seed"]) for k, v in payload["classes"].items()}
        print(f"[SeedBank] 已加载固定单样本 seed: {bank_path}")
        return seed_map

    model = load_cfm_model(pretrain_model_path, device)
    model.eval()
    classifier.eval()

    seed_map: Dict[int, int] = {}
    details: Dict[str, Dict[str, float]] = {}
    base_seed = int(CONFIG["seed"]) + int(CONFIG.get("single_vis_seed_offset", 7000))

    for cls_idx in class_indices:
        candidate_infos = []
        cls_base = base_seed + cls_idx * 100000
        for i in range(candidate_k):
            candidate_seed = cls_base + i
            info = score_single_seed(
                model=model,
                classifier=classifier,
                class_idx=cls_idx,
                seed=candidate_seed,
                device=device,
                strategy=strategy,
                sharpness_weight=sharpness_weight,
            )
            candidate_infos.append(info)
        best = max(candidate_infos, key=lambda x: x["score"])
        seed_map[cls_idx] = int(best["seed"])
        details[str(cls_idx)] = best
        print(
            f"[SeedBank] class={CLASS_NAMES[cls_idx]:>10} | seed={best['seed']} | "
            f"conf={best['target_confidence']:.4f} | sharp={best['sharpness']:.4f} | score={best['score']:.4f}"
        )

    payload = {
        "pretrain_model_path": str(pretrain_model_path),
        "strategy": strategy,
        "candidate_k": candidate_k,
        "sharpness_weight": sharpness_weight,
        "classes": details,
    }
    bank_path.parent.mkdir(parents=True, exist_ok=True)
    bank_path.write_text(json.dumps(payload, indent=2, ensure_ascii=False), encoding="utf-8")
    print(f"[SeedBank] 已保存固定单样本 seed bank: {bank_path}")
    return seed_map


def resolve_resample_mode(name: str) -> int:
    """将配置字符串映射为 PIL 插值模式。"""
    mapping = {
        "nearest": Image.Resampling.NEAREST,
        "bilinear": Image.Resampling.BILINEAR,
        "bicubic": Image.Resampling.BICUBIC,
        "lanczos": Image.Resampling.LANCZOS,
    }
    return mapping.get(str(name).lower(), Image.Resampling.LANCZOS)


def tensor_to_upscaled_pil(image: torch.Tensor, upscale: int) -> Image.Image:
    """把单张张量图像转成高分辨率 PIL 图（显式反归一化后再放大）。"""
    image_01 = ((image.clamp(-1.0, 1.0) + 1.0) / 2.0).float()
    img = ToPILImage()(image_01)
    if upscale > 1:
        img = img.resize(
            (img.size[0] * upscale, img.size[1] * upscale),
            resample=resolve_resample_mode(CONFIG.get("single_vis_resample", "lanczos")),
        )
    return img


def _paste_images_horizontally(images: List[Image.Image], save_path: Path, gap: int, padding: int) -> None:
    """无标题横向拼接图片。"""
    widths, heights = zip(*(img.size for img in images))
    canvas = Image.new(
        "RGB",
        (sum(widths) + gap * (len(images) - 1) + padding * 2, max(heights) + padding * 2),
        color=(255, 255, 255),
    )
    x = padding
    for img in images:
        y = padding + (max(heights) - img.size[1]) // 2
        canvas.paste(img, (x, y))
        x += img.size[0] + gap
    save_path.parent.mkdir(parents=True, exist_ok=True)
    canvas.save(save_path, format="PNG", optimize=True)


def _paste_stage_triplets(
    panels: List[List[Image.Image]],
    save_path: Path,
    gap_x: int,
    gap_y: int,
    padding: int,
) -> None:
    """每个阶段一个竖向三图面板，再按阶段横向拼接。"""
    stage_widths = []
    stage_heights = []
    stage_canvases = []
    for panel in panels:
        widths, heights = zip(*(img.size for img in panel))
        stage_w = max(widths)
        stage_h = sum(heights) + gap_y * (len(panel) - 1)
        stage_canvas = Image.new("RGB", (stage_w, stage_h), color=(255, 255, 255))
        y = 0
        for img in panel:
            x = (stage_w - img.size[0]) // 2
            stage_canvas.paste(img, (x, y))
            y += img.size[1] + gap_y
        stage_canvases.append(stage_canvas)
        stage_widths.append(stage_w)
        stage_heights.append(stage_h)

    canvas = Image.new(
        "RGB",
        (sum(stage_widths) + gap_x * (len(stage_canvases) - 1) + padding * 2, max(stage_heights) + padding * 2),
        color=(255, 255, 255),
    )
    x = padding
    for panel_img in stage_canvases:
        y = padding + (max(stage_heights) - panel_img.size[1]) // 2
        canvas.paste(panel_img, (x, y))
        x += panel_img.size[0] + gap_x
    save_path.parent.mkdir(parents=True, exist_ok=True)
    canvas.save(save_path, format="PNG", optimize=True)


def _paste_image_matrix(
    matrix: List[List[Image.Image]],
    save_path: Path,
    gap_x: int,
    gap_y: int,
    padding: int,
) -> None:
    """无标题矩阵拼图。"""
    cell_w = max(img.size[0] for row in matrix for img in row)
    cell_h = max(img.size[1] for row in matrix for img in row)
    n_rows = len(matrix)
    n_cols = len(matrix[0]) if matrix else 0
    canvas = Image.new(
        "RGB",
        (padding * 2 + n_cols * cell_w + (n_cols - 1) * gap_x, padding * 2 + n_rows * cell_h + (n_rows - 1) * gap_y),
        color=(255, 255, 255),
    )
    for row_idx, row in enumerate(matrix):
        for col_idx, img in enumerate(row):
            x = padding + col_idx * (cell_w + gap_x) + (cell_w - img.size[0]) // 2
            y = padding + row_idx * (cell_h + gap_y) + (cell_h - img.size[1]) // 2
            canvas.paste(img, (x, y))
    save_path.parent.mkdir(parents=True, exist_ok=True)
    canvas.save(save_path, format="PNG", optimize=True)


def _stage_folder_name(stage_idx: int) -> str:
    """把阶段索引映射为稳定目录名。"""
    if stage_idx == 0:
        return "stage0_initial"
    forget_idx = FORGET_SEQUENCE[stage_idx - 1]
    return f"stage{stage_idx}_forget_{CLASS_NAMES[forget_idx]}"


def save_stage_single_sample_bank(
    figures_dir: Path,
    stage_panels: List[List[Image.Image]],
    class_order: List[int],
) -> Dict[str, Dict[str, str]]:
    """按阶段输出固定单样本图：每阶段一个文件夹，每类一张图。"""
    root = figures_dir / "single_samples_by_stage"
    exported: Dict[str, Dict[str, str]] = {}

    for stage_idx, panel in enumerate(stage_panels):
        stage_name = _stage_folder_name(stage_idx)
        stage_dir = root / stage_name
        stage_dir.mkdir(parents=True, exist_ok=True)

        class_paths: Dict[str, str] = {}
        for cls_idx, img in zip(class_order, panel):
            class_name = CLASS_NAMES[cls_idx]
            save_path = stage_dir / f"{class_name}.png"
            img.save(save_path, format="PNG", optimize=True)
            class_paths[class_name] = str(save_path)

        exported[stage_name] = class_paths
        print(f"[Figure] 已保存阶段单样本目录: {stage_dir}")

    return exported


@torch.no_grad()
def build_fixed_eval_bundle(n_samples: int, device: torch.device, seed: int) -> Dict[str, torch.Tensor]:
    """固定评估标签与初始噪声，确保跨阶段对比的统计一致性。"""
    g = torch.Generator(device=device)
    g.manual_seed(seed)
    labels = torch.randint(0, CONFIG["num_classes"], (n_samples,), generator=g, device=device)
    z0 = torch.randn(n_samples, *CONFIG["data_shape"], generator=g, device=device)
    return {"labels": labels, "z0": z0}


@torch.no_grad()
def evaluate_generation_rates(
    model: nn.Module,
    classifier: nn.Module,
    device: torch.device,
    target_classes: List[int],
    n_samples: int,
    eval_bundle: Dict[str, torch.Tensor] = None,
) -> Dict[int, float]:
    """返回目标遗忘类的生成占比（%）。"""
    model.eval()
    classifier.eval()

    if eval_bundle is None:
        labels = torch.randint(0, CONFIG["num_classes"], (n_samples,), device=device)
        z0_all = None
    else:
        labels = eval_bundle["labels"]
        z0_all = eval_bundle["z0"]

    samples = generate_samples(
        model=model,
        labels=labels,
        device=device,
        batch_size=CONFIG["sample_batch_size"],
        z0_all=z0_all,
        strategy="eval",
    )

    preds = []
    for start in range(0, samples.size(0), CONFIG["sample_batch_size"]):
        batch = samples[start:start + CONFIG["sample_batch_size"]]
        logits = classifier(batch)
        preds.append(logits.argmax(dim=1).cpu())

    preds = torch.cat(preds, dim=0).numpy()
    rates = {}
    total = max(preds.shape[0], 1)
    for cls_idx in target_classes:
        rates[cls_idx] = float((preds == cls_idx).sum() / total * 100.0)

    return rates


@torch.no_grad()
def evaluate_conditional_generation_quality(
    model: nn.Module,
    classifier: nn.Module,
    device: torch.device,
    samples_per_class: int,
    eval_bundle: Dict[str, torch.Tensor] = None,
) -> Dict[str, object]:
    """评估条件生成的类别命中率与目标置信度。"""
    if eval_bundle is None:
        eval_bundle = build_balanced_eval_bundle(
            samples_per_class=samples_per_class,
            device=device,
            seed=CONFIG["seed"] + 2024,
        )
    labels = eval_bundle["labels"]
    samples = generate_samples(
        model=model,
        labels=labels,
        device=device,
        batch_size=CONFIG["sample_batch_size"],
        z0_all=eval_bundle["z0"],
        strategy="eval",
    )
    logits_all = []
    for start in range(0, samples.size(0), CONFIG["sample_batch_size"]):
        batch = samples[start:start + CONFIG["sample_batch_size"]]
        logits_all.append(classifier(batch).cpu())
    logits = torch.cat(logits_all, dim=0)
    preds = logits.argmax(dim=1)
    labels_cpu = labels.cpu()
    probs = logits.softmax(dim=1)
    gathered = probs.gather(1, labels_cpu.unsqueeze(1)).squeeze(1)

    per_class_accuracy = {}
    for class_idx, class_name in enumerate(CLASS_NAMES):
        mask = labels_cpu == class_idx
        per_class_accuracy[class_name] = float((preds[mask] == labels_cpu[mask]).float().mean().item() * 100.0)

    return {
        "accuracy": float((preds == labels_cpu).float().mean().item() * 100.0),
        "target_confidence": float(gathered.mean().item() * 100.0),
        "per_class_accuracy": per_class_accuracy,
    }


def save_grid_image(grid: torch.Tensor, save_path: Path, upscale: int = 1) -> None:
    """保存单张网格图，供预训练过程追踪。"""
    img = ToPILImage()(grid.cpu())
    if upscale > 1:
        img = img.resize(
            (img.size[0] * upscale, img.size[1] * upscale),
            resample=Image.Resampling.BICUBIC,
        )
    save_path.parent.mkdir(parents=True, exist_ok=True)
    img.save(save_path, format="PNG", optimize=True)


@torch.no_grad()
def collect_class_samples(
    data_loader: DataLoader,
    class_idx: int,
    max_samples: int,
    device: torch.device,
) -> torch.Tensor:
    """从数据集中提取某一类的若干归一化样本，用于向量场投影可视化。"""
    chunks = []
    collected = 0
    for x, y in data_loader:
        mask = (y == class_idx)
        if mask.any():
            picked = x[mask]
            remaining = max_samples - collected
            chunks.append(picked[:remaining])
            collected += min(picked.size(0), remaining)
            if collected >= max_samples:
                break
    if not chunks:
        raise RuntimeError(f"未能从 data_loader 中提取类别 {CLASS_NAMES[class_idx]} 的样本。")
    samples = torch.cat(chunks, dim=0)[:max_samples]
    return samples.to(device)


def build_plane_from_samples(
    real_samples: torch.Tensor,
    noise_samples: torch.Tensor,
) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
    """在图像空间中构造二维主平面，用于把高维向量场投影成 quiver 图。"""
    real_flat = real_samples.reshape(real_samples.size(0), -1).detach().cpu()
    noise_flat = noise_samples.reshape(noise_samples.size(0), -1).detach().cpu()
    combined = torch.cat([real_flat, noise_flat], dim=0)
    center = combined.mean(dim=0, keepdim=True)
    _, _, v = torch.pca_lowrank(combined - center, q=2, center=False)
    basis = v[:, :2]
    proj_real = (real_flat - center) @ basis
    proj_noise = (noise_flat - center) @ basis
    return center.squeeze(0), basis, proj_real, proj_noise


def build_plane_grid(
    proj_real: torch.Tensor,
    proj_noise: torch.Tensor,
    grid_size: int,
    margin_ratio: float,
) -> Tuple[np.ndarray, np.ndarray, torch.Tensor]:
    """在二维投影平面上生成规则网格。"""
    proj_all = torch.cat([proj_real, proj_noise], dim=0)
    mins = proj_all.min(dim=0).values
    maxs = proj_all.max(dim=0).values
    spans = (maxs - mins).clamp(min=1e-3)
    mins = mins - spans * margin_ratio
    maxs = maxs + spans * margin_ratio
    xs = np.linspace(float(mins[0]), float(maxs[0]), grid_size)
    ys = np.linspace(float(mins[1]), float(maxs[1]), grid_size)
    xx, yy = np.meshgrid(xs, ys)
    coords = torch.from_numpy(np.stack([xx, yy], axis=-1).reshape(-1, 2)).float()
    return xx, yy, coords


@torch.no_grad()
def evaluate_projected_vector_field(
    model: nn.Module,
    class_idx: int,
    t_scalar: float,
    coords_2d: torch.Tensor,
    center: torch.Tensor,
    basis: torch.Tensor,
    device: torch.device,
) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """在二维主平面网格上评估条件向量场，并投影为 quiver 箭头。"""
    flat_points = center.unsqueeze(0) + coords_2d @ basis.T
    x_points = flat_points.view(-1, *CONFIG["data_shape"]).to(device)
    labels = torch.full((x_points.size(0),), class_idx, dtype=torch.long, device=device)
    t = torch.full((x_points.size(0),), float(t_scalar), dtype=x_points.dtype, device=device)

    vt_chunks = []
    batch_size = int(CONFIG.get("field_vis_batch_size", 256))
    for start in range(0, x_points.size(0), batch_size):
        end = start + batch_size
        vt = model(t[start:end], x_points[start:end], labels[start:end])
        vt_chunks.append(vt.reshape(vt.size(0), -1).detach().cpu())
    vt_flat = torch.cat(vt_chunks, dim=0)
    uv = vt_flat @ basis
    mag = torch.linalg.norm(uv, dim=1)
    return (
        uv[:, 0].numpy(),
        uv[:, 1].numpy(),
        mag.numpy(),
    )


def style_field_axis(ax: plt.Axes) -> None:
    """统一 quiver 子图风格，使其更接近论文插图效果。"""
    ax.set_xticks([])
    ax.set_yticks([])
    ax.set_aspect("equal")
    ax.set_facecolor("#f8fafc")
    for spine in ax.spines.values():
        spine.set_visible(True)
        spine.set_linewidth(0.85)
        spine.set_edgecolor("#dbe4f0")


def compute_marker_layout(
    p0_mean: np.ndarray,
    x1_mean: np.ndarray,
    min_distance: float = 0.52,
) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """当 p0 和 x1 过近时，适度拉开显示位置与标签位置，避免遮挡。"""
    p0_disp = p0_mean.astype(np.float64).copy()
    x1_disp = x1_mean.astype(np.float64).copy()
    delta = x1_disp - p0_disp
    dist = float(np.linalg.norm(delta))
    if dist < 1e-8:
        direction = np.array([1.0, 0.0], dtype=np.float64)
    else:
        direction = delta / dist
    if dist < min_distance:
        shift = 0.5 * (min_distance - dist) + 0.10
        p0_disp = p0_disp - direction * shift
        x1_disp = x1_disp + direction * shift
    p0_text = p0_disp + np.array([-0.12, 0.06], dtype=np.float64)
    x1_text = x1_disp + np.array([0.10, 0.06], dtype=np.float64)
    return p0_disp, x1_disp, p0_text, x1_text


@torch.no_grad()
def save_conditional_quiver_figure(
    model: nn.Module,
    class_idx: int,
    data_loader: DataLoader,
    device: torch.device,
    mode_name: str,
    stage_name: str,
    save_path: Path,
) -> None:
    """为某个遗忘阶段输出论文风格的条件向量场 quiver 图。"""
    model.eval()
    n_real = int(CONFIG.get("field_vis_class_samples", 72))
    n_pairs = int(CONFIG.get("field_vis_pairs", n_real))
    grid_size = int(CONFIG.get("field_vis_grid_size", 17))
    margin_ratio = float(CONFIG.get("field_vis_margin_ratio", 0.18))
    quiver_scale = float(CONFIG.get("field_vis_quiver_scale", 10.5))
    time_points = tuple(float(v) for v in CONFIG.get("field_vis_time_points", (0.0, 1.0 / 3.0, 2.0 / 3.0, 1.0)))

    real_samples = collect_class_samples(data_loader, class_idx, n_real, device)
    noise_samples = torch.randn(n_pairs, *CONFIG["data_shape"], device=device)
    real_for_pairs = real_samples[:n_pairs]
    center, basis, proj_real, proj_noise = build_plane_from_samples(real_for_pairs, noise_samples)
    xx, yy, coords_2d = build_plane_grid(proj_real, proj_noise, grid_size=grid_size, margin_ratio=margin_ratio)

    p0_mean = proj_noise.mean(dim=0).numpy()
    x1_mean = proj_real.mean(dim=0).numpy()
    fig, axes = plt.subplots(1, len(time_points), figsize=(5.15 * len(time_points), 6.1), dpi=CONFIG["paper_fig_dpi"])
    if len(time_points) == 1:
        axes = [axes]

    p0_disp, x1_disp, p0_text, x1_text = compute_marker_layout(
        p0_mean=p0_mean,
        x1_mean=x1_mean,
        min_distance=0.65 if class_idx == 3 else 0.52,
    )

    quiver_artist = None
    for ax, t_scalar in zip(axes, time_points):
        xt_refs = (1.0 - t_scalar) * noise_samples + t_scalar * real_for_pairs
        xt_proj = ((xt_refs.reshape(xt_refs.size(0), -1).detach().cpu() - center.unsqueeze(0)) @ basis).numpy()
        u, v, mag = evaluate_projected_vector_field(
            model=model,
            class_idx=class_idx,
            t_scalar=t_scalar,
            coords_2d=coords_2d,
            center=center,
            basis=basis,
            device=device,
        )

        mag_norm = np.percentile(mag, 90)
        if mag_norm > 1e-8:
            u_plot = u / mag_norm
            v_plot = v / mag_norm
        else:
            u_plot = u
            v_plot = v

        ax.scatter(
            xt_proj[:, 0],
            xt_proj[:, 1],
            s=18,
            c="#111827",
            alpha=0.08,
            linewidths=0,
            zorder=1,
        )
        quiver_artist = ax.quiver(
            xx,
            yy,
            u_plot.reshape(xx.shape),
            v_plot.reshape(xx.shape),
            mag.reshape(xx.shape),
            cmap="turbo",
            angles="xy",
            scale_units="xy",
            scale=quiver_scale * 0.92,
            width=0.0082,
            headwidth=4.6,
            headlength=6.0,
            headaxislength=4.8,
            pivot="mid",
            alpha=0.95,
            zorder=2,
        )
        ax.scatter([p0_disp[0]], [p0_disp[1]], s=54, c="#0f172a", marker="s", zorder=3)
        ax.scatter([x1_disp[0]], [x1_disp[1]], s=64, c="#111827", marker="o", zorder=3)
        ax.text(p0_text[0], p0_text[1], r"$p_0$", fontsize=12.5, color="#0f172a", va="center", ha="right")
        ax.text(x1_text[0], x1_text[1], r"$x_1$", fontsize=12.5, color="#111827", va="center", ha="left")
        if abs(t_scalar) < 1e-8:
            title = "t = 0.0"
        elif abs(t_scalar - 1.0 / 3.0) < 1e-6:
            title = r"t = $1/3$"
        elif abs(t_scalar - 2.0 / 3.0) < 1e-6:
            title = r"t = $2/3$"
        elif abs(t_scalar - 1.0) < 1e-8:
            title = "t = 1.0"
        else:
            title = rf"t = {t_scalar:.2f}"
        ax.set_title(title, fontsize=16, pad=12)
        style_field_axis(ax)

    if quiver_artist is not None:
        cax = fig.add_axes([0.12, 0.10, 0.76, 0.038])
        cbar = fig.colorbar(quiver_artist, cax=cax, orientation="horizontal")
        cbar.set_label("Projected magnitude", fontsize=11, labelpad=5)
        cbar.ax.tick_params(labelsize=9)
    fig.subplots_adjust(left=0.03, right=0.99, top=0.86, bottom=0.19, wspace=0.12)
    save_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(save_path, bbox_inches="tight", dpi=CONFIG["paper_fig_dpi"])
    plt.close(fig)
    print(f"[Figure] 已保存向量场 quiver 图: {save_path}")


def _vector_component(
    xx: np.ndarray,
    yy: np.ndarray,
    center_xy: Tuple[float, float],
    mode: str,
    eps: float,
) -> Tuple[np.ndarray, np.ndarray]:
    """二维示意流场的单个吸引/排斥分量。"""
    cx, cy = center_xy
    dx = cx - xx
    dy = cy - yy
    denom = dx * dx + dy * dy + eps
    if mode == "attr":
        return dx / denom, dy / denom
    return -dx / denom, -dy / denom


def _build_schematic_field(
    centers: Dict[str, Tuple[float, float]],
    stage_spec: Dict[str, str],
    repulsion_scale: float,
    eps: float,
) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """构造论文示意图用的 2D 连续遗忘向量场。"""
    x_min, x_max, y_min, y_max = CONFIG.get("schematic_extent", (-3.0, 3.0, -2.6, 2.8))
    grid_size = int(CONFIG.get("schematic_grid_size", 25))
    xs = np.linspace(x_min, x_max, grid_size)
    ys = np.linspace(y_min, y_max, grid_size)
    xx, yy = np.meshgrid(xs, ys)

    u = np.zeros_like(xx)
    v = np.zeros_like(yy)
    potential = np.zeros_like(xx)
    for key, mode in stage_spec.items():
        weight = repulsion_scale if mode == "rep" else 1.0
        du, dv = _vector_component(xx, yy, centers[key], mode=mode, eps=eps)
        u += weight * du
        v += weight * dv
        mag = np.sqrt(du ** 2 + dv ** 2)
        if mode == "rep":
            potential += weight * mag
        else:
            potential -= 0.55 * weight * mag
    magnitude = np.sqrt(u ** 2 + v ** 2)
    return xx, yy, u, v, potential + 0.15 * magnitude


def _plot_schematic_panel(
    ax: plt.Axes,
    xx: np.ndarray,
    yy: np.ndarray,
    u: np.ndarray,
    v: np.ndarray,
    potential: np.ndarray,
    centers: Dict[str, Tuple[float, float]],
    title: str,
) -> None:
    """绘制单个 2D 连续遗忘示意流场子图。"""
    stride = max(1, int(CONFIG.get("schematic_quiver_stride", 2)))
    bg = ax.pcolormesh(
        xx,
        yy,
        potential,
        cmap="coolwarm",
        shading="gouraud",
        alpha=0.92,
    )
    ax.quiver(
        xx[::stride, ::stride],
        yy[::stride, ::stride],
        u[::stride, ::stride],
        v[::stride, ::stride],
        color="#111827",
        angles="xy",
        scale_units="xy",
        scale=9.5,
        width=0.0038,
        headwidth=3.8,
        headlength=5.0,
        headaxislength=4.2,
        pivot="mid",
        alpha=0.86,
        zorder=3,
    )

    label_cfg = {
        "retain": ("Retain", "#111827", "o", (0.00, -0.24), 10.5),
        "forget_a": ("A", "#1f2937", "o", (0.00, 0.30), 12.6),
        "forget_b": ("B", "#1f2937", "o", (-0.16, 0.32), 12.6),
        "forget_c": ("C", "#1f2937", "o", (0.18, 0.30), 12.6),
    }
    for key, (label, color, marker, offset, font_size) in label_cfg.items():
        cx, cy = centers[key]
        ax.scatter(cx, cy, s=34, c=color, marker=marker, zorder=4)
        ax.text(
            cx + offset[0],
            cy + offset[1],
            label,
            fontsize=font_size,
            color=color,
            va="center",
            ha="center",
            zorder=6,
        )

    ax.set_title(title, fontsize=13, pad=8)
    ax.set_aspect("equal")
    ax.set_xticks([])
    ax.set_yticks([])
    ax.set_xlim(xx.min(), xx.max())
    ax.set_ylim(yy.min(), yy.max())
    ax.set_facecolor("white")
    for spine in ax.spines.values():
        spine.set_visible(True)
        spine.set_linewidth(0.9)
        spine.set_edgecolor("#d1d5db")
    return bg


def save_sequential_unlearning_schematic(mode: str, save_path: Path) -> None:
    """输出论文单栏友好的 1x4 连续遗忘示意流场图。"""
    centers = {
        key: tuple(value)
        for key, value in CONFIG.get("schematic_centers", {}).items()
    }
    repulsion_scale = float(CONFIG.get("schematic_repulsion_scale", 2.8))
    eps = float(CONFIG.get("schematic_eps", 0.06))

    assert mode in {"naive", "temporal"}
    if mode == "naive":
        fig_title = "Naive Sequential Unlearning"
        stage_specs = [
            ("Pretrain", {"retain": "attr", "forget_a": "attr", "forget_b": "attr", "forget_c": "attr"}),
            ("Stage 1: Forget A", {"retain": "attr", "forget_a": "rep", "forget_b": "attr", "forget_c": "attr"}),
            ("Stage 2: Forget B", {"retain": "attr", "forget_a": "attr", "forget_b": "rep", "forget_c": "attr"}),
            ("Stage 3: Forget C", {"retain": "attr", "forget_a": "attr", "forget_b": "attr", "forget_c": "rep"}),
        ]
    else:
        fig_title = "CFMU Temporal Shield"
        stage_specs = [
            ("Pretrain", {"retain": "attr", "forget_a": "attr", "forget_b": "attr", "forget_c": "attr"}),
            ("Stage 1: Forget A", {"retain": "attr", "forget_a": "rep", "forget_b": "attr", "forget_c": "attr"}),
            ("Stage 2: Forget B", {"retain": "attr", "forget_a": "rep", "forget_b": "rep", "forget_c": "attr"}),
            ("Stage 3: Forget C", {"retain": "attr", "forget_a": "rep", "forget_b": "rep", "forget_c": "rep"}),
        ]

    save_dpi = int(CONFIG.get("schematic_save_dpi", CONFIG["paper_fig_dpi"]))
    fig, axes = plt.subplots(1, 4, figsize=(17.2, 4.8), dpi=save_dpi)
    color_mesh = None
    for ax, (stage_title, stage_spec) in zip(axes, stage_specs):
        xx, yy, u, v, potential = _build_schematic_field(
            centers=centers,
            stage_spec=stage_spec,
            repulsion_scale=repulsion_scale,
            eps=eps,
        )
        color_mesh = _plot_schematic_panel(
            ax=ax,
            xx=xx,
            yy=yy,
            u=u,
            v=v,
            potential=potential,
            centers=centers,
            title=stage_title,
        )

    if color_mesh is not None:
        cax = fig.add_axes([0.10, 0.125, 0.80, 0.042])
        cbar = fig.colorbar(color_mesh, cax=cax, orientation="horizontal")
        cbar.ax.invert_xaxis()
        cbar.set_label("")
        cbar.ax.tick_params(labelsize=8)
        cbar.ax.text(
            -0.03,
            0.5,
            "Attraction-low",
            transform=cbar.ax.transAxes,
            ha="right",
            va="center",
            fontsize=10,
            color="#1f2937",
        )
        cbar.ax.text(
            1.03,
            0.5,
            "Repulsion-high",
            transform=cbar.ax.transAxes,
            ha="left",
            va="center",
            fontsize=10,
            color="#1f2937",
        )
    fig.subplots_adjust(left=0.015, right=0.995, top=0.96, bottom=0.17, wspace=0.16)
    save_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(save_path, bbox_inches="tight", dpi=save_dpi)
    plt.close(fig)
    print(f"[Figure] 已保存连续遗忘示意流场图: {save_path}")


def save_unlearning_degradation_figure(
    naive_history: List[Dict[str, float]],
    temporal_history: List[Dict[str, float]],
    save_path: Path,
) -> None:
    """输出“历史遗忘反弹抑制”图：左趋势，右反弹条形图。"""
    stage_ids = list(range(len(naive_history)))
    stage_labels = [row["stage"] for row in naive_history]

    fig, axes = plt.subplots(1, 2, figsize=(16, 6), dpi=CONFIG["paper_fig_dpi"])

    # 左图：三遗忘类在各阶段的生成率曲线（Naive vs Ours）
    ax = axes[0]
    colors = ["#d62728", "#1f77b4", "#2ca02c"]
    for c_idx, cls_idx in enumerate(FORGET_SEQUENCE):
        cname = CLASS_NAMES[cls_idx]
        y_naive = [row[cname] for row in naive_history]
        y_temporal = [row[cname] for row in temporal_history]
        ax.plot(stage_ids, y_naive, "--o", color=colors[c_idx], label=f"Naive-{cname}", alpha=0.85)
        ax.plot(stage_ids, y_temporal, "-o", color=colors[c_idx], label=f"Ours-{cname}", linewidth=2.0)

    ax.set_title("Sequential Forgetting Rates")
    ax.set_xlabel("Stage")
    ax.set_ylabel("Generated Rate (%)")
    ax.set_xticks(stage_ids)
    ax.set_xticklabels([f"S{i}" if i > 0 else "Pre" for i in stage_ids])
    ax.grid(alpha=0.25)
    ax.legend(fontsize=8, ncol=2)

    # 右图：仅统计“历史类”在后续阶段的正向反弹幅度
    ax = axes[1]
    x_labels = []
    naive_rebound = []
    temporal_rebound = []

    for stage_idx in range(1, len(FORGET_SEQUENCE) + 1):
        hist_classes = FORGET_SEQUENCE[: max(0, stage_idx - 1)]
        if not hist_classes:
            continue
        for cls_idx in hist_classes:
            cname = CLASS_NAMES[cls_idx]
            delta_key = f"{cname}_delta"
            x_labels.append(f"S{stage_idx}-{cname}")
            naive_rebound.append(max(0.0, float(naive_history[stage_idx].get(delta_key, 0.0))))
            temporal_rebound.append(max(0.0, float(temporal_history[stage_idx].get(delta_key, 0.0))))

    xs = np.arange(len(x_labels))
    bw = 0.36
    ax.bar(xs - bw / 2, naive_rebound, width=bw, label="Naive", color="#ef4444", alpha=0.85)
    ax.bar(xs + bw / 2, temporal_rebound, width=bw, label="Ours (Temporal)", color="#2563eb", alpha=0.9)
    ax.set_title("Historical Rebound (Positive Delta Only)")
    ax.set_ylabel("Rebound (%)")
    ax.set_xticks(xs)
    ax.set_xticklabels(x_labels, rotation=25, ha="right")
    ax.grid(axis="y", alpha=0.25)
    ax.legend()

    fig.suptitle("Unlearning Degradation Suppression", fontsize=15)
    fig.tight_layout(rect=[0, 0.02, 1, 0.95])

    save_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(save_path, bbox_inches="tight", dpi=CONFIG["paper_fig_dpi"])
    plt.close(fig)
    print(f"[Figure] 已保存反弹抑制分析图: {save_path}")


def build_stage_report_line(
    stage_name: str,
    rates: Dict[int, float],
    previous_rates: Dict[int, float] = None,
) -> Dict[str, float]:
    """把单阶段速记结果整理成统一 dict。"""
    row = {"stage": stage_name}
    for cls_idx in FORGET_SEQUENCE:
        key = CLASS_NAMES[cls_idx]
        row[key] = float(rates[cls_idx])
        if previous_rates is not None:
            row[f"{key}_delta"] = float(rates[cls_idx] - previous_rates[key])
    return row


def print_table3_style(history_rows: List[Dict[str, float]], title: str) -> None:
    """按论文展示风格打印三遗忘类占比。"""
    print(f"\n[{title}] Table 3 style generation rates (%)")
    header = ["stage"] + FORGET_SEQUENCE_NAMES
    print(" | ".join(f"{item:>14}" for item in header))
    print("-" * 68)
    for row in history_rows:
        values = [row["stage"]] + [f"{row[name]:.2f}" for name in FORGET_SEQUENCE_NAMES]
        print(" | ".join(f"{item:>14}" for item in values))


def _format_rate_with_optional_delta(rate: float, delta: float, show_delta: bool) -> str:
    """Table 3 单元格格式：xx.xx% 或 xx.xx% (+/-yy.yy)。"""
    if not show_delta:
        return f"{rate:.2f}%"
    sign = "+" if delta >= 0 else ""
    return f"{rate:.2f}% ({sign}{delta:.2f})"


def _short_forget_name(class_name: str) -> str:
    """与论文表格习惯对齐的短名称。"""
    mapping = {
        "automobile": "Auto",
        "cat": "Cat",
        "deer": "Deer",
    }
    return mapping.get(class_name, class_name.title())


def build_cifar_table3_rows(mode_name: str, history_rows: List[Dict[str, float]]) -> List[Dict[str, str]]:
    """按论文 Table 3 规则，把单方法历史转换为表格行。

    规则：
    - 包含 Pretrain + 三个遗忘阶段。
    - 在第 k 阶段，仅对历史已遗忘类（前 k-1 个）显示 delta。
    - delta 使用当前阶段相对上一阶段的变化量。
    """
    if len(history_rows) < 4:
        raise ValueError("history_rows 长度不足，至少需要 Pretrain + 3 个阶段。")

    rows: List[Dict[str, str]] = []

    # Pretrain
    pre = history_rows[0]
    rows.append(
        {
            "Method": mode_name,
            "Stage": "Pretrain",
            "Step 1": f"{pre[CLASS_NAMES[FORGET_SEQUENCE[0]]]:.2f}%",
            "Step 2": f"{pre[CLASS_NAMES[FORGET_SEQUENCE[1]]]:.2f}%",
            "Step 3": f"{pre[CLASS_NAMES[FORGET_SEQUENCE[2]]]:.2f}%",
        }
    )

    # Stage 1..3
    for stage_idx in range(1, 4):
        row_src = history_rows[stage_idx]
        forget_cls = CLASS_NAMES[FORGET_SEQUENCE[stage_idx - 1]]
        stage_name = f"Forget {_short_forget_name(forget_cls)}"

        out_row: Dict[str, str] = {
            "Method": mode_name if stage_idx == 1 else "",
            "Stage": stage_name,
        }

        for step_idx, cls_idx in enumerate(FORGET_SEQUENCE):
            cname = CLASS_NAMES[cls_idx]
            rate = float(row_src[cname])
            delta = float(row_src.get(f"{cname}_delta", 0.0))
            show_delta = step_idx < (stage_idx - 1)
            out_row[f"Step {step_idx + 1}"] = _format_rate_with_optional_delta(rate, delta, show_delta)

        rows.append(out_row)

    return rows


def save_cifar_table3_outputs(
    naive_history: List[Dict[str, float]],
    temporal_history: List[Dict[str, float]],
    output_dir: Path,
) -> Dict[str, str]:
    """统一输出 CIFAR-10 的 Table 3 量化结果（JSON/CSV/MD/PNG）。"""
    output_dir.mkdir(parents=True, exist_ok=True)

    rows = []
    rows.extend(build_cifar_table3_rows("Naive Sequential", naive_history))
    rows.extend(build_cifar_table3_rows("Ours (Temporal)", temporal_history))

    # 1) JSON
    json_path = output_dir / "table3_cifar_quantitative.json"
    with open(json_path, "w", encoding="utf-8") as f:
        json.dump(rows, f, indent=2, ensure_ascii=False)

    # 2) CSV
    csv_path = output_dir / "table3_cifar_quantitative.csv"
    fieldnames = ["Method", "Stage", "Step 1", "Step 2", "Step 3"]
    with open(csv_path, "w", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)

    # 3) Markdown
    md_path = output_dir / "table3_cifar_quantitative.md"
    md_lines = [
        "| Method | Stage | Step 1 | Step 2 | Step 3 |",
        "|---|---|---:|---:|---:|",
    ]
    for row in rows:
        md_lines.append(
            f"| {row['Method']} | {row['Stage']} | {row['Step 1']} | {row['Step 2']} | {row['Step 3']} |"
        )
    md_path.write_text("\n".join(md_lines), encoding="utf-8")

    # 4) 高分辨率 PNG 表格
    fig_path = output_dir / "table3_cifar_quantitative.png"
    headers = ["Method", "Stage", "Step 1", "Step 2", "Step 3"]
    cell_text = [[row[h] for h in headers] for row in rows]

    fig_h = max(4.0, 0.62 * len(cell_text) + 1.8)
    fig, ax = plt.subplots(figsize=(16, fig_h), dpi=CONFIG["paper_fig_dpi"])
    ax.axis("off")
    table = ax.table(
        cellText=cell_text,
        colLabels=headers,
        cellLoc="center",
        loc="center",
    )
    table.auto_set_font_size(False)
    table.set_fontsize(11)
    table.scale(1.0, 1.45)

    for (r, c), cell in table.get_celld().items():
        if r == 0:
            cell.set_text_props(weight="bold")
            cell.set_facecolor("#f5f5f5")

    ax.set_title("Table 3 (CIFAR-10): Sequential Unlearning Quantitative Results", fontsize=14, pad=16)
    fig.savefig(fig_path, bbox_inches="tight", dpi=CONFIG["paper_fig_dpi"])
    plt.close(fig)

    print("[Table3] CIFAR-10 量化表已输出:")
    print(f"  - JSON: {json_path}")
    print(f"  - CSV : {csv_path}")
    print(f"  - MD  : {md_path}")
    print(f"  - PNG : {fig_path}")

    return {
        "json": str(json_path),
        "csv": str(csv_path),
        "markdown": str(md_path),
        "png": str(fig_path),
    }


def save_cfm_state_dict(model: nn.Module, path: Path) -> None:
    """保存 CFM 学生模型状态字典。"""
    path.parent.mkdir(parents=True, exist_ok=True)
    torch.save(model.state_dict(), path)



def save_history_json(history_rows: List[Dict[str, float]], save_path: Path) -> None:
    """将每个阶段输出写入 json，便于外部制图或比对。"""
    serializable = []
    for row in history_rows:
        serializable.append({
            key: (float(value) if isinstance(value, (np.floating, float, int)) else value)
            for key, value in row.items()
        })
    save_path.parent.mkdir(parents=True, exist_ok=True)
    with open(save_path, "w", encoding="utf-8") as f:
        json.dump(serializable, f, indent=2, ensure_ascii=False)
    print(f"[Summary] 已保存结果摘要: {save_path}")


# In[9]:


# 8. 顺序遗忘主流程：支持 Naive 与 Temporal 两条线
# --------------------------------------------------
# run_sequential_experiment 会在一个 mode 下依次执行三个遗忘阶段：
# Stage-1 forget automobile，Stage-2 forget cat，Stage-3 forget deer。
# 每个阶段训练结束后会做一次生成率评估并输出 delta，
# 同时保存阶段 ckpt 与可视化对比图，最后写入 JSON 报告。
# --------------------------------------------------


def run_sequential_experiment(
    mode: str,
    teacher_checkpoint_path: str,
    classifier: nn.Module,
    feature_net: nn.Module,
    centroids: torch.Tensor,
    train_loader: DataLoader,
    field_vis_loader: DataLoader,
    single_seed_map: Dict[int, int],
    device: torch.device,
    output_dir: Path,
) -> Dict[str, object]:
    """在单一 mode 下执行 3 个连续遗忘阶段。"""
    assert mode in {"naive", "temporal"}
    mode_name = "Naive Sequential" if mode == "naive" else "Ours (Temporal)"
    # 默认保持论文中的 Temporal Shield 更新路径本身，不把 EMA 混入方法定义。
    # 若后续仅做工程对比，可手动打开 sequential_use_ema。
    use_stage_ema = bool(CONFIG.get("sequential_use_ema", True))
    stage_ema_decay = float(CONFIG.get("sequential_ema_decay", 0.999))

    teacher = load_cfm_model(teacher_checkpoint_path, device)
    teacher.eval()
    student = load_cfm_model(teacher_checkpoint_path, device)
    student.train()

    # 每种方法的输出统一拆分为 checkpoints / figures / summaries，便于论文复盘。
    checkpoints_dir = output_dir / "checkpoints"
    figures_dir = output_dir / "figures"
    summaries_dir = output_dir / "summaries"
    checkpoints_dir.mkdir(parents=True, exist_ok=True)
    figures_dir.mkdir(parents=True, exist_ok=True)
    summaries_dir.mkdir(parents=True, exist_ok=True)

    checkpoint_paths = {}
    pretrain_ckpt_path = checkpoints_dir / f"cfm_cifar_{mode}_stage0_pretrain.pt"
    save_cfm_state_dict(teacher, pretrain_ckpt_path)
    checkpoint_paths["stage0_pretrain"] = str(pretrain_ckpt_path)
    print(f"[Checkpoint] 已保存初始预训练权重: {pretrain_ckpt_path}")

    fm = ConditionalFlowMatcher(sigma=CONFIG["sigma"])

    # 固定评估采样噪声与条件标签，保证 Pretrain/Stage 间可比。
    eval_bundle = build_fixed_eval_bundle(
        n_samples=CONFIG["n_eval_samples"],
        device=device,
        seed=CONFIG["seed"] + (0 if mode == "naive" else 1000),
    )

    # Baseline 统计：记录初始预训练模型在遗忘目标类上的生成率
    baseline_rates = evaluate_generation_rates(
        model=teacher,
        classifier=classifier,
        device=device,
        target_classes=FORGET_SEQUENCE,
        n_samples=CONFIG["n_eval_samples"],
        eval_bundle=eval_bundle,
    )
    history_rows = [build_stage_report_line("Pretrain", baseline_rates)]
    print_table3_style(history_rows, mode_name)

    # 单样本可视化：每类固定一个 seed，跨阶段/跨图复用，保证严格可比。
    single_stage_panels: List[List[Image.Image]] = []

    def render_single_samples(model_to_render: nn.Module) -> List[Image.Image]:
        panel: List[Image.Image] = []
        strategy = str(CONFIG.get("single_vis_generation_strategy", "eval"))
        upscale = int(CONFIG.get("single_vis_upscale", 7))
        for cls_idx in FORGET_SEQUENCE:
            seed = int(single_seed_map[cls_idx])
            z0 = make_single_sample_z0(seed=seed, device=device)
            img_tensor = generate_single_class_image(
                model=model_to_render,
                class_idx=cls_idx,
                device=device,
                z0=z0,
                strategy=strategy,
            )
            img_pil = tensor_to_upscaled_pil(img_tensor, upscale=upscale)
            panel.append(img_pil)
        return panel

    single_stage_panels.append(render_single_samples(teacher))
    field_figure_paths = {}

    forgotten_classes: List[int] = []

    # 按论文设置的任务序列执行
    for stage_idx, forget_class in enumerate(FORGET_SEQUENCE):
        historical_forgotten = list(forgotten_classes)
        forgotten_classes.append(forget_class)
        expected_cumulative = list(FORGET_SEQUENCE[: stage_idx + 1])
        assert forgotten_classes == expected_cumulative, (
            "累计遗忘顺序异常，请检查 FORGET_SEQUENCE 与阶段推进逻辑。"
        )
        stage_name = f"Stage {stage_idx + 1}: Forget {CLASS_NAMES[forget_class]}"
        print(f"\n=== [{mode_name}] {stage_name} ===")
        student_ema = None
        if use_stage_ema:
            student_ema = instantiate_cfm_model(device)
            student_ema.load_state_dict(student.state_dict())
            student_ema = freeze_module(student_ema)

        # Temporal 模式需要把历史学生模型冻结后作为 v_{theta}^{k-1} 参考，
        # 供 OT 项计算 <Delta v, v_prev>。
        student_prev = None
        if mode == "temporal" and historical_forgotten:
            student_prev = freeze_module(copy.deepcopy(student).to(device))

        optimizer = torch.optim.Adam(
            student.parameters(),
            lr=CONFIG["lr"],
            weight_decay=CONFIG["weight_decay"],
        )
        stage_total_steps = CONFIG["epochs_per_stage"] * len(train_loader)
        scheduler = CosineAnnealingLR(optimizer, T_max=max(stage_total_steps, 1))

        if mode == "temporal":
            # Temporal 阶段需要 E_attr 与 E_rep 所需原型集合
            prototypes = build_stage_prototypes(centroids, forgotten_classes)
            retain_proto = prototypes["retain"]
            forget_proto = prototypes["forget"]

        # 仅做当前阶段训练
        global_step = 0
        for epoch in range(CONFIG["epochs_per_stage"]):
            student.train()
            loss_main_meter = 0.0
            loss_ortho_meter = 0.0
            loss_distill_meter = 0.0
            step_count = 0

            for x_real, y in train_loader:
                x_real, y = x_real.to(device), y.to(device)
                optimizer.zero_grad(set_to_none=True)

                # 构造 flow matching 的 (t, x_t, u_t) 监督对
                x_noise = torch.randn_like(x_real)
                x_target = x_real.clone()
                forgotten_tensor = torch.tensor(forgotten_classes, device=device)
                mask_forget = torch.isin(y, forgotten_tensor)
                if mask_forget.any():
                    # 对当前阶段要遗忘的类，x1 = z（高噪声目标），让模型学会“避开”该类
                    x_target[mask_forget] = x_noise[mask_forget]

                t, xt, ut = fm.sample_location_and_conditional_flow(x_noise, x_target)
                vt = student(t, xt, y)
                loss_distill = torch.tensor(0.0, device=device)

                if mode == "naive":
                    # 只优化 MSE 回归误差；这是论文中的对照线
                    loss_main = per_sample_mse(vt, ut).mean()
                    loss_ortho = torch.tensor(0.0, device=device)
                else:
                    # Temporal：能量权重应基于原始样本语义，而不是已被替换成噪声的 x_target。
                    with torch.no_grad():
                        energy = compute_energy(x_real, feature_net, retain_proto, forget_proto)
                        lambda_k = lambda_schedule(global_step, stage_total_steps, CONFIG["lambda_max"])
                        weights = torch.sigmoid(-lambda_k * energy).clamp(
                            min=CONFIG["weight_floor"], max=1.0
                        )
                    loss_main = weighted_cfm_loss(vt, ut, weights)

                    # OT 正交约束：只施加在历史已遗忘类别样本上
                    loss_ortho = torch.tensor(0.0, device=device)
                    if student_prev is not None and historical_forgotten:
                        historical_tensor = torch.tensor(historical_forgotten, device=device)
                        mask_hist = torch.isin(y, historical_tensor)
                        if mask_hist.any():
                            loss_ortho = compute_orthogonality_loss(
                                student=student,
                                student_prev=student_prev,
                                t_hist=t[mask_hist],
                                xt_hist=xt[mask_hist],
                                y_hist=y[mask_hist],
                            )

                    # 可选 retain distillation：默认关闭，但在高语义耦合场景下可进一步稳住保留子空间。
                    if student_prev is not None and CONFIG["distill_gamma"] > 0.0:
                        mask_retain = ~mask_forget
                        if mask_retain.any():
                            loss_distill = compute_retain_distillation_loss(
                                student=student,
                                student_prev=student_prev,
                                t_retain=t[mask_retain],
                                xt_retain=xt[mask_retain],
                                y_retain=y[mask_retain],
                            )

                total_loss = (
                    loss_main
                    + CONFIG["beta"] * loss_ortho
                    + CONFIG["distill_gamma"] * loss_distill
                )
                total_loss.backward()
                torch.nn.utils.clip_grad_norm_(student.parameters(), max_norm=CONFIG["grad_clip_norm"])
                optimizer.step()
                if student_ema is not None:
                    update_ema_model(student, student_ema, stage_ema_decay)
                scheduler.step()

                loss_main_meter += loss_main.item()
                loss_ortho_meter += loss_ortho.item()
                loss_distill_meter += loss_distill.item()
                step_count += 1
                global_step += 1

            avg_main = loss_main_meter / max(step_count, 1)
            avg_ortho = loss_ortho_meter / max(step_count, 1)
            avg_distill = loss_distill_meter / max(step_count, 1)
            print(
                f"[{mode_name}] epoch {epoch + 1:02d}/{CONFIG['epochs_per_stage']:02d} | "
                f"main {avg_main:.6f} | ortho {avg_ortho:.6f} | distill {avg_distill:.6f}"
            )

        # 阶段评估
        student.eval()
        eval_model = student_ema if student_ema is not None else student
        eval_model.eval()
        stage_rates = evaluate_generation_rates(
            model=eval_model,
            classifier=classifier,
            device=device,
            target_classes=FORGET_SEQUENCE,
            n_samples=CONFIG["n_eval_samples"],
            eval_bundle=eval_bundle,
        )

        previous_rates = history_rows[-1]
        current_row = build_stage_report_line(stage_name, stage_rates)
        for cls_idx in FORGET_SEQUENCE:
            class_name = CLASS_NAMES[cls_idx]
            current_row[f"{class_name}_delta"] = float(stage_rates[cls_idx] - previous_rates[class_name])
        history_rows.append(current_row)

        for cls_idx in FORGET_SEQUENCE:
            delta = current_row[f"{CLASS_NAMES[cls_idx]}_delta"]
            print(
                f"  {CLASS_NAMES[cls_idx]:>10}: {stage_rates[cls_idx]:6.2f}% | "
                f"delta vs prev stage {delta:+6.2f}%"
            )

        # 保存模型和可视化素材
        ckpt_path = checkpoints_dir / f"cfm_cifar_{mode}_stage{stage_idx + 1}.pt"
        save_cfm_state_dict(eval_model, ckpt_path)
        checkpoint_paths[f"stage{stage_idx + 1}"] = str(ckpt_path)
        print(f"[Checkpoint] 已保存阶段权重: {ckpt_path}")
        if student_ema is not None:
            raw_ckpt_path = checkpoints_dir / f"cfm_cifar_{mode}_stage{stage_idx + 1}_raw.pt"
            save_cfm_state_dict(student, raw_ckpt_path)
            checkpoint_paths[f"stage{stage_idx + 1}_raw"] = str(raw_ckpt_path)

        single_stage_panels.append(render_single_samples(eval_model))

        quiver_path = figures_dir / f"cifar_{mode}_stage{stage_idx + 1}_field_{CLASS_NAMES[forget_class]}.png"
        save_conditional_quiver_figure(
            model=eval_model,
            class_idx=forget_class,
            data_loader=field_vis_loader,
            device=device,
            mode_name=mode_name,
            stage_name=stage_name,
            save_path=quiver_path,
        )
        field_figure_paths[f"stage{stage_idx + 1}_{CLASS_NAMES[forget_class]}"] = str(quiver_path)

    final_eval_model = student_ema if use_stage_ema and student_ema is not None else student
    final_ckpt_path = checkpoints_dir / f"cfm_cifar_{mode}_final.pt"
    save_cfm_state_dict(final_eval_model, final_ckpt_path)
    checkpoint_paths["final"] = str(final_ckpt_path)
    print(f"[Checkpoint] 已保存最终权重: {final_ckpt_path}")
    if use_stage_ema and student_ema is not None:
        final_raw_ckpt_path = checkpoints_dir / f"cfm_cifar_{mode}_final_raw.pt"
        save_cfm_state_dict(student, final_raw_ckpt_path)
        checkpoint_paths["final_raw"] = str(final_raw_ckpt_path)

    stage_single_sample_paths = save_stage_single_sample_bank(
        figures_dir=figures_dir,
        stage_panels=single_stage_panels,
        class_order=FORGET_SEQUENCE,
    )

    summary_path = summaries_dir / f"cifar_{mode}_history.json"
    save_history_json(history_rows, summary_path)
    print_table3_style(history_rows, mode_name)

    return {
        "mode": mode,
        "history": history_rows,
        "checkpoint_paths": checkpoint_paths,
        "stage_single_sample_paths": stage_single_sample_paths,
        "field_figures": field_figure_paths,
        "summary_path": str(summary_path),
    }


# In[10]:


# 9. 顶层 orchestrator：准备环境并串联 naive/temporal 两条实验线
# --------------------------------------------------
# main 只做三件事：
# 1) 数据与路径检查
# 2) 构建（或加载）冻结分类器 + 特征质心
# 3) 按配置执行 run_sequential_experiment
# --------------------------------------------------


def run_preflight_check(device: torch.device) -> Dict[str, object]:
    """运行前自检：检查路径、数据完整性、样本规格和关键超参数。"""
    report: Dict[str, object] = {}
    report["device"] = str(device)
    report["torch_version"] = str(torch.__version__)
    report["torch_cuda_version"] = str(torch.version.cuda)

    # 1) 关键路径检查
    pretrain_path = Path(CONFIG["pretrain_model_path"])
    report["pretrain_config_path"] = str(pretrain_path)
    report["pretrain_exists"] = pretrain_path.exists()
    if not report["pretrain_exists"]:
        print(
            "[Preflight] 未在 CONFIG['pretrain_model_path'] 找到预训练权重；"
            "将先执行 CFM 预训练并保存该权重，再进入连续遗忘阶段。"
        )

    output_root = Path(CONFIG["output_root"])
    output_root.mkdir(parents=True, exist_ok=True)
    report["output_root"] = str(output_root)

    # 2) CIFAR-10 原始批文件完整性检查
    cifar_raw = Path(CONFIG["data_root"]) / "cifar-10-batches-py"
    required_files = [
        "batches.meta",
        "data_batch_1",
        "data_batch_2",
        "data_batch_3",
        "data_batch_4",
        "data_batch_5",
        "test_batch",
    ]
    missing = [name for name in required_files if not (cifar_raw / name).exists()]
    report["cifar_raw_missing"] = missing
    if missing:
        print(
            "[Preflight] CIFAR-10 原始文件不完整，将在主流程 download=True 时自动补齐。"
            " 缺失文件: " + ", ".join(missing)
        )

    # 3) 可加载性与数据格式检查
    tf = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5)),
    ])
    trainset_probe = datasets.CIFAR10(
        root=CONFIG["data_root"],
        train=True,
        download=True,
        transform=tf,
    )
    testset_probe = datasets.CIFAR10(
        root=CONFIG["data_root"],
        train=False,
        download=True,
        transform=tf,
    )

    report["train_size"] = len(trainset_probe)
    report["test_size"] = len(testset_probe)
    if len(trainset_probe) != 50000 or len(testset_probe) != 10000:
        raise RuntimeError(
            f"CIFAR-10 样本数异常: train={len(trainset_probe)}, test={len(testset_probe)}"
        )

    x0, y0 = trainset_probe[0]
    report["sample_shape"] = tuple(x0.shape)
    report["sample_dtype"] = str(x0.dtype)
    report["sample_label"] = int(y0)
    if tuple(x0.shape) != tuple(CONFIG["data_shape"]):
        raise RuntimeError(
            f"样本形状与 CONFIG['data_shape'] 不一致: {tuple(x0.shape)} vs {CONFIG['data_shape']}"
        )

    targets = np.asarray(trainset_probe.targets)
    uniq = np.unique(targets)
    report["label_min"] = int(uniq.min())
    report["label_max"] = int(uniq.max())
    report["num_unique_labels"] = int(uniq.size)
    if uniq.min() != 0 or uniq.max() != 9 or uniq.size != 10:
        raise RuntimeError(
            f"标签空间异常: min={uniq.min()}, max={uniq.max()}, unique={uniq.size}"
        )

    # 4) 关键超参数的基本合法性检查
    if CONFIG["epochs_per_stage"] <= 0:
        raise ValueError("CONFIG['epochs_per_stage'] 必须 > 0")
    if CONFIG["batch_size"] <= 0:
        raise ValueError("CONFIG['batch_size'] 必须 > 0")
    if CONFIG["pretrain_epochs"] <= 0:
        raise ValueError("CONFIG['pretrain_epochs'] 必须 > 0")
    if CONFIG["pretrain_lr"] <= 0:
        raise ValueError("CONFIG['pretrain_lr'] 必须 > 0")
    if CONFIG["model_num_channels"] <= 0:
        raise ValueError("CONFIG['model_num_channels'] 必须 > 0")
    if get_total_condition_classes() < CONFIG["num_classes"]:
        raise ValueError("条件标签总数不能小于真实类别数")
    if CONFIG.get("model_num_res_blocks", 0) <= 0:
        raise ValueError("CONFIG['model_num_res_blocks'] 必须 > 0")
    if CONFIG.get("model_num_heads", 0) <= 0:
        raise ValueError("CONFIG['model_num_heads'] 必须 > 0")
    if not (0.0 <= CONFIG["pretrain_warmup_ratio"] < 1.0):
        raise ValueError("CONFIG['pretrain_warmup_ratio'] 必须在 [0, 1) 区间")
    if not (0.0 < CONFIG["pretrain_ema_decay"] < 1.0):
        raise ValueError("CONFIG['pretrain_ema_decay'] 必须在 (0, 1) 区间")
    if CONFIG["pretrain_eval_batches"] <= 0:
        raise ValueError("CONFIG['pretrain_eval_batches'] 必须 > 0")
    if CONFIG.get("pretrain_grad_accum_steps", 1) <= 0:
        raise ValueError("CONFIG['pretrain_grad_accum_steps'] 必须 > 0")
    if CONFIG.get("pretrain_checkpoint_every", 1) <= 0:
        raise ValueError("CONFIG['pretrain_checkpoint_every'] 必须 > 0")
    if CONFIG.get("pretrain_early_stop_patience", 0) < 0:
        raise ValueError("CONFIG['pretrain_early_stop_patience'] 必须 >= 0")
    if CONFIG.get("pretrain_early_stop_min_epochs", 0) < 0:
        raise ValueError("CONFIG['pretrain_early_stop_min_epochs'] 必须 >= 0")
    if CONFIG.get("pretrain_early_stop_min_improvement", 0.0) < 0.0:
        raise ValueError("CONFIG['pretrain_early_stop_min_improvement'] 必须 >= 0")
    if CONFIG.get("pretrain_target_opt_steps_gpu", 1) <= 0 or CONFIG.get("pretrain_target_opt_steps_cpu", 1) <= 0:
        raise ValueError("CONFIG['pretrain_target_opt_steps_gpu'] 和 CONFIG['pretrain_target_opt_steps_cpu'] 必须 > 0")
    if CONFIG.get("pretrain_max_epochs_gpu", 1) <= 0 or CONFIG.get("pretrain_max_epochs_cpu", 1) <= 0:
        raise ValueError("CONFIG['pretrain_max_epochs_gpu'] 和 CONFIG['pretrain_max_epochs_cpu'] 必须 > 0")
    if CONFIG.get("batch_size_gpu", 1) <= 0 or CONFIG.get("batch_size_cpu", 1) <= 0:
        raise ValueError("CONFIG['batch_size_gpu'] 和 CONFIG['batch_size_cpu'] 必须 > 0")
    if CONFIG.get("pretrain_batch_size_gpu", 1) <= 0 or CONFIG.get("pretrain_batch_size_cpu", 1) <= 0:
        raise ValueError("CONFIG['pretrain_batch_size_gpu'] 和 CONFIG['pretrain_batch_size_cpu'] 必须 > 0")
    if CONFIG.get("sample_time_steps_eval", 1) < 2 or CONFIG.get("sample_time_steps_vis", 1) < 2:
        raise ValueError("采样时间网格至少需要 2 个时间点")
    if CONFIG.get("single_seed_candidates", 0) <= 0:
        raise ValueError("CONFIG['single_seed_candidates'] 必须 > 0")
    if CONFIG.get("single_vis_generation_strategy", "eval") not in {"eval", "vis"}:
        raise ValueError("CONFIG['single_vis_generation_strategy'] 必须是 eval 或 vis")
    if CONFIG.get("single_vis_resample", "lanczos").lower() not in {"nearest", "bilinear", "bicubic", "lanczos"}:
        raise ValueError("CONFIG['single_vis_resample'] 不在支持列表")
    if not (0.0 <= CONFIG.get("pretrain_label_dropout", 0.0) < 1.0):
        raise ValueError("CONFIG['pretrain_label_dropout'] 必须在 [0, 1) 区间")
    if CONFIG.get("pretrain_eval_samples_per_class", 0) <= 0 or CONFIG.get("pretrain_vis_samples_per_class", 0) <= 0:
        raise ValueError("预训练评估/可视化的每类样本数必须 > 0")
    if not (0.0 <= CONFIG.get("train_random_flip_p", 0.0) <= 1.0):
        raise ValueError("CONFIG['train_random_flip_p'] 必须在 [0, 1] 区间")
    if CONFIG.get("sequential_use_ema", False) and not (0.0 < CONFIG.get("sequential_ema_decay", 0.0) < 1.0):
        raise ValueError("CONFIG['sequential_ema_decay'] 必须在 (0, 1) 区间")
    if CONFIG.get("device_preference", "cuda") not in {"cuda", "gpu", "cpu", "auto"}:
        raise ValueError("CONFIG['device_preference'] 必须是 cuda/gpu/cpu/auto")
    if CONFIG["beta"] < 0 or CONFIG["distill_gamma"] < 0:
        raise ValueError("CONFIG['beta'] 和 CONFIG['distill_gamma'] 必须 >= 0")

    print("[Preflight] 自检通过：路径、数据完整性、样本格式、超参数均正常。")
    return report


@torch.no_grad()
def evaluate_pretrain_flow_loss(
    model: nn.Module,
    data_loader: DataLoader,
    device: torch.device,
    max_batches: int,
) -> float:
    """在固定批次数上评估 CFM 回归损失，作为预训练收敛信号。"""
    fm = ConditionalFlowMatcher(sigma=CONFIG["sigma"])
    was_training = model.training
    model.eval()

    loss_meter = 0.0
    count = 0
    for batch_idx, (x_real, y) in enumerate(data_loader):
        if batch_idx >= max_batches:
            break
        x_real, y = x_real.to(device), y.to(device)
        x_noise = torch.randn_like(x_real)
        t, xt, ut = fm.sample_location_and_conditional_flow(x_noise, x_real)
        vt = model(t, xt, y)
        loss_meter += per_sample_mse(vt, ut).mean().item()
        count += 1

    if was_training:
        model.train()
    else:
        model.eval()
    return float(loss_meter / max(count, 1))


def infer_pretrain_epoch_offset(save_path: Path, summaries_dir: Path) -> int:
    """推断已有预训练轮次上限，用于续训命名避免与旧 checkpoint 冲突。"""
    max_epoch = 0

    history_json_path = summaries_dir / "pretrain_history.json"
    if history_json_path.exists():
        try:
            payload = json.loads(history_json_path.read_text(encoding="utf-8"))
            if isinstance(payload, list):
                for row in payload:
                    if isinstance(row, dict) and "epoch" in row:
                        max_epoch = max(max_epoch, int(float(row["epoch"])))
        except Exception:
            pass

    history_csv_path = summaries_dir / "pretrain_history.csv"
    if history_csv_path.exists():
        try:
            with open(history_csv_path, "r", encoding="utf-8", newline="") as f:
                reader = csv.DictReader(f)
                for row in reader:
                    if "epoch" in row and row["epoch"] not in {None, ""}:
                        max_epoch = max(max_epoch, int(float(row["epoch"])))
        except Exception:
            pass

    pattern = re.compile(rf"^{re.escape(save_path.stem)}_epoch(\d+)\.pt$")
    for ckpt_path in save_path.parent.glob(f"{save_path.stem}_epoch*.pt"):
        match = pattern.match(ckpt_path.name)
        if match:
            max_epoch = max(max_epoch, int(match.group(1)))

    fig_dir = summaries_dir.parent / "figures"
    fig_pattern = re.compile(r"^pretrain_epoch(\d+)\.png$")
    for fig_path in fig_dir.glob("pretrain_epoch*.png"):
        match = fig_pattern.match(fig_path.name)
        if match:
            max_epoch = max(max_epoch, int(match.group(1)))

    return max_epoch


def train_pretrained_cfm(
    train_loader: DataLoader,
    eval_loader: DataLoader,
    device: torch.device,
    save_path: Path,
    classifier: nn.Module = None,
) -> Path:
    """从零训练 CIFAR CFM 预训练模型，并将高质量权重保存到指定路径。"""
    pretrain_root = Path(CONFIG["output_root"]) / "pretrain"
    pretrain_figures_dir = pretrain_root / "figures"
    pretrain_summaries_dir = pretrain_root / "summaries"
    pretrain_figures_dir.mkdir(parents=True, exist_ok=True)
    pretrain_summaries_dir.mkdir(parents=True, exist_ok=True)

    eval_model_uses_ema = bool(CONFIG.get("pretrain_eval_use_ema", True))
    base_lr = float(CONFIG["pretrain_lr"])
    accum_steps = max(1, int(CONFIG.get("pretrain_grad_accum_steps", 1)))
    steps_per_epoch = max((len(train_loader) + accum_steps - 1) // accum_steps, 1)
    if device.type == "cuda":
        target_opt_steps = int(CONFIG.get("pretrain_target_opt_steps_gpu", 30000))
        max_epochs = int(CONFIG.get("pretrain_max_epochs_gpu", 360))
    else:
        target_opt_steps = int(CONFIG.get("pretrain_target_opt_steps_cpu", 2500))
        max_epochs = int(CONFIG.get("pretrain_max_epochs_cpu", 24))

    fm = ConditionalFlowMatcher(sigma=CONFIG["sigma"])
    use_amp = bool(CONFIG.get("pretrain_use_amp", True)) and device.type == "cuda"
    scaler: Optional[torch.amp.GradScaler] = None
    if device.type == "cuda":
        scaler = torch.amp.GradScaler("cuda", enabled=use_amp)

    quality_bundle = None
    if classifier is not None:
        quality_bundle = build_balanced_eval_bundle(
            samples_per_class=int(CONFIG.get("pretrain_eval_samples_per_class", 32)),
            device=device,
            seed=CONFIG["seed"] + 2024,
        )
    vis_bundle = build_balanced_eval_bundle(
        samples_per_class=int(CONFIG.get("pretrain_vis_samples_per_class", 10)),
        device=device,
        seed=CONFIG["seed"] + 2048,
    )

    model = instantiate_cfm_model(device)
    model.train()
    ema_model = instantiate_cfm_model(device)
    ema_model.load_state_dict(model.state_dict())
    ema_model = freeze_module(ema_model)
    optimizer = torch.optim.AdamW(
        model.parameters(),
        lr=base_lr,
        weight_decay=CONFIG["pretrain_weight_decay"],
        betas=(0.9, 0.999),
    )

    label_dropout = float(CONFIG.get("pretrain_label_dropout", 0.0))
    visualize_every = max(1, int(CONFIG.get("pretrain_visualize_every", 5)))
    eval_batches = int(CONFIG.get("pretrain_eval_batches", 120))
    checkpoint_every = int(CONFIG.get("pretrain_checkpoint_every", 5))
    early_stop_enabled = bool(CONFIG.get("pretrain_early_stop_enabled", True))
    early_stop_patience = int(CONFIG.get("pretrain_early_stop_patience", 0))
    early_stop_min_epochs = int(CONFIG.get("pretrain_early_stop_min_epochs", 0))
    early_stop_min_improvement = float(CONFIG.get("pretrain_early_stop_min_improvement", 1e-4))
    min_lr_scale = 0.08
    ema_decay = float(CONFIG.get("pretrain_ema_decay", 0.999))
    best_ckpt_path = save_path.with_name(f"{save_path.stem}_best.pt")

    history_rows: List[Dict[str, float]] = []
    best_model_state_dict: Optional[Dict[str, torch.Tensor]] = None
    best_eval = float("inf")
    best_epoch = -1
    best_cond_acc = float("-inf")
    no_improve_epochs = 0
    local_epochs = int(CONFIG["pretrain_epochs"])
    epochs_from_target = int(np.ceil(target_opt_steps / steps_per_epoch))
    local_epochs = max(local_epochs, epochs_from_target)
    local_epochs = min(local_epochs, max_epochs)
    display_total_epochs = local_epochs
    total_opt_steps = max(local_epochs * steps_per_epoch, 1)
    warmup_steps = max(1, int(CONFIG.get("pretrain_warmup_ratio", 0.05) * total_opt_steps))
    optimizer_step = 0
    lr_now = base_lr
    best_model_state_dict = copy.deepcopy((ema_model if eval_model_uses_ema else model).state_dict())

    print(
        f"[Pretrain Plan] local_epochs={local_epochs} | "
        f"steps_per_epoch={steps_per_epoch} | total_opt_steps={total_opt_steps} | "
        f"target_opt_steps={target_opt_steps} | accum_steps={accum_steps} | display_epoch_end={display_total_epochs}"
    )

    def set_lr_by_step(step_idx: int) -> float:
        if step_idx < warmup_steps:
            scale = float(step_idx + 1) / float(warmup_steps)
        else:
            denom = max(1, total_opt_steps - warmup_steps)
            progress = float(step_idx - warmup_steps) / float(denom)
            cosine_scale = 0.5 * (1.0 + np.cos(np.pi * progress))
            scale = min_lr_scale + (1.0 - min_lr_scale) * cosine_scale
        lr = base_lr * scale
        for pg in optimizer.param_groups:
            pg["lr"] = lr
        return lr

    for epoch in range(local_epochs):
        model.train()
        loss_meter = 0.0
        step_count = 0
        optimizer.zero_grad(set_to_none=True)

        for batch_idx, (x_real, y) in enumerate(train_loader):
            x_real, y = x_real.to(device), y.to(device)
            y_cond = apply_label_condition_dropout(y, label_dropout)

            if batch_idx % accum_steps == 0:
                lr_now = set_lr_by_step(optimizer_step)

            x_noise = torch.randn_like(x_real)
            t, xt, ut = fm.sample_location_and_conditional_flow(x_noise, x_real)

            with torch.amp.autocast(device_type=device.type, enabled=use_amp):
                vt = model(t, xt, y_cond)
                loss = per_sample_mse(vt, ut).mean()
                scaled_loss = loss / accum_steps

            if use_amp and scaler is not None:
                scaler.scale(scaled_loss).backward()
            else:
                scaled_loss.backward()

            should_step = ((batch_idx + 1) % accum_steps == 0) or ((batch_idx + 1) == len(train_loader))
            if should_step:
                if use_amp and scaler is not None:
                    scaler.unscale_(optimizer)
                torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=CONFIG["grad_clip_norm"])

                if use_amp and scaler is not None:
                    scaler.step(optimizer)
                    scaler.update()
                else:
                    optimizer.step()

                optimizer.zero_grad(set_to_none=True)
                update_ema_model(model, ema_model, ema_decay)
                optimizer_step += 1

            loss_meter += loss.item()
            step_count += 1

        train_loss = loss_meter / max(step_count, 1)
        eval_model = ema_model if eval_model_uses_ema else model
        eval_loss = evaluate_pretrain_flow_loss(
            model=eval_model,
            data_loader=eval_loader,
            device=device,
            max_batches=eval_batches,
        )
        quality_metrics = None
        if classifier is not None:
            quality_metrics = evaluate_conditional_generation_quality(
                model=eval_model,
                classifier=classifier,
                device=device,
                samples_per_class=int(CONFIG.get("pretrain_eval_samples_per_class", 32)),
                eval_bundle=quality_bundle,
            )

        is_best = False
        if quality_metrics is not None:
            cond_acc = float(quality_metrics["accuracy"])
            better_acc = cond_acc > best_cond_acc + 1e-6
            tie_better_loss = abs(cond_acc - best_cond_acc) <= 1e-6 and eval_loss < best_eval
            if better_acc or tie_better_loss:
                best_cond_acc = cond_acc
                best_eval = float(eval_loss)
                best_epoch = epoch + 1
                best_model_state_dict = copy.deepcopy(eval_model.state_dict())
                save_cfm_state_dict(eval_model, best_ckpt_path)
                is_best = True
        elif eval_loss < best_eval:
            best_eval = float(eval_loss)
            best_epoch = epoch + 1
            best_model_state_dict = copy.deepcopy(eval_model.state_dict())
            save_cfm_state_dict(eval_model, best_ckpt_path)
            is_best = True

        if early_stop_enabled:
            if is_best:
                no_improve_epochs = 0
            else:
                current_gap = 0.0
                if quality_metrics is not None and best_cond_acc > float("-inf"):
                    current_gap = float(best_cond_acc - float(quality_metrics["accuracy"]))
                else:
                    current_gap = float(eval_loss - best_eval)
                if current_gap >= early_stop_min_improvement:
                    no_improve_epochs += 1
                else:
                    no_improve_epochs = 0

        display_epoch = epoch + 1
        row: Dict[str, float] = {
            "epoch": float(display_epoch),
            "train_loss": float(train_loss),
            "eval_flow_loss": float(eval_loss),
            "lr": float(lr_now),
            "optimizer_steps": float(optimizer_step),
        }
        if quality_metrics is not None:
            row["cond_acc"] = float(quality_metrics["accuracy"])
            row["target_confidence"] = float(quality_metrics["target_confidence"])
            for class_name, class_acc in quality_metrics["per_class_accuracy"].items():
                row[f"acc_{class_name}"] = float(class_acc)
        history_rows.append(row)

        if is_best or ((epoch + 1) % visualize_every == 0):
            grid = make_class_grid(
                eval_model,
                device,
                z0=vis_bundle["z0"].clone(),
                samples_per_class=int(CONFIG.get("pretrain_vis_samples_per_class", 10)),
                strategy="vis",
            )
            save_grid_image(
                grid,
                pretrain_figures_dir / f"pretrain_epoch{display_epoch:03d}.png",
                upscale=CONFIG.get("figure_strip_upscale", 1),
            )
            if is_best:
                save_grid_image(
                    grid,
                    pretrain_figures_dir / "pretrain_best.png",
                    upscale=CONFIG.get("figure_strip_upscale", 1),
                )

        if ((epoch + 1) % checkpoint_every == 0) or (epoch + 1 == local_epochs):
            epoch_ckpt = save_path.with_name(f"{save_path.stem}_epoch{display_epoch:03d}.pt")
            save_cfm_state_dict(eval_model, epoch_ckpt)

        log_line = (
            f"[Pretrain] epoch {display_epoch:03d}/{display_total_epochs:03d}"
            f" (local {epoch + 1:02d}/{local_epochs:02d}) | "
            f"train_loss {train_loss:.6f} | eval_loss {eval_loss:.6f}"
        )
        if quality_metrics is not None:
            log_line += (
                f" | cond_acc {quality_metrics['accuracy']:.2f}%"
                f" | target_conf {quality_metrics['target_confidence']:.2f}%"
            )
        log_line += (
            f" | best_eval {best_eval:.6f}@E{best_epoch:03d}"
            f" | lr {lr_now:.2e} | opt_steps {optimizer_step}/{total_opt_steps}"
        )
        if best_cond_acc > float("-inf"):
            log_line += f" | best_cond_acc {best_cond_acc:.2f}%"
        if early_stop_enabled:
            log_line += f" | early_stop_counter {no_improve_epochs}/{max(early_stop_patience, 0)}"
        print(log_line)

        if early_stop_enabled and early_stop_patience > 0:
            enough_epochs = display_epoch >= early_stop_min_epochs
            if enough_epochs and no_improve_epochs >= early_stop_patience:
                print(
                    f"[Pretrain] 触发早停：连续 {no_improve_epochs} 个 epoch 无显著提升，"
                    f"在 epoch {display_epoch} 提前结束。"
                )
                break

    save_path.parent.mkdir(parents=True, exist_ok=True)
    if history_rows:
        history_json_path = pretrain_summaries_dir / "pretrain_history.json"
        save_history_json(history_rows, history_json_path)
        history_csv_path = pretrain_summaries_dir / "pretrain_history.csv"
        fieldnames = sorted({key for row in history_rows for key in row.keys()})
        with open(history_csv_path, "w", newline="", encoding="utf-8") as f:
            writer = csv.DictWriter(f, fieldnames=fieldnames)
            writer.writeheader()
            for row in history_rows:
                writer.writerow(row)
        print(f"[Summary] 已保存预训练日志: {history_csv_path}")

    if best_model_state_dict is not None:
        save_cfm_state_dict(load_cfm_model(str(best_ckpt_path), device), save_path)
        if best_epoch > 0:
            best_epoch_path = save_path.with_name(f"{save_path.stem}_best_epoch{int(best_epoch):03d}.pt")
            save_cfm_state_dict(load_cfm_model(str(best_ckpt_path), device), best_epoch_path)
            print(f"[Pretrain] 已额外保存累计轮次命名的最优权重: {best_epoch_path}")
    else:
        fallback_model = ema_model if eval_model_uses_ema else model
        save_cfm_state_dict(fallback_model, save_path)

    print(f"[Pretrain] 预训练完成，权重已保存: {save_path}")
    print(f"[Pretrain] 最优检查点: {best_ckpt_path} | best_eval_loss={best_eval:.6f}")
    return save_path


def resolve_or_prepare_pretrain_checkpoint(
    train_loader: DataLoader,
    eval_loader: DataLoader,
    device: torch.device,
    classifier: nn.Module = None,
) -> Path:
    """按“先预训练、后遗忘”的流程准备起始权重。"""
    configured_path = Path(CONFIG["pretrain_model_path"]).expanduser()
    archive_candidate = Path(CONFIG["output_root"]) / "pretrain" / "cfm_cifar_pretrain.pt"
    force_retrain = bool(CONFIG.get("force_retrain_pretrain", False))
    run_pretrain_first = bool(CONFIG.get("run_pretrain_first", True))

    # 兼容“历史权重已归档到 outputs 目录”的场景：主路径不存在时自动回退。
    if not configured_path.exists() and archive_candidate.exists() and not force_retrain:
        print(
            f"[Pretrain] 主路径未找到权重，自动回退到归档权重: {archive_candidate}"
        )
        configured_path = archive_candidate
        CONFIG["pretrain_model_path"] = str(configured_path)

    if configured_path.exists() and not force_retrain:
        _ = load_cfm_model(str(configured_path), device)
        print(f"[Pretrain] 使用已存在的预训练权重: {configured_path}")
        return configured_path

    if run_pretrain_first:
        if configured_path.exists() and force_retrain:
            print(f"[Pretrain] force_retrain_pretrain=True，覆盖重训: {configured_path}")
        else:
            print(f"[Pretrain] 未找到预训练权重，开始先训后忘流程，目标保存: {configured_path}")

        resolved = train_pretrained_cfm(
            train_loader=train_loader,
            eval_loader=eval_loader,
            device=device,
            save_path=configured_path,
            classifier=classifier,
        )

        # 额外保存一份标准归档路径，便于后续重复实验直接复用。
        archive_path = Path(CONFIG["output_root"]) / "pretrain" / "cfm_cifar_pretrain.pt"
        archive_path.parent.mkdir(parents=True, exist_ok=True)
        if archive_path.resolve() != resolved.resolve():
            torch.save(torch.load(resolved, map_location="cpu"), archive_path)
            print(f"[Pretrain] 已同步归档预训练权重: {archive_path}")

        CONFIG["pretrain_model_path"] = str(resolved)
        return resolved

    raise FileNotFoundError(
        "未找到预训练 CFM 权重，且已关闭 run_pretrain_first。"
        f" 请先生成并保存到: {configured_path}"
    )


def resolve_runtime_device() -> torch.device:
    """按配置解析运行设备；默认强制使用 GPU。"""
    pref = str(CONFIG.get("device_preference", "cuda")).lower()
    allow_cpu = bool(CONFIG.get("allow_cpu_fallback", False))

    if pref in {"cuda", "gpu"}:
        if torch.cuda.is_available():
            device = torch.device("cuda")
            cuda_idx = torch.cuda.current_device()
            cuda_name = torch.cuda.get_device_name(cuda_idx)
            total_gb = torch.cuda.get_device_properties(cuda_idx).total_memory / (1024 ** 3)
            if hasattr(torch.backends, "cudnn"):
                torch.backends.cudnn.benchmark = True
                if hasattr(torch.backends.cudnn, "allow_tf32"):
                    torch.backends.cudnn.allow_tf32 = True
            if hasattr(torch.backends, "cuda") and hasattr(torch.backends.cuda, "matmul"):
                torch.backends.cuda.matmul.allow_tf32 = True
            print(
                f"使用设备: cuda | GPU={cuda_name} | 显存={total_gb:.1f}GB | "
                f"torch_cuda={torch.version.cuda}"
            )
            return device

        msg = (
            "当前配置要求使用 GPU，但 torch.cuda.is_available() 为 False。\n"
            f"- torch={torch.__version__}\n"
            f"- torch_cuda={torch.version.cuda}\n"
            "- 请检查 NVIDIA 驱动是否正常加载（nvidia-smi）\n"
            "- 若驱动版本偏旧，请安装与驱动匹配的 PyTorch CUDA 版本（例如 cu128）\n"
            "  参考命令：\n"
            "  pip uninstall -y torch torchvision torchaudio && "
            "pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu128"
        )
        if not allow_cpu:
            raise RuntimeError(msg)

        print("[Warning] " + msg + " 已按 allow_cpu_fallback=True 回退到 CPU。")
        return torch.device("cpu")

    if pref == "cpu":
        raise RuntimeError(
            "当前脚本已固定为 GPU 训练流程，不支持 device_preference='cpu'。"
            " 请改回 CONFIG['device_preference'] = 'cuda' 并确保服务器 GPU 可用。"
        )

    if torch.cuda.is_available():
        print("使用设备: cuda（auto）")
        return torch.device("cuda")

    raise RuntimeError(
        "当前脚本要求全程 GPU 运行，但未检测到可用 CUDA 设备。"
        " 请检查服务器上的 NVIDIA 驱动、CUDA 版 PyTorch 与 `nvidia-smi` 状态。"
    )


def resolve_loader_kwargs(device: torch.device) -> Dict[str, object]:
    """按设备给出推荐 DataLoader 配置。"""
    if device.type == "cuda":
        batch_size = int(CONFIG.get("batch_size_gpu", CONFIG["batch_size"]))
        num_workers = int(CONFIG.get("num_workers_gpu", CONFIG.get("num_workers", 2)))
        pin_memory = True
    else:
        batch_size = int(CONFIG.get("batch_size_cpu", CONFIG["batch_size"]))
        num_workers = int(CONFIG.get("num_workers_cpu", CONFIG.get("num_workers", 2)))
        pin_memory = bool(CONFIG.get("pin_memory", True))

    num_workers = max(num_workers, 0)
    kwargs = {
        "batch_size": max(batch_size, 1),
        "num_workers": num_workers,
        "pin_memory": pin_memory,
    }
    if num_workers > 0:
        kwargs["persistent_workers"] = True
        kwargs["prefetch_factor"] = 4 if device.type == "cuda" else 2
    return kwargs


def main() -> Dict[str, Dict[str, object]]:
    set_seed(CONFIG["seed"])
    if hasattr(torch, "set_float32_matmul_precision"):
        torch.set_float32_matmul_precision("high")

    device = resolve_runtime_device()

    output_root = Path(CONFIG["output_root"])
    output_root.mkdir(parents=True, exist_ok=True)
    summaries_root = output_root / "summaries"
    paper_figures_root = output_root / "paper_figures"
    summaries_root.mkdir(parents=True, exist_ok=True)
    paper_figures_root.mkdir(parents=True, exist_ok=True)

    # 运行前自检：路径、数据完整性、数据格式、关键超参数。
    preflight_report = run_preflight_check(device)

    # 准备 CIFAR-10 训练/测试数据
    tf_eval = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5)),
    ])
    tf_train = transforms.Compose([
        transforms.RandomHorizontalFlip(p=float(CONFIG.get("train_random_flip_p", 0.5))),
        transforms.ToTensor(),
        transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5)),
    ])
    tf_classifier_train = tf_train if CONFIG.get("classifier_use_augmentation", True) else tf_eval
    tf_pretrain = tf_train
    tf_unlearning = tf_train if CONFIG.get("align_pretrain_and_unlearning_augmentation", True) else tf_eval

    pretrain_trainset = datasets.CIFAR10(
        root=CONFIG["data_root"],
        train=True,
        download=True,
        transform=tf_pretrain,
    )
    trainset = datasets.CIFAR10(
        root=CONFIG["data_root"],
        train=True,
        download=True,
        transform=tf_unlearning,
    )
    trainset_eval = datasets.CIFAR10(
        root=CONFIG["data_root"],
        train=True,
        download=True,
        transform=tf_eval,
    )
    trainset_cls = datasets.CIFAR10(
        root=CONFIG["data_root"],
        train=True,
        download=True,
        transform=tf_classifier_train,
    )
    testset = datasets.CIFAR10(
        root=CONFIG["data_root"],
        train=False,
        download=True,
        transform=tf_eval,
    )

    loader_kwargs = resolve_loader_kwargs(device)
    pretrain_loader_kwargs = dict(loader_kwargs)
    if device.type == "cuda":
        pretrain_loader_kwargs["batch_size"] = int(CONFIG.get("pretrain_batch_size_gpu", loader_kwargs["batch_size"]))
    else:
        pretrain_loader_kwargs["batch_size"] = int(CONFIG.get("pretrain_batch_size_cpu", loader_kwargs["batch_size"]))

    print(
        f"[DataLoader] train_bs={loader_kwargs['batch_size']} | pretrain_bs={pretrain_loader_kwargs['batch_size']} | "
        f"num_workers={loader_kwargs['num_workers']} | pin_memory={loader_kwargs['pin_memory']}"
    )
    pretrain_loader = DataLoader(pretrain_trainset, shuffle=True, drop_last=True, **pretrain_loader_kwargs)
    train_loader = DataLoader(trainset, shuffle=True, drop_last=True, **loader_kwargs)
    train_loader_eval = DataLoader(trainset_eval, shuffle=False, drop_last=False, **loader_kwargs)
    train_loader_cls = DataLoader(trainset_cls, shuffle=True, drop_last=False, **loader_kwargs)
    test_loader = DataLoader(testset, shuffle=False, drop_last=False, **loader_kwargs)

    # 1) 先准备分类器，让预训练阶段也能挑到更强的 CIFAR10 条件生成基座；
    #    这一步只服务于预训练质量提升，不改变后续 CFMU Temporal Shield 的定义。
    classifier = build_or_load_classifier(
        path=CONFIG["classifier_path"],
        device=device,
        train_loader=train_loader_cls,
        test_loader=test_loader,
        epochs=CONFIG["classifier_epochs"],
        lr=CONFIG["classifier_lr"],
    )
    feature_net = freeze_module(copy.deepcopy(classifier).to(device))

    # Step-A 再准备 CFM 预训练权重（存在则复用，不存在则先训练并保存）。
    # 这里强化的是“生成模型基座”，而连续遗忘阶段的能量权重 + OT 正交约束保持论文一致。
    pretrain_path = resolve_or_prepare_pretrain_checkpoint(
        train_loader=pretrain_loader,
        eval_loader=train_loader_eval,
        device=device,
        classifier=classifier,
    )
    preflight_report["resolved_pretrain_path"] = str(pretrain_path)
    preflight_report["pretrain_exists_after_resolve"] = pretrain_path.exists()

    single_seed_map = resolve_single_seed_bank(
        pretrain_model_path=str(pretrain_path),
        classifier=classifier,
        device=device,
    )
    preflight_report["single_seed_map"] = {CLASS_NAMES[k]: int(v) for k, v in single_seed_map.items()}

    # 2) 计算每类冻结特征质心（仅一次）
    centroids = compute_class_feature_centroids(
        feature_net=feature_net,
        data_loader=train_loader_eval,
        device=device,
        cache_path=CONFIG["centroid_cache_path"],
    )

    # 3) 运行 Naive 与 Temporal 两条线
    results: Dict[str, Dict[str, object]] = {}
    if CONFIG["run_naive"]:
        results["naive"] = run_sequential_experiment(
            mode="naive",
            teacher_checkpoint_path=str(pretrain_path),
            classifier=classifier,
            feature_net=feature_net,
            centroids=centroids,
            train_loader=train_loader,
            field_vis_loader=train_loader_eval,
            single_seed_map=single_seed_map,
            device=device,
            output_dir=output_root / "naive",
        )

    if CONFIG["run_temporal"]:
        results["temporal"] = run_sequential_experiment(
            mode="temporal",
            teacher_checkpoint_path=str(pretrain_path),
            classifier=classifier,
            feature_net=feature_net,
            centroids=centroids,
            train_loader=train_loader,
            field_vis_loader=train_loader_eval,
            single_seed_map=single_seed_map,
            device=device,
            output_dir=output_root / "temporal",
        )

    merged_summary_path = summaries_root / "cifar_temporal_full_summary.json"
    merged_payload = {}
    for key, value in results.items():
        merged_payload[key] = value["history"]

    with open(merged_summary_path, "w", encoding="utf-8") as f:
        json.dump(merged_payload, f, indent=2, ensure_ascii=False)

    paper_figure_entries = {}
    if "naive" in results:
        naive_schematic_path = paper_figures_root / "Naive_Sequential.png"
        save_sequential_unlearning_schematic(mode="naive", save_path=naive_schematic_path)
        paper_figure_entries["naive_sequential_schematic"] = str(naive_schematic_path)
    if "temporal" in results:
        temporal_schematic_path = paper_figures_root / "CFMU_Temporal.png"
        save_sequential_unlearning_schematic(mode="temporal", save_path=temporal_schematic_path)
        paper_figure_entries["cfmu_temporal_schematic"] = str(temporal_schematic_path)

    # 4) 论文图与量化：反弹抑制图 + Table 3 量化表
    if "naive" in results and "temporal" in results:
        rebound_fig_path = paper_figures_root / "cifar_unlearning_degradation_suppression.png"
        save_unlearning_degradation_figure(
            naive_history=results["naive"]["history"],
            temporal_history=results["temporal"]["history"],
            save_path=rebound_fig_path,
        )

        table3_root = output_root / "table3_quantitative"
        table3_paths = save_cifar_table3_outputs(
            naive_history=results["naive"]["history"],
            temporal_history=results["temporal"]["history"],
            output_dir=table3_root,
        )
        results["table3_cifar"] = table3_paths
        paper_figure_entries["unlearning_degradation_suppression"] = str(rebound_fig_path)

    if paper_figure_entries:
        results["paper_figures"] = paper_figure_entries

    results["preflight"] = preflight_report

    print(f"\n[Summary] 全部实验结果已保存: {merged_summary_path}")
    print("[Summary] 权重文件目录：<output_root>/<mode>/checkpoints")
    print("[Summary] 论文图目录：<output_root>/paper_figures")
    print("[Summary] Table 3 量化表目录：<output_root>/table3_quantitative")
    return results


# In[11]:


# 10. 运行实验（最后一格）
# --------------------------------------------------
# 运行前检查（建议逐项确认）：
# 1) 先预训练再遗忘：若 CONFIG['pretrain_model_path'] 不存在，程序会先训练 CFM 并保存到该路径。
# 2) 默认强制 GPU（device_preference='cuda' 且 allow_cpu_fallback=False），若环境不满足会直接报错并给出 CUDA 版本修复提示。
# 3) 如需每次都重训预训练模型，可设置 CONFIG['force_retrain_pretrain'] = True。
# 4) 预训练期间会结合条件命中率与 flow-loss 选择 best checkpoint，并把最优权重写回 pretrain_model_path。
# 5) 训练步数由 pretrain_target_opt_steps_* 与 pretrain_epochs 联合决定；为论文质量优先，默认会跑较长。
# 6) 当前版本已移除续训/断点续跑逻辑；若开启 force_retrain_pretrain，会直接从零开始重训预训练模型。
# 7) 输出目录 CONFIG['output_root'] 有写权限。
# 8) 若只跑论文主方法（Temporal），可关闭 naive 分支。
#
# 可选开关：
# - 只跑 Temporal：
#     CONFIG["run_naive"] = False
#     CONFIG["run_temporal"] = True
# - 只跑 Naive：
#     CONFIG["run_naive"] = True
#     CONFIG["run_temporal"] = False
# --------------------------------------------------

results = main()
results


# In[ ]:


# 运行提示：请执行上方 1~10 号单元。
# 本单元仅作占位注释，不含任何实验逻辑。


# In[ ]:


# 复盘建议：
# - 先看 cell 8 的损失构成与阶段循环。
# - 再看 cell 9 的输出文件组织和论文图生成逻辑。


# In[ ]:


# 快速调试（可选）：只跑 Temporal 主方法。
# CONFIG['run_naive'] = False
# CONFIG['run_temporal'] = True
# CONFIG['epochs_per_stage'] = 4  # 调试时可暂时降低

