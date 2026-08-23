"""结果可视化工具。"""

from __future__ import annotations

from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import torch

LEAD_NAMES = [
    "I", "II", "III", "aVR", "aVL", "aVF",
    "V1", "V2", "V3", "V4", "V5", "V6",
]


def plot_ecg_comparison(
    pred: torch.Tensor,
    target: torch.Tensor,
    sample_idx: int = 0,
    save_path: str | Path | None = None,
    fs: int = 125,
) -> plt.Figure:
    """绘制预测与真实 ECG 对比图。

    Args:
        pred:   [B, 12, L]
        target: [B, 12, L]
        sample_idx: 选择哪个样本
        save_path: 保存路径 (None 则返回 Figure)
        fs: 采样率
    """
    p = pred[sample_idx].cpu().numpy()  # [12, L]
    t = target[sample_idx].cpu().numpy()

    L = p.shape[1]
    time = np.arange(L) / fs

    fig, axes = plt.subplots(12, 1, figsize=(12, 20), sharex=True)
    for lead_idx in range(12):
        ax = axes[lead_idx]
        ax.plot(time, t[lead_idx], color="tab:blue", label="True", linewidth=0.8)
        ax.plot(time, p[lead_idx], color="tab:orange", label="Pred", linewidth=0.8, alpha=0.8)
        ax.set_ylabel(LEAD_NAMES[lead_idx], fontsize=9)
        ax.tick_params(labelsize=7)
        if lead_idx == 0:
            ax.legend(fontsize=7, loc="upper right")

    axes[-1].set_xlabel("Time (s)", fontsize=9)
    fig.suptitle("ECG Reconstruction: Predicted vs True", fontsize=12)
    plt.tight_layout()

    if save_path:
        fig.savefig(save_path, dpi=150, bbox_inches="tight")
        plt.close(fig)
    return fig


def plot_training_curves(
    history: dict[str, list[float]],
    save_path: str | Path | None = None,
) -> plt.Figure:
    """绘制训练曲线。

    Args:
        history: {"train_loss": [...], "val_loss": [...], ...}
    """
    fig, axes = plt.subplots(1, 2, figsize=(12, 4))

    # Loss 曲线
    ax = axes[0]
    if "train_loss" in history:
        ax.plot(history["train_loss"], label="Train")
    if "val_loss" in history:
        ax.plot(history["val_loss"], label="Val")
    ax.set_xlabel("Epoch")
    ax.set_ylabel("Loss")
    ax.set_title("Training Loss")
    ax.legend()
    ax.set_yscale("log")

    # 学习率曲线
    ax = axes[1]
    if "lr" in history:
        ax.plot(history["lr"])
    ax.set_xlabel("Epoch")
    ax.set_ylabel("Learning Rate")
    ax.set_title("Learning Rate Schedule")

    plt.tight_layout()

    if save_path:
        fig.savefig(save_path, dpi=150, bbox_inches="tight")
        plt.close(fig)
    return fig


def plot_ppg_ecg_pair(
    ppg: torch.Tensor,
    ecg: torch.Tensor,
    sample_idx: int = 0,
    save_path: str | Path | None = None,
    fs: int = 125,
) -> plt.Figure:
    """绘制 PPG 和 ECG (II 导联) 配对信号。"""
    p = ppg[sample_idx].cpu().numpy()
    e = ecg[sample_idx].cpu().numpy()

    L = len(p)
    time = np.arange(L) / fs

    fig, axes = plt.subplots(2, 1, figsize=(12, 5), sharex=True)
    axes[0].plot(time, p, color="tab:green", linewidth=0.8)
    axes[0].set_ylabel("PPG", fontsize=10)
    axes[0].set_title("PPG Signal")

    # 找到 II 导联的索引
    lead_ii_idx = LEAD_NAMES.index("II")
    axes[1].plot(time, e[lead_ii_idx], color="tab:red", linewidth=0.8)
    axes[1].set_ylabel("ECG (II)", fontsize=10)
    axes[1].set_xlabel("Time (s)")

    plt.tight_layout()

    if save_path:
        fig.savefig(save_path, dpi=150, bbox_inches="tight")
        plt.close(fig)
    return fig
