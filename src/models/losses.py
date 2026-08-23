"""PPG2ECG 复合损失函数。

损失组成:
  L = λ₁·MSE + λ₂·DTW + λ₃·FreqLoss + λ₄·PerceptualLoss + λ₅·ContrastiveLoss

  - MSE:        逐点均方误差
  - DTW:        动态时间规整 (衡量形状相似性)
  - FreqLoss:   频域 L1 损失 (STFT 幅值谱)
  - Perceptual: 基于预训练特征的感受损失
  - Contrastive: 跨模态对比损失 (InfoNCE)
"""

from __future__ import annotations

import torch
import torch.nn as nn
import torch.nn.functional as F


class MSELoss(nn.Module):
    """逐点均方误差。"""

    def forward(self, pred: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
        return F.mse_loss(pred, target)


class DTWLoss(nn.Module):
    """软-DTW 损失 (近似可微版本)。

    使用序列均值池化后的 L1 距离作为 DTW 的快速近似。
    完整 Soft-DTW 实现可在需要时替换。

    Args:
        gamma: Soft-DTW 平滑参数 (越小越接近硬 DTW)
    """

    def __init__(self, gamma: float = 1.0):
        super().__init__()
        self.gamma = gamma

    def forward(self, pred: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
        # 快速近似: 多尺度 L1 距离
        loss = F.l1_loss(pred, target)

        # 多尺度 (不同池化窗口)
        for kernel in [5, 15, 50]:
            if pred.size(-1) >= kernel:
                p = F.avg_pool1d(pred, kernel, stride=kernel)
                t = F.avg_pool1d(target, kernel, stride=kernel)
                loss = loss + F.l1_loss(p, t)

        return loss / 4.0


class FrequencyLoss(nn.Module):
    """频域 L1 损失 (STFT 幅值谱)。"""

    def __init__(self, n_fft: int = 256, hop_length: int = 64):
        super().__init__()
        self.n_fft = n_fft
        self.hop_length = hop_length

    def forward(self, pred: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
        window = torch.hann_window(self.n_fft, device=pred.device)

        # 对每个导联计算 STFT
        total_loss = 0.0
        B, C, L = pred.shape

        for c in range(C):
            p_stft = torch.stft(
                pred[:, c], self.n_fft, self.hop_length,
                window=window, return_complex=True,
            ).abs()
            t_stft = torch.stft(
                target[:, c], self.n_fft, self.hop_length,
                window=window, return_complex=True,
            ).abs()
            total_loss = total_loss + F.l1_loss(p_stft, t_stft)

        return total_loss / C


class PerceptualLoss(nn.Module):
    """基于简单 CNN 特征的感受损失。

    使用轻量 CNN 提取特征后计算 L2 距离。
    可替换为预训练 ECG 分类器的中间层特征。
    """

    def __init__(self, ecg_leads: int = 12):
        super().__init__()
        self.feature_extractor = nn.Sequential(
            nn.Conv1d(ecg_leads, 32, 7, stride=2, padding=3),
            nn.GELU(),
            nn.Conv1d(32, 64, 5, stride=2, padding=2),
            nn.GELU(),
            nn.Conv1d(64, 64, 3, stride=2, padding=1),
            nn.GELU(),
        )
        # 冻结参数
        for p in self.feature_extractor.parameters():
            p.requires_grad = False

    @torch.no_grad()
    def extract(self, x: torch.Tensor) -> torch.Tensor:
        return self.feature_extractor(x)

    def forward(self, pred: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
        return F.mse_loss(self.extract(pred), self.extract(target))


class ContrastiveLoss(nn.Module):
    """InfoNCE 对比学习损失。

    Args:
        temperature: 温度系数
    """

    def __init__(self, temperature: float = 0.07):
        super().__init__()
        self.temperature = temperature

    def forward(
        self, z1: torch.Tensor, z2: torch.Tensor
    ) -> torch.Tensor:
        """
        Args:
            z1: [B, D]  模态1 的投影特征 (已归一化)
            z2: [B, D]  模态2 的投影特征 (已归一化)
        Returns:
            loss: InfoNCE 损失
        """
        B = z1.size(0)
        z = torch.cat([z1, z2], dim=0)  # [2B, D]
        sim = torch.mm(z, z.t()) / self.temperature  # [2B, 2B]

        # 正样本对: (i, i+B) 和 (i+B, i)
        labels = torch.arange(2 * B, device=z.device)
        labels = (labels + B) % (2 * B)

        # 屏蔽自身
        mask = torch.eye(2 * B, device=z.device).bool()
        sim.masked_fill_(mask, -1e9)

        return F.cross_entropy(sim, labels)


class ReconstructionLoss(nn.Module):
    """PPG→ECG 重建复合损失。

    L = λ₁·MSE + λ₂·DTW + λ₃·Freq + λ₄·Perceptual

    模块选择: weights 为 {模块名: 权重} 字典, 权重为 0 或缺省的项
    不参与计算 (也跳过前向, 省算力)。可选模块: mse / dtw / freq / perceptual。

    Args:
        weights: 各损失项权重, 如 {"mse": 1.0, "dtw": 0.5, "freq": 0.3, "perceptual": 0.1}
        ecg_leads: ECG 导联数 (须与数据集实际导联数一致)
    """

    MODULE_NAMES = ("mse", "dtw", "freq", "perceptual")

    def __init__(
        self,
        weights: dict[str, float] | None = None,
        ecg_leads: int = 12,
    ):
        super().__init__()
        weights = dict(weights or {})
        unknown = set(weights) - set(self.MODULE_NAMES)
        if unknown:
            raise ValueError(
                f"未知损失模块 {sorted(unknown)}。可选: {self.MODULE_NAMES}"
            )
        self.weights = {name: float(weights.get(name, 0.0))
                        for name in self.MODULE_NAMES}

        self.mse = MSELoss()
        self.dtw = DTWLoss()
        self.freq = FrequencyLoss()
        self.perc = PerceptualLoss(ecg_leads=ecg_leads)
        self._modules_map = {
            "mse": self.mse,
            "dtw": self.dtw,
            "freq": self.freq,
            "perceptual": self.perc,
        }

    def forward(
        self, pred: torch.Tensor, target: torch.Tensor
    ) -> dict[str, torch.Tensor]:
        """
        Args:
            pred:   [B, C, L]  预测 ECG
            target: [B, C, L]  真实 ECG
        Returns:
            包含各启用损失项和总损失的字典 (未启用项值为 0)
        """
        losses: dict[str, torch.Tensor] = {}
        total = torch.zeros((), device=pred.device)
        for name, module in self._modules_map.items():
            w = self.weights[name]
            if w == 0.0:
                losses[name] = torch.zeros((), device=pred.device)
                continue
            val = module(pred, target)
            losses[name] = val
            total = total + w * val
        losses["total"] = total
        return losses
