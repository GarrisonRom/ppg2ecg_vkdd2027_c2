"""心脏疾病分类器。

支持三种路径:
  - Path A: 重建 ECG -> 独立分类器
  - Path B: 端到端联合训练
  - Path C: 跨模态对比学习 (推荐)

当前实现 Path A 的分类器骨架。
"""

from __future__ import annotations

import torch
import torch.nn as nn
import torch.nn.functional as F


class ECGClassifier(nn.Module):
    """基于 1D-CNN + Transformer 的 ECG 分类器。

    Args:
        ecg_leads: ECG 导联数 (默认 12)
        signal_length: 信号长度 (默认 1250)
        num_classes: 疾病类别数
        base_channels: 基础通道数
    """

    def __init__(
        self,
        ecg_leads: int = 12,
        signal_length: int = 1250,
        num_classes: int = 5,
        base_channels: int = 64,
    ):
        super().__init__()

        # CNN 特征提取
        self.features = nn.Sequential(
            nn.Conv1d(ecg_leads, base_channels, 15, stride=2, padding=7),
            nn.BatchNorm1d(base_channels),
            nn.GELU(),
            nn.Conv1d(base_channels, base_channels * 2, 7, stride=2, padding=3),
            nn.BatchNorm1d(base_channels * 2),
            nn.GELU(),
            nn.Conv1d(base_channels * 2, base_channels * 4, 5, stride=2, padding=2),
            nn.BatchNorm1d(base_channels * 4),
            nn.GELU(),
            nn.Conv1d(base_channels * 4, base_channels * 4, 3, stride=2, padding=1),
            nn.BatchNorm1d(base_channels * 4),
            nn.GELU(),
        )

        # 计算展平后的序列长度
        with torch.no_grad():
            dummy = torch.zeros(1, ecg_leads, signal_length)
            feat = self.features(dummy)
            self.feat_channels, self.feat_length = feat.size(1), feat.size(2)

        # Transformer 编码器
        encoder_layer = nn.TransformerEncoderLayer(
            d_model=self.feat_channels,
            nhead=8,
            dim_feedforward=self.feat_channels * 4,
            dropout=0.1,
            batch_first=True,
            activation="gelu",
        )
        self.transformer = nn.TransformerEncoder(encoder_layer, num_layers=2)

        # 分类头
        self.classifier = nn.Sequential(
            nn.LayerNorm(self.feat_channels),
            nn.Linear(self.feat_channels, self.feat_channels // 2),
            nn.GELU(),
            nn.Dropout(0.2),
            nn.Linear(self.feat_channels // 2, num_classes),
        )

    def forward(self, ecg: torch.Tensor) -> torch.Tensor:
        """
        Args:
            ecg: [B, 12, L]
        Returns:
            logits: [B, num_classes]
        """
        x = self.features(ecg)           # [B, C, L']
        x = x.transpose(1, 2)            # [B, L', C]
        x = self.transformer(x)          # [B, L', C]
        x = x.mean(dim=1)                # 全局平均池化 [B, C]
        return self.classifier(x)        # [B, num_classes]


# 兼容别名
DiseaseClassifier = ECGClassifier


class ContrastiveHead(nn.Module):
    """跨模态对比学习投影头 (Path C)。

    将 ECG 和 PPG 的特征投影到共享的对比学习空间。
    """

    def __init__(self, in_dim: int, proj_dim: int = 128):
        super().__init__()
        self.projector = nn.Sequential(
            nn.Linear(in_dim, in_dim),
            nn.GELU(),
            nn.Linear(in_dim, proj_dim),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = F.normalize(self.projector(x), dim=-1)
        return x
