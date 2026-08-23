"""多尺度 ECG 恢复解码器。

将潜在表示解码为 12 导联 ECG 信号，采用多尺度上采样策略:
  - Stage 1: 潜在表示 -> 中间特征 (低频轮廓)
  - Stage 2: 上采样 -> 中频细节
  - Stage 3: 上采样 -> 全分辨率波形

输入:  潜在表示 [B, D, T']
输出:  ECG 信号 [B, 12, L]  (L = 1250, 12 导联)
"""

from __future__ import annotations

import torch
import torch.nn as nn


class ResBlock1D(nn.Module):
    """1D 残差块。"""

    def __init__(self, in_ch: int, out_ch: int, kernel_size: int = 5):
        super().__init__()
        self.conv1 = nn.Conv1d(in_ch, out_ch, kernel_size, padding=kernel_size // 2)
        self.norm1 = nn.GroupNorm(8, out_ch)
        self.conv2 = nn.Conv1d(out_ch, out_ch, kernel_size, padding=kernel_size // 2)
        self.norm2 = nn.GroupNorm(8, out_ch)
        self.act = nn.GELU()
        self.skip = (
            nn.Conv1d(in_ch, out_ch, 1) if in_ch != out_ch else nn.Identity()
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        h = self.act(self.norm1(self.conv1(x)))
        h = self.norm2(self.conv2(h))
        return self.act(h + self.skip(x))


class UpsampleBlock(nn.Module):
    """转置卷积上采样块。"""

    def __init__(self, in_ch: int, out_ch: int, scale_factor: int = 2):
        super().__init__()
        self.up = nn.ConvTranspose1d(
            in_ch, out_ch, kernel_size=scale_factor * 2,
            stride=scale_factor, padding=scale_factor // 2,
        )
        self.block = ResBlock1D(out_ch, out_ch)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.block(self.up(x))


class ECGDecoder(nn.Module):
    """多尺度 ECG 解码器。

    Args:
        latent_dim: 潜在表示通道数
        ecg_leads: ECG 导联数 (默认 12)
        signal_length: 目标信号长度 (默认 1250)
        base_channels: 基础通道数
    """

    def __init__(
        self,
        latent_dim: int = 128,
        ecg_leads: int = 12,
        signal_length: int = 1250,
        base_channels: int = 32,
    ):
        super().__init__()
        self.ecg_leads = ecg_leads
        self.signal_length = signal_length

        # Stage 1: 潜在表示 -> 底层特征
        self.stage1 = nn.Sequential(
            ResBlock1D(latent_dim, base_channels * 4),
            ResBlock1D(base_channels * 4, base_channels * 4),
        )

        # Stage 2: 上采样 x4
        self.stage2 = nn.Sequential(
            UpsampleBlock(base_channels * 4, base_channels * 2, scale_factor=2),
            ResBlock1D(base_channels * 2, base_channels * 2),
            UpsampleBlock(base_channels * 2, base_channels * 2, scale_factor=2),
            ResBlock1D(base_channels * 2, base_channels * 2),
        )

        # Stage 3: 上采样到目标长度
        self.stage3 = nn.Sequential(
            ResBlock1D(base_channels * 2, base_channels),
            ResBlock1D(base_channels, base_channels),
        )

        # 输出投影
        self.out_conv = nn.Conv1d(base_channels, ecg_leads, 7, padding=3)

    def forward(self, latent: torch.Tensor) -> torch.Tensor:
        """
        Args:
            latent: [B, D, T']
        Returns:
            ecg: [B, 12, L]  (插值到 signal_length)
        """
        x = self.stage1(latent)
        x = self.stage2(x)
        x = self.stage3(x)
        ecg = self.out_conv(x)

        # 插值到目标长度
        if ecg.size(-1) != self.signal_length:
            ecg = nn.functional.interpolate(
                ecg, size=self.signal_length, mode="linear", align_corners=False
            )
        return ecg
