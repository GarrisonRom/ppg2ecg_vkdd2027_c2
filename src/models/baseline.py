"""可扩展的 PPG→ECG 基线模型。

该模块只负责一个干净的监督式时序映射:

    4 路 PPG -> 1D Res-Encoder -> 时序 latent -> 1D Decoder -> 4 路 ECG

不包含 VAE、GAN、Flow、GRL 或额外的生理损失。后续实验可以通过注册表
替换 encoder/decoder，避免把不同研究假设混在第一版 baseline 中。
"""

from __future__ import annotations

from typing import Any

import torch
import torch.nn as nn
import torch.nn.functional as F


def _group_norm(channels: int, max_groups: int = 8) -> nn.GroupNorm:
    """选择一个能整除通道数的 GroupNorm 分组数。"""
    groups = min(max_groups, channels)
    while groups > 1 and channels % groups != 0:
        groups -= 1
    return nn.GroupNorm(groups, channels)


class BaselineResBlock1D(nn.Module):
    """带可选通道 Dropout 的 1D 残差块。"""

    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        stride: int = 1,
        dropout: float = 0.0,
    ):
        super().__init__()
        if not 0.0 <= dropout < 1.0:
            raise ValueError(f"dropout must be in [0, 1), got {dropout}")

        self.conv1 = nn.Conv1d(
            in_channels, out_channels, kernel_size=7,
            stride=stride, padding=3,
        )
        self.norm1 = _group_norm(out_channels)
        self.conv2 = nn.Conv1d(
            out_channels, out_channels, kernel_size=5, padding=2,
        )
        self.norm2 = _group_norm(out_channels)
        self.act = nn.GELU()
        self.dropout = nn.Dropout1d(dropout) if dropout > 0 else nn.Identity()
        self.skip = (
            nn.Conv1d(in_channels, out_channels, kernel_size=1, stride=stride)
            if in_channels != out_channels or stride != 1
            else nn.Identity()
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        h = self.act(self.norm1(self.conv1(x)))
        h = self.dropout(h)
        h = self.norm2(self.conv2(h))
        return self.act(h + self.skip(x))


class BaselinePPGEncoder(nn.Module):
    """四路 PPG 时序编码器。

    输入为 ``[B, ppg_channels, T]``，输出是一个字典:

    ``{"latent": [B, latent_dim, T/16], "skips": (...)}``

    skip features 供 baseline decoder 使用；latent 保留时间维度，避免把
    逐搏时序压成单个全局向量。
    """

    def __init__(
        self,
        signal_length: int = 2000,
        latent_dim: int = 128,
        ppg_channels: int = 4,
        base_channels: int = 32,
        dropout: float = 0.1,
    ):
        super().__init__()
        if ppg_channels < 1:
            raise ValueError("ppg_channels must be positive")
        if base_channels < 8:
            raise ValueError("base_channels must be at least 8")

        self.signal_length = signal_length
        self.latent_dim = latent_dim
        self.ppg_channels = ppg_channels
        self.base_channels = base_channels

        self.stem = BaselineResBlock1D(
            ppg_channels, base_channels, stride=2, dropout=dropout,
        )
        self.enc1 = BaselineResBlock1D(
            base_channels, base_channels * 2, stride=2, dropout=dropout,
        )
        self.enc2 = BaselineResBlock1D(
            base_channels * 2, base_channels * 4, stride=2, dropout=dropout,
        )
        self.bottleneck = BaselineResBlock1D(
            base_channels * 4, latent_dim, stride=2, dropout=dropout,
        )

    def forward(self, ppg: torch.Tensor) -> dict[str, Any]:
        if ppg.dim() == 2:
            ppg = ppg.unsqueeze(1)
        if ppg.dim() != 3:
            raise ValueError(f"PPG must have shape [B,C,T], got {tuple(ppg.shape)}")
        if ppg.size(1) != self.ppg_channels:
            raise ValueError(
                f"expected {self.ppg_channels} PPG channels, got {ppg.size(1)}"
            )

        skip0 = self.stem(ppg)       # [B, base, T/2]
        skip1 = self.enc1(skip0)     # [B, 2*base, T/4]
        skip2 = self.enc2(skip1)     # [B, 4*base, T/8]
        latent = self.bottleneck(skip2)  # [B, latent_dim, T/16]
        return {"latent": latent, "skips": (skip0, skip1, skip2)}


class BaselineUpsampleBlock1D(nn.Module):
    """线性插值 + 卷积上采样，避免转置卷积的棋盘伪影。"""

    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        dropout: float = 0.0,
    ):
        super().__init__()
        self.proj = nn.Conv1d(in_channels, out_channels, kernel_size=3, padding=1)
        self.norm = _group_norm(out_channels)
        self.act = nn.GELU()
        self.block = BaselineResBlock1D(
            out_channels, out_channels, dropout=dropout,
        )

    def forward(self, x: torch.Tensor, target_length: int) -> torch.Tensor:
        x = F.interpolate(x, size=target_length, mode="linear", align_corners=False)
        x = self.act(self.norm(self.proj(x)))
        return self.block(x)


class BaselineECGDecoder(nn.Module):
    """将时序 latent 解码为多导联 ECG。"""

    def __init__(
        self,
        signal_length: int = 2000,
        latent_dim: int = 128,
        ecg_leads: int = 4,
        base_channels: int = 32,
        dropout: float = 0.0,
    ):
        super().__init__()
        if ecg_leads < 1:
            raise ValueError("ecg_leads must be positive")
        if base_channels < 8:
            raise ValueError("base_channels must be at least 8")

        self.signal_length = signal_length
        self.ecg_leads = ecg_leads
        self.base_channels = base_channels

        c = base_channels
        self.bottleneck = BaselineResBlock1D(latent_dim, c * 4, dropout=dropout)

        self.up1 = BaselineUpsampleBlock1D(c * 4, c * 4, dropout=dropout)
        self.dec1 = BaselineResBlock1D(c * 8, c * 4, dropout=dropout)

        self.up2 = BaselineUpsampleBlock1D(c * 4, c * 2, dropout=dropout)
        self.dec2 = BaselineResBlock1D(c * 4, c * 2, dropout=dropout)

        self.up3 = BaselineUpsampleBlock1D(c * 2, c, dropout=dropout)
        self.dec3 = BaselineResBlock1D(c * 2, c, dropout=dropout)

        self.up4 = BaselineUpsampleBlock1D(c, c // 2, dropout=dropout)
        self.dec4 = BaselineResBlock1D(c // 2, c // 2, dropout=dropout)
        self.out_conv = nn.Conv1d(c // 2, ecg_leads, kernel_size=7, padding=3)

    @staticmethod
    def _skip_or_zeros(
        x: torch.Tensor,
        skip: torch.Tensor | None,
    ) -> torch.Tensor:
        """兼容只有 latent 的调用，同时保持 decoder 接口可扩展。"""
        if skip is None:
            return torch.zeros_like(x)
        if skip.size(-1) != x.size(-1):
            skip = F.interpolate(skip, size=x.size(-1), mode="linear", align_corners=False)
        if skip.size(1) != x.size(1):
            raise ValueError(
                f"skip channels {skip.size(1)} do not match decoder channels {x.size(1)}"
            )
        return skip

    def forward(self, encoded: torch.Tensor | dict[str, Any]) -> torch.Tensor:
        if isinstance(encoded, dict):
            latent = encoded["latent"]
            skips = encoded.get("skips")
            if skips is not None and len(skips) != 3:
                raise ValueError("baseline encoder skips must contain three tensors")
        else:
            latent = encoded
            skips = None

        x = self.bottleneck(latent)
        skip0, skip1, skip2 = (skips if skips is not None else (None, None, None))

        target = skip2.size(-1) if skip2 is not None else max(1, x.size(-1) * 2)
        x = self.up1(x, target)
        x = self.dec1(torch.cat([x, self._skip_or_zeros(x, skip2)], dim=1))

        target = skip1.size(-1) if skip1 is not None else max(1, x.size(-1) * 2)
        x = self.up2(x, target)
        x = self.dec2(torch.cat([x, self._skip_or_zeros(x, skip1)], dim=1))

        target = skip0.size(-1) if skip0 is not None else max(1, x.size(-1) * 2)
        x = self.up3(x, target)
        x = self.dec3(torch.cat([x, self._skip_or_zeros(x, skip0)], dim=1))

        x = self.up4(x, self.signal_length)
        x = self.dec4(x)
        return self.out_conv(x)


__all__ = ["BaselinePPGEncoder", "BaselineECGDecoder"]
