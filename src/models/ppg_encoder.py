"""PPG 信号编码器。

将 PPG 信号编码为潜在表示，包含三个分支:
  1. 频域分支: STFT -> U-Net 提取频谱特征
  2. 时域分支: 1D-CNN / Transformer 提取波形特征
  3. Cross-Attention 融合: 频域和时域特征交互

输入:  PPG 信号 [B, L]  (L = 1250, 10s @ 125Hz)
输出:  潜在表示 [B, D, L']  (D = latent_dim)
"""

from __future__ import annotations

import torch
import torch.nn as nn


class STFTBranch(nn.Module):
    """频域分支: STFT 变换后用卷积提取频谱特征。"""

    def __init__(
        self,
        n_fft: int = 256,
        hop_length: int = 64,
        base_channels: int = 32,
    ):
        super().__init__()
        self.n_fft = n_fft
        self.hop_length = hop_length

        # 对幅值谱进行卷积 (n_fft//2 + 1 个频率bin)
        in_ch = n_fft // 2 + 1
        self.conv = nn.Sequential(
            nn.Conv1d(in_ch, base_channels, 3, padding=1),
            nn.GroupNorm(8, base_channels),
            nn.GELU(),
            nn.Conv1d(base_channels, base_channels * 2, 3, padding=1),
            nn.GroupNorm(8, base_channels * 2),
            nn.GELU(),
            nn.Conv1d(base_channels * 2, base_channels * 4, 3, padding=1),
            nn.GroupNorm(8, base_channels * 4),
            nn.GELU(),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """x: [B, L] -> [B, C, T']"""
        # STFT: [B, freq_bins, time_frames]
        stft = torch.stft(
            x,
            n_fft=self.n_fft,
            hop_length=self.hop_length,
            return_complex=True,
            window=torch.hann_window(self.n_fft, device=x.device),
        )
        magnitude = stft.abs()  # [B, freq_bins, T']
        return self.conv(magnitude)


class TemporalBranch(nn.Module):
    """时域分支: 1D-CNN 提取波形特征。"""

    def __init__(self, in_channels: int = 1, base_channels: int = 32):
        super().__init__()
        self.conv = nn.Sequential(
            nn.Conv1d(in_channels, base_channels, 7, stride=2, padding=3),
            nn.GroupNorm(8, base_channels),
            nn.GELU(),
            nn.Conv1d(base_channels, base_channels * 2, 5, stride=2, padding=2),
            nn.GroupNorm(8, base_channels * 2),
            nn.GELU(),
            nn.Conv1d(base_channels * 2, base_channels * 4, 5, stride=2, padding=2),
            nn.GroupNorm(8, base_channels * 4),
            nn.GELU(),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """x: [B, 1, L] -> [B, C, L']"""
        return self.conv(x)


class CrossAttentionFusion(nn.Module):
    """频域-时域 Cross-Attention 融合模块。"""

    def __init__(self, dim: int, num_heads: int = 4):
        super().__init__()
        self.cross_attn = nn.MultiheadAttention(dim, num_heads, batch_first=True)
        self.norm1 = nn.LayerNorm(dim)
        self.ffn = nn.Sequential(
            nn.Linear(dim, dim * 2),
            nn.GELU(),
            nn.Linear(dim * 2, dim),
        )
        self.norm2 = nn.LayerNorm(dim)

    def forward(
        self, freq_feat: torch.Tensor, temp_feat: torch.Tensor
    ) -> torch.Tensor:
        """
        freq_feat: [B, C, T]  (Query)
        temp_feat: [B, C, T]  (Key/Value)
        -> [B, C, T]
        """
        # 转为序列格式 [B, T, C]
        q = freq_feat.transpose(1, 2)
        kv = temp_feat.transpose(1, 2)

        attn_out, _ = self.cross_attn(q, kv, kv)
        x = self.norm1(q + attn_out)
        x = self.norm2(x + self.ffn(x))
        return x.transpose(1, 2)  # 回到 [B, C, T]


class PPGEncoder(nn.Module):
    """PPG 编码器: 频域 + 时域 + Cross-Attention 融合。

    Args:
        signal_length: 输入信号长度 (默认 1250, 即 10s @ 125Hz)
        latent_dim: 潜在表示通道数
        base_channels: 基础通道数
    """

    def __init__(
        self,
        signal_length: int = 1250,
        latent_dim: int = 128,
        base_channels: int = 32,
    ):
        super().__init__()
        self.signal_length = signal_length
        self.latent_dim = latent_dim

        self.stft_branch = STFTBranch(base_channels=base_channels)
        self.temporal_branch = TemporalBranch(base_channels=base_channels)

        # Cross-Attention 融合
        self.fusion = CrossAttentionFusion(dim=base_channels * 4)

        # 投影到 latent_dim
        self.proj = nn.Conv1d(base_channels * 4, latent_dim, 1)

    def forward(self, ppg: torch.Tensor) -> torch.Tensor:
        """
        Args:
            ppg: [B, L] 或 [B, 1, L]
        Returns:
            latent: [B, latent_dim, T']
        """
        if ppg.dim() == 2:
            ppg = ppg.unsqueeze(1)  # [B, 1, L]

        # 频域分支 (需要 [B, L])
        freq_feat = self.stft_branch(ppg.squeeze(1))  # [B, C, T_f]

        # 时域分支
        temp_feat = self.temporal_branch(ppg)  # [B, C, T_t]

        # 对齐时间维度 (取较短的)
        T_min = min(freq_feat.size(-1), temp_feat.size(-1))
        freq_feat = freq_feat[..., :T_min]
        temp_feat = temp_feat[..., :T_min]

        # Cross-Attention 融合
        fused = self.fusion(freq_feat, temp_feat)  # [B, C, T_min]

        # 投影
        latent = self.proj(fused)  # [B, latent_dim, T_min]
        return latent
