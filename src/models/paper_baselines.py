"""Small, self-contained reimplementations of published PPG->ECG cores.

These modules are intentionally independent from the project's main registry.
They support a controlled comparison on SensSmartTech without changing the
existing v0.x training path.

Implemented paper cores:
  - CardioGAN: attention U-Net generators and time/frequency discriminators.
  - RDDM: conditional 1D denoisers for ROI-guided forward and reverse steps.
  - QRS-TransAttn: temporal/channel attention encoder-decoder.
  - P2E-WGAN: paired U-Net generator and conditional critic.
  - LightweightPPG2ECG: compact multi-kernel residual attention network.

The original papers use single-channel signals and different sampling and
normalization conventions. The classes therefore accept arbitrary channel
counts and do not apply a final tanh by default.
"""

from __future__ import annotations

import math
from typing import Iterable

import torch
import torch.nn as nn
import torch.nn.functional as F


def _groups(channels: int, maximum: int = 8) -> int:
    groups = min(maximum, channels)
    while groups > 1 and channels % groups:
        groups -= 1
    return groups


class ConvBlock1D(nn.Module):
    """Two-convolution block used by the attention U-Net."""

    def __init__(self, in_channels: int, out_channels: int, stride: int = 1):
        super().__init__()
        self.conv1 = nn.Conv1d(
            in_channels, out_channels, kernel_size=7, stride=stride, padding=3,
        )
        self.norm1 = nn.GroupNorm(_groups(out_channels), out_channels)
        self.conv2 = nn.Conv1d(out_channels, out_channels, kernel_size=5, padding=2)
        self.norm2 = nn.GroupNorm(_groups(out_channels), out_channels)
        self.act = nn.LeakyReLU(0.2, inplace=True)
        self.skip = (
            nn.Conv1d(in_channels, out_channels, 1, stride=stride)
            if in_channels != out_channels or stride != 1 else nn.Identity()
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        h = self.act(self.norm1(self.conv1(x)))
        h = self.norm2(self.conv2(h))
        return self.act(h + self.skip(x))


class AttentionGate1D(nn.Module):
    """Self-gated soft attention for a U-Net skip feature."""

    def __init__(self, skip_channels: int, gate_channels: int, hidden: int):
        super().__init__()
        self.skip_proj = nn.Conv1d(skip_channels, hidden, 1, bias=False)
        self.gate_proj = nn.Conv1d(gate_channels, hidden, 1, bias=False)
        self.psi = nn.Conv1d(hidden, 1, 1)

    def forward(self, skip: torch.Tensor, gate: torch.Tensor) -> torch.Tensor:
        if gate.size(-1) != skip.size(-1):
            gate = F.interpolate(gate, size=skip.size(-1), mode="linear", align_corners=False)
        score = torch.relu(self.skip_proj(skip) + self.gate_proj(gate))
        return skip * torch.sigmoid(self.psi(score))


class AttentionUNet1D(nn.Module):
    """Attention U-Net generator adapted to arbitrary 1D channel counts."""

    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        base_channels: int = 16,
        output_activation: str = "none",
    ):
        super().__init__()
        if base_channels < 8:
            raise ValueError("base_channels must be at least 8")
        c0, c1, c2, c3, c4 = (
            base_channels,
            base_channels * 2,
            base_channels * 4,
            base_channels * 8,
            base_channels * 16,
        )
        self.output_activation = output_activation
        self.enc0 = ConvBlock1D(in_channels, c0)
        self.enc1 = ConvBlock1D(c0, c1, stride=2)
        self.enc2 = ConvBlock1D(c1, c2, stride=2)
        self.enc3 = ConvBlock1D(c2, c3, stride=2)
        self.bottleneck = ConvBlock1D(c3, c4, stride=2)

        self.att3 = AttentionGate1D(c3, c4, max(8, c3 // 2))
        self.att2 = AttentionGate1D(c2, c3, max(8, c2 // 2))
        self.att1 = AttentionGate1D(c1, c2, max(8, c1 // 2))
        self.att0 = AttentionGate1D(c0, c1, max(8, c0 // 2))
        self.dec3 = ConvBlock1D(c4 + c3, c3)
        self.dec2 = ConvBlock1D(c3 + c2, c2)
        self.dec1 = ConvBlock1D(c2 + c1, c1)
        self.dec0 = ConvBlock1D(c1 + c0, c0)
        self.out_conv = nn.Conv1d(c0, out_channels, kernel_size=7, padding=3)

    @staticmethod
    def _up(x: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
        return F.interpolate(x, size=target.size(-1), mode="linear", align_corners=False)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        e0 = self.enc0(x)
        e1 = self.enc1(e0)
        e2 = self.enc2(e1)
        e3 = self.enc3(e2)
        h = self.bottleneck(e3)

        h = self.dec3(torch.cat([self._up(h, e3), self.att3(e3, h)], dim=1))
        h = self.dec2(torch.cat([self._up(h, e2), self.att2(e2, h)], dim=1))
        h = self.dec1(torch.cat([self._up(h, e1), self.att1(e1, h)], dim=1))
        h = self.dec0(torch.cat([self._up(h, e0), self.att0(e0, h)], dim=1))
        y = self.out_conv(h)
        if self.output_activation == "tanh":
            y = torch.tanh(y)
        elif self.output_activation != "none":
            raise ValueError(f"unknown output_activation: {self.output_activation}")
        return y


class PatchDiscriminator1D(nn.Module):
    """PatchGAN-style discriminator for the time domain."""

    def __init__(self, in_channels: int, base_channels: int = 16):
        super().__init__()
        channels = [in_channels, base_channels, base_channels * 2, base_channels * 4]
        layers: list[nn.Module] = []
        for i in range(len(channels) - 1):
            layers.append(
                nn.Conv1d(channels[i], channels[i + 1], 5, stride=2, padding=2)
            )
            if i > 0:
                layers.append(nn.InstanceNorm1d(channels[i + 1]))
            layers.append(nn.LeakyReLU(0.2, inplace=True))
        layers.append(nn.Conv1d(channels[-1], 1, 3, padding=1))
        self.net = nn.Sequential(*layers)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.net(x)


def stft_log_magnitude(
    x: torch.Tensor,
    n_fft: int = 128,
    hop_length: int = 32,
) -> torch.Tensor:
    """Return `[B, C, F, frames]` log-magnitude spectrograms."""
    b, c, t = x.shape
    window = torch.hann_window(n_fft, device=x.device, dtype=x.dtype)
    flat = x.reshape(b * c, t)
    spec = torch.stft(
        flat,
        n_fft=n_fft,
        hop_length=hop_length,
        win_length=n_fft,
        window=window,
        center=False,
        return_complex=True,
    )
    mag = torch.log(spec.abs().clamp_min(1e-6))
    return mag.reshape(b, c, mag.size(-2), mag.size(-1))


class SpectrogramDiscriminator(nn.Module):
    """2D discriminator operating on log STFT magnitudes."""

    def __init__(self, in_channels: int, base_channels: int = 8):
        super().__init__()
        c = [in_channels, base_channels, base_channels * 2, base_channels * 4]
        layers: list[nn.Module] = []
        for i in range(len(c) - 1):
            layers.append(nn.Conv2d(c[i], c[i + 1], 5, stride=2, padding=2))
            if i > 0:
                layers.append(nn.InstanceNorm2d(c[i + 1]))
            layers.append(nn.LeakyReLU(0.2, inplace=True))
        layers.append(nn.Conv2d(c[-1], 1, 3, padding=1))
        self.net = nn.Sequential(*layers)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.net(stft_log_magnitude(x))


def _time_embedding(t: torch.Tensor, dim: int) -> torch.Tensor:
    half = dim // 2
    freq = torch.exp(
        -math.log(10000.0)
        * torch.arange(half, device=t.device, dtype=torch.float32)
        / max(1, half)
    )
    args = t.float()[:, None] * freq[None]
    emb = torch.cat([torch.sin(args), torch.cos(args)], dim=-1)
    if dim % 2:
        emb = F.pad(emb, (0, 1))
    return emb


class ConditionalResBlock1D(nn.Module):
    """Residual block with diffusion-time injection."""

    def __init__(self, in_channels: int, out_channels: int, time_dim: int, stride: int = 1):
        super().__init__()
        self.norm1 = nn.GroupNorm(_groups(in_channels), in_channels)
        self.conv1 = nn.Conv1d(in_channels, out_channels, 3, stride=stride, padding=1)
        self.time_proj = nn.Linear(time_dim, out_channels)
        self.norm2 = nn.GroupNorm(_groups(out_channels), out_channels)
        self.conv2 = nn.Conv1d(out_channels, out_channels, 3, padding=1)
        self.act = nn.SiLU()
        self.skip = (
            nn.Conv1d(in_channels, out_channels, 1, stride=stride)
            if in_channels != out_channels or stride != 1 else nn.Identity()
        )

    def forward(self, x: torch.Tensor, t_emb: torch.Tensor) -> torch.Tensor:
        h = self.conv1(self.act(self.norm1(x)))
        h = h + self.time_proj(t_emb)[:, :, None]
        h = self.conv2(self.act(self.norm2(h)))
        return h + self.skip(x)


class ConditionalUNet1D(nn.Module):
    """Lightweight conditional signal-space U-Net used by the RDDM core."""

    def __init__(
        self,
        signal_channels: int,
        condition_channels: int,
        base_channels: int = 16,
        time_dim: int = 128,
    ):
        super().__init__()
        c0, c1, c2, c3 = (
            base_channels,
            base_channels * 2,
            base_channels * 4,
            base_channels * 8,
        )
        self.time_dim = time_dim
        self.time_mlp = nn.Sequential(
            nn.Linear(time_dim, time_dim * 2),
            nn.SiLU(),
            nn.Linear(time_dim * 2, time_dim),
        )
        self.in_conv = nn.Conv1d(signal_channels + condition_channels, c0, 3, padding=1)
        self.enc0 = ConditionalResBlock1D(c0, c0, time_dim)
        self.enc1 = ConditionalResBlock1D(c0, c1, time_dim, stride=2)
        self.enc2 = ConditionalResBlock1D(c1, c2, time_dim, stride=2)
        self.enc3 = ConditionalResBlock1D(c2, c3, time_dim, stride=2)
        self.mid = ConditionalResBlock1D(c3, c3, time_dim)
        self.cond0 = nn.Conv1d(condition_channels, c0, 1)
        self.cond1 = nn.Conv1d(condition_channels, c1, 1)
        self.cond2 = nn.Conv1d(condition_channels, c2, 1)
        self.cond3 = nn.Conv1d(condition_channels, c3, 1)
        self.dec2 = ConditionalResBlock1D(c3 + c2, c2, time_dim)
        self.dec1 = ConditionalResBlock1D(c2 + c1, c1, time_dim)
        self.dec0 = ConditionalResBlock1D(c1 + c0, c0, time_dim)
        self.out = nn.Sequential(
            nn.GroupNorm(_groups(c0), c0),
            nn.SiLU(),
            nn.Conv1d(c0, signal_channels, 3, padding=1),
        )

    @staticmethod
    def _condition(condition: torch.Tensor, proj: nn.Module, target: torch.Tensor) -> torch.Tensor:
        c = F.interpolate(condition, size=target.size(-1), mode="linear", align_corners=False)
        return proj(c)

    def forward(self, signal: torch.Tensor, condition: torch.Tensor, t: torch.Tensor) -> torch.Tensor:
        temb = self.time_mlp(_time_embedding(t, self.time_dim))
        h = self.in_conv(torch.cat([signal, condition], dim=1))
        h = self.enc0(h, temb)
        h = h + self._condition(condition, self.cond0, h)
        s0 = h
        h = self.enc1(h, temb)
        h = h + self._condition(condition, self.cond1, h)
        s1 = h
        h = self.enc2(h, temb)
        h = h + self._condition(condition, self.cond2, h)
        s2 = h
        h = self.enc3(h, temb)
        h = h + self._condition(condition, self.cond3, h)
        h = self.mid(h, temb)
        h = F.interpolate(h, size=s2.size(-1), mode="linear", align_corners=False)
        h = self.dec2(torch.cat([h, s2], dim=1), temb)
        h = F.interpolate(h, size=s1.size(-1), mode="linear", align_corners=False)
        h = self.dec1(torch.cat([h, s1], dim=1), temb)
        h = F.interpolate(h, size=s0.size(-1), mode="linear", align_corners=False)
        h = self.dec0(torch.cat([h, s0], dim=1), temb)
        return self.out(h)


class RDDMCore(nn.Module):
    """ROI-guided conditional diffusion core from the RDDM paper.

    The training mask is derived from the target ECG (as in the paper's
    training-time R-peak ROI). It is not required during sampling: the second
    network predicts the ROI-guided intermediate signal from the standard
    noisy signal and PPG condition.
    """

    def __init__(
        self,
        signal_channels: int,
        condition_channels: int,
        timesteps: int = 1000,
        beta_end: float = 0.2,
        base_channels: int = 16,
        roi_gamma: int = 32,
        roi_threshold: float = 1.5,
        lambda_roi: float = 100.0,
        lambda_global: float = 1.0,
    ):
        super().__init__()
        self.signal_channels = signal_channels
        self.condition_channels = condition_channels
        self.timesteps = timesteps
        self.roi_gamma = int(roi_gamma)
        self.roi_threshold = float(roi_threshold)
        self.lambda_roi = float(lambda_roi)
        self.lambda_global = float(lambda_global)
        betas = torch.linspace(1e-4, beta_end, timesteps)
        alphas = 1.0 - betas
        alpha_bar = torch.cumprod(alphas, dim=0)
        self.register_buffer("betas", betas)
        self.register_buffer("alphas", alphas)
        self.register_buffer("alpha_bar", alpha_bar)
        self.register_buffer("sqrt_alpha_bar", alpha_bar.sqrt())
        self.register_buffer("sqrt_one_minus_alpha_bar", (1.0 - alpha_bar).sqrt())
        self.roi_net = ConditionalUNet1D(
            signal_channels, condition_channels, base_channels=base_channels,
        )
        self.eps_net = ConditionalUNet1D(
            signal_channels, condition_channels, base_channels=base_channels,
        )

    def roi_mask(self, ecg: torch.Tensor) -> torch.Tensor:
        """Approximate QRS windows from target ECG without storing annotations."""
        lead = ecg[:, 1:2] if ecg.size(1) > 1 else ecg[:, :1]
        baseline = F.avg_pool1d(lead, kernel_size=125, stride=1, padding=62)
        energy = (lead - baseline).abs()
        scale = energy.flatten(1).std(dim=1).clamp_min(1e-4)[:, None, None]
        seeds = (energy > self.roi_threshold * scale).float()
        width = max(3, 2 * self.roi_gamma + 1)
        mask = F.max_pool1d(seeds, kernel_size=width, stride=1, padding=width // 2)
        return mask.expand(-1, ecg.size(1), -1).clamp(0.0, 1.0)

    def q_sample(
        self,
        x0: torch.Tensor,
        t: torch.Tensor,
        noise: torch.Tensor,
    ) -> torch.Tensor:
        a = self.sqrt_alpha_bar[t][:, None, None]
        b = self.sqrt_one_minus_alpha_bar[t][:, None, None]
        return a * x0 + b * noise

    def training_loss(self, condition: torch.Tensor, target: torch.Tensor) -> dict[str, torch.Tensor]:
        b = target.size(0)
        t = torch.randint(0, self.timesteps, (b,), device=target.device)
        noise = torch.randn_like(target)
        mask = self.roi_mask(target)
        x_t = self.q_sample(target, t, noise)
        x_m = self.q_sample(target, t, mask * noise)
        eps_pred = self.eps_net(x_m, condition, t)
        x_p = self.roi_net(x_t, condition, t)
        roi_loss = F.mse_loss(eps_pred, mask * noise)
        global_loss = F.mse_loss(x_p, x_m)
        total = self.lambda_roi * roi_loss + self.lambda_global * global_loss
        return {
            "total": total,
            "roi": roi_loss,
            "global": global_loss,
        }

    @torch.no_grad()
    def sample(
        self,
        condition: torch.Tensor,
        steps: int = 10,
        generator: torch.Generator | None = None,
    ) -> torch.Tensor:
        """Deterministic DDIM-style sampling using the paper's two networks."""
        b, _, length = condition.shape
        x = torch.randn(
            b, self.signal_channels, length,
            device=condition.device, dtype=condition.dtype, generator=generator,
        )
        steps = max(1, int(steps))
        times = torch.linspace(
            self.timesteps - 1, 0, steps, device=condition.device, dtype=torch.long,
        )
        for i, t_value in enumerate(times):
            t_int = int(t_value.item())
            t = torch.full((b,), t_int, device=condition.device, dtype=torch.long)
            x_p = self.roi_net(x, condition, t)
            eps = self.eps_net(x_p, condition, t)
            # The paper's beta range reaches 0.2.  With a 1000-step schedule
            # the cumulative product can underflow in float32, so protect the
            # reverse update from division by zero on the first sampling step.
            alpha_t = self.alpha_bar[t_int].clamp_min(1e-5)
            if i + 1 < len(times):
                prev_int = int(times[i + 1].item())
                alpha_prev = self.alpha_bar[prev_int].clamp_min(1e-5)
            else:
                alpha_prev = torch.tensor(1.0, device=condition.device, dtype=condition.dtype)
            x0 = (x_p - (1.0 - alpha_t).sqrt() * eps) / alpha_t.sqrt()
            x0 = torch.nan_to_num(x0, nan=0.0, posinf=20.0, neginf=-20.0)
            x0 = x0.clamp(-20.0, 20.0)
            x = alpha_prev.sqrt() * x0 + (1.0 - alpha_prev).sqrt() * eps
        return x0


class TemporalChannelAttention1D(nn.Module):
    """Lightweight temporal/channel attention used by QRS-focused baselines."""

    def __init__(self, channels: int, reduction: int = 4):
        super().__init__()
        hidden = max(4, channels // reduction)
        self.channel = nn.Sequential(
            nn.AdaptiveAvgPool1d(1),
            nn.Conv1d(channels, hidden, 1),
            nn.GELU(),
            nn.Conv1d(hidden, channels, 1),
            nn.Sigmoid(),
        )
        self.temporal = nn.Sequential(
            nn.Conv1d(channels, hidden, 7, padding=3),
            nn.GELU(),
            nn.Conv1d(hidden, 1, 1),
            nn.Sigmoid(),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return x * self.channel(x) * self.temporal(x)


class _AttnResBlock1D(nn.Module):
    def __init__(self, in_channels: int, out_channels: int, stride: int = 1):
        super().__init__()
        self.conv1 = nn.Conv1d(in_channels, out_channels, 7, stride=stride, padding=3)
        self.norm1 = nn.GroupNorm(_groups(out_channels), out_channels)
        self.conv2 = nn.Conv1d(out_channels, out_channels, 5, padding=2)
        self.norm2 = nn.GroupNorm(_groups(out_channels), out_channels)
        self.attn = TemporalChannelAttention1D(out_channels)
        self.act = nn.GELU()
        self.skip = (nn.Conv1d(in_channels, out_channels, 1, stride=stride)
                     if in_channels != out_channels or stride != 1 else nn.Identity())

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        h = self.act(self.norm1(self.conv1(x)))
        h = self.norm2(self.conv2(h))
        h = self.attn(h)
        return self.act(h + self.skip(x))


class QRSTransAttnNet(nn.Module):
    """Attention CNN encoder-decoder adapted from QRS-TransAttn.

    The paper uses temporal/channel attention and an explicit QRS-focused
    objective.  The implementation keeps that mechanism while accepting an
    arbitrary signal length and channel count for controlled local tests.
    """

    def __init__(self, in_channels: int = 1, out_channels: int = 1,
                 base_channels: int = 16, output_activation: str = "tanh"):
        super().__init__()
        c0, c1, c2, c3 = base_channels, base_channels * 2, base_channels * 4, base_channels * 8
        self.output_activation = output_activation
        self.e0 = _AttnResBlock1D(in_channels, c0)
        self.e1 = _AttnResBlock1D(c0, c1, stride=2)
        self.e2 = _AttnResBlock1D(c1, c2, stride=2)
        self.e3 = _AttnResBlock1D(c2, c3, stride=2)
        self.mid = _AttnResBlock1D(c3, c3)
        self.d2 = _AttnResBlock1D(c3 + c2, c2)
        self.d1 = _AttnResBlock1D(c2 + c1, c1)
        self.d0 = _AttnResBlock1D(c1 + c0, c0)
        self.out = nn.Conv1d(c0, out_channels, 7, padding=3)

    @staticmethod
    def _up(x: torch.Tensor, ref: torch.Tensor) -> torch.Tensor:
        return F.interpolate(x, size=ref.size(-1), mode="linear", align_corners=False)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        e0 = self.e0(x)
        e1 = self.e1(e0)
        e2 = self.e2(e1)
        e3 = self.e3(e2)
        h = self.mid(e3)
        h = self.d2(torch.cat([self._up(h, e2), e2], dim=1))
        h = self.d1(torch.cat([self._up(h, e1), e1], dim=1))
        h = self.d0(torch.cat([self._up(h, e0), e0], dim=1))
        y = self.out(h)
        if self.output_activation == "tanh":
            y = torch.tanh(y)
        elif self.output_activation != "none":
            raise ValueError(f"unknown output_activation: {self.output_activation}")
        return y


class P2EWGANGenerator(AttentionUNet1D):
    """Paired U-Net generator used by the P2E-WGAN reproduction."""

    def __init__(self, in_channels: int = 1, out_channels: int = 1,
                 base_channels: int = 16):
        super().__init__(in_channels, out_channels, base_channels=base_channels,
                         output_activation="tanh")


class ConditionalPatchDiscriminator1D(nn.Module):
    """Conditional PatchGAN critic for concatenated PPG and ECG signals."""

    def __init__(self, condition_channels: int = 1, target_channels: int = 1,
                 base_channels: int = 16):
        super().__init__()
        self.net = nn.Sequential(
            nn.Conv1d(condition_channels + target_channels, base_channels, 7, 2, 3),
            nn.LeakyReLU(0.2, inplace=True),
            nn.Conv1d(base_channels, base_channels * 2, 5, 2, 2),
            nn.InstanceNorm1d(base_channels * 2), nn.LeakyReLU(0.2, inplace=True),
            nn.Conv1d(base_channels * 2, base_channels * 4, 5, 2, 2),
            nn.InstanceNorm1d(base_channels * 4), nn.LeakyReLU(0.2, inplace=True),
            nn.Conv1d(base_channels * 4, base_channels * 8, 5, 2, 2),
            nn.InstanceNorm1d(base_channels * 8), nn.LeakyReLU(0.2, inplace=True),
            nn.Conv1d(base_channels * 8, 1, 3, padding=1),
        )

    def forward(self, condition: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
        return self.net(torch.cat([condition, target], dim=1))


class LightweightPPG2ECG(nn.Module):
    """Compact multi-kernel residual network inspired by Li et al. (2024)."""

    def __init__(self, in_channels: int = 1, out_channels: int = 1,
                 width: int = 32, output_activation: str = "tanh"):
        super().__init__()
        self.output_activation = output_activation
        self.stem = nn.Conv1d(in_channels, width, 7, padding=3)
        self.branches = nn.ModuleList([
            nn.Conv1d(width, width // 4, 3, padding=1, groups=1),
            nn.Conv1d(width, width // 4, 5, padding=2, groups=1),
            nn.Conv1d(width, width // 4, 7, padding=3, groups=1),
            nn.Conv1d(width, width // 4, 11, padding=5, groups=1),
        ])
        self.mix = nn.Conv1d(width, width, 1)
        self.norm = nn.GroupNorm(_groups(width), width)
        self.attn = TemporalChannelAttention1D(width)
        self.res = nn.Sequential(
            nn.Conv1d(width, width, 5, padding=2), nn.GELU(),
            nn.Conv1d(width, width, 5, padding=2),
        )
        self.out = nn.Conv1d(width, out_channels, 7, padding=3)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        h = self.stem(x)
        h = self.mix(torch.cat([branch(h) for branch in self.branches], dim=1))
        h = self.attn(self.norm(h))
        h = F.gelu(h + self.res(h))
        y = self.out(h)
        if self.output_activation == "tanh":
            y = torch.tanh(y)
        elif self.output_activation != "none":
            raise ValueError(f"unknown output_activation: {self.output_activation}")
        return y


__all__ = [
    "AttentionUNet1D",
    "PatchDiscriminator1D",
    "SpectrogramDiscriminator",
    "RDDMCore",
    "stft_log_magnitude",
    "QRSTransAttnNet",
    "P2EWGANGenerator",
    "ConditionalPatchDiscriminator1D",
    "LightweightPPG2ECG",
]
