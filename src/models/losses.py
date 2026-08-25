"""PPG2ECG 复合损失函数。

损失组成:
  L = λ₁·MSE + λ₂·L1 + λ₃·QRS-L1 + λ₄·Derivative
      + λ₅·MultiScale-STFT + λ₆·DTW + λ₇·PerceptualLoss

  - MSE:        逐点均方误差
  - L1:         全局逐点绝对误差
  - QRS-L1:     目标 ECG QRS 区域的加权绝对误差
  - QRS-amplitude: QRS 局部峰值与能量幅度监督
  - Derivative: 一阶差分绝对误差，约束上升沿/下降沿
  - DTW:        动态时间规整 (衡量形状相似性)
  - FreqLoss:   频域 L1 损失 (STFT 幅值谱)
  - Perceptual: 基于预训练特征的感受损失
  - Contrastive: 跨模态对比损失 (InfoNCE)
"""

from __future__ import annotations

from typing import Any

import torch
import torch.nn as nn
import torch.nn.functional as F

from .wavelet import Symlet4SWT
from .wavelet_decoder import HaarWavelet1D


class MSELoss(nn.Module):
    """逐点均方误差。"""

    def forward(self, pred: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
        return F.mse_loss(pred, target)


class L1Loss(nn.Module):
    """全局逐点绝对误差。"""

    def forward(self, pred: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
        return F.l1_loss(pred, target)


def _same_avg_pool1d(x: torch.Tensor, kernel_size: int) -> torch.Tensor:
    """Reflect-padded average pooling with unchanged temporal length."""
    kernel_size = max(1, int(kernel_size))
    if kernel_size == 1:
        return x
    if kernel_size % 2 == 0:
        kernel_size += 1
    pad = kernel_size // 2
    return F.avg_pool1d(F.pad(x, (pad, pad), mode="reflect"), kernel_size, stride=1)


class QRSWeightedL1Loss(nn.Module):
    """QRS-focused L1 using a target-only soft temporal mask.

    The mask is an auxiliary training signal; it is never passed to the
    encoder, flow, decoder, or evaluator at inference time.
    """

    def __init__(
        self,
        sample_rate: float = 250.0,
        qrs_width_ms: float = 120.0,
        baseline_ms: float = 400.0,
        peak_window_ms: float = 80.0,
        peak_threshold: float = 0.45,
        lead_index: int = 1,
        scale_floor: float = 0.05,
    ):
        super().__init__()
        self.sample_rate = float(sample_rate)
        self.qrs_width_ms = float(qrs_width_ms)
        self.baseline_ms = float(baseline_ms)
        self.peak_window_ms = float(peak_window_ms)
        self.peak_threshold = float(peak_threshold)
        self.lead_index = int(lead_index)
        self.scale_floor = float(scale_floor)

    def estimate_mask(self, target: torch.Tensor) -> torch.Tensor:
        if target.dim() != 3:
            raise ValueError(f"expected target [B,C,T], got {tuple(target.shape)}")
        lead = min(max(self.lead_index, 0), target.size(1) - 1)
        signal = target[:, lead:lead + 1]
        baseline_window = max(3, round(self.baseline_ms * self.sample_rate / 1000.0))
        detrended = signal - _same_avg_pool1d(signal, baseline_window)
        score = detrended.abs()

        peak_window = max(3, round(self.peak_window_ms * self.sample_rate / 1000.0))
        if peak_window % 2 == 0:
            peak_window += 1
        local_max = F.max_pool1d(
            score, kernel_size=peak_window, stride=1, padding=peak_window // 2,
        )
        scale = score.amax(dim=-1, keepdim=True).clamp_min(1e-6)
        candidates = (
            (score >= local_max - 1e-6)
            & (score >= self.peak_threshold * scale)
        ).to(score.dtype)

        qrs_window = max(3, round(self.qrs_width_ms * self.sample_rate / 1000.0))
        if qrs_window % 2 == 0:
            qrs_window += 1
        mask = F.max_pool1d(
            candidates, kernel_size=qrs_window, stride=1, padding=qrs_window // 2,
        ).clamp(0.0, 1.0)
        fallback = (score / scale).clamp(0.0, 1.0)
        has_region = mask.sum(dim=-1, keepdim=True) > 0
        return torch.where(has_region, mask, fallback)

    def forward(self, pred: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
        mask = self.estimate_mask(target).detach()
        diff = torch.abs(pred - target)
        weighted = diff * mask
        denominator = mask.sum() * pred.size(1)
        return weighted.sum() / denominator.clamp_min(1.0)


class QRSAmplitudeLoss(nn.Module):
    """Explicitly supervise local QRS amplitude and energy.

    The target-only QRS mask is shared with :class:`QRSWeightedL1Loss`, but
    the objective is deliberately amplitude-sensitive: it compares the
    detrended absolute peak and RMS excursion for every ECG lead.  This keeps
    a model from satisfying a high-frequency loss with a narrow, low-amplitude
    spike.  The mask is detached and is never an inference-time input.
    """

    def __init__(
        self,
        sample_rate: float = 250.0,
        qrs_width_ms: float = 120.0,
        baseline_ms: float = 400.0,
        peak_window_ms: float = 80.0,
        peak_threshold: float = 0.45,
        lead_index: int = 1,
        scale_floor: float = 0.05,
    ):
        super().__init__()
        self.mask_estimator = QRSWeightedL1Loss(
            sample_rate=sample_rate,
            qrs_width_ms=qrs_width_ms,
            baseline_ms=baseline_ms,
            peak_window_ms=peak_window_ms,
            peak_threshold=peak_threshold,
            lead_index=lead_index,
        )
        self.sample_rate = float(sample_rate)
        self.baseline_ms = float(baseline_ms)
        self.scale_floor = float(scale_floor)

    def forward(
        self,
        pred: torch.Tensor,
        target: torch.Tensor,
        mask_target: torch.Tensor | None = None,
    ) -> torch.Tensor:
        # ``mask_target`` lets branch-local losses use the full ECG to locate
        # QRS complexes while still comparing amplitudes in the branch domain.
        mask_source = target if mask_target is None else mask_target
        mask = self.mask_estimator.estimate_mask(mask_source).detach()
        baseline_window = max(
            3, round(self.baseline_ms * self.sample_rate / 1000.0),
        )
        pred_excursion = pred - _same_avg_pool1d(pred, baseline_window)
        target_excursion = target - _same_avg_pool1d(target, baseline_window)
        mask = mask.expand_as(pred)
        count = mask.sum(dim=-1).clamp_min(1.0)

        pred_abs = pred_excursion.abs()
        target_abs = target_excursion.abs()
        pred_peak = (pred_abs * mask).amax(dim=-1)
        target_peak = (target_abs * mask).amax(dim=-1)
        pred_rms = ((pred_abs.square() * mask).sum(dim=-1) / count).sqrt()
        target_rms = ((target_abs.square() * mask).sum(dim=-1) / count).sqrt()

        peak_scale = target_peak.detach().clamp_min(self.scale_floor)
        rms_scale = target_rms.detach().clamp_min(self.scale_floor)
        peak_error = (pred_peak - target_peak).abs() / peak_scale
        rms_error = (pred_rms - target_rms).abs() / rms_scale
        return 0.5 * (peak_error.mean() + rms_error.mean())


class DerivativeLoss(nn.Module):
    """First-difference L1 loss for ECG slopes and sharp transitions."""

    def forward(self, pred: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
        pred_diff = pred[..., 1:] - pred[..., :-1]
        target_diff = target[..., 1:] - target[..., :-1]
        return F.l1_loss(pred_diff, target_diff)


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
    """Multi-scale log-magnitude STFT L1 auxiliary loss."""

    def __init__(self, n_ffts: tuple[int, ...] = (64, 128, 256, 512)):
        super().__init__()
        self.n_ffts = tuple(int(n) for n in n_ffts)

    def forward(self, pred: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
        batch, channels, length = pred.shape
        # STFT kernels are more portable in float32 than under autocast half.
        pred_flat = pred.float().reshape(batch * channels, length)
        target_flat = target.float().reshape(batch * channels, length)
        losses = []
        for n_fft in self.n_ffts:
            if n_fft > length:
                continue
            hop_length = max(1, n_fft // 4)
            window = torch.hann_window(
                n_fft, device=pred_flat.device, dtype=pred_flat.dtype,
            )
            p_stft = torch.stft(
                pred_flat, n_fft, hop_length, window=window,
                return_complex=True, center=True,
            ).abs()
            t_stft = torch.stft(
                target_flat, n_fft, hop_length, window=window,
                return_complex=True, center=True,
            ).abs()
            losses.append(F.l1_loss(torch.log1p(p_stft), torch.log1p(t_stft)))
        if not losses:
            return pred.new_zeros(())
        return torch.stack(losses).mean()


class WaveletCoefficientLoss(nn.Module):
    """Energy-normalized multi-scale Symlet-4 SWT coefficient loss.

    Unlike a global FFT band error, each coefficient retains its time index.
    This lets the loss distinguish a periodic QRS transient from diffuse high-
    frequency noise. The normalization prevents small QRS coefficients from
    disappearing under the low-frequency ECG energy.
    """

    def __init__(
        self,
        levels: int = 5,
        approx_weight: float = 0.25,
        detail_weights: tuple[float, ...] | list[float] | None = None,
        normalize: bool = True,
        scale_floor: float = 0.05,
    ):
        super().__init__()
        self.transform = Symlet4SWT(levels=levels)
        configured = tuple(detail_weights or (0.25, 1.0, 1.0, 1.0, 0.5))
        if len(configured) != levels:
            raise ValueError("detail_weights must have one value per SWT level")
        if any(float(value) < 0.0 for value in configured) or approx_weight < 0.0:
            raise ValueError("wavelet weights must be non-negative")
        self.approx_weight = float(approx_weight)
        self.detail_weights = tuple(float(value) for value in configured)
        self.normalize = bool(normalize)
        self.scale_floor = float(scale_floor)

    def _term(self, prediction: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
        difference = torch.abs(prediction - target)
        if not self.normalize:
            return difference.mean()
        scale = target.abs().mean(dim=-1, keepdim=True).clamp_min(self.scale_floor)
        return (difference / scale).mean()

    def forward(self, pred: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
        pred_coeffs = self.transform(pred)
        target_coeffs = self.transform(target)
        terms = [self.approx_weight * self._term(
            pred_coeffs["approx"], target_coeffs["approx"],
        )]
        for weight, prediction, reference in zip(
            self.detail_weights,
            pred_coeffs["details"],
            target_coeffs["details"],
        ):
            terms.append(weight * self._term(prediction, reference))
        denominator = self.approx_weight + sum(self.detail_weights)
        if denominator == 0.0:
            return pred.new_zeros(())
        return torch.stack(terms).sum() / denominator


class WaveletQRSConsistencyLoss(nn.Module):
    """Match the time-localized QRS energy envelope in SWT detail levels.

    At 250 Hz, detail levels 2-4 cover approximately 8-62.5 Hz. The target
    envelope is normalized per lead/window, so this term supervises where a
    periodic transient occurs rather than rewarding diffuse high-frequency
    energy everywhere.
    """

    def __init__(
        self,
        levels: int = 5,
        qrs_levels: tuple[int, ...] | list[int] = (2, 3, 4),
        smooth_kernel: int = 3,
        scale_floor: float = 0.05,
    ):
        super().__init__()
        self.transform = Symlet4SWT(levels=levels)
        selected = tuple(int(level) for level in qrs_levels)
        if not selected or any(level < 1 or level > levels for level in selected):
            raise ValueError("qrs_levels must be valid SWT detail levels")
        self.qrs_levels = selected
        self.smooth_kernel = max(1, int(smooth_kernel))
        self.scale_floor = float(scale_floor)

    def _envelope(
        self,
        coeffs: dict[str, torch.Tensor | tuple[torch.Tensor, ...]],
    ) -> torch.Tensor:
        details = coeffs["details"]
        energy = sum(details[level - 1].square() for level in self.qrs_levels)
        if self.smooth_kernel > 1:
            kernel = self.smooth_kernel
            if kernel % 2 == 0:
                kernel += 1
            pad = kernel // 2
            energy = F.avg_pool1d(
                F.pad(energy, (pad, pad), mode="reflect"),
                kernel, stride=1,
            )
        return energy

    def forward(self, pred: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
        pred_energy = self._envelope(self.transform(pred))
        target_energy = self._envelope(self.transform(target))
        scale = target_energy.amax(dim=-1, keepdim=True).clamp_min(self.scale_floor)
        return F.l1_loss(pred_energy / scale, target_energy / scale)


class HaarCoefficientLoss(nn.Module):
    """Compare directly generated Haar coefficients with target coefficients."""

    def __init__(
        self,
        levels: int = 4,
        approx_weight: float = 0.25,
        detail_weights: tuple[float, ...] | list[float] | None = None,
        scale_floor: float = 0.05,
    ):
        super().__init__()
        self.transform = HaarWavelet1D(levels=levels)
        self.approx_weight = float(approx_weight)
        self.detail_weights = tuple(
            float(value) for value in (detail_weights or (0.25, 1.0, 1.0, 1.0))
        )
        if len(self.detail_weights) != levels:
            raise ValueError("haar detail_weights must have one value per level")
        self.scale_floor = float(scale_floor)

    @staticmethod
    def _as_float(coefficients: dict[str, Any]) -> dict[str, Any]:
        return {
            "approx": coefficients["approx"].float(),
            "details": tuple(value.float() for value in coefficients["details"]),
        }

    def _term(self, prediction: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
        scale = target.abs().mean(dim=-1, keepdim=True).clamp_min(self.scale_floor)
        return (torch.abs(prediction - target) / scale).mean()

    def forward(
        self,
        pred: torch.Tensor,
        target: torch.Tensor,
        pred_coeffs: dict[str, Any] | None = None,
    ) -> torch.Tensor:
        prediction = self._as_float(pred_coeffs) if pred_coeffs is not None else self.transform(pred)
        reference = self.transform(target)
        terms = [self.approx_weight * self._term(
            prediction["approx"], reference["approx"],
        )]
        for weight, value, target_value in zip(
            self.detail_weights, prediction["details"], reference["details"],
        ):
            terms.append(weight * self._term(value, target_value))
        denominator = self.approx_weight + sum(self.detail_weights)
        return torch.stack(terms).sum() / max(denominator, 1e-6)


class HaarQRSWaveletEnvelopeLoss(nn.Module):
    """Match time-localized QRS energy from Haar detail levels."""

    def __init__(
        self,
        levels: int = 4,
        qrs_levels: tuple[int, ...] | list[int] = (2, 3, 4),
        smooth_kernel: int = 5,
        scale_floor: float = 0.05,
        lead_index: int = 1,
    ):
        super().__init__()
        self.transform = HaarWavelet1D(levels=levels)
        selected = tuple(int(level) for level in qrs_levels)
        if not selected or any(level < 1 or level > levels for level in selected):
            raise ValueError("haar qrs_levels must be valid detail levels")
        self.qrs_levels = selected
        self.smooth_kernel = max(1, int(smooth_kernel))
        self.scale_floor = float(scale_floor)
        self.lead_index = int(lead_index)

    def envelope(
        self,
        coefficients: dict[str, Any],
        signal_length: int,
    ) -> torch.Tensor:
        details = coefficients["details"]
        # Each detail has a different temporal grid. Interpolate every scale
        # before summing; adding D2/D3/D4 directly would mix incompatible
        # lengths (and would fail for the normal 2000-point window).
        energy = sum(
            F.interpolate(
                details[level - 1].float().square(),
                size=signal_length,
                mode="linear",
                align_corners=False,
            )
            for level in self.qrs_levels
        )
        if self.smooth_kernel > 1:
            kernel = self.smooth_kernel
            if kernel % 2 == 0:
                kernel += 1
            pad = kernel // 2
            energy = F.avg_pool1d(
                F.pad(energy, (pad, pad), mode="reflect"), kernel, stride=1,
            )
        return energy

    def forward(
        self,
        pred: torch.Tensor,
        target: torch.Tensor,
        pred_coeffs: dict[str, Any] | None = None,
    ) -> torch.Tensor:
        prediction = pred_coeffs if pred_coeffs is not None else self.transform(pred)
        reference = self.transform(target)
        pred_energy = self.envelope(prediction, target.size(-1))
        target_energy = self.envelope(reference, target.size(-1))
        scale = target_energy.amax(dim=-1, keepdim=True).clamp_min(self.scale_floor)
        return F.l1_loss(pred_energy / scale, target_energy / scale)


class PeakIntervalLoss(nn.Module):
    """Differentiable relative RR-interval loss using target-only peak anchors.

    Target envelope maxima identify candidate R-peak locations. For each target
    anchor, the generated envelope is softly localized in a small neighborhood;
    only the resulting predicted interval differences receive gradients.
    """

    def __init__(
        self,
        levels: int = 4,
        qrs_levels: tuple[int, ...] | list[int] = (2, 3, 4),
        sample_rate: float = 250.0,
        peak_window_ms: float = 100.0,
        search_radius_ms: float = 80.0,
        peak_threshold: float = 0.25,
        softmax_temperature: float = 0.05,
        max_peaks: int = 16,
        lead_index: int = 1,
    ):
        super().__init__()
        self.envelope_loss = HaarQRSWaveletEnvelopeLoss(
            levels=levels, qrs_levels=qrs_levels, smooth_kernel=3,
            lead_index=lead_index,
        )
        self.sample_rate = float(sample_rate)
        self.peak_window_ms = float(peak_window_ms)
        self.search_radius_ms = float(search_radius_ms)
        self.peak_threshold = float(peak_threshold)
        self.softmax_temperature = float(softmax_temperature)
        self.max_peaks = max(2, int(max_peaks))
        self.lead_index = int(lead_index)

    def forward(
        self,
        pred: torch.Tensor,
        target: torch.Tensor,
        pred_coeffs: dict[str, Any] | None = None,
    ) -> torch.Tensor:
        prediction = pred_coeffs if pred_coeffs is not None else self.envelope_loss.transform(pred)
        reference = self.envelope_loss.transform(target)
        pred_env = self.envelope_loss.envelope(prediction, target.size(-1))
        target_env = self.envelope_loss.envelope(reference, target.size(-1))
        lead = min(max(self.lead_index, 0), target_env.size(1) - 1)
        pred_env = pred_env[:, lead]
        target_env = target_env[:, lead]

        length = target_env.size(-1)
        kernel = max(3, round(self.peak_window_ms * self.sample_rate / 1000.0))
        if kernel % 2 == 0:
            kernel += 1
        local_max = F.max_pool1d(
            target_env.unsqueeze(1), kernel, stride=1, padding=kernel // 2,
        ).squeeze(1)
        scale = target_env.amax(dim=-1, keepdim=True).clamp_min(1e-6)
        candidates = (
            (target_env >= local_max - 1e-7)
            & (target_env >= self.peak_threshold * scale)
        )
        k = min(self.max_peaks, length)
        masked = target_env.masked_fill(~candidates, float("-inf"))
        values, indices = torch.topk(masked, k=k, dim=-1)
        valid = torch.isfinite(values)
        indices, order = torch.sort(indices, dim=-1)
        valid = torch.gather(valid, -1, order)

        radius = max(1, round(self.search_radius_ms * self.sample_rate / 1000.0))
        offsets = torch.arange(-radius, radius + 1, device=target.device)
        positions = indices.unsqueeze(-1) + offsets.view(1, 1, -1)
        positions = positions.clamp(0, length - 1)
        local_values = torch.gather(
            pred_env.unsqueeze(1).expand(-1, k, -1), 2, positions,
        )
        temperature = max(self.softmax_temperature, 1e-4)
        soft_weights = torch.softmax(local_values / temperature, dim=-1)
        centers = (soft_weights * positions.float()).sum(dim=-1)

        pair_valid = valid[:, 1:] & valid[:, :-1]
        target_intervals = indices[:, 1:].float() - indices[:, :-1].float()
        pred_intervals = centers[:, 1:] - centers[:, :-1]
        minimum_interval = max(1.0, 0.25 * self.sample_rate)
        pair_valid &= target_intervals >= minimum_interval
        if not bool(pair_valid.any()):
            return pred.sum() * 0.0
        relative_error = torch.abs(pred_intervals - target_intervals) / target_intervals.clamp_min(1.0)
        return relative_error[pair_valid].mean()


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

    L = λ₁·MSE + λ₂·L1 + λ₃·QRS + λ₄·Derivative
        + λ₅·Freq + λ₆·DTW + λ₇·Perceptual

    模块选择: weights 为 {模块名: 权重} 字典, 权重为 0 或缺省的项
    不参与计算 (也跳过前向, 省算力)。可选模块: mse / l1 / qrs_weighted /
    qrs_amplitude / derivative / freq / wavelet / wavelet_qrs / dtw /
    perceptual。

    Args:
        weights: 各损失项权重, 如 {"mse": 1.0, "dtw": 0.5, "freq": 0.3, "perceptual": 0.1}
        ecg_leads: ECG 导联数 (须与数据集实际导联数一致)
    """

    MODULE_NAMES = (
        "mse", "l1", "qrs_weighted", "qrs_amplitude", "derivative", "freq", "wavelet",
        "wavelet_qrs", "haar_wavelet", "haar_qrs", "peak_interval",
        "dtw", "perceptual",
    )

    def __init__(
        self,
        weights: dict[str, float] | None = None,
        ecg_leads: int = 12,
        sample_rate: float = 250.0,
        qrs_config: dict | None = None,
        wavelet_config: dict | None = None,
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

        qrs_config = dict(qrs_config or {})
        self.mse = MSELoss()
        self.l1 = L1Loss()
        self.qrs_weighted = QRSWeightedL1Loss(
            sample_rate=sample_rate, **qrs_config,
        )
        self.qrs_amplitude = QRSAmplitudeLoss(
            sample_rate=sample_rate, **qrs_config,
        )
        self.derivative = DerivativeLoss()
        self.dtw = DTWLoss()
        self.freq = FrequencyLoss()
        wavelet_config = dict(wavelet_config or {})
        levels = int(wavelet_config.get("levels", 5))
        detail_weights = wavelet_config.get("detail_weights")
        self.wavelet = WaveletCoefficientLoss(
            levels=levels,
            approx_weight=float(wavelet_config.get("approx_weight", 0.25)),
            detail_weights=detail_weights,
            normalize=bool(wavelet_config.get("normalize", True)),
            scale_floor=float(wavelet_config.get("scale_floor", 0.05)),
        )
        self.wavelet_qrs = WaveletQRSConsistencyLoss(
            levels=levels,
            qrs_levels=wavelet_config.get("qrs_levels", (2, 3, 4)),
            smooth_kernel=int(wavelet_config.get("smooth_kernel", 3)),
            scale_floor=float(wavelet_config.get("scale_floor", 0.05)),
        )
        haar_levels = int(wavelet_config.get("haar_levels", 4))
        self.haar_wavelet = HaarCoefficientLoss(
            levels=haar_levels,
            approx_weight=float(wavelet_config.get("haar_approx_weight", 0.25)),
            detail_weights=wavelet_config.get("haar_detail_weights"),
            scale_floor=float(wavelet_config.get("scale_floor", 0.05)),
        )
        self.haar_qrs = HaarQRSWaveletEnvelopeLoss(
            levels=haar_levels,
            qrs_levels=wavelet_config.get("haar_qrs_levels", (2, 3, 4)),
            smooth_kernel=int(wavelet_config.get("smooth_kernel", 5)),
            scale_floor=float(wavelet_config.get("scale_floor", 0.05)),
        )
        self.peak_interval = PeakIntervalLoss(
            levels=haar_levels,
            qrs_levels=wavelet_config.get("haar_qrs_levels", (2, 3, 4)),
            sample_rate=sample_rate,
            peak_window_ms=float(wavelet_config.get("peak_window_ms", 100.0)),
            search_radius_ms=float(wavelet_config.get("search_radius_ms", 80.0)),
            peak_threshold=float(wavelet_config.get("peak_threshold", 0.25)),
            softmax_temperature=float(wavelet_config.get("softmax_temperature", 0.05)),
            max_peaks=int(wavelet_config.get("max_peaks", 16)),
            lead_index=int(wavelet_config.get("lead_index", 1)),
        )
        self.perc = PerceptualLoss(ecg_leads=ecg_leads)
        self._modules_map = {
            "mse": self.mse,
            "l1": self.l1,
            "qrs_weighted": self.qrs_weighted,
            "qrs_amplitude": self.qrs_amplitude,
            "derivative": self.derivative,
            "dtw": self.dtw,
            "freq": self.freq,
            "wavelet": self.wavelet,
            "wavelet_qrs": self.wavelet_qrs,
            "haar_wavelet": self.haar_wavelet,
            "haar_qrs": self.haar_qrs,
            "peak_interval": self.peak_interval,
            "perceptual": self.perc,
        }

    def forward(
        self,
        pred: torch.Tensor,
        target: torch.Tensor,
        pred_coeffs: dict[str, Any] | None = None,
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
            if name in {"haar_wavelet", "haar_qrs", "peak_interval"}:
                val = module(pred, target, pred_coeffs=pred_coeffs)
            else:
                val = module(pred, target)
            losses[name] = val
            total = total + w * val
        losses["total"] = total
        return losses
