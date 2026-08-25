"""Coefficient-domain ECG decoder with an exact Haar IDWT synthesis path."""

from __future__ import annotations

from typing import Any

import torch
import torch.nn as nn

from .baseline import BaselineResBlock1D, BaselineUpsampleBlock1D


class HaarWavelet1D(nn.Module):
    """Orthonormal, boundary-free Haar DWT/IDWT for even-length signals.

    The SensSmartTech windows have length 2000, which is divisible by 2**4.
    Keeping this first coefficient decoder on Haar avoids ambiguous padding and
    makes the inverse path exactly reconstruct the predicted coefficients.
    """

    def __init__(self, levels: int = 4):
        super().__init__()
        if levels < 1 or levels > 6:
            raise ValueError("levels must be in [1, 6]")
        self.levels = int(levels)
        self.register_buffer(
            "inv_sqrt2", torch.tensor(2.0 ** -0.5, dtype=torch.float32),
            persistent=False,
        )

    def decompose(self, signal: torch.Tensor) -> dict[str, Any]:
        if signal.dim() != 3:
            raise ValueError(f"signal must have shape [B,C,T], got {tuple(signal.shape)}")
        required = 2 ** self.levels
        if signal.size(-1) % required != 0:
            raise ValueError(
                f"signal length {signal.size(-1)} must be divisible by {required}"
            )
        approx = signal
        details: list[torch.Tensor] = []
        scale = self.inv_sqrt2.to(signal)
        for _ in range(self.levels):
            even = approx[..., 0::2]
            odd = approx[..., 1::2]
            approx = (even + odd) * scale
            details.append((even - odd) * scale)
        return {"approx": approx, "details": tuple(details)}

    def reconstruct(self, coefficients: dict[str, Any]) -> torch.Tensor:
        approx = coefficients["approx"]
        details = coefficients["details"]
        if len(details) != self.levels:
            raise ValueError("coefficient detail count does not match levels")
        scale = self.inv_sqrt2.to(approx)
        for detail in reversed(details):
            if detail.shape != approx.shape:
                raise ValueError(
                    f"approx/detail shape mismatch: {tuple(approx.shape)} vs {tuple(detail.shape)}"
                )
            even = (approx + detail) * scale
            odd = (approx - detail) * scale
            approx = torch.stack((even, odd), dim=-1).reshape(
                *even.shape[:-1], even.size(-1) * 2,
            )
        return approx

    def forward(self, signal: torch.Tensor) -> dict[str, Any]:
        return self.decompose(signal)


class _CoefficientHead(nn.Module):
    def __init__(self, in_channels: int, out_channels: int, hidden: int):
        super().__init__()
        self.net = nn.Sequential(
            nn.Conv1d(in_channels, hidden, kernel_size=7, padding=3),
            nn.GELU(),
            nn.Conv1d(hidden, out_channels, kernel_size=7, padding=3),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.net(x)


def _zero_coefficient_head_output(head: _CoefficientHead) -> None:
    """Start an optional residual coefficient head as an exact zero path."""
    output = head.net[-1]
    if not isinstance(output, nn.Conv1d):
        raise TypeError("coefficient head must end with a Conv1d")
    nn.init.zeros_(output.weight)
    if output.bias is not None:
        nn.init.zeros_(output.bias)


class _LocalResidualBlock1D(nn.Module):
    """Amplitude-preserving local context block for high-rate coefficients."""

    def __init__(self, channels: int, dilation: int = 1):
        super().__init__()
        dilation = max(1, int(dilation))
        padding = dilation
        self.conv1 = nn.Conv1d(
            channels, channels, kernel_size=3,
            padding=padding, dilation=dilation,
        )
        self.conv2 = nn.Conv1d(
            channels, channels, kernel_size=3,
            padding=padding, dilation=dilation,
        )
        self.act = nn.GELU()
        # Start as a near-identity path so the new branch cannot erase the
        # morphology learned by the v0.55 decoder at initialization.
        self.residual_scale = nn.Parameter(torch.tensor(0.1))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        update = self.conv2(self.act(self.conv1(x)))
        return x + self.residual_scale * update


class WaveletECGDecoder(nn.Module):
    """Generate wavelet coefficients and synthesize ECG with a fixed Haar IDWT.

    The temporal decoder mirrors the existing residual baseline until the
    coefficient resolutions are reached:

    ``A4/D4: T/16, D3: T/8, D2: T/4, D1: T/2``.
    """

    def __init__(
        self,
        signal_length: int = 2000,
        latent_dim: int = 128,
        ecg_leads: int = 4,
        base_channels: int = 32,
        dropout: float = 0.0,
        levels: int = 4,
        skip_base_channels: int | None = None,
        highfreq_refine: bool = False,
        highfreq_blocks: int = 2,
        highfreq_hidden: int | None = None,
        highfreq_gain: float = 1.0,
        highfreq_residual: bool = False,
    ):
        super().__init__()
        if signal_length % (2 ** levels) != 0:
            raise ValueError("signal_length must be divisible by 2**levels")
        if ecg_leads < 1 or base_channels < 8:
            raise ValueError("ecg_leads must be positive and base_channels >= 8")

        self.signal_length = int(signal_length)
        self.ecg_leads = int(ecg_leads)
        self.levels = int(levels)
        self.base_channels = int(base_channels)
        self.skip_base_channels = int(skip_base_channels or base_channels)
        if self.skip_base_channels < 8:
            raise ValueError("skip_base_channels must be at least 8")
        self.wavelet = HaarWavelet1D(levels=levels)
        self.highfreq_refine = bool(highfreq_refine)
        self.highfreq_blocks = max(0, int(highfreq_blocks)) if highfreq_refine else 0
        if highfreq_hidden is None:
            highfreq_hidden = max(int(base_channels), 32)
        self.highfreq_hidden = max(8, int(highfreq_hidden))
        self.highfreq_gain = float(highfreq_gain)
        if self.highfreq_gain <= 0.0:
            raise ValueError("highfreq_gain must be positive")
        self.highfreq_residual = bool(highfreq_residual and self.highfreq_refine)

        c = int(base_channels)
        self.bottleneck = BaselineResBlock1D(latent_dim, c * 4, dropout=dropout)
        self.up1 = BaselineUpsampleBlock1D(c * 4, c * 4, dropout=dropout)
        self.dec1 = BaselineResBlock1D(c * 8, c * 4, dropout=dropout)
        self.up2 = BaselineUpsampleBlock1D(c * 4, c * 2, dropout=dropout)
        self.dec2 = BaselineResBlock1D(c * 4, c * 2, dropout=dropout)
        self.up3 = BaselineUpsampleBlock1D(c * 2, c, dropout=dropout)
        self.dec3 = BaselineResBlock1D(c * 2, c, dropout=dropout)

        # A wider encoder can retain more information in its skip tensors
        # without forcing the wavelet decoder's output branches to widen too.
        # These projections keep the coefficient decoder a controlled
        # high-resolution bottleneck rather than silently dropping skips.
        skip_c = self.skip_base_channels
        self.skip2_proj = (
            nn.Identity() if skip_c == c
            else nn.Conv1d(skip_c * 4, c * 4, kernel_size=1)
        )
        self.skip1_proj = (
            nn.Identity() if skip_c == c
            else nn.Conv1d(skip_c * 2, c * 2, kernel_size=1)
        )
        self.skip0_proj = (
            nn.Identity() if skip_c == c
            else nn.Conv1d(skip_c, c, kernel_size=1)
        )

        # D1/D2 retain the finest temporal grids.  These blocks use no
        # normalization, so a sharp coefficient excursion is not converted
        # into a small batch-relative activation before the output head.
        detail2_blocks = tuple(
            _LocalResidualBlock1D(c * 2, dilation=2 ** (idx % 3))
            for idx in range(self.highfreq_blocks)
        )
        detail1_blocks = tuple(
            _LocalResidualBlock1D(c, dilation=2 ** (idx % 3))
            for idx in range(self.highfreq_blocks)
        )
        self.detail2_refine = nn.Sequential(*detail2_blocks)
        self.detail1_refine = nn.Sequential(*detail1_blocks)

        hidden = max(c // 2, 8)
        high_hidden = self.highfreq_hidden if self.highfreq_refine else hidden
        self.approx_head = _CoefficientHead(c * 4, ecg_leads, hidden)
        self.detail4_head = _CoefficientHead(c * 4, ecg_leads, hidden)
        self.detail3_head = _CoefficientHead(c * 4, ecg_leads, hidden)
        detail_hidden = hidden if self.highfreq_residual else high_hidden
        self.detail2_head = _CoefficientHead(c * 2, ecg_leads, detail_hidden)
        self.detail1_head = _CoefficientHead(c, ecg_leads, detail_hidden)
        if self.highfreq_residual:
            # Keep the v0.55 coefficient heads as the base morphology path.
            # The new local heads start at zero, so training cannot erase that
            # path before it has learned a useful high-frequency correction.
            self.detail2_residual_head = _CoefficientHead(
                c * 2, ecg_leads, high_hidden,
            )
            self.detail1_residual_head = _CoefficientHead(
                c, ecg_leads, high_hidden,
            )
            _zero_coefficient_head_output(self.detail2_residual_head)
            _zero_coefficient_head_output(self.detail1_residual_head)
        else:
            self.detail2_residual_head = None
            self.detail1_residual_head = None

    @staticmethod
    def _skip_or_zeros(
        x: torch.Tensor,
        skip: torch.Tensor | None,
        projection: nn.Module,
    ) -> torch.Tensor:
        if skip is None:
            return torch.zeros_like(x)
        skip = projection(skip)
        if skip.size(-1) != x.size(-1):
            skip = torch.nn.functional.interpolate(
                skip, size=x.size(-1), mode="linear", align_corners=False,
            )
        if skip.size(1) != x.size(1):
            raise ValueError(
                f"skip channels {skip.size(1)} do not match decoder channels {x.size(1)}"
            )
        return skip

    def _decode_coefficients(
        self, encoded: torch.Tensor | dict[str, Any]
    ) -> dict[str, Any]:
        if isinstance(encoded, dict):
            latent = encoded["latent"]
            skips = encoded.get("skips")
            if skips is not None and len(skips) != 3:
                raise ValueError("wavelet encoder skips must contain three tensors")
        else:
            latent = encoded
            skips = None

        x = self.bottleneck(latent)  # T/16 -> A4/D4
        approx = self.approx_head(x)
        detail4 = self.detail4_head(x)

        skip0, skip1, skip2 = (skips if skips is not None else (None, None, None))
        target = skip2.size(-1) if skip2 is not None else x.size(-1) * 2
        x = self.up1(x, target)
        x = self.dec1(torch.cat([
            x, self._skip_or_zeros(x, skip2, self.skip2_proj),
        ], dim=1))
        detail3 = self.detail3_head(x)

        target = skip1.size(-1) if skip1 is not None else x.size(-1) * 2
        x = self.up2(x, target)
        x = self.dec2(torch.cat([
            x, self._skip_or_zeros(x, skip1, self.skip1_proj),
        ], dim=1))
        if self.highfreq_residual:
            detail2 = self.detail2_head(x) + self.highfreq_gain * (
                self.detail2_residual_head(self.detail2_refine(x))
            )
        else:
            detail2 = self.highfreq_gain * self.detail2_head(
                self.detail2_refine(x),
            )

        target = skip0.size(-1) if skip0 is not None else x.size(-1) * 2
        x = self.up3(x, target)
        x = self.dec3(torch.cat([
            x, self._skip_or_zeros(x, skip0, self.skip0_proj),
        ], dim=1))
        if self.highfreq_residual:
            detail1 = self.detail1_head(x) + self.highfreq_gain * (
                self.detail1_residual_head(self.detail1_refine(x))
            )
        else:
            detail1 = self.highfreq_gain * self.detail1_head(
                self.detail1_refine(x),
            )

        return {
            "approx": approx,
            # Detail order is fine-to-coarse: D1, D2, D3, D4.
            "details": (detail1, detail2, detail3, detail4),
        }

    def forward(
        self,
        encoded: torch.Tensor | dict[str, Any],
        return_coeffs: bool = False,
    ) -> torch.Tensor | dict[str, Any]:
        coefficients = self._decode_coefficients(encoded)
        fused = self.wavelet.reconstruct(coefficients)
        if fused.size(-1) != self.signal_length:
            raise RuntimeError(
                f"IDWT returned length {fused.size(-1)}, expected {self.signal_length}"
            )
        if return_coeffs:
            return {"fused": fused, "coefficients": coefficients}
        return fused


__all__ = ["HaarWavelet1D", "WaveletECGDecoder"]
