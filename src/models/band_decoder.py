"""Three-band ECG decoder for the v0.52 frequency-disentangled experiment.

The decoder still predicts in the time domain.  Each branch is projected into a
fixed FFT band before the three components are fused, so the experiment can be
interpreted as low-frequency baseline, morphology, and QRS-detail generation
without introducing a wavelet coefficient interface.
"""

from __future__ import annotations

from typing import Any

import torch
import torch.nn as nn

from .baseline import BaselineResBlock1D, BaselineUpsampleBlock1D


class GatedHighSkip1D(nn.Module):
    """Project and gate a length-preserving skip for the high band only."""

    def __init__(
        self,
        feature_channels: int,
        skip_channels: int,
        hidden_channels: int,
        dropout: float = 0.1,
    ):
        super().__init__()
        hidden_channels = max(8, int(hidden_channels))
        self.feature_proj = nn.Conv1d(feature_channels, hidden_channels, 1, bias=False)
        self.skip_proj = nn.Conv1d(skip_channels, hidden_channels, 1, bias=False)
        self.gate = nn.Conv1d(hidden_channels, 1, 1)
        self.out_proj = nn.Conv1d(skip_channels, hidden_channels, 1)
        self.dropout = nn.Dropout1d(dropout) if dropout > 0 else nn.Identity()

    def forward(self, features: torch.Tensor, skip: torch.Tensor) -> torch.Tensor:
        if skip.size(-1) != features.size(-1):
            skip = nn.functional.interpolate(
                skip, size=features.size(-1), mode="linear", align_corners=False,
            )
        score = torch.relu(self.feature_proj(features) + self.skip_proj(skip))
        gate = torch.sigmoid(self.gate(score))
        return self.dropout(self.out_proj(skip)) * gate


class MultiBandECGDecoder(nn.Module):
    """Generate low/mid/high ECG components and fuse them in the time domain."""

    band_names = ("low", "mid", "high")

    def __init__(
        self,
        signal_length: int = 2000,
        latent_dim: int = 128,
        ecg_leads: int = 4,
        base_channels: int = 32,
        dropout: float = 0.0,
        sample_rate: float = 250.0,
        low_max_hz: float = 0.5,
        mid_max_hz: float = 10.0,
        high_max_hz: float = 40.0,
        gated_high_skip: bool = False,
        full_skip_channels: int = 0,
        high_skip_dropout: float = 0.1,
    ):
        super().__init__()
        if signal_length < 16:
            raise ValueError("signal_length must be at least 16")
        if ecg_leads < 1:
            raise ValueError("ecg_leads must be positive")
        if base_channels < 8:
            raise ValueError("base_channels must be at least 8")
        if not 0.0 < low_max_hz < mid_max_hz < high_max_hz:
            raise ValueError("band cutoffs must satisfy 0 < low < mid < high")
        nyquist = float(sample_rate) / 2.0
        if high_max_hz >= nyquist:
            raise ValueError("high_max_hz must be below the Nyquist frequency")

        self.signal_length = int(signal_length)
        self.ecg_leads = int(ecg_leads)
        self.sample_rate = float(sample_rate)
        self.cutoffs_hz = (float(low_max_hz), float(mid_max_hz), float(high_max_hz))
        self.gated_high_skip = bool(gated_high_skip)
        self.full_skip_channels = int(full_skip_channels)

        c = int(base_channels)
        self.bottleneck = BaselineResBlock1D(latent_dim, c * 4, dropout=dropout)
        self.up1 = BaselineUpsampleBlock1D(c * 4, c * 4, dropout=dropout)
        self.dec1 = BaselineResBlock1D(c * 8, c * 4, dropout=dropout)
        self.up2 = BaselineUpsampleBlock1D(c * 4, c * 2, dropout=dropout)
        self.dec2 = BaselineResBlock1D(c * 4, c * 2, dropout=dropout)
        self.up3 = BaselineUpsampleBlock1D(c * 2, c, dropout=dropout)
        self.dec3 = BaselineResBlock1D(c * 2, c, dropout=dropout)
        self.up4 = BaselineUpsampleBlock1D(c, c // 2, dropout=dropout)
        self.dec4 = BaselineResBlock1D(c // 2, c // 2, dropout=dropout)

        if self.gated_high_skip:
            if self.full_skip_channels < 1:
                raise ValueError(
                    "full_skip_channels must be positive when gated_high_skip is enabled"
                )
            self.high_skip = GatedHighSkip1D(
                feature_channels=c // 2,
                skip_channels=self.full_skip_channels,
                hidden_channels=max(8, c // 2),
                dropout=high_skip_dropout,
            )
            self.high_skip_fuse = BaselineResBlock1D(
                c // 2 + max(8, c // 2), c // 2, dropout=dropout,
            )
            # Start exactly at the control path.  The high-resolution feature
            # can only influence the decoder after this scalar learns a
            # non-zero residual gain, which prevents an untrained skip from
            # replacing the stable low/mid/high backbone at epoch 1.
            self.high_skip_residual_scale = nn.Parameter(torch.zeros(1))
        else:
            self.high_skip = None
            self.high_skip_fuse = None
            self.high_skip_residual_scale = None

        branch_hidden = max(c // 2, 8)
        self.band_heads = nn.ModuleDict({
            name: nn.Sequential(
                nn.Conv1d(c // 2, branch_hidden, kernel_size=7, padding=3),
                nn.GELU(),
                nn.Conv1d(branch_hidden, ecg_leads, kernel_size=7, padding=3),
            )
            for name in self.band_names
        })
        # Identity-like initialization: initial fusion is a simple sum of the
        # three projected bands, while later training may learn lead-specific
        # mixing through this pointwise layer.
        self.fusion = nn.Conv1d(3 * ecg_leads, ecg_leads, kernel_size=1)
        nn.init.zeros_(self.fusion.weight)
        nn.init.zeros_(self.fusion.bias)
        with torch.no_grad():
            for band_index in range(3):
                for lead in range(ecg_leads):
                    self.fusion.weight[lead, band_index * ecg_leads + lead, 0] = 1.0

        frequencies = torch.fft.rfftfreq(self.signal_length, d=1.0 / self.sample_rate)
        self.register_buffer("frequencies", frequencies, persistent=False)
        masks = self._make_masks(frequencies)
        self.register_buffer("band_masks", masks, persistent=False)

    def _make_masks(self, frequencies: torch.Tensor) -> torch.Tensor:
        low_max, mid_max, high_max = self.cutoffs_hz
        masks = torch.stack([
            (frequencies <= low_max),
            (frequencies > low_max) & (frequencies <= mid_max),
            (frequencies > mid_max) & (frequencies <= high_max),
        ]).to(torch.float32)
        return masks[:, None, :]

    def project_bands(self, signal: torch.Tensor) -> dict[str, torch.Tensor]:
        """Project a time-domain signal into the fixed FFT bands."""
        if signal.dim() != 3 or signal.size(-1) != self.signal_length:
            raise ValueError(
                f"signal must have shape [B,C,{self.signal_length}], got {tuple(signal.shape)}"
            )
        original_dtype = signal.dtype
        spectrum = torch.fft.rfft(signal.float(), dim=-1)
        components = {}
        for index, name in enumerate(self.band_names):
            component = torch.fft.irfft(
                spectrum * self.band_masks[index],
                n=self.signal_length,
                dim=-1,
            )
            components[name] = component.to(original_dtype)
        return components

    @staticmethod
    def _skip_or_zeros(x: torch.Tensor, skip: torch.Tensor | None) -> torch.Tensor:
        if skip is None:
            return torch.zeros_like(x)
        if skip.size(-1) != x.size(-1):
            skip = torch.nn.functional.interpolate(
                skip, size=x.size(-1), mode="linear", align_corners=False,
            )
        if skip.size(1) != x.size(1):
            raise ValueError(
                f"skip channels {skip.size(1)} do not match decoder channels {x.size(1)}"
            )
        return skip

    def _decode_features(self, encoded: torch.Tensor | dict[str, Any]) -> torch.Tensor:
        if isinstance(encoded, dict):
            latent = encoded["latent"]
            skips = encoded.get("skips")
            if skips is not None and len(skips) != 3:
                raise ValueError("multiband encoder skips must contain three tensors")
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
        return self.dec4(x)

    def forward(
        self,
        encoded: torch.Tensor | dict[str, Any],
        return_bands: bool = False,
    ) -> torch.Tensor | dict[str, Any]:
        features = self._decode_features(encoded)
        high_features = features
        full_skip = encoded.get("full_skip") if isinstance(encoded, dict) else None
        if self.high_skip is not None and full_skip is not None:
            skip_residual = self.high_skip_fuse(
                torch.cat([features, self.high_skip(features, full_skip)], dim=1)
            )
            high_features = features + torch.tanh(self.high_skip_residual_scale) * skip_residual
        raw_bands = {
            name: self.band_heads[name](
                high_features if name == "high" else features
            )
            for name in self.band_names
        }
        bands = {}
        for index, name in enumerate(self.band_names):
            spectrum = torch.fft.rfft(raw_bands[name].float(), dim=-1)
            bands[name] = torch.fft.irfft(
                spectrum * self.band_masks[index],
                n=self.signal_length,
                dim=-1,
            ).to(raw_bands[name].dtype)
        fused = self.fusion(torch.cat([bands[name] for name in self.band_names], dim=1))
        if return_bands:
            return {"fused": fused, "bands": bands, "raw_bands": raw_bands}
        return fused

    def forward_with_bands(
        self, encoded: torch.Tensor | dict[str, Any]
    ) -> tuple[torch.Tensor, dict[str, torch.Tensor]]:
        output = self.forward(encoded, return_bands=True)
        return output["fused"], output["bands"]


__all__ = ["MultiBandECGDecoder", "GatedHighSkip1D"]
