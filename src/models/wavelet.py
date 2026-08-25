"""Differentiable, shift-stable wavelet utilities for ECG losses.

The transform is a stationary wavelet transform (SWT): every coefficient keeps
the original time resolution, so a local QRS complex remains local in the
coefficient sequence. Filters are fixed buffers and therefore add no trainable
parameters. The implementation intentionally avoids a NumPy/PyWavelets hop in
the training graph.
"""

from __future__ import annotations

import torch
import torch.nn as nn
import torch.nn.functional as F


# Symlet-4 analysis filters. Keeping the coefficients here makes the loss
# self-contained and reproducible across environments.
_SYM4_DEC_LO = (
    -0.07576571478927333,
    -0.02963552764599851,
    0.49761866763201545,
    0.8037387518059161,
    0.29785779560527736,
    -0.09921954357684722,
    -0.012603967262037833,
    0.0322231006040427,
)
_SYM4_DEC_HI = (
    -0.0322231006040427,
    -0.012603967262037833,
    0.09921954357684722,
    0.29785779560527736,
    -0.8037387518059161,
    0.49761866763201545,
    0.02963552764599851,
    -0.07576571478927333,
)


class Symlet4SWT(nn.Module):
    """Fixed Symlet-4 stationary wavelet decomposition.

    At 250 Hz, levels 2-4 roughly cover the 8-62.5 Hz QRS range, while the
    coarser approximation captures baseline and slower P/T morphology.
    """

    def __init__(self, levels: int = 5):
        super().__init__()
        if levels < 1 or levels > 8:
            raise ValueError("levels must be in [1, 8]")
        self.levels = int(levels)
        self.register_buffer(
            "dec_lo", torch.tensor(_SYM4_DEC_LO, dtype=torch.float32),
            persistent=False,
        )
        self.register_buffer(
            "dec_hi", torch.tensor(_SYM4_DEC_HI, dtype=torch.float32),
            persistent=False,
        )

    @staticmethod
    def _dilated_filter(filter_: torch.Tensor, level: int) -> torch.Tensor:
        dilation = 2 ** (level - 1)
        length = (filter_.numel() - 1) * dilation + 1
        result = filter_.new_zeros(length)
        result[::dilation] = filter_
        # Conv1d is cross-correlation; reverse the taps for wavelet convolution.
        return result.flip(0)

    @staticmethod
    def _same_depthwise_conv(
        signal: torch.Tensor,
        filter_: torch.Tensor,
    ) -> torch.Tensor:
        channels = signal.size(1)
        kernel = filter_.view(1, 1, -1).expand(channels, -1, -1)
        kernel_size = filter_.numel()
        left = kernel_size // 2
        right = kernel_size - 1 - left
        padded = F.pad(signal, (left, right), mode="reflect")
        return F.conv1d(padded, kernel, groups=channels)

    def forward(
        self, signal: torch.Tensor,
    ) -> dict[str, torch.Tensor | tuple[torch.Tensor, ...]]:
        if signal.dim() != 3:
            raise ValueError(f"signal must have shape [B,C,T], got {tuple(signal.shape)}")
        if signal.size(-1) <= 2 ** (self.levels - 1) * 8:
            raise ValueError("signal is too short for the requested SWT levels")

        current = signal.float()
        details: list[torch.Tensor] = []
        for level in range(1, self.levels + 1):
            low = self._dilated_filter(self.dec_lo, level).to(current)
            high = self._dilated_filter(self.dec_hi, level).to(current)
            details.append(self._same_depthwise_conv(current, high))
            current = self._same_depthwise_conv(current, low)
        return {"approx": current, "details": tuple(details)}


__all__ = ["Symlet4SWT"]
