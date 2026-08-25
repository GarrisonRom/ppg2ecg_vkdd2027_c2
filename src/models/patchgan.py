"""Lightweight conditional 1-D PatchGAN for PPG-to-ECG reconstruction.

The discriminator receives a paired signal ``[PPG, ECG]`` and returns a
sequence of patch logits rather than one global real/fake score.  It is kept
small and local so that the adversarial term can sharpen short ECG morphology
without replacing the supervised physiological objectives.
"""

from __future__ import annotations

import torch
import torch.nn as nn


class ConditionalPatchGAN1D(nn.Module):
    """Conditional 1-D PatchGAN discriminator with hinge-logit output."""

    def __init__(
        self,
        ppg_channels: int = 4,
        ecg_leads: int = 4,
        base_channels: int = 32,
        num_layers: int = 4,
        max_channels: int = 256,
        dropout: float = 0.0,
    ):
        super().__init__()
        if ppg_channels < 1 or ecg_leads < 1:
            raise ValueError("ppg_channels and ecg_leads must be positive")
        if base_channels < 8:
            raise ValueError("base_channels must be at least 8")
        if num_layers < 2:
            raise ValueError("num_layers must be at least 2")
        if not 0.0 <= dropout < 1.0:
            raise ValueError("dropout must be in [0, 1)")

        self.ppg_channels = int(ppg_channels)
        self.ecg_leads = int(ecg_leads)
        self.input_channels = self.ppg_channels + self.ecg_leads
        self.base_channels = int(base_channels)
        self.num_layers = int(num_layers)

        layers: list[nn.Module] = []
        in_channels = self.input_channels
        for layer_index in range(num_layers):
            out_channels = min(max_channels, base_channels * (2 ** layer_index))
            # The first block has no normalization, following the standard
            # PatchGAN convention; later blocks use GroupNorm for stable small
            # batch behaviour (the subject-balanced batches are only 16).
            layers.append(
                nn.Conv1d(
                    in_channels, out_channels, kernel_size=7,
                    stride=2, padding=3,
                )
            )
            if layer_index > 0:
                groups = min(8, out_channels)
                while groups > 1 and out_channels % groups != 0:
                    groups -= 1
                layers.append(nn.GroupNorm(groups, out_channels))
            layers.append(nn.LeakyReLU(0.2, inplace=True))
            if dropout > 0:
                layers.append(nn.Dropout1d(dropout))
            in_channels = out_channels

        # A final stride-1 convolution keeps a spatial patch map while adding
        # a little context beyond the last downsampling block.
        layers.append(nn.Conv1d(in_channels, 1, kernel_size=3, padding=1))
        self.net = nn.Sequential(*layers)

    def forward(self, ppg: torch.Tensor, ecg: torch.Tensor) -> torch.Tensor:
        if ppg.dim() != 3 or ecg.dim() != 3:
            raise ValueError("ppg and ecg must have shape [B,C,T]")
        if ppg.size(0) != ecg.size(0) or ppg.size(-1) != ecg.size(-1):
            raise ValueError("ppg and ecg batch/length dimensions must match")
        if ppg.size(1) != self.ppg_channels:
            raise ValueError(
                f"expected {self.ppg_channels} PPG channels, got {ppg.size(1)}"
            )
        if ecg.size(1) != self.ecg_leads:
            raise ValueError(
                f"expected {self.ecg_leads} ECG leads, got {ecg.size(1)}"
            )
        return self.net(torch.cat([ppg, ecg], dim=1))


def patchgan_hinge_discriminator_loss(
    real_logits: torch.Tensor,
    fake_logits: torch.Tensor,
) -> torch.Tensor:
    """Hinge loss for the discriminator."""
    return 0.5 * (
        torch.relu(1.0 - real_logits).mean()
        + torch.relu(1.0 + fake_logits).mean()
    )


def patchgan_hinge_generator_loss(fake_logits: torch.Tensor) -> torch.Tensor:
    """Non-saturating hinge generator objective."""
    return -fake_logits.mean()


__all__ = [
    "ConditionalPatchGAN1D",
    "patchgan_hinge_discriminator_loss",
    "patchgan_hinge_generator_loss",
]
