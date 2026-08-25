"""VAE encoder, conditional flow generator, and adversarial subject head.

The module is deliberately separate from the v0.1 baseline.  The encoder
factorizes the PPG representation into content and style latents.  The flow
generator models ECG residual uncertainty conditioned on the complete latent,
while the subject discriminator receives only content through a GRL.
"""

from __future__ import annotations

import math
from typing import Any

import torch
import torch.nn as nn
import torch.nn.functional as F

from .baseline import BaselineECGDecoder, BaselineResBlock1D


def _kl_normal(mu: torch.Tensor, logvar: torch.Tensor) -> torch.Tensor:
    """Mean KL(q(z|x) || N(0, I)) over batch, channels, and time."""
    safe_logvar = torch.clamp(logvar, min=-8.0, max=8.0)
    return -0.5 * (1.0 + safe_logvar - mu.square() - safe_logvar.exp()).mean()


class VAEPPGEncoder(nn.Module):
    """Temporal VAE encoder with explicit content/style posterior heads.

    Input ``[B, C_ppg, T]`` is reduced by 16x in time.  The returned
    ``latent`` concatenates sampled content and style tensors, while
    ``z_content`` and ``z_style`` remain available for auxiliary objectives.
    In evaluation mode the posterior means are used instead of random samples.
    """

    def __init__(
        self,
        signal_length: int = 2000,
        latent_dim: int = 128,
        ppg_channels: int = 4,
        content_dim: int | None = None,
        style_dim: int | None = None,
        base_channels: int = 32,
        dropout: float = 0.1,
        full_skip_channels: int = 0,
    ):
        super().__init__()
        if ppg_channels < 1:
            raise ValueError("ppg_channels must be positive")
        if latent_dim < 4:
            raise ValueError("latent_dim must be at least 4")
        if base_channels < 8:
            raise ValueError("base_channels must be at least 8")

        content_dim = content_dim or latent_dim // 2
        style_dim = style_dim or latent_dim - content_dim
        if content_dim < 1 or style_dim < 1 or content_dim + style_dim != latent_dim:
            raise ValueError(
                "content_dim + style_dim must equal latent_dim and both must be positive"
            )

        self.signal_length = signal_length
        self.latent_dim = latent_dim
        self.ppg_channels = ppg_channels
        self.content_dim = content_dim
        self.style_dim = style_dim
        self.full_skip_channels = max(0, int(full_skip_channels))

        # Optional length-preserving feature for the gated high-frequency
        # decoder.  It is disabled by default so the v0.2/v0.61 encoder
        # checkpoints keep exactly the old parameterization.
        if self.full_skip_channels > 0:
            self.full_skip = nn.Sequential(
                nn.Conv1d(ppg_channels, self.full_skip_channels, kernel_size=7, padding=3),
                nn.GroupNorm(
                    min(8, self.full_skip_channels), self.full_skip_channels,
                ),
                nn.GELU(),
                nn.Dropout1d(dropout) if dropout > 0 else nn.Identity(),
            )
        else:
            self.full_skip = None

        self.stem = BaselineResBlock1D(
            ppg_channels, base_channels, stride=2, dropout=dropout,
        )
        self.enc1 = BaselineResBlock1D(
            base_channels, base_channels * 2, stride=2, dropout=dropout,
        )
        self.enc2 = BaselineResBlock1D(
            base_channels * 2, base_channels * 4, stride=2, dropout=dropout,
        )
        self.shared = BaselineResBlock1D(
            base_channels * 4, latent_dim, stride=2, dropout=dropout,
        )

        self.mu_content = nn.Conv1d(latent_dim, content_dim, kernel_size=3, padding=1)
        self.logvar_content = nn.Conv1d(latent_dim, content_dim, kernel_size=3, padding=1)
        self.mu_style = nn.Conv1d(latent_dim, style_dim, kernel_size=3, padding=1)
        self.logvar_style = nn.Conv1d(latent_dim, style_dim, kernel_size=3, padding=1)

    @staticmethod
    def _sample(mu: torch.Tensor, logvar: torch.Tensor, training: bool) -> torch.Tensor:
        logvar = torch.clamp(logvar, min=-8.0, max=8.0)
        if not training:
            return mu
        return mu + torch.randn_like(mu) * torch.exp(0.5 * logvar)

    def forward(self, ppg: torch.Tensor) -> dict[str, Any]:
        if ppg.dim() == 2:
            ppg = ppg.unsqueeze(1)
        if ppg.dim() != 3:
            raise ValueError(f"PPG must have shape [B,C,T], got {tuple(ppg.shape)}")
        if ppg.size(1) != self.ppg_channels:
            raise ValueError(
                f"expected {self.ppg_channels} PPG channels, got {ppg.size(1)}"
            )

        full_skip = self.full_skip(ppg) if self.full_skip is not None else None
        skip0 = self.stem(ppg)
        skip1 = self.enc1(skip0)
        skip2 = self.enc2(skip1)
        h = skip2
        h = self.shared(h)

        mu_c = self.mu_content(h)
        logvar_c = torch.clamp(self.logvar_content(h), min=-8.0, max=8.0)
        mu_s = self.mu_style(h)
        logvar_s = torch.clamp(self.logvar_style(h), min=-8.0, max=8.0)
        z_c = self._sample(mu_c, logvar_c, self.training)
        z_s = self._sample(mu_s, logvar_s, self.training)

        return {
            "latent": torch.cat([z_c, z_s], dim=1),
            # Keep the high-resolution temporal features available to
            # decoders that need local morphology (for example the v0.52
            # multi-band decoder).  These skips are deterministic features
            # from the encoder trunk; the VAE sampling remains confined to
            # the latent path above.
            "skips": (skip0, skip1, skip2),
            "full_skip": full_skip,
            # Posterior moments are exposed explicitly for cross-modal
            # alignment and latent transport.  ``latent`` remains the
            # sampled representation used by the older v0.2/v0.3 paths.
            "mu": torch.cat([mu_c, mu_s], dim=1),
            "logvar": torch.cat([logvar_c, logvar_s], dim=1),
            "z_content": z_c,
            "z_style": z_s,
            "mu_content": mu_c,
            "logvar_content": logvar_c,
            "mu_style": mu_s,
            "logvar_style": logvar_s,
            "kl_content": _kl_normal(mu_c, logvar_c),
            "kl_style": _kl_normal(mu_s, logvar_s),
        }


class CardioAlignEncoder(VAEPPGEncoder):
    """VAE encoder used by the PPGFlowECG-inspired paired pathway.

    The architecture is intentionally shared with the existing VAE encoder;
    the distinct class/registry name makes the experimental assumption
    explicit and keeps v0.2/v0.3 checkpoints untouched.
    """

    pass


class LatentRectifiedFlow(nn.Module):
    """A compact conditional rectified flow over temporal latent tensors.

    The training path transports a PPG posterior mean ``source`` to an ECG
    posterior mean ``target`` along linear interpolants.  At inference,
    deterministic Euler integration starts at the PPG latent and follows the
    learned vector field to an ECG latent endpoint.
    """

    def __init__(
        self,
        latent_dim: int = 128,
        hidden_channels: int = 128,
        num_layers: int = 4,
        dropout: float = 0.0,
        max_steps: int = 8,
    ):
        super().__init__()
        if latent_dim < 1 or hidden_channels < 8:
            raise ValueError("latent_dim must be positive and hidden_channels >= 8")
        if num_layers < 2:
            raise ValueError("num_layers must be at least 2")
        if not 0.0 <= dropout < 1.0:
            raise ValueError("dropout must be in [0, 1)")
        self.latent_dim = int(latent_dim)
        self.hidden_channels = int(hidden_channels)
        self.max_steps = int(max_steps)

        blocks: list[nn.Module] = [
            nn.Conv1d(2 * latent_dim + 1, hidden_channels, kernel_size=3, padding=1),
            nn.GroupNorm(min(8, hidden_channels), hidden_channels),
            nn.GELU(),
        ]
        for _ in range(num_layers - 2):
            blocks.extend([
                nn.Conv1d(hidden_channels, hidden_channels, kernel_size=3, padding=1),
                nn.GroupNorm(min(8, hidden_channels), hidden_channels),
                nn.GELU(),
                nn.Dropout1d(dropout) if dropout > 0 else nn.Identity(),
            ])
        blocks.append(nn.Conv1d(hidden_channels, latent_dim, kernel_size=3, padding=1))
        self.net = nn.Sequential(*blocks)
        # An initially small vector field avoids destabilizing the decoder
        # before the paired alignment terms have formed a useful latent space.
        nn.init.zeros_(self.net[-1].weight)
        nn.init.zeros_(self.net[-1].bias)

    def vector_field(
        self,
        x_t: torch.Tensor,
        condition: torch.Tensor,
        t: torch.Tensor | float,
    ) -> torch.Tensor:
        if x_t.shape != condition.shape or x_t.dim() != 3:
            raise ValueError(
                "x_t and condition must have the same shape [B, latent_dim, T]"
            )
        if not torch.is_tensor(t):
            t = x_t.new_tensor(float(t))
        if t.dim() == 0:
            t = t.view(1, 1, 1).expand(x_t.size(0), 1, x_t.size(-1))
        elif t.dim() == 1:
            t = t.view(-1, 1, 1).expand(-1, 1, x_t.size(-1))
        elif t.dim() == 2:
            t = t.unsqueeze(-1).expand(-1, 1, x_t.size(-1))
        inp = torch.cat([x_t, condition, t], dim=1)
        return self.net(inp)

    def loss(self, source: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
        if source.shape != target.shape:
            raise ValueError("source and target latent tensors must have equal shapes")
        batch = source.size(0)
        t = torch.rand(batch, device=source.device, dtype=source.dtype)
        t_view = t.view(batch, 1, 1)
        x_t = (1.0 - t_view) * source + t_view * target
        velocity = target - source
        return F.mse_loss(self.vector_field(x_t, source, t), velocity)

    @torch.no_grad()
    def integrate(
        self,
        source: torch.Tensor,
        steps: int | None = None,
    ) -> torch.Tensor:
        steps = int(steps or self.max_steps)
        if steps < 1:
            raise ValueError("steps must be positive")
        x = source
        dt = 1.0 / steps
        for index in range(steps):
            t = source.new_tensor(float(index) / steps)
            x = x + dt * self.vector_field(x, source, t)
        return x

    def forward(
        self,
        source: torch.Tensor,
        steps: int | None = None,
    ) -> torch.Tensor:
        # Keep gradients for the training-time endpoint reconstruction.
        steps = int(steps or self.max_steps)
        if steps < 1:
            raise ValueError("steps must be positive")
        x = source
        dt = 1.0 / steps
        for index in range(steps):
            t = source.new_tensor(float(index) / steps)
            x = x + dt * self.vector_field(x, source, t)
        return x


class ConditionalAffineCoupling1D(nn.Module):
    """A channel-wise affine coupling layer conditioned on ECG context."""

    def __init__(
        self,
        channels: int,
        condition_channels: int,
        hidden_channels: int = 32,
        mask_first: bool = True,
        max_log_scale: float = 0.8,
    ):
        super().__init__()
        if channels < 2:
            raise ValueError("flow channels must be at least 2")
        self.channels = channels
        self.max_log_scale = max_log_scale
        mask = torch.zeros(channels)
        split = channels // 2
        if mask_first:
            mask[:split] = 1.0
        else:
            mask[split:] = 1.0
        self.register_buffer("mask", mask.view(1, channels, 1))

        self.net = nn.Sequential(
            nn.Conv1d(channels + condition_channels, hidden_channels, 5, padding=2),
            nn.GroupNorm(min(8, hidden_channels), hidden_channels),
            nn.GELU(),
            nn.Conv1d(hidden_channels, hidden_channels, 3, padding=1),
            nn.GroupNorm(min(8, hidden_channels), hidden_channels),
            nn.GELU(),
            nn.Conv1d(hidden_channels, channels * 2, 3, padding=1),
        )
        # Start as identity so the VAE reconstruction path is stable at epoch 0.
        nn.init.zeros_(self.net[-1].weight)
        nn.init.zeros_(self.net[-1].bias)

    def _affine_parameters(self, x: torch.Tensor, condition: torch.Tensor):
        x_a = x * self.mask
        h = self.net(torch.cat([x_a, condition], dim=1))
        scale, shift = h.chunk(2, dim=1)
        scale = torch.tanh(scale) * self.max_log_scale * (1.0 - self.mask)
        shift = shift * (1.0 - self.mask)
        return scale, shift

    def forward(
        self,
        x: torch.Tensor,
        condition: torch.Tensor,
        inverse: bool = False,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        scale, shift = self._affine_parameters(x, condition)
        x_a = x * self.mask
        x_b = x * (1.0 - self.mask)
        if inverse:
            y_b = x_b * torch.exp(scale) + shift
            y = x_a + y_b
            logdet = scale.sum(dim=(1, 2))
        else:
            y_b = (x_b - shift) * torch.exp(-scale)
            y = x_a + y_b
            logdet = -scale.sum(dim=(1, 2))
        return y, logdet


class ConditionalFlowECGGenerator(nn.Module):
    """Conditional residual normalizing flow for ECG generation.

    ``forward`` returns the deterministic zero-noise mode for compatibility
    with the existing evaluator.  ``nll_per_sample`` trains the complete flow
    likelihood and ``sample`` exposes stochastic ECG generation for later work.
    """

    def __init__(
        self,
        signal_length: int = 2000,
        latent_dim: int = 128,
        ecg_leads: int = 4,
        base_channels: int = 24,
        num_flow_layers: int = 4,
        flow_hidden_channels: int = 32,
        dropout: float = 0.0,
    ):
        super().__init__()
        if ecg_leads < 2:
            raise ValueError("ecg_leads must be at least 2 for channel coupling")
        if num_flow_layers < 1:
            raise ValueError("num_flow_layers must be positive")

        self.signal_length = signal_length
        self.latent_dim = latent_dim
        self.ecg_leads = ecg_leads
        self.context_decoder = BaselineECGDecoder(
            signal_length=signal_length,
            latent_dim=latent_dim,
            ecg_leads=ecg_leads,
            base_channels=base_channels,
            dropout=dropout,
        )
        self.log_scale_head = nn.Conv1d(ecg_leads, ecg_leads, kernel_size=1)
        nn.init.zeros_(self.log_scale_head.weight)
        nn.init.zeros_(self.log_scale_head.bias)
        self.couplings = nn.ModuleList([
            ConditionalAffineCoupling1D(
                channels=ecg_leads,
                condition_channels=ecg_leads,
                hidden_channels=flow_hidden_channels,
                mask_first=(i % 2 == 0),
            )
            for i in range(num_flow_layers)
        ])

    def _context(self, encoded: torch.Tensor | dict[str, Any]) -> torch.Tensor:
        latent = encoded["latent"] if isinstance(encoded, dict) else encoded
        if latent.dim() != 3 or latent.size(1) != self.latent_dim:
            raise ValueError(
                f"expected latent [B,{self.latent_dim},T_latent], got {tuple(latent.shape)}"
            )
        return self.context_decoder(latent)

    def _log_scale(self, context: torch.Tensor) -> torch.Tensor:
        return torch.clamp(self.log_scale_head(context), min=-3.0, max=3.0)

    def _to_base(
        self,
        target: torch.Tensor,
        context: torch.Tensor,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        log_scale = self._log_scale(context)
        z = (target - context) * torch.exp(-log_scale)
        logdet = -log_scale.sum(dim=(1, 2))
        for coupling in self.couplings:
            z, layer_logdet = coupling(z, context, inverse=False)
            logdet = logdet + layer_logdet
        return z, logdet

    def nll_per_sample(
        self,
        target: torch.Tensor,
        encoded: torch.Tensor | dict[str, Any],
    ) -> torch.Tensor:
        context = self._context(encoded)
        z, logdet = self._to_base(target, context)
        log_base = 0.5 * (z.square() + math.log(2.0 * math.pi)).sum(dim=(1, 2))
        # Normalize by signal size so this term has a usable scale beside MSE.
        return (log_base - logdet) / float(target.size(1) * target.size(2))

    def nll(self, target: torch.Tensor, encoded: torch.Tensor | dict[str, Any]) -> torch.Tensor:
        return self.nll_per_sample(target, encoded).mean()

    def sample(
        self,
        encoded: torch.Tensor | dict[str, Any],
        noise: torch.Tensor | None = None,
    ) -> torch.Tensor:
        context = self._context(encoded)
        if noise is None:
            noise = torch.randn_like(context)
        if noise.shape != context.shape:
            raise ValueError(
                f"noise shape {tuple(noise.shape)} must match {tuple(context.shape)}"
            )
        x = noise
        for coupling in reversed(self.couplings):
            x, _ = coupling(x, context, inverse=True)
        x = x * torch.exp(self._log_scale(context)) + context
        return x

    def forward(self, encoded: torch.Tensor | dict[str, Any]) -> torch.Tensor:
        context = self._context(encoded)
        return self.sample(encoded, noise=torch.zeros_like(context))


class _GradientReversal(torch.autograd.Function):
    @staticmethod
    def forward(ctx, x: torch.Tensor, coefficient: float):
        ctx.coefficient = float(coefficient)
        return x.view_as(x)

    @staticmethod
    def backward(ctx, grad_output: torch.Tensor):
        return -ctx.coefficient * grad_output, None


def grad_reverse(x: torch.Tensor, coefficient: float = 1.0) -> torch.Tensor:
    """Reverse gradients into the encoder while keeping discriminator updates normal."""
    return _GradientReversal.apply(x, coefficient)


class SubjectDiscriminator(nn.Module):
    """Subject classifier attached only to the VAE content representation."""

    def __init__(
        self,
        content_dim: int,
        num_subjects: int,
        hidden_dim: int = 128,
        dropout: float = 0.1,
    ):
        super().__init__()
        if content_dim < 1 or num_subjects < 2:
            raise ValueError("content_dim must be positive and num_subjects >= 2")
        self.content_dim = content_dim
        self.num_subjects = num_subjects
        self.classifier = nn.Sequential(
            nn.Linear(content_dim, hidden_dim),
            nn.LayerNorm(hidden_dim),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, num_subjects),
        )

    def forward(self, z_content: torch.Tensor, grl_lambda: float = 1.0) -> torch.Tensor:
        if z_content.dim() == 3:
            z_content = z_content.mean(dim=-1)
        if z_content.dim() != 2 or z_content.size(1) != self.content_dim:
            raise ValueError(
                f"expected content [B,{self.content_dim}], got {tuple(z_content.shape)}"
            )
        return self.classifier(grad_reverse(z_content, grl_lambda))


__all__ = [
    "VAEPPGEncoder",
    "CardioAlignEncoder",
    "LatentRectifiedFlow",
    "ConditionalFlowECGGenerator",
    "SubjectDiscriminator",
    "grad_reverse",
]
