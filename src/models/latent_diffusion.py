"""潜空间扩散模型 (Latent Diffusion, DDIM)。

在 PPG 编码器输出的潜在空间中进行扩散去噪:
  - 训练阶段: 对潜在表示添加噪声，训练 U-Net 预测噪声
  - 推理阶段: DDIM 采样从纯噪声恢复潜在表示

参考: Rombach et al., "High-Resolution Image Synthesis with Latent Diffusion Models"
"""

from __future__ import annotations

import math
import torch
import torch.nn as nn
import torch.nn.functional as F


class SinusoidalPositionEmbedding(nn.Module):
    """正弦位置编码 (时间步嵌入)。"""

    def __init__(self, dim: int):
        super().__init__()
        self.dim = dim

    def forward(self, t: torch.Tensor) -> torch.Tensor:
        half = self.dim // 2
        freqs = torch.exp(
            -math.log(10000) * torch.arange(half, device=t.device) / half
        )
        args = t[:, None].float() * freqs[None]
        return torch.cat([torch.cos(args), torch.sin(args)], dim=-1)


class ResBlock1DWithTime(nn.Module):
    """带时间步条件注入的 1D 残差块。"""

    def __init__(self, in_ch: int, out_ch: int, time_dim: int):
        super().__init__()
        self.norm1 = nn.GroupNorm(8, in_ch)
        self.conv1 = nn.Conv1d(in_ch, out_ch, 3, padding=1)
        self.time_proj = nn.Linear(time_dim, out_ch)
        self.norm2 = nn.GroupNorm(8, out_ch)
        self.conv2 = nn.Conv1d(out_ch, out_ch, 3, padding=1)
        self.act = nn.GELU()
        self.skip = nn.Conv1d(in_ch, out_ch, 1) if in_ch != out_ch else nn.Identity()

    def forward(self, x: torch.Tensor, t_emb: torch.Tensor) -> torch.Tensor:
        h = self.act(self.norm1(x))
        h = self.conv1(h)
        h = h + self.time_proj(t_emb)[:, :, None]
        h = self.act(self.norm2(h))
        h = self.conv2(h)
        return h + self.skip(x)


class TimeConditionalSequential(nn.Module):
    """按顺序执行多个 ResBlock1DWithTime，传递时间嵌入。"""

    def __init__(self, *layers: ResBlock1DWithTime):
        super().__init__()
        self.layers = nn.ModuleList(layers)

    def forward(self, x: torch.Tensor, t_emb: torch.Tensor) -> torch.Tensor:
        for layer in self.layers:
            x = layer(x, t_emb)
        return x


class DiffusionUNet(nn.Module):
    """用于潜在空间去噪的轻量 1D U-Net。

    Args:
        in_channels: 潜在表示通道数
        base_channels: 基础通道数
        num_res_blocks: 每层的残差块数
        time_emb_dim: 时间步嵌入维度
    """

    def __init__(
        self,
        in_channels: int = 128,
        base_channels: int = 64,
        num_res_blocks: int = 2,
        time_emb_dim: int = 128,
    ):
        super().__init__()

        self.time_embed = nn.Sequential(
            SinusoidalPositionEmbedding(time_emb_dim),
            nn.Linear(time_emb_dim, time_emb_dim * 4),
            nn.GELU(),
            nn.Linear(time_emb_dim * 4, time_emb_dim),
        )

        # 编码器
        self.encoder = nn.ModuleList([
            self._make_layer(in_channels, base_channels, num_res_blocks, time_emb_dim),
            self._make_layer(base_channels, base_channels * 2, num_res_blocks, time_emb_dim),
            self._make_layer(base_channels * 2, base_channels * 4, num_res_blocks, time_emb_dim),
        ])

        # 瓶颈层
        self.mid = self._make_layer(
            base_channels * 4, base_channels * 4, num_res_blocks, time_emb_dim
        )

        # 解码器 (带 skip connection)
        self.decoder = nn.ModuleList([
            self._make_layer(base_channels * 8, base_channels * 4, num_res_blocks, time_emb_dim),
            self._make_layer(base_channels * 6, base_channels * 2, num_res_blocks, time_emb_dim),
            self._make_layer(base_channels * 3, base_channels, num_res_blocks, time_emb_dim),
        ])

        self.out_conv = nn.Conv1d(base_channels, in_channels, 1)

    @staticmethod
    def _make_layer(in_ch, out_ch, num_blocks, time_dim):
        layers = [ResBlock1DWithTime(in_ch, out_ch, time_dim)]
        for _ in range(num_blocks - 1):
            layers.append(ResBlock1DWithTime(out_ch, out_ch, time_dim))
        return TimeConditionalSequential(*layers)

    def forward(self, x: torch.Tensor, t: torch.Tensor) -> torch.Tensor:
        """
        Args:
            x: [B, C, T']  带噪声的潜在表示
            t: [B]  扩散时间步
        Returns:
            noise_pred: [B, C, T']  预测的噪声
        """
        t_emb = self.time_embed(t)

        # 编码器
        skips = []
        for enc in self.encoder:
            x = enc(x, t_emb)
            skips.append(x)
            x = F.avg_pool1d(x, 2)

        # 瓶颈
        x = self.mid(x, t_emb)

        # 解码器
        for dec, skip in zip(self.decoder, reversed(skips)):
            x = F.interpolate(x, size=skip.size(-1), mode="linear", align_corners=False)
            x = torch.cat([x, skip], dim=1)
            x = dec(x, t_emb)

        return self.out_conv(x)


class LatentDiffusion(nn.Module):
    """潜空间扩散模型。

    管理前向扩散 (加噪) 和反向去噪 (采样) 过程。

    Args:
        latent_dim: 潜在表示通道数
        num_timesteps: 扩散步数 (默认 1000)
        beta_schedule: 噪声调度类型 ('linear' 或 'cosine')
    """

    def __init__(
        self,
        latent_dim: int = 128,
        num_timesteps: int = 1000,
        beta_schedule: str = "linear",
    ):
        super().__init__()
        self.latent_dim = latent_dim
        self.num_timesteps = num_timesteps

        # U-Net 去噪网络
        self.unet = DiffusionUNet(in_channels=latent_dim)

        # 噪声调度
        if beta_schedule == "linear":
            betas = torch.linspace(1e-4, 0.02, num_timesteps)
        elif beta_schedule == "cosine":
            steps = num_timesteps + 1
            x = torch.linspace(0, num_timesteps, steps)
            alphas_cumprod = torch.cos(
                ((x / num_timesteps) + 0.008) / 1.008 * math.pi * 0.5
            ) ** 2
            alphas_cumprod = alphas_cumprod / alphas_cumprod[0]
            betas = 1 - (alphas_cumprod[1:] / alphas_cumprod[:-1])
            betas = torch.clamp(betas, 0, 0.999)
        else:
            raise ValueError(f"Unknown beta_schedule: {beta_schedule}")

        alphas = 1.0 - betas
        alphas_cumprod = torch.cumprod(alphas, dim=0)

        # 注册为 buffer (不参与梯度)
        self.register_buffer("betas", betas)
        self.register_buffer("alphas", alphas)
        self.register_buffer("alphas_cumprod", alphas_cumprod)
        self.register_buffer(
            "sqrt_alphas_cumprod", torch.sqrt(alphas_cumprod)
        )
        self.register_buffer(
            "sqrt_one_minus_alphas_cumprod", torch.sqrt(1.0 - alphas_cumprod)
        )

    def q_sample(
        self, x0: torch.Tensor, t: torch.Tensor, noise: torch.Tensor | None = None
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """前向扩散: 给 x0 添加 t 步噪声。

        Returns:
            (x_t, noise): 加噪后的样本和所用噪声
        """
        if noise is None:
            noise = torch.randn_like(x0)

        sqrt_a = self.sqrt_alphas_cumprod[t][:, None, None]
        sqrt_1m = self.sqrt_one_minus_alphas_cumprod[t][:, None, None]
        x_t = sqrt_a * x0 + sqrt_1m * noise
        return x_t, noise

    def training_loss(
        self, x0: torch.Tensor, t: torch.Tensor | None = None
    ) -> torch.Tensor:
        """计算去噪训练损失 (简单 MSE)。"""
        B = x0.size(0)
        if t is None:
            t = torch.randint(0, self.num_timesteps, (B,), device=x0.device)

        x_t, noise = self.q_sample(x0, t)
        noise_pred = self.unet(x_t, t)
        return F.mse_loss(noise_pred, noise)

    @torch.no_grad()
    def ddim_sample(
        self,
        shape: tuple[int, ...],
        device: torch.device,
        num_steps: int = 50,
        eta: float = 0.0,
    ) -> torch.Tensor:
        """DDIM 采样。

        Args:
            shape: 输出形状 (B, C, T')
            num_steps: 采样步数 (少于 num_timesteps)
            eta: 随机性参数 (0 = 确定性, 1 = DDPM)
        Returns:
            x0: 去噪后的潜在表示
        """
        B = shape[0]
        x = torch.randn(shape, device=device)

        # 选取时间步子序列
        step_indices = torch.linspace(0, self.num_timesteps - 1, num_steps, dtype=torch.long)
        step_indices = step_indices.flip(0)  # 从大到小

        for i, t in enumerate(step_indices):
            t_batch = torch.full((B,), t, device=device, dtype=torch.long)
            noise_pred = self.unet(x, t_batch)

            alpha_t = self.alphas_cumprod[t]
            alpha_prev = self.alphas_cumprod[step_indices[i + 1]] if i + 1 < len(step_indices) else torch.tensor(1.0, device=device)

            x0_pred = (x - (1 - alpha_t).sqrt() * noise_pred) / alpha_t.sqrt()

            # DDIM 更新
            sigma = eta * ((1 - alpha_prev) / (1 - alpha_t) * (1 - alpha_t / alpha_prev)).sqrt()
            dir_xt = (1 - alpha_prev - sigma ** 2).sqrt() * noise_pred
            x = alpha_prev.sqrt() * x0_pred + dir_xt
            if sigma > 0:
                x = x + sigma * torch.randn_like(x)

        return x
