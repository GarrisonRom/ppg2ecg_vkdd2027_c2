"""PPG→ECG 重建训练器。

训练流程:
  1. PPG 编码器提取潜在表示
  2. (可选) 潜空间扩散模型精炼
  3. ECG 解码器重建 12 导联 ECG
  4. 复合损失函数计算梯度

支持:
  - 混合精度训练 (AMP)
  - 梯度裁剪
  - 学习率调度
  - 检查点保存/恢复
  - WandB / TensorBoard 日志
"""

from __future__ import annotations

import os
from pathlib import Path

import torch
import torch.nn as nn
from torch.optim import AdamW
from torch.optim.lr_scheduler import CosineAnnealingLR
from tqdm import tqdm

from ..models import (
    ReconstructionLoss,
    build_decoder,
    build_diffusion,
    build_encoder,
)


class PPG2ECGTrainer:
    """PPG→ECG 重建训练器。

    模型模块按 config["model"] 分节经注册表构建:
        model.encoder / model.decoder / model.diffusion 均为
        {name: 注册名, ...构造参数}; diffusion 为 null 表示不启用。
        signal_length / ecg_leads 由数据集实例传入, 优先级高于配置。

    Args:
        config: 完整实验配置 (data/model/training/...)
        train_loader: 训练数据 DataLoader
        val_loader: 验证数据 DataLoader
        device: 计算设备
        signal_length: 窗口长度 (来自数据集)
        ecg_leads: ECG 导联数 (来自数据集)
    """

    def __init__(
        self,
        config: dict,
        train_loader,
        val_loader,
        device: torch.device | None = None,
        signal_length: int = 1250,
        ecg_leads: int = 12,
        ppg_channels: int = 1,
    ):
        self.config = config
        self.train_loader = train_loader
        self.val_loader = val_loader
        self.device = device or torch.device(
            "cuda" if torch.cuda.is_available() else "cpu"
        )

        model_cfg = config.get("model", {})
        latent_dim = model_cfg.get("latent_dim", 128)

        # 模型 (按注册表构建; 数据集派生参数强制覆盖配置)
        self.encoder = build_encoder(
            model_cfg.get("encoder"),
            signal_length=signal_length,
            latent_dim=latent_dim,
            ppg_channels=ppg_channels,
        ).to(self.device)

        self.decoder = build_decoder(
            model_cfg.get("decoder"),
            signal_length=signal_length,
            latent_dim=latent_dim,
            ecg_leads=ecg_leads,
        ).to(self.device)

        # 扩散模型 (可选)
        self.diffusion = build_diffusion(
            model_cfg.get("diffusion"),
            latent_dim=latent_dim,
        )
        self.diffusion = self.diffusion.to(self.device) if self.diffusion else None

        # 损失函数 (模块按权重选择)
        loss_cfg = model_cfg.get("loss", {})
        self.criterion = ReconstructionLoss(
            weights=loss_cfg.get("weights"),
            ecg_leads=ecg_leads,
        ).to(self.device)

        # 优化器 (training 分节)
        train_cfg = config.get("training", {})
        params = list(self.encoder.parameters()) + list(self.decoder.parameters())
        if self.diffusion:
            params += list(self.diffusion.parameters())

        self.optimizer = AdamW(
            params,
            lr=train_cfg.get("lr", 1e-4),
            weight_decay=train_cfg.get("weight_decay", 1e-5),
        )

        # 学习率调度
        self.scheduler = CosineAnnealingLR(
            self.optimizer,
            T_max=train_cfg.get("epochs", 100),
            eta_min=train_cfg.get("min_lr", 1e-6),
        )

        # 训练状态
        self.epoch = 0
        self.best_val_loss = float("inf")
        self.amp_scaler = torch.amp.GradScaler() if self.device.type == "cuda" else None

        # 输出目录
        self.output_dir = Path(config.get("output_dir", "checkpoints"))
        self.output_dir.mkdir(parents=True, exist_ok=True)
        self.diff_weight = train_cfg.get("diff_weight", 0.1)
        self.train_cfg = train_cfg

    def train_one_epoch(self) -> dict[str, float]:
        """训练一个 epoch。"""
        self.encoder.train()
        self.decoder.train()
        if self.diffusion:
            self.diffusion.train()

        total_losses = {}
        num_batches = 0

        pbar = tqdm(self.train_loader, desc=f"Epoch {self.epoch + 1}")
        for batch in pbar:
            ppg = batch["ppg"].to(self.device)
            ecg = batch["ecg"].to(self.device)

            self.optimizer.zero_grad()

            # 混合精度前向
            use_amp = self.amp_scaler is not None
            with torch.amp.autocast("cuda", enabled=use_amp):
                encoded = self.encoder(ppg)
                # baseline encoder 返回 {latent, skips}; 旧 encoder 仍直接返回 tensor。
                latent = encoded["latent"] if isinstance(encoded, dict) else encoded

                if self.diffusion:
                    # 扩散损失 (辅助)
                    diff_loss = self.diffusion.training_loss(latent)
                    # 使用扩散采样后的表示 (训练时直接用原始 latent)
                    ecg_pred = self.decoder(encoded)
                else:
                    diff_loss = torch.tensor(0.0, device=self.device)
                    ecg_pred = self.decoder(encoded)

                losses = self.criterion(ecg_pred, ecg)
                loss = losses["total"] + diff_loss * self.diff_weight

            # 反向传播
            if use_amp:
                self.amp_scaler.scale(loss).backward()
                self.amp_scaler.unscale_(self.optimizer)
                torch.nn.utils.clip_grad_norm_(
                    self._trainable_parameters(),
                    self.train_cfg.get("max_grad_norm", 1.0),
                )
                self.amp_scaler.step(self.optimizer)
                self.amp_scaler.update()
            else:
                loss.backward()
                torch.nn.utils.clip_grad_norm_(
                    self._trainable_parameters(),
                    self.train_cfg.get("max_grad_norm", 1.0),
                )
                self.optimizer.step()

            # 累计损失
            for k, v in losses.items():
                total_losses[k] = total_losses.get(k, 0) + v.item()
            total_losses["diffusion"] = total_losses.get("diffusion", 0) + diff_loss.item()
            num_batches += 1

            pbar.set_postfix({
                "loss": f"{loss.item():.4f}",
                "lr": f"{self.optimizer.param_groups[0]['lr']:.2e}",
            })

        return {k: v / num_batches for k, v in total_losses.items()}

    @torch.no_grad()
    def validate(self) -> dict[str, float]:
        """验证。"""
        self.encoder.eval()
        self.decoder.eval()

        total_losses = {}
        num_batches = 0

        for batch in tqdm(self.val_loader, desc="Validation"):
            ppg = batch["ppg"].to(self.device)
            ecg = batch["ecg"].to(self.device)

            encoded = self.encoder(ppg)
            ecg_pred = self.decoder(encoded)
            losses = self.criterion(ecg_pred, ecg)

            for k, v in losses.items():
                total_losses[k] = total_losses.get(k, 0) + v.item()
            num_batches += 1

        return {k: v / num_batches for k, v in total_losses.items()}

    def train(self, num_epochs: int):
        """完整训练流程。"""
        print(f"Training on {self.device}")
        print(f"Encoder params: {sum(p.numel() for p in self.encoder.parameters()):,}")
        print(f"Decoder params: {sum(p.numel() for p in self.decoder.parameters()):,}")
        if self.diffusion:
            print(f"Diffusion params: {sum(p.numel() for p in self.diffusion.parameters()):,}")
        print(f"Output dir: {self.output_dir}")
        print("-" * 60)

        for epoch in range(num_epochs):
            self.epoch = epoch

            train_losses = self.train_one_epoch()
            val_losses = self.validate()

            self.scheduler.step()

            # 打印摘要
            print(f"\nEpoch {epoch + 1}/{num_epochs}")
            print(f"  Train loss: {train_losses.get('total', 0):.4f}")
            print(f"  Val   loss: {val_losses.get('total', 0):.4f}")

            # 保存最佳模型
            val_loss = val_losses.get("total", float("inf"))
            if val_loss < self.best_val_loss:
                self.best_val_loss = val_loss
                self.save_checkpoint("best.pth")
                print(f"  ** Best model saved (val_loss={val_loss:.4f})")

            # 定期保存
            if (epoch + 1) % self.train_cfg.get("save_every", 10) == 0:
                self.save_checkpoint(f"epoch_{epoch + 1}.pth")

        # 保存最终模型
        self.save_checkpoint("final.pth")
        print(f"\nTraining complete. Best val loss: {self.best_val_loss:.4f}")

    def _trainable_parameters(self):
        """返回 encoder/decoder/(可选 diffusion) 的全部可训练参数。"""
        params = list(self.encoder.parameters()) + list(self.decoder.parameters())
        if self.diffusion:
            params += list(self.diffusion.parameters())
        return [p for p in params if p.requires_grad]

    def save_checkpoint(self, filename: str):
        """保存检查点。"""
        ckpt = {
            "epoch": self.epoch,
            "encoder": self.encoder.state_dict(),
            "decoder": self.decoder.state_dict(),
            "optimizer": self.optimizer.state_dict(),
            "scheduler": self.scheduler.state_dict(),
            "best_val_loss": self.best_val_loss,
            "config": self.config,
        }
        if self.diffusion:
            ckpt["diffusion"] = self.diffusion.state_dict()

        path = self.output_dir / filename
        torch.save(ckpt, path)

    def load_checkpoint(self, path: str | Path):
        """加载检查点。"""
        ckpt = torch.load(path, map_location=self.device, weights_only=False)
        self.encoder.load_state_dict(ckpt["encoder"])
        self.decoder.load_state_dict(ckpt["decoder"])
        if self.diffusion and "diffusion" in ckpt:
            self.diffusion.load_state_dict(ckpt["diffusion"])
        self.optimizer.load_state_dict(ckpt["optimizer"])
        self.scheduler.load_state_dict(ckpt["scheduler"])
        self.epoch = ckpt["epoch"]
        self.best_val_loss = ckpt["best_val_loss"]
        print(f"Checkpoint loaded: {path} (epoch {self.epoch})")
