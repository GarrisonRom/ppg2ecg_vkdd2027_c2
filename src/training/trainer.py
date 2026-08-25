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

import json
import os
from pathlib import Path

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.optim import AdamW
from torch.optim.lr_scheduler import CosineAnnealingLR
from tqdm import tqdm

from ..models import (
    ConditionalPatchGAN1D,
    ReconstructionLoss,
    SubjectDiscriminator,
    build_decoder,
    build_diffusion,
    build_encoder,
    build_latent_flow,
    patchgan_hinge_discriminator_loss,
    patchgan_hinge_generator_loss,
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
        sample_rate: float = 250.0,
    ):
        self.config = config
        self.train_loader = train_loader
        self.val_loader = val_loader
        self.device = device or torch.device(
            "cuda" if torch.cuda.is_available() else "cpu"
        )

        model_cfg = config.get("model", {})
        train_cfg = config.get("training", {})
        latent_dim = model_cfg.get("latent_dim", 128)

        # Decoder warm-up is useful when comparing a new decoder against a
        # previously validated encoder.  During this stage the encoder is
        # frozen and VAE posterior means are used, so decoder optimization is
        # not obscured by posterior sampling noise.  The encoder is unfrozen
        # automatically for the remaining joint fine-tuning epochs.
        self.encoder_freeze_epochs = max(
            0, int(train_cfg.get("encoder_freeze_epochs", 0))
        )
        self.use_posterior_mean_when_frozen = bool(
            train_cfg.get("use_posterior_mean_when_frozen", True)
        )
        self.encoder_frozen = False
        self.encoder_stage = "joint"

        # 模型 (按注册表构建; 数据集派生参数强制覆盖配置)
        self.encoder = build_encoder(
            model_cfg.get("encoder"),
            signal_length=signal_length,
            latent_dim=latent_dim,
            ppg_channels=ppg_channels,
        ).to(self.device)
        encoder_init = model_cfg.get("init_encoder_checkpoint")
        if encoder_init:
            self._load_encoder_weights(encoder_init)

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

        encoder_name = (model_cfg.get("encoder") or {}).get("name")
        decoder_name = (model_cfg.get("decoder") or {}).get("name")
        # ``gated_multiband_decoder`` reuses the same three-band interface as
        # the original decoder.  Keep it on the structured return path so the
        # configured band reconstruction loss is applied to both variants.
        self.is_band_decoder = decoder_name in {
            "multiband_decoder", "gated_multiband_decoder",
        }
        self.is_wavelet_decoder = decoder_name == "wavelet_decoder"
        self.is_vae = encoder_name in {
            "vae_encoder", "cardio_align_encoder", "cardio_ppg_encoder",
            "cardio_ecg_encoder",
        }
        self.is_flow = decoder_name == "flow_generator" and hasattr(
            self.decoder, "nll_per_sample"
        )

        # v0.4 PPGFlowECG-inspired paired pathway.  By default the PPG and
        # ECG branches share the same VAE weights, matching the shared
        # CardioAlign encoder idea.  A separate ECG branch remains available
        # for later ablations via share_cardio_encoder: false.
        latent_flow_cfg = model_cfg.get("latent_flow")
        self.is_cardio_flow = (
            latent_flow_cfg is not None
            and encoder_name in {
                "cardio_align_encoder", "cardio_ppg_encoder",
                "cardio_ecg_encoder",
            }
        )
        self.ecg_encoder = None
        self.latent_flow = None
        self.cardio_share_encoder = True
        self.cardio_flow_steps = 8
        self.cardio_weights: dict[str, float] = {}
        if self.is_cardio_flow:
            self.cardio_share_encoder = bool(model_cfg.get("share_cardio_encoder", True))
            if self.cardio_share_encoder:
                self.ecg_encoder = self.encoder
            else:
                ecg_cfg = model_cfg.get("ecg_encoder") or model_cfg.get("encoder")
                self.ecg_encoder = build_encoder(
                    ecg_cfg,
                    signal_length=signal_length,
                    latent_dim=latent_dim,
                    ppg_channels=ecg_leads,
                ).to(self.device)
            self.latent_flow = build_latent_flow(
                latent_flow_cfg,
                latent_dim=latent_dim,
            ).to(self.device)
            cardio_cfg = model_cfg.get("cardio_align", {}) or {}
            self.cardio_flow_steps = int(cardio_cfg.get("integration_steps", 8))
            self.cardio_weights = {
                "direct_recon": float(cardio_cfg.get("direct_recon_weight", 1.0)),
                "cross_recon": float(cardio_cfg.get("cross_recon_weight", 0.25)),
                "flow_recon": float(cardio_cfg.get("flow_recon_weight", 1.0)),
                "distribution": float(cardio_cfg.get("distribution_weight", 0.25)),
                "contrastive": float(cardio_cfg.get("contrastive_weight", 0.05)),
                "latent_flow": float(cardio_cfg.get("latent_flow_weight", 1.0)),
            }
            self.cardio_temperature = float(cardio_cfg.get("temperature", 0.1))
            projection_dim = int(cardio_cfg.get("projection_dim", 128))
            self.ppg_projection = nn.Sequential(
                nn.Linear(latent_dim, projection_dim),
                nn.LayerNorm(projection_dim),
                nn.GELU(),
                nn.Linear(projection_dim, projection_dim),
            ).to(self.device)
            self.ecg_projection = nn.Sequential(
                nn.Linear(latent_dim, projection_dim),
                nn.LayerNorm(projection_dim),
                nn.GELU(),
                nn.Linear(projection_dim, projection_dim),
            ).to(self.device)
        else:
            self.ppg_projection = None
            self.ecg_projection = None

        # v0.5 bidirectional cycle pathway.  The reverse model is a separate
        # ECG->PPG encoder/decoder pair.  It is trained directly on paired data
        # and then used to constrain PPG->ECG->PPG.  Keeping the parameters
        # separate avoids accidentally treating an ordinary ConvNet as an
        # exactly invertible function.
        cycle_cfg = model_cfg.get("cycle_consistency", {}) or {}
        self.cycle_enabled = bool(cycle_cfg.get("enabled", False))
        self.reverse_encoder = None
        self.reverse_decoder = None
        self.cycle_direct_weight = 0.0
        self.cycle_reconstruction_weight = 0.0
        self.cycle_loss_type = "l1"
        self.cycle_reverse_pretrain_epochs = 0
        self.cycle_freeze_reverse = False
        self.reverse_frozen = False
        self.cycle_pretrain_ecg_weight = 1.0
        if self.cycle_enabled:
            if self.is_cardio_flow:
                raise ValueError("cycle_consistency and latent_flow cannot be enabled together yet")
            reverse_encoder_cfg = (
                cycle_cfg.get("reverse_encoder")
                or cycle_cfg.get("ecg_encoder")
                or model_cfg.get("encoder")
            )
            reverse_decoder_cfg = (
                cycle_cfg.get("reverse_decoder")
                or cycle_cfg.get("ppg_decoder")
                or model_cfg.get("decoder")
            )
            self.reverse_encoder = build_encoder(
                reverse_encoder_cfg,
                signal_length=signal_length,
                latent_dim=latent_dim,
                ppg_channels=ecg_leads,
            ).to(self.device)
            self.reverse_decoder = build_decoder(
                reverse_decoder_cfg,
                signal_length=signal_length,
                latent_dim=latent_dim,
                ecg_leads=ppg_channels,
            ).to(self.device)
            self.cycle_direct_weight = float(
                cycle_cfg.get("direct_ecg2ppg_weight", 1.0)
            )
            self.cycle_reconstruction_weight = float(
                cycle_cfg.get("ppg_cycle_weight", 1.0)
            )
            self.cycle_reverse_pretrain_epochs = max(
                0, int(cycle_cfg.get("reverse_pretrain_epochs", 0))
            )
            self.cycle_freeze_reverse = bool(cycle_cfg.get("freeze_reverse", False))
            self.cycle_pretrain_ecg_weight = float(
                cycle_cfg.get("pretrain_ecg_weight", 1.0)
            )
            self.cycle_loss_type = str(cycle_cfg.get("signal_loss", "l1")).lower()
            if self.cycle_loss_type not in {"l1", "mse"}:
                raise ValueError("cycle_consistency.signal_loss must be 'l1' or 'mse'")
            if self.cycle_freeze_reverse and self.cycle_reverse_pretrain_epochs == 0:
                self._set_reverse_frozen(True)

        # VAE/Flow/对抗/IRM 的权重都保持为小的显式开关，便于后续逐项消融。
        self.vae_kl_weight = float(train_cfg.get("vae_kl_weight", 0.0))
        self.flow_weight = float(train_cfg.get("flow_weight", 0.0))

        adv_cfg = model_cfg.get("adversarial", {}) or {}
        self.adv_enabled = bool(adv_cfg.get("enabled", False)) and self.is_vae
        self.adv_loss_weight = float(adv_cfg.get("loss_weight", 0.0))
        self.grl_max_lambda = float(adv_cfg.get("grl_max_lambda", 0.0))
        self.grl_warmup_epochs = int(adv_cfg.get("grl_warmup_epochs", 0))
        self.subject_to_index: dict[int, int] = {}
        self.subject_discriminator: SubjectDiscriminator | None = None
        if self.adv_enabled:
            train_metadata = getattr(getattr(train_loader, "dataset", None), "metadata", None)
            if train_metadata is not None and "subject_id" in train_metadata.columns:
                raw_subjects = sorted({int(v) for v in train_metadata["subject_id"].tolist()})
                self.subject_to_index = {
                    subject_id: idx for idx, subject_id in enumerate(raw_subjects)
                }
        if len(self.subject_to_index) >= 2:
                enc_cfg = model_cfg.get("encoder", {}) or {}
                content_dim = int(
                    getattr(
                        self.encoder,
                        "content_dim",
                        enc_cfg.get("content_dim", latent_dim // 2),
                    )
                )
                self.subject_discriminator = SubjectDiscriminator(
                    content_dim=content_dim,
                    num_subjects=len(self.subject_to_index),
                    hidden_dim=int(adv_cfg.get("hidden_dim", 128)),
                    dropout=float(adv_cfg.get("dropout", 0.1)),
                ).to(self.device)
        else:
            self.adv_enabled = False

        # Optional conditional PatchGAN.  It is deliberately separate from
        # the subject adversary: the latter removes subject information from
        # the content latent, while PatchGAN only judges local realism of the
        # paired [PPG, ECG] waveform.  Keeping a distinct optimizer makes the
        # adversarial contribution auditable and leaves all old experiments
        # unchanged when the section is absent or disabled.
        patch_cfg = model_cfg.get("patchgan", {}) or {}
        self.patchgan_enabled = bool(patch_cfg.get("enabled", False))
        self.patchgan_loss_weight = float(patch_cfg.get("generator_weight", 0.0))
        self.patchgan_discriminator = None
        self.patchgan_optimizer = None
        if self.patchgan_enabled and self.patchgan_loss_weight > 0.0:
            self.patchgan_discriminator = ConditionalPatchGAN1D(
                ppg_channels=ppg_channels,
                ecg_leads=ecg_leads,
                base_channels=int(patch_cfg.get("base_channels", 32)),
                num_layers=int(patch_cfg.get("num_layers", 4)),
                max_channels=int(patch_cfg.get("max_channels", 256)),
                dropout=float(patch_cfg.get("dropout", 0.0)),
            ).to(self.device)
            self.patchgan_lr = float(patch_cfg.get("discriminator_lr", 2e-4))
            self.patchgan_weight_decay = float(
                patch_cfg.get("weight_decay", 1e-5)
            )
        else:
            self.patchgan_enabled = False
            self.patchgan_loss_weight = 0.0
            self.patchgan_lr = 0.0
            self.patchgan_weight_decay = 0.0

        irm_cfg = model_cfg.get("irm", {}) or {}
        self.irm_enabled = bool(irm_cfg.get("enabled", False)) and self.is_vae
        self.irm_mode = str(irm_cfg.get("mode", "vrex")).lower()
        if self.irm_mode != "vrex":
            raise ValueError("Only the stable V-REx-style IRM auxiliary is currently supported")
        self.irm_weight = float(irm_cfg.get("loss_weight", 0.0))
        self.irm_min_env_samples = int(irm_cfg.get("min_env_samples", 2))

        # 损失函数 (模块按权重选择)
        loss_cfg = model_cfg.get("loss", {})
        self.criterion = ReconstructionLoss(
            weights=loss_cfg.get("weights"),
            ecg_leads=ecg_leads,
            sample_rate=sample_rate,
            qrs_config=loss_cfg.get("qrs"),
            wavelet_config=loss_cfg.get("wavelet"),
        ).to(self.device)

        # v0.52 multi-band auxiliary objective. The decoder itself performs
        # fixed FFT projections; these weights only balance their supervised
        # time-domain components and never expose band labels to the encoder.
        band_cfg = model_cfg.get("band_loss", {}) or {}
        self.band_loss_enabled = bool(band_cfg.get("enabled", False)) and self.is_band_decoder
        self.band_loss_weight = float(band_cfg.get("total_weight", 0.0))
        self.band_loss_normalize = bool(band_cfg.get("normalize", True))
        self.band_loss_scale_floor = float(band_cfg.get("scale_floor", 0.05))
        configured_band_weights = band_cfg.get("weights", {}) or {}
        self.band_loss_band_weights = {
            "low": float(configured_band_weights.get("low", 0.5)),
            "mid": float(configured_band_weights.get("mid", 1.0)),
            "high": float(configured_band_weights.get("high", 2.0)),
        }
        # Optional branch-local amplitude supervision.  Applying this to the
        # high-frequency branch keeps QRS peak/RMS calibration from pulling the
        # fused low- and mid-frequency morphology away from the v0.52 baseline.
        self.band_qrs_amplitude_weight = float(
            band_cfg.get("qrs_amplitude_weight", 0.0)
        )

        # 优化器 (training 分节)
        params = list(self.encoder.parameters()) + list(self.decoder.parameters())
        if self.ecg_encoder is not None and not self.cardio_share_encoder:
            params += list(self.ecg_encoder.parameters())
        if self.latent_flow:
            params += list(self.latent_flow.parameters())
        if self.ppg_projection:
            params += list(self.ppg_projection.parameters())
        if self.ecg_projection:
            params += list(self.ecg_projection.parameters())
        if self.reverse_encoder:
            params += list(self.reverse_encoder.parameters())
        if self.reverse_decoder:
            params += list(self.reverse_decoder.parameters())
        if self.diffusion:
            params += list(self.diffusion.parameters())
        if self.subject_discriminator:
            params += list(self.subject_discriminator.parameters())

        base_lr = float(train_cfg.get("lr", 1e-4))
        encoder_lr = float(train_cfg.get("encoder_lr", base_lr))
        decoder_lr = float(train_cfg.get("decoder_lr", base_lr))
        encoder_ids = {id(parameter) for parameter in self.encoder.parameters()}
        unique_params: list[torch.nn.Parameter] = []
        seen_ids: set[int] = set()
        for parameter in params:
            if id(parameter) not in seen_ids:
                unique_params.append(parameter)
                seen_ids.add(id(parameter))
        encoder_params = [
            parameter for parameter in unique_params if id(parameter) in encoder_ids
        ]
        other_params = [
            parameter for parameter in unique_params if id(parameter) not in encoder_ids
        ]
        param_groups = []
        if encoder_params:
            param_groups.append({"params": encoder_params, "lr": encoder_lr})
        if other_params:
            param_groups.append({"params": other_params, "lr": decoder_lr})

        self.optimizer = AdamW(
            param_groups,
            lr=base_lr,
            weight_decay=train_cfg.get("weight_decay", 1e-5),
        )
        if self.patchgan_discriminator is not None:
            self.patchgan_optimizer = AdamW(
                self.patchgan_discriminator.parameters(),
                lr=self.patchgan_lr,
                betas=(0.5, 0.999),
                weight_decay=self.patchgan_weight_decay,
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
        self.best_epoch = 0
        self.epochs_without_improvement = 0
        self.early_stopping_patience = max(
            0, int(train_cfg.get("early_stopping_patience", 0))
        )
        self.early_stopping_min_delta = float(
            train_cfg.get("early_stopping_min_delta", 0.0)
        )
        self.amp_scaler = torch.amp.GradScaler() if self.device.type == "cuda" else None

        # 输出目录
        self.output_dir = Path(config.get("output_dir", "checkpoints"))
        self.output_dir.mkdir(parents=True, exist_ok=True)
        self.diff_weight = float(train_cfg.get("diff_weight", 0.1))
        self.train_cfg = train_cfg
        self.history: list[dict] = []
        self._write_subject_mapping()

    def train_one_epoch(self) -> dict[str, float]:
        """训练一个 epoch。"""
        if self.encoder_frozen:
            self.encoder.eval()
        else:
            self.encoder.train()
        self.decoder.train()
        if self.ecg_encoder is not None and self.ecg_encoder is not self.encoder:
            self.ecg_encoder.train()
        if self.reverse_encoder:
            self.reverse_encoder.train()
        if self.reverse_decoder:
            self.reverse_decoder.train()
        if self.reverse_frozen:
            self.reverse_encoder.eval()
            self.reverse_decoder.eval()
        if self.latent_flow:
            self.latent_flow.train()
        if self.diffusion:
            self.diffusion.train()
        if self.subject_discriminator:
            self.subject_discriminator.train()
        if self.patchgan_discriminator:
            self.patchgan_discriminator.train()

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
                if self.cycle_enabled and self._cycle_reverse_pretrain_active():
                    losses = self._compute_reverse_pretrain_objective(ppg, ecg)
                else:
                    encoded = self._prepare_encoded_for_stage(self.encoder(ppg))
                    # baseline encoder 返回 {latent, skips}; 旧 encoder 仍直接返回 tensor。
                    latent = encoded["latent"] if isinstance(encoded, dict) else encoded

                    if self.is_cardio_flow:
                        encoded_ecg = self.ecg_encoder(ecg)
                        losses = self._compute_cardio_objective(
                            encoded, encoded_ecg, ecg, batch,
                        )
                    elif self.cycle_enabled:
                        losses = self._compute_cycle_objective(
                            encoded, ppg, ecg,
                        )
                    elif self.diffusion:
                        # 扩散损失 (辅助)
                        diff_loss = self.diffusion.training_loss(latent)
                        # 使用扩散采样后的表示 (训练时直接用原始 latent)
                        ecg_pred, band_outputs, pred_coeffs = self._decode_main(encoded)
                        losses = self._compute_objective(
                            encoded, ecg_pred, ecg, batch, diff_loss,
                            band_outputs=band_outputs,
                            pred_coeffs=pred_coeffs,
                        )
                    else:
                        diff_loss = torch.tensor(0.0, device=self.device)
                        ecg_pred, band_outputs, pred_coeffs = self._decode_main(encoded)
                        losses = self._compute_objective(
                            encoded, ecg_pred, ecg, batch, diff_loss,
                            band_outputs=band_outputs,
                            pred_coeffs=pred_coeffs,
                        )
                    if (
                        self.patchgan_discriminator is not None
                        and not self.is_cardio_flow
                        and not self.cycle_enabled
                        and not self._cycle_reverse_pretrain_active()
                    ):
                        d_loss, g_adv = self._patchgan_step(
                            ppg, ecg, ecg_pred,
                        )
                        losses["patchgan_d"] = d_loss.detach()
                        losses["patchgan_g"] = g_adv
                        losses["objective"] = (
                            losses["objective"]
                            + self.patchgan_loss_weight * g_adv
                        )
                loss = losses["objective"]

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
                total_losses[k] = total_losses.get(k, 0) + float(v.detach().item())
            num_batches += 1

            flow_metric = losses.get("latent_flow", losses.get("flow_nll", loss.new_zeros(())))
            pbar.set_postfix({
                "loss": f"{loss.item():.4f}",
                "flow": f"{flow_metric.item():.3f}",
                "lr": f"{self.optimizer.param_groups[0]['lr']:.2e}",
            })

        return {k: v / num_batches for k, v in total_losses.items()}

    @torch.no_grad()
    def validate(self) -> dict[str, float]:
        """验证。"""
        self.encoder.eval()
        self.decoder.eval()
        if self.ecg_encoder is not None and self.ecg_encoder is not self.encoder:
            self.ecg_encoder.eval()
        if self.reverse_encoder:
            self.reverse_encoder.eval()
        if self.reverse_decoder:
            self.reverse_decoder.eval()
        if self.latent_flow:
            self.latent_flow.eval()
        if self.subject_discriminator:
            self.subject_discriminator.eval()
        if self.patchgan_discriminator:
            self.patchgan_discriminator.eval()

        total_losses = {}
        num_batches = 0

        for batch in tqdm(self.val_loader, desc="Validation"):
            ppg = batch["ppg"].to(self.device)
            ecg = batch["ecg"].to(self.device)

            encoded = self._prepare_encoded_for_stage(self.encoder(ppg))
            if self.cycle_enabled and self._cycle_reverse_pretrain_active():
                losses = self._compute_reverse_pretrain_objective(ppg, ecg)
            elif self.is_cardio_flow:
                encoded_ecg = self.ecg_encoder(ecg)
                losses = self._compute_cardio_objective(
                    encoded, encoded_ecg, ecg, batch,
                )
            elif self.cycle_enabled:
                losses = self._compute_cycle_objective(
                    encoded, ppg, ecg,
                )
            else:
                ecg_pred, band_outputs, pred_coeffs = self._decode_main(encoded)
                losses = self._compute_objective(
                    encoded, ecg_pred, ecg, batch,
                    torch.zeros((), device=self.device),
                    include_adversarial=False,
                    band_outputs=band_outputs,
                    pred_coeffs=pred_coeffs,
                )

            for k, v in losses.items():
                total_losses[k] = total_losses.get(k, 0) + float(v.detach().item())
            num_batches += 1

        return {k: v / num_batches for k, v in total_losses.items()}

    def train(self, num_epochs: int):
        """完整训练流程。"""
        print(f"Training on {self.device}")
        print(f"Encoder params: {sum(p.numel() for p in self.encoder.parameters()):,}")
        print(f"Decoder params: {sum(p.numel() for p in self.decoder.parameters()):,}")
        if self.diffusion:
            print(f"Diffusion params: {sum(p.numel() for p in self.diffusion.parameters()):,}")
        if self.reverse_encoder:
            print(
                f"Reverse ECG->PPG params: "
                f"{sum(p.numel() for p in self.reverse_encoder.parameters()) + sum(p.numel() for p in self.reverse_decoder.parameters()):,}"
            )
        if self.patchgan_discriminator:
            print(
                f"PatchGAN params: "
                f"{sum(p.numel() for p in self.patchgan_discriminator.parameters()):,} "
                f"(G weight={self.patchgan_loss_weight:.4f}, "
                f"D lr={self.patchgan_lr:.2e})"
            )
        print(f"Output dir: {self.output_dir}")
        print("-" * 60)

        for epoch in range(num_epochs):
            self.epoch = epoch
            self._prepare_encoder_stage()
            self._prepare_cycle_stage()

            train_losses = self.train_one_epoch()
            val_losses = self.validate()

            self.scheduler.step()

            # 打印摘要
            print(f"\nEpoch {epoch + 1}/{num_epochs}")
            print(f"  Encoder stage: {self.encoder_stage}")
            if self.cycle_enabled:
                print(f"  Cycle stage: {self.cycle_stage}")
            print(f"  Train loss: {train_losses.get('total', 0):.4f}")
            print(f"  Val   loss: {val_losses.get('total', 0):.4f}")
            if self.subject_discriminator:
                print(
                    f"  Subject: loss={train_losses.get('subject_loss', 0):.4f}, "
                    f"acc={train_losses.get('subject_acc', 0):.4f}, "
                    f"GRL={train_losses.get('grl_lambda', 0):.3f}"
                )
            if self.patchgan_discriminator:
                print(
                    f"  PatchGAN: D={train_losses.get('patchgan_d', 0):.4f}, "
                    f"G={train_losses.get('patchgan_g', 0):.4f}"
                )
            if self.irm_enabled:
                print(f"  V-REx: {train_losses.get('irm_aux', 0):.6f}")

            epoch_record = {
                "epoch": epoch + 1,
                "encoder_stage": self.encoder_stage,
                "train": train_losses,
                "val": val_losses,
            }
            if self.cycle_enabled:
                epoch_record["cycle_stage"] = self.cycle_stage
            self.history.append(epoch_record)
            self._write_history()

            # Reverse pretraining produces a reusable forward model but is not
            # eligible for the deployable PPG->ECG best checkpoint.  Once the
            # frozen cycle stage starts, select by the configured ECG metric.
            if not (self.cycle_enabled and self._cycle_reverse_pretrain_active()):
                selection_metric = str(
                    self.train_cfg.get("selection_metric", "objective")
                )
                val_loss = val_losses.get(
                    selection_metric,
                    val_losses.get("objective", val_losses.get("total", float("inf"))),
                )
                improved = val_loss < self.best_val_loss - self.early_stopping_min_delta
                if improved:
                    self.best_val_loss = val_loss
                    self.best_epoch = epoch + 1
                    self.epochs_without_improvement = 0
                    self.save_checkpoint("best.pth")
                    print(
                        f"  ** Best model saved ({selection_metric}={val_loss:.4f})"
                    )
                elif self.early_stopping_patience > 0:
                    self.epochs_without_improvement += 1
                    print(
                        "  Early-stop monitor: "
                        f"{self.epochs_without_improvement}/"
                        f"{self.early_stopping_patience}"
                    )

            # 定期保存
            if (epoch + 1) % self.train_cfg.get("save_every", 10) == 0:
                self.save_checkpoint(f"epoch_{epoch + 1}.pth")

            if (
                self.early_stopping_patience > 0
                and self.epochs_without_improvement >= self.early_stopping_patience
            ):
                print(
                    f"Early stopping at epoch {epoch + 1}; "
                    f"best epoch={self.best_epoch}, "
                    f"best val={self.best_val_loss:.6f}"
                )
                break

        # 保存最终模型
        self.save_checkpoint("final.pth")
        print(f"\nTraining complete. Best val loss: {self.best_val_loss:.4f}")

    def _trainable_parameters(self):
        """返回 encoder/decoder/(可选 diffusion) 的全部可训练参数。"""
        params = list(self.encoder.parameters()) + list(self.decoder.parameters())
        if self.ecg_encoder is not None and self.ecg_encoder is not self.encoder:
            params += list(self.ecg_encoder.parameters())
        if self.reverse_encoder:
            params += list(self.reverse_encoder.parameters())
        if self.reverse_decoder:
            params += list(self.reverse_decoder.parameters())
        if self.latent_flow:
            params += list(self.latent_flow.parameters())
        if self.ppg_projection:
            params += list(self.ppg_projection.parameters())
        if self.ecg_projection:
            params += list(self.ecg_projection.parameters())
        if self.diffusion:
            params += list(self.diffusion.parameters())
        if self.subject_discriminator:
            params += list(self.subject_discriminator.parameters())
        return [p for p in params if p.requires_grad]

    def _write_subject_mapping(self):
        """保存训练 subject 的连续分类索引，便于复现实验和解释日志。"""
        if not self.subject_to_index:
            return
        path = self.output_dir / "subject_mapping.json"
        with path.open("w", encoding="utf-8") as f:
            json.dump(
                {str(k): int(v) for k, v in self.subject_to_index.items()},
                f, ensure_ascii=False, indent=2,
            )

    def _cycle_reverse_pretrain_active(self) -> bool:
        """Whether the current epoch belongs to ECG->PPG pretraining."""
        return bool(
            self.cycle_enabled
            and self.cycle_reverse_pretrain_epochs > 0
            and self.epoch < self.cycle_reverse_pretrain_epochs
        )

    def _set_reverse_frozen(self, frozen: bool) -> None:
        """Freeze/unfreeze reverse ECG->PPG parameters without blocking input gradients."""
        if self.reverse_encoder is None or self.reverse_decoder is None:
            return
        self.reverse_frozen = bool(frozen)
        for module in (self.reverse_encoder, self.reverse_decoder):
            for parameter in module.parameters():
                parameter.requires_grad_(not frozen)

    def _prepare_cycle_stage(self) -> None:
        """Transition from reverse pretraining to the frozen-cycle stage."""
        if not self.cycle_enabled:
            return
        if self._cycle_reverse_pretrain_active():
            self.cycle_stage = "reverse_pretrain"
            return
        if self.cycle_freeze_reverse:
            if not self.reverse_frozen and self.cycle_reverse_pretrain_epochs > 0:
                # At this point the preceding epoch has trained the direct
                # reverse path. Keep a named artifact before freezing it.
                self.save_checkpoint("reverse_pretrained.pth")
            self._set_reverse_frozen(True)
            self.cycle_stage = "frozen_cycle"
        else:
            self.cycle_stage = "joint_cycle"

    def _write_history(self):
        path = self.output_dir / "training_history.json"
        with path.open("w", encoding="utf-8") as f:
            json.dump(self.history, f, ensure_ascii=False, indent=2)

    def _load_encoder_weights(self, checkpoint_path: str | Path) -> None:
        """Initialize only the main encoder from a compatible checkpoint.

        The default path remains strict for ordinary experiments.  A widened
        VAE can opt into ``init_encoder_mode: overlap``; in that mode matching
        tensor prefixes are copied and newly added channels keep their fresh
        initialization.  This makes a latent-capacity ablation distinguish
        capacity from an entirely different random representation.
        """
        path = Path(checkpoint_path)
        if not path.exists():
            raise FileNotFoundError(
                f"Configured encoder checkpoint does not exist: {path}"
            )
        checkpoint = torch.load(path, map_location=self.device, weights_only=False)
        state = checkpoint.get("encoder") if isinstance(checkpoint, dict) else None
        if state is None:
            raise KeyError(f"Checkpoint {path} does not contain an 'encoder' state")
        init_mode = str(
            (self.config.get("model", {}) or {}).get(
                "init_encoder_mode", "strict",
            )
        ).lower()
        if init_mode not in {"strict", "overlap"}:
            raise ValueError(
                "model.init_encoder_mode must be 'strict' or 'overlap'"
            )

        if init_mode == "strict":
            missing, unexpected = self.encoder.load_state_dict(state, strict=False)
            if missing or unexpected:
                raise RuntimeError(
                    f"Incompatible encoder checkpoint {path}: "
                    f"missing={missing}, unexpected={unexpected}"
                )
        else:
            current = self.encoder.state_dict()
            adapted: dict[str, torch.Tensor] = {}
            copied_partial: list[str] = []
            skipped: list[str] = []
            for key, value in state.items():
                if key not in current or not torch.is_tensor(value):
                    skipped.append(key)
                    continue
                target = current[key]
                if target.shape == value.shape:
                    adapted[key] = value.to(device=target.device, dtype=target.dtype)
                    continue
                if target.ndim != value.ndim:
                    skipped.append(key)
                    continue
                # Copy the common prefix along every tensor dimension.  This
                # handles widened Conv/Norm/posterior-head tensors while
                # leaving the newly introduced channels at initialization.
                overlap = tuple(
                    slice(0, min(int(dst), int(src)))
                    for dst, src in zip(target.shape, value.shape)
                )
                merged = target.clone()
                merged[overlap] = value.to(
                    device=target.device, dtype=target.dtype,
                )[overlap]
                adapted[key] = merged
                copied_partial.append(key)
            missing, unexpected = self.encoder.load_state_dict(adapted, strict=False)
            print(
                "Initialized encoder with overlap transfer: "
                f"exact={len(adapted) - len(copied_partial)}, "
                f"partial={len(copied_partial)}, skipped={len(skipped)}, "
                f"missing={len(missing)}, unexpected={len(unexpected)}"
            )
        self.encoder_init_checkpoint = str(path)
        print(f"Initialized PPG encoder from: {path}")

    def _set_encoder_frozen(self, frozen: bool) -> None:
        """Freeze/unfreeze the main PPG encoder for decoder warm-up."""
        self.encoder_frozen = bool(frozen)
        for parameter in self.encoder.parameters():
            parameter.requires_grad_(not frozen)

    def _prepare_encoder_stage(self) -> None:
        """Apply the configured decoder-warmup/joint-training schedule."""
        should_freeze = (
            self.encoder_freeze_epochs > 0
            and self.epoch < self.encoder_freeze_epochs
        )
        self._set_encoder_frozen(should_freeze)
        self.encoder_stage = "decoder_warmup" if should_freeze else "joint"

    def _prepare_encoded_for_stage(self, encoded: dict | torch.Tensor):
        """Use deterministic VAE posterior means during encoder warm-up.

        The shallow copy preserves posterior statistics for logging while
        replacing only the tensors consumed by the decoder and auxiliary
        heads.  Non-VAE encoders and joint-training epochs are unchanged.
        """
        if not (
            self.encoder_frozen
            and self.use_posterior_mean_when_frozen
            and isinstance(encoded, dict)
            and "mu" in encoded
        ):
            return encoded
        prepared = dict(encoded)
        prepared["latent"] = encoded["mu"]
        if "mu_content" in encoded:
            prepared["z_content"] = encoded["mu_content"]
        if "mu_style" in encoded:
            prepared["z_style"] = encoded["mu_style"]
        return prepared

    def _current_grl_lambda(self) -> float:
        if not self.adv_enabled or self.subject_discriminator is None:
            return 0.0
        if self.grl_warmup_epochs <= 0:
            return self.grl_max_lambda
        progress = min(1.0, float(self.epoch + 1) / self.grl_warmup_epochs)
        return self.grl_max_lambda * progress

    def _subject_terms(
        self,
        encoded: dict | torch.Tensor,
        batch: dict,
        include_adversarial: bool = True,
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """Return discriminator CE, accuracy, and active GRL coefficient."""
        reference = encoded["latent"] if isinstance(encoded, dict) else encoded
        zero = reference.sum() * 0.0
        if (
            not include_adversarial
            or not self.adv_enabled
            or self.subject_discriminator is None
            or not isinstance(encoded, dict)
            or "z_content" not in encoded
            or "subject_id" not in batch
        ):
            return zero, zero.detach(), zero.detach()

        raw = batch["subject_id"].to(self.device).long().view(-1)
        labels = torch.full_like(raw, -1)
        for raw_id, mapped_id in self.subject_to_index.items():
            labels[raw == raw_id] = mapped_id
        valid = labels >= 0
        if int(valid.sum().item()) == 0:
            return zero, zero.detach(), zero.detach()

        grl_lambda = self._current_grl_lambda()
        logits = self.subject_discriminator(encoded["z_content"][valid], grl_lambda)
        subject_loss = F.cross_entropy(logits, labels[valid])
        subject_acc = (logits.argmax(dim=1) == labels[valid]).float().mean()
        return subject_loss, subject_acc.detach(), reference.new_tensor(grl_lambda)

    def _irm_penalty(
        self,
        sample_risk: torch.Tensor,
        batch: dict,
        reference: torch.Tensor,
    ) -> torch.Tensor:
        """Stable V-REx penalty: variance of per-subject environment risks."""
        zero = reference.sum() * 0.0
        if not self.irm_enabled or "subject_id" not in batch:
            return zero
        env_ids = batch["subject_id"].to(self.device).long().view(-1)
        risks = []
        for subject_id in torch.unique(env_ids):
            mask = env_ids == subject_id
            if int(mask.sum().item()) >= self.irm_min_env_samples:
                risks.append(sample_risk[mask].mean())
        if len(risks) < 2:
            return zero
        return torch.stack(risks).var(unbiased=False)

    @staticmethod
    def _posterior_mean(encoded: dict | torch.Tensor) -> torch.Tensor:
        if isinstance(encoded, dict):
            return encoded.get("mu", encoded["latent"])
        return encoded

    @staticmethod
    def _symmetric_diag_kl(
        mu_a: torch.Tensor,
        logvar_a: torch.Tensor,
        mu_b: torch.Tensor,
        logvar_b: torch.Tensor,
    ) -> torch.Tensor:
        """Symmetric KL for diagonal temporal Gaussian posteriors."""
        logvar_a = torch.clamp(logvar_a, min=-8.0, max=8.0)
        logvar_b = torch.clamp(logvar_b, min=-8.0, max=8.0)
        var_a = logvar_a.exp()
        var_b = logvar_b.exp()
        kl_ab = 0.5 * (
            logvar_b - logvar_a
            + (var_a + (mu_a - mu_b).square()) / var_b
            - 1.0
        ).mean()
        kl_ba = 0.5 * (
            logvar_a - logvar_b
            + (var_b + (mu_b - mu_a).square()) / var_a
            - 1.0
        ).mean()
        return 0.5 * (kl_ab + kl_ba)

    def _paired_info_nce(
        self,
        mu_ppg: torch.Tensor,
        mu_ecg: torch.Tensor,
    ) -> torch.Tensor:
        """InfoNCE over paired PPG/ECG posterior means."""
        if self.ppg_projection is None or self.ecg_projection is None:
            return mu_ppg.sum() * 0.0
        ppg = F.normalize(self.ppg_projection(mu_ppg.mean(dim=-1)), dim=-1)
        ecg = F.normalize(self.ecg_projection(mu_ecg.mean(dim=-1)), dim=-1)
        logits = ppg @ ecg.transpose(0, 1)
        logits = logits / max(self.cardio_temperature, 1e-4)
        labels = torch.arange(ppg.size(0), device=ppg.device)
        return 0.5 * (F.cross_entropy(logits, labels) + F.cross_entropy(logits.transpose(0, 1), labels))

    def _compute_cardio_objective(
        self,
        encoded_ppg: dict | torch.Tensor,
        encoded_ecg: dict | torch.Tensor,
        ecg: torch.Tensor,
        batch: dict,
    ) -> dict[str, torch.Tensor]:
        """Compute CardioAlign + paired latent Flow objectives for v0.4."""
        source = self._posterior_mean(encoded_ppg)
        target = self._posterior_mean(encoded_ecg)
        direct_pred = self.decoder(encoded_ppg)
        cross_pred = self.decoder(target)
        flow_latent = self.latent_flow(source, steps=self.cardio_flow_steps)
        flow_pred = self.decoder(flow_latent)

        direct_losses = self.criterion(direct_pred, ecg)
        cross_losses = self.criterion(cross_pred, ecg)
        flow_losses = self.criterion(flow_pred, ecg)

        zero = source.sum() * 0.0
        if isinstance(encoded_ppg, dict) and isinstance(encoded_ecg, dict):
            mu_ppg = encoded_ppg.get("mu", source)
            mu_ecg = encoded_ecg.get("mu", target)
            align_mse = F.mse_loss(mu_ppg, mu_ecg)
            if "logvar" in encoded_ppg and "logvar" in encoded_ecg:
                align_kl = self._symmetric_diag_kl(
                    mu_ppg, encoded_ppg["logvar"],
                    mu_ecg, encoded_ecg["logvar"],
                )
            else:
                align_kl = zero
            distribution = align_mse + align_kl
            contrastive = self._paired_info_nce(mu_ppg, mu_ecg)
            kl = (
                encoded_ppg.get("kl_content", zero)
                + encoded_ppg.get("kl_style", zero)
                + encoded_ecg.get("kl_content", zero)
                + encoded_ecg.get("kl_style", zero)
            )
        else:
            distribution = zero
            contrastive = zero
            kl = zero

        latent_flow_loss = self.latent_flow.loss(source, target)
        objective = (
            self.cardio_weights["direct_recon"] * direct_losses["total"]
            + self.cardio_weights["cross_recon"] * cross_losses["total"]
            + self.cardio_weights["flow_recon"] * flow_losses["total"]
            + self.cardio_weights["distribution"] * distribution
            + self.cardio_weights["contrastive"] * contrastive
            + self.cardio_weights["latent_flow"] * latent_flow_loss
            + self.vae_kl_weight * kl
        )
        losses = {
            "total": flow_losses["total"],
            "direct_total": direct_losses["total"],
            "cross_recon": cross_losses["total"],
            "flow_recon": flow_losses["total"],
            "align_distribution": distribution,
            "align_contrastive": contrastive,
            "latent_flow": latent_flow_loss,
            "kl_content": kl,
            "kl_style": zero,
            "objective": objective,
            "subject_loss": zero,
            "subject_acc": zero,
            "grl_lambda": zero,
            "irm_aux": zero,
            "diffusion": zero,
        }
        # Keep the individual reconstruction components visible in history.
        for name, value in flow_losses.items():
            if name != "total":
                losses[f"flow_{name}"] = value
        return losses

    def _decode_main(
        self,
        encoded: dict | torch.Tensor,
    ) -> tuple[torch.Tensor, dict | None, dict | None]:
        """Decode the main PPG path and expose optional structured outputs.

        The wavelet decoder returns the synthesized ECG together with the
        coefficient tensors that produced it.  Keeping those tensors in the
        training path is important: comparing a re-transformed fused signal
        would supervise the IDWT output, but would not directly train each
        coefficient head.
        """
        if self.is_wavelet_decoder:
            output = self.decoder(encoded, return_coeffs=True)
            return output["fused"], None, output["coefficients"]
        if self.is_band_decoder:
            output = self.decoder(encoded, return_bands=True)
            return output["fused"], output["bands"], None
        return self.decoder(encoded), None, None

    def _patchgan_step(
        self,
        ppg: torch.Tensor,
        ecg: torch.Tensor,
        ecg_pred: torch.Tensor,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """Update the conditional PatchGAN and return its generator loss.

        The discriminator update uses a detached fake signal.  For the
        generator update discriminator parameters are temporarily frozen, so
        the returned gradient flows through its patch logits into ``ecg_pred``
        without accumulating a second discriminator gradient.
        """
        if self.patchgan_discriminator is None or self.patchgan_optimizer is None:
            zero = ecg_pred.sum() * 0.0
            return zero, zero

        discriminator = self.patchgan_discriminator
        self.patchgan_optimizer.zero_grad(set_to_none=True)
        real_logits = discriminator(ppg, ecg)
        fake_logits = discriminator(ppg, ecg_pred.detach())
        d_loss = patchgan_hinge_discriminator_loss(real_logits, fake_logits)

        if self.amp_scaler is not None:
            self.amp_scaler.scale(d_loss).backward()
            self.amp_scaler.unscale_(self.patchgan_optimizer)
            torch.nn.utils.clip_grad_norm_(
                discriminator.parameters(),
                self.train_cfg.get("patchgan_max_grad_norm", 1.0),
            )
            self.amp_scaler.step(self.patchgan_optimizer)
            self.amp_scaler.update()
        else:
            d_loss.backward()
            torch.nn.utils.clip_grad_norm_(
                discriminator.parameters(),
                self.train_cfg.get("patchgan_max_grad_norm", 1.0),
            )
            self.patchgan_optimizer.step()

        # Do not let the generator update create stale discriminator grads.
        self.patchgan_optimizer.zero_grad(set_to_none=True)
        for parameter in discriminator.parameters():
            parameter.requires_grad_(False)
        try:
            fake_logits_for_g = discriminator(ppg, ecg_pred)
            g_loss = patchgan_hinge_generator_loss(fake_logits_for_g)
        finally:
            for parameter in discriminator.parameters():
                parameter.requires_grad_(True)
        return d_loss.detach(), g_loss

    def _band_terms(
        self,
        bands: dict[str, torch.Tensor] | None,
        target: torch.Tensor,
        reference: torch.Tensor,
    ) -> dict[str, torch.Tensor]:
        """Return energy-normalized low/mid/high reconstruction terms."""
        zero = reference.sum() * 0.0
        empty = {
            "band_total": zero,
            "band_low": zero,
            "band_mid": zero,
            "band_high": zero,
            "band_qrs_amplitude": zero,
        }
        if (
            not self.band_loss_enabled
            or (
                self.band_loss_weight == 0.0
                and self.band_qrs_amplitude_weight == 0.0
            )
            or bands is None
            or not hasattr(self.decoder, "project_bands")
        ):
            return empty

        target_bands = self.decoder.project_bands(target)
        values: dict[str, torch.Tensor] = {}
        for name in ("low", "mid", "high"):
            prediction = bands[name]
            band_target = target_bands[name].to(dtype=prediction.dtype)
            difference = torch.abs(prediction - band_target)
            if self.band_loss_normalize:
                scale = band_target.abs().mean(dim=-1, keepdim=True)
                scale = scale.clamp_min(self.band_loss_scale_floor)
                value = (difference / scale).mean()
            else:
                value = difference.mean()
            values[name] = value

        if self.band_qrs_amplitude_weight != 0.0:
            # ``bands["high"]`` and ``target_bands["high"]`` retain the same
            # time grid.  The QRS mask is estimated from the target only by
            # QRSAmplitudeLoss, so no target information enters inference.
            values["qrs_amplitude"] = self.criterion.qrs_amplitude(
                bands["high"],
                target_bands["high"].to(dtype=bands["high"].dtype),
                mask_target=target,
            )
        else:
            values["qrs_amplitude"] = zero

        total = self.band_loss_weight * sum(
            self.band_loss_band_weights[name] * values[name]
            for name in ("low", "mid", "high")
        ) + self.band_qrs_amplitude_weight * values["qrs_amplitude"]
        return {
            "band_total": total,
            "band_low": values["low"],
            "band_mid": values["mid"],
            "band_high": values["high"],
            "band_qrs_amplitude": values["qrs_amplitude"],
        }

    def _cycle_signal_loss(
        self,
        prediction: torch.Tensor,
        target: torch.Tensor,
    ) -> torch.Tensor:
        """Signal loss for the reverse ECG->PPG branch and PPG cycle."""
        if self.cycle_loss_type == "mse":
            return F.mse_loss(prediction, target)
        return F.l1_loss(prediction, target)

    def _compute_reverse_pretrain_objective(
        self,
        ppg: torch.Tensor,
        ecg: torch.Tensor,
    ) -> dict[str, torch.Tensor]:
        """Train only the directly supervised ECG->PPG forward model."""
        if self.reverse_encoder is None or self.reverse_decoder is None:
            raise RuntimeError("reverse pretraining requested without reverse modules")
        encoded_ppg = self.encoder(ppg)
        ecg_pred, band_outputs, pred_coeffs = self._decode_main(encoded_ppg)
        ecg_losses = self.criterion(ecg_pred, ecg, pred_coeffs=pred_coeffs)
        reference = encoded_ppg["latent"] if isinstance(encoded_ppg, dict) else encoded_ppg
        band_losses = self._band_terms(band_outputs, ecg, reference)
        ppg_direct = self.reverse_decoder(self.reverse_encoder(ecg))
        direct_loss = self._cycle_signal_loss(ppg_direct, ppg)
        objective = (
            self.cycle_pretrain_ecg_weight * (
                ecg_losses["total"] + band_losses["band_total"]
            )
            + self.cycle_direct_weight * direct_loss
        )
        zero = objective.detach() * 0.0
        return {
            "total": objective,
            "objective": objective,
            "ecg_total": ecg_losses["total"],
            "mse": ecg_losses["mse"],
            "l1": ecg_losses["l1"],
            "qrs_weighted": ecg_losses["qrs_weighted"],
            "qrs_amplitude": ecg_losses["qrs_amplitude"],
            "derivative": ecg_losses["derivative"],
            "freq": ecg_losses["freq"],
            "wavelet": ecg_losses["wavelet"],
            "wavelet_qrs": ecg_losses["wavelet_qrs"],
            "haar_wavelet": ecg_losses["haar_wavelet"],
            "haar_qrs": ecg_losses["haar_qrs"],
            "peak_interval": ecg_losses["peak_interval"],
            "band_total": band_losses["band_total"],
            "band_low": band_losses["band_low"],
            "band_mid": band_losses["band_mid"],
            "band_high": band_losses["band_high"],
            "band_qrs_amplitude": band_losses["band_qrs_amplitude"],
            "ppg_direct": direct_loss,
            "ppg_cycle": zero,
            "subject_loss": zero,
            "subject_acc": zero,
            "grl_lambda": zero,
            "irm_aux": zero,
            "flow_nll": zero,
            "kl_content": zero,
            "kl_style": zero,
            "diffusion": zero,
        }

    def _compute_cycle_objective(
        self,
        encoded_ppg: dict | torch.Tensor,
        ppg: torch.Tensor,
        ecg: torch.Tensor,
    ) -> dict[str, torch.Tensor]:
        """Compute v0.5 bidirectional ECG<->PPG objective.

        The direct reverse path anchors the learned ECG->PPG mapping on paired
        observations.  The cycle path uses the generated ECG, so a PPG->ECG
        prediction must remain compatible with that forward physiological map.
        It does not force ECG->PPG->ECG identity, which would incorrectly
        demand recovery of ECG details absent from PPG.
        """
        if self.reverse_encoder is None or self.reverse_decoder is None:
            raise RuntimeError("cycle objective requested without reverse modules")

        ecg_pred, band_outputs, pred_coeffs = self._decode_main(encoded_ppg)
        encoded_ecg = self.reverse_encoder(ecg)
        ppg_direct = self.reverse_decoder(encoded_ecg)
        encoded_generated_ecg = self.reverse_encoder(ecg_pred)
        ppg_cycle = self.reverse_decoder(encoded_generated_ecg)

        ecg_losses = self.criterion(ecg_pred, ecg, pred_coeffs=pred_coeffs)
        reference = encoded_ppg["latent"] if isinstance(encoded_ppg, dict) else encoded_ppg
        band_losses = self._band_terms(band_outputs, ecg, reference)
        direct_loss = self._cycle_signal_loss(ppg_direct, ppg)
        cycle_loss = self._cycle_signal_loss(ppg_cycle, ppg)
        objective = (
            ecg_losses["total"] + band_losses["band_total"]
            + self.cycle_direct_weight * direct_loss
            + self.cycle_reconstruction_weight * cycle_loss
        )
        zero = objective.detach() * 0.0
        return {
            # Keep ``total`` as the optimized value so the generic logger and
            # best-checkpoint selection report the complete v0.5 objective.
            "total": objective,
            "objective": objective,
            "ecg_total": ecg_losses["total"],
            "mse": ecg_losses["mse"],
            "l1": ecg_losses["l1"],
            "qrs_weighted": ecg_losses["qrs_weighted"],
            "qrs_amplitude": ecg_losses["qrs_amplitude"],
            "derivative": ecg_losses["derivative"],
            "freq": ecg_losses["freq"],
            "wavelet": ecg_losses["wavelet"],
            "wavelet_qrs": ecg_losses["wavelet_qrs"],
            "haar_wavelet": ecg_losses["haar_wavelet"],
            "haar_qrs": ecg_losses["haar_qrs"],
            "peak_interval": ecg_losses["peak_interval"],
            "band_total": band_losses["band_total"],
            "band_low": band_losses["band_low"],
            "band_mid": band_losses["band_mid"],
            "band_high": band_losses["band_high"],
            "band_qrs_amplitude": band_losses["band_qrs_amplitude"],
            "ppg_direct": direct_loss,
            "ppg_cycle": cycle_loss,
            "subject_loss": zero,
            "subject_acc": zero,
            "grl_lambda": zero,
            "irm_aux": zero,
            "flow_nll": zero,
            "kl_content": zero,
            "kl_style": zero,
            "diffusion": zero,
        }

    def _compute_objective(
        self,
        encoded: dict | torch.Tensor,
        ecg_pred: torch.Tensor,
        ecg: torch.Tensor,
        batch: dict,
        diff_loss: torch.Tensor,
        include_adversarial: bool = True,
        band_outputs: dict[str, torch.Tensor] | None = None,
        pred_coeffs: dict | None = None,
    ) -> dict[str, torch.Tensor]:
        """Compute reconstruction and all optional advanced objectives."""
        recon_losses = self.criterion(ecg_pred, ecg, pred_coeffs=pred_coeffs)
        reference = encoded["latent"] if isinstance(encoded, dict) else encoded
        zero = reference.sum() * 0.0
        band_losses = self._band_terms(band_outputs, ecg, reference)

        if self.is_flow and hasattr(self.decoder, "nll_per_sample"):
            flow_per_sample = self.decoder.nll_per_sample(ecg, encoded)
            flow_nll = flow_per_sample.mean()
        else:
            flow_per_sample = torch.zeros(ecg.size(0), device=self.device) + zero
            flow_nll = zero

        if self.is_vae and isinstance(encoded, dict):
            kl_content = encoded.get("kl_content", zero)
            kl_style = encoded.get("kl_style", zero)
        else:
            kl_content = zero
            kl_style = zero

        aux_active = not self.encoder_frozen
        subject_loss, subject_acc, grl_lambda = self._subject_terms(
            encoded,
            batch,
            include_adversarial=include_adversarial and aux_active,
        )
        sample_recon = (ecg_pred - ecg).square().mean(dim=(1, 2))
        sample_risk = sample_recon + self.flow_weight * flow_per_sample
        irm_aux = (
            self._irm_penalty(sample_risk, batch, reference)
            if aux_active
            else zero
        )

        effective_kl_weight = self.vae_kl_weight if aux_active else 0.0
        effective_flow_weight = self.flow_weight if aux_active else 0.0

        objective = (
            recon_losses["total"]
            + band_losses["band_total"]
            + effective_flow_weight * flow_nll
            + effective_kl_weight * (kl_content + kl_style)
            + self.adv_loss_weight * subject_loss
            + self.irm_weight * irm_aux
            + self.diff_weight * diff_loss
        )
        losses = dict(recon_losses)
        losses.update({
            "objective": objective,
            "band_total": band_losses["band_total"],
            "band_low": band_losses["band_low"],
            "band_mid": band_losses["band_mid"],
            "band_high": band_losses["band_high"],
            "band_qrs_amplitude": band_losses["band_qrs_amplitude"],
            "flow_nll": flow_nll,
            "kl_content": kl_content,
            "kl_style": kl_style,
            "subject_loss": subject_loss,
            "subject_acc": subject_acc,
            "grl_lambda": grl_lambda,
            "irm_aux": irm_aux,
            "diffusion": diff_loss,
        })
        return losses

    def save_checkpoint(self, filename: str):
        """保存检查点。"""
        ckpt = {
            "epoch": self.epoch,
            "encoder": self.encoder.state_dict(),
            "decoder": self.decoder.state_dict(),
            "optimizer": self.optimizer.state_dict(),
            "scheduler": self.scheduler.state_dict(),
            "best_val_loss": self.best_val_loss,
            "best_epoch": self.best_epoch,
            "epochs_without_improvement": self.epochs_without_improvement,
            "config": self.config,
        }
        if self.diffusion:
            ckpt["diffusion"] = self.diffusion.state_dict()
        if self.ecg_encoder is not None and self.ecg_encoder is not self.encoder:
            ckpt["ecg_encoder"] = self.ecg_encoder.state_dict()
        if self.reverse_encoder:
            ckpt["reverse_encoder"] = self.reverse_encoder.state_dict()
        if self.reverse_decoder:
            ckpt["reverse_decoder"] = self.reverse_decoder.state_dict()
        if self.latent_flow:
            ckpt["latent_flow"] = self.latent_flow.state_dict()
        if self.ppg_projection:
            ckpt["ppg_projection"] = self.ppg_projection.state_dict()
        if self.ecg_projection:
            ckpt["ecg_projection"] = self.ecg_projection.state_dict()
        if self.subject_discriminator:
            ckpt["subject_discriminator"] = self.subject_discriminator.state_dict()
            ckpt["subject_to_index"] = self.subject_to_index
        if self.patchgan_discriminator:
            ckpt["patchgan_discriminator"] = self.patchgan_discriminator.state_dict()
            if self.patchgan_optimizer is not None:
                ckpt["patchgan_optimizer"] = self.patchgan_optimizer.state_dict()
        ckpt["history"] = self.history

        path = self.output_dir / filename
        torch.save(ckpt, path)

    def load_checkpoint(self, path: str | Path):
        """加载检查点。"""
        ckpt = torch.load(path, map_location=self.device, weights_only=False)
        self.encoder.load_state_dict(ckpt["encoder"])
        self.decoder.load_state_dict(ckpt["decoder"])
        if self.diffusion and "diffusion" in ckpt:
            self.diffusion.load_state_dict(ckpt["diffusion"])
        if self.ecg_encoder is not None and self.ecg_encoder is not self.encoder and "ecg_encoder" in ckpt:
            self.ecg_encoder.load_state_dict(ckpt["ecg_encoder"])
        if self.reverse_encoder is not None and "reverse_encoder" in ckpt:
            self.reverse_encoder.load_state_dict(ckpt["reverse_encoder"])
        if self.reverse_decoder is not None and "reverse_decoder" in ckpt:
            self.reverse_decoder.load_state_dict(ckpt["reverse_decoder"])
        if self.latent_flow and "latent_flow" in ckpt:
            self.latent_flow.load_state_dict(ckpt["latent_flow"])
        if self.ppg_projection and "ppg_projection" in ckpt:
            self.ppg_projection.load_state_dict(ckpt["ppg_projection"])
        if self.ecg_projection and "ecg_projection" in ckpt:
            self.ecg_projection.load_state_dict(ckpt["ecg_projection"])
        if self.subject_discriminator and "subject_discriminator" in ckpt:
            self.subject_discriminator.load_state_dict(ckpt["subject_discriminator"])
        if self.patchgan_discriminator and "patchgan_discriminator" in ckpt:
            self.patchgan_discriminator.load_state_dict(
                ckpt["patchgan_discriminator"]
            )
            if self.patchgan_optimizer is not None and "patchgan_optimizer" in ckpt:
                self.patchgan_optimizer.load_state_dict(ckpt["patchgan_optimizer"])
        self.optimizer.load_state_dict(ckpt["optimizer"])
        self.scheduler.load_state_dict(ckpt["scheduler"])
        self.epoch = ckpt["epoch"]
        self.best_val_loss = ckpt["best_val_loss"]
        self.best_epoch = int(ckpt.get("best_epoch", self.epoch))
        self.epochs_without_improvement = int(
            ckpt.get("epochs_without_improvement", 0)
        )
        self.history = ckpt.get("history", [])
        print(f"Checkpoint loaded: {path} (epoch {self.epoch})")
