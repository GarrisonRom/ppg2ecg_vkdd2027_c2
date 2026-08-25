#!/usr/bin/env python
"""Train and evaluate controlled CardioGAN/RDDM reimplementations.

Examples (from the project root):

  D:\\Anaconda\\envs\\cuda126_env\\python.exe scripts/train_paper_baselines.py \
      --method cardiogan --config configs/exp_cardiogan_repro.yaml
  D:\\Anaconda\\envs\\cuda126_env\\python.exe scripts/train_paper_baselines.py \
      --method rddm --config configs/exp_rddm_repro.yaml

The script keeps the current SensSmartTech subject-wise 22/5/5 split and
produces `best.pth`, `pred_<split>.npz`, and the project's standard
`eval_<split>.json` files. These are controlled adaptations of the papers to
4-channel, 250 Hz, 2000-point data; they are not claims of an exact original
dataset reproduction.
"""

from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F
from torch.optim import Adam
from tqdm import tqdm

PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

from src.data import create_dataloaders
from src.evaluation.metrics import evaluate_all
from src.models.paper_baselines import (
    AttentionUNet1D,
    PatchDiscriminator1D,
    RDDMCore,
    SpectrogramDiscriminator,
)
from src.utils import set_seed
from src.utils.config import _deep_merge, load_config, save_config


DEFAULTS = {
    "experiment": {"name": "senssmarttech_paper_repro"},
    "data": {
        "dataset": "senssmarttech",
        "root": "data/processed/SensSmartTech/subjectwise_per-lead",
        "ppg_channel": None,
        "ecg_lead": None,
        "batch_size": 8,
        "num_workers": 0,
    },
    "method": "cardiogan",
    "training": {
        "epochs": 20,
        "lr": 1e-4,
        "weight_decay": 1e-5,
        "max_grad_norm": 1.0,
        "save_every": 5,
    },
    "cardiogan": {
        "base_channels": 16,
        "disc_base_channels": 16,
        "freq_disc_base_channels": 8,
        "alpha_time": 3.0,
        "beta_frequency": 1.0,
        "cycle_weight": 30.0,
        "output_activation": "none",
    },
    "rddm": {
        "base_channels": 16,
        "timesteps": 1000,
        "beta_end": 0.2,
        "roi_gamma": 32,
        "roi_threshold": 1.5,
        "lambda_roi": 100.0,
        "lambda_global": 1.0,
        "sampling_steps": 10,
    },
    "seed": 42,
    "device": "auto",
}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="CardioGAN/RDDM controlled reimplementation")
    parser.add_argument("--method", choices=["cardiogan", "rddm"], required=True)
    parser.add_argument("--config", type=Path, default=None)
    parser.add_argument("--name", type=str, default=None)
    parser.add_argument("--epochs", type=int, default=None)
    parser.add_argument("--batch-size", type=int, default=None)
    parser.add_argument("--seed", type=int, default=None)
    parser.add_argument("--device", choices=["auto", "cuda", "cpu"], default=None)
    parser.add_argument("--output", type=Path, default=None)
    parser.add_argument("--sample-steps", type=int, default=None)
    return parser.parse_args()


def build_config(args: argparse.Namespace) -> dict:
    cfg = _deep_merge({}, DEFAULTS)
    if args.config is not None:
        cfg = _deep_merge(cfg, load_config(args.config))
    cfg["method"] = args.method
    if args.name is not None:
        cfg.setdefault("experiment", {})["name"] = args.name
    if args.epochs is not None:
        cfg.setdefault("training", {})["epochs"] = args.epochs
    if args.batch_size is not None:
        cfg.setdefault("data", {})["batch_size"] = args.batch_size
    if args.seed is not None:
        cfg["seed"] = args.seed
    if args.device is not None:
        cfg["device"] = args.device
    if args.sample_steps is not None:
        cfg.setdefault("rddm", {})["sampling_steps"] = args.sample_steps
    return cfg


def resolve_device(value: str) -> torch.device:
    if value == "auto":
        return torch.device("cuda" if torch.cuda.is_available() else "cpu")
    return torch.device(value)


def _target(device: torch.device, prediction: torch.Tensor) -> torch.Tensor:
    return torch.ones_like(prediction, device=device)


def _source(device: torch.device, prediction: torch.Tensor) -> torch.Tensor:
    return torch.zeros_like(prediction, device=device)


class CardioGANTrainer:
    """CycleGAN training with dual time/frequency discriminators."""

    def __init__(self, cfg: dict, loaders: dict, device: torch.device, out_dir: Path):
        self.cfg = cfg
        self.device = device
        self.out_dir = out_dir
        train_ds = loaders["train"].dataset
        ppg_channels = train_ds.num_ppg_channels
        ecg_channels = train_ds.ecg_leads
        mcfg = cfg.get("cardiogan", {})
        base = int(mcfg.get("base_channels", 16))

        self.g_ecg = AttentionUNet1D(
            ppg_channels, ecg_channels, base_channels=base,
            output_activation=str(mcfg.get("output_activation", "none")),
        ).to(device)
        self.g_ppg = AttentionUNet1D(
            ecg_channels, ppg_channels, base_channels=base,
            output_activation=str(mcfg.get("output_activation", "none")),
        ).to(device)
        dbase = int(mcfg.get("disc_base_channels", 16))
        fbase = int(mcfg.get("freq_disc_base_channels", 8))
        self.d_ecg_time = PatchDiscriminator1D(ecg_channels, dbase).to(device)
        self.d_ppg_time = PatchDiscriminator1D(ppg_channels, dbase).to(device)
        self.d_ecg_freq = SpectrogramDiscriminator(ecg_channels, fbase).to(device)
        self.d_ppg_freq = SpectrogramDiscriminator(ppg_channels, fbase).to(device)

        lr = float(cfg.get("training", {}).get("lr", 1e-4))
        wd = float(cfg.get("training", {}).get("weight_decay", 1e-5))
        self.opt_g = Adam(
            list(self.g_ecg.parameters()) + list(self.g_ppg.parameters()),
            lr=lr, betas=(0.5, 0.999), weight_decay=wd,
        )
        self.opt_d = Adam(
            list(self.d_ecg_time.parameters()) + list(self.d_ppg_time.parameters())
            + list(self.d_ecg_freq.parameters()) + list(self.d_ppg_freq.parameters()),
            lr=lr, betas=(0.5, 0.999), weight_decay=wd,
        )
        self.alpha_time = float(mcfg.get("alpha_time", 3.0))
        self.beta_frequency = float(mcfg.get("beta_frequency", 1.0))
        self.cycle_weight = float(mcfg.get("cycle_weight", 30.0))
        self.max_grad_norm = float(cfg.get("training", {}).get("max_grad_norm", 1.0))
        self.history: list[dict] = []
        self.best_val = float("inf")

    @staticmethod
    def _gan_loss(logits: torch.Tensor, real: bool) -> torch.Tensor:
        return F.mse_loss(logits, torch.ones_like(logits) if real else torch.zeros_like(logits))

    def _discriminator_loss(self, disc, real: torch.Tensor, fake: torch.Tensor) -> torch.Tensor:
        return 0.5 * (self._gan_loss(disc(real), True) + self._gan_loss(disc(fake.detach()), False))

    def train_epoch(self, loader) -> dict[str, float]:
        self.g_ecg.train(); self.g_ppg.train()
        for module in (
            self.d_ecg_time, self.d_ppg_time, self.d_ecg_freq, self.d_ppg_freq,
        ):
            module.train()
        totals: dict[str, float] = {}
        n_batches = 0
        for batch in tqdm(loader, desc="CardioGAN", leave=False):
            ppg = batch["ppg"].to(self.device, non_blocking=True)
            ecg = batch["ecg"].to(self.device, non_blocking=True)
            # The original CardioGAN trains the two domains in an unpaired way.
            ppg_real = ppg[torch.randperm(ppg.size(0), device=self.device)]
            ecg_real = ecg[torch.randperm(ecg.size(0), device=self.device)]

            with torch.no_grad():
                fake_ecg = self.g_ecg(ppg_real)
                fake_ppg = self.g_ppg(ecg_real)
            d_loss = (
                self._discriminator_loss(self.d_ecg_time, ecg_real, fake_ecg)
                + self._discriminator_loss(self.d_ecg_freq, ecg_real, fake_ecg)
                + self._discriminator_loss(self.d_ppg_time, ppg_real, fake_ppg)
                + self._discriminator_loss(self.d_ppg_freq, ppg_real, fake_ppg)
            )
            self.opt_d.zero_grad(set_to_none=True)
            d_loss.backward()
            torch.nn.utils.clip_grad_norm_(self._discriminator_parameters(), self.max_grad_norm)
            self.opt_d.step()

            fake_ecg = self.g_ecg(ppg_real)
            fake_ppg = self.g_ppg(ecg_real)
            rec_ppg = self.g_ppg(fake_ecg)
            rec_ecg = self.g_ecg(fake_ppg)
            adv_time = (
                self._gan_loss(self.d_ecg_time(fake_ecg), True)
                + self._gan_loss(self.d_ppg_time(fake_ppg), True)
            )
            adv_freq = (
                self._gan_loss(self.d_ecg_freq(fake_ecg), True)
                + self._gan_loss(self.d_ppg_freq(fake_ppg), True)
            )
            cycle = F.l1_loss(rec_ppg, ppg_real) + F.l1_loss(rec_ecg, ecg_real)
            g_loss = self.alpha_time * adv_time + self.beta_frequency * adv_freq + self.cycle_weight * cycle
            self.opt_g.zero_grad(set_to_none=True)
            g_loss.backward()
            torch.nn.utils.clip_grad_norm_(self._generator_parameters(), self.max_grad_norm)
            self.opt_g.step()

            vals = {"g_total": g_loss, "d_total": d_loss, "adv_time": adv_time,
                    "adv_freq": adv_freq, "cycle": cycle}
            for key, value in vals.items():
                totals[key] = totals.get(key, 0.0) + float(value.detach().item())
            n_batches += 1
        return {k: v / max(1, n_batches) for k, v in totals.items()}

    def _generator_parameters(self):
        yield from self.g_ecg.parameters()
        yield from self.g_ppg.parameters()

    def _discriminator_parameters(self):
        for module in (self.d_ecg_time, self.d_ppg_time, self.d_ecg_freq, self.d_ppg_freq):
            yield from module.parameters()

    @torch.no_grad()
    def validation_l1(self, loader) -> float:
        self.g_ecg.eval()
        total = 0.0
        count = 0
        for batch in loader:
            ppg = batch["ppg"].to(self.device)
            ecg = batch["ecg"].to(self.device)
            total += float(F.l1_loss(self.g_ecg(ppg), ecg).item())
            count += 1
        return total / max(1, count)

    def save(self, path: Path, epoch: int):
        torch.save({
            "method": "cardiogan",
            "epoch": epoch,
            "generator_ecg": self.g_ecg.state_dict(),
            "generator_ppg": self.g_ppg.state_dict(),
            "discriminator_ecg_time": self.d_ecg_time.state_dict(),
            "discriminator_ppg_time": self.d_ppg_time.state_dict(),
            "discriminator_ecg_freq": self.d_ecg_freq.state_dict(),
            "discriminator_ppg_freq": self.d_ppg_freq.state_dict(),
            "optimizer_g": self.opt_g.state_dict(),
            "optimizer_d": self.opt_d.state_dict(),
            "history": self.history,
            "config": self.cfg,
        }, path)

    def load(self, path: Path):
        ckpt = torch.load(path, map_location=self.device, weights_only=False)
        self.g_ecg.load_state_dict(ckpt["generator_ecg"])
        self.g_ppg.load_state_dict(ckpt["generator_ppg"])
        self.d_ecg_time.load_state_dict(ckpt["discriminator_ecg_time"])
        self.d_ppg_time.load_state_dict(ckpt["discriminator_ppg_time"])
        self.d_ecg_freq.load_state_dict(ckpt["discriminator_ecg_freq"])
        self.d_ppg_freq.load_state_dict(ckpt["discriminator_ppg_freq"])
        self.history = ckpt.get("history", [])


class RDDMTrainer:
    """Training wrapper for the ROI-guided diffusion core."""

    def __init__(self, cfg: dict, loaders: dict, device: torch.device, out_dir: Path):
        self.cfg = cfg
        self.device = device
        self.out_dir = out_dir
        train_ds = loaders["train"].dataset
        rcfg = cfg.get("rddm", {})
        self.model = RDDMCore(
            signal_channels=train_ds.ecg_leads,
            condition_channels=train_ds.num_ppg_channels,
            timesteps=int(rcfg.get("timesteps", 1000)),
            beta_end=float(rcfg.get("beta_end", 0.2)),
            base_channels=int(rcfg.get("base_channels", 16)),
            roi_gamma=int(rcfg.get("roi_gamma", 32)),
            roi_threshold=float(rcfg.get("roi_threshold", 1.5)),
            lambda_roi=float(rcfg.get("lambda_roi", 100.0)),
            lambda_global=float(rcfg.get("lambda_global", 1.0)),
        ).to(device)
        train_cfg = cfg.get("training", {})
        self.optimizer = Adam(
            self.model.parameters(),
            lr=float(train_cfg.get("lr", 1e-4)),
            weight_decay=float(train_cfg.get("weight_decay", 1e-5)),
        )
        self.max_grad_norm = float(train_cfg.get("max_grad_norm", 1.0))
        self.sample_steps = int(rcfg.get("sampling_steps", 10))
        self.history: list[dict] = []
        self.best_val = float("inf")

    def train_epoch(self, loader) -> dict[str, float]:
        self.model.train()
        totals: dict[str, float] = {}
        n_batches = 0
        for batch in tqdm(loader, desc="RDDM", leave=False):
            ppg = batch["ppg"].to(self.device, non_blocking=True)
            ecg = batch["ecg"].to(self.device, non_blocking=True)
            losses = self.model.training_loss(ppg, ecg)
            self.optimizer.zero_grad(set_to_none=True)
            losses["total"].backward()
            torch.nn.utils.clip_grad_norm_(self.model.parameters(), self.max_grad_norm)
            self.optimizer.step()
            for key, value in losses.items():
                totals[key] = totals.get(key, 0.0) + float(value.detach().item())
            n_batches += 1
        return {k: v / max(1, n_batches) for k, v in totals.items()}

    @torch.no_grad()
    def validation_loss(self, loader) -> float:
        self.model.eval()
        total = 0.0
        count = 0
        for batch in loader:
            ppg = batch["ppg"].to(self.device)
            ecg = batch["ecg"].to(self.device)
            total += float(self.model.training_loss(ppg, ecg)["total"].item())
            count += 1
        return total / max(1, count)

    def save(self, path: Path, epoch: int):
        torch.save({
            "method": "rddm",
            "epoch": epoch,
            "model": self.model.state_dict(),
            "optimizer": self.optimizer.state_dict(),
            "history": self.history,
            "config": self.cfg,
        }, path)

    def load(self, path: Path):
        ckpt = torch.load(path, map_location=self.device, weights_only=False)
        self.model.load_state_dict(ckpt["model"])
        self.history = ckpt.get("history", [])


@torch.no_grad()
def predict_generator(model, dataset, device: torch.device, batch_size: int, seed: int) -> tuple[np.ndarray, float]:
    model.eval()
    x = torch.from_numpy(dataset._x).float()
    preds = []
    start = time.perf_counter()
    torch.manual_seed(seed)
    if device.type == "cuda":
        torch.cuda.manual_seed_all(seed)
    for i in range(0, len(x), batch_size):
        ppg = x[i:i + batch_size].to(device)
        preds.append(model(ppg).cpu().numpy())
    return np.concatenate(preds, axis=0), time.perf_counter() - start


@torch.no_grad()
def predict_rddm(model: RDDMCore, dataset, device: torch.device, batch_size: int,
                 steps: int, seed: int) -> tuple[np.ndarray, float]:
    model.eval()
    x = torch.from_numpy(dataset._x).float()
    preds = []
    start = time.perf_counter()
    torch.manual_seed(seed)
    if device.type == "cuda":
        torch.cuda.manual_seed_all(seed)
    for i in tqdm(range(0, len(x), batch_size), desc="RDDM sample", leave=False):
        ppg = x[i:i + batch_size].to(device)
        preds.append(model.sample(ppg, steps=steps).cpu().numpy())
    return np.concatenate(preds, axis=0), time.perf_counter() - start


def evaluate_method(
    method: str,
    model,
    loaders: dict,
    out_dir: Path,
    device: torch.device,
    cfg: dict,
    seed: int,
):
    batch_size = int(cfg.get("data", {}).get("batch_size", 8))
    steps = int(cfg.get("rddm", {}).get("sampling_steps", 10))
    for split in ("train", "val", "test"):
        if split not in loaders:
            continue
        ds = loaders[split].dataset
        if method == "cardiogan":
            pred, elapsed = predict_generator(model.g_ecg, ds, device, batch_size, seed)
        else:
            pred, elapsed = predict_rddm(model.model, ds, device, batch_size, steps, seed)
        target = ds._y
        subject_ids = ds.metadata["subject_id"].to_numpy()
        result = {
            "run": str(out_dir),
            "method": method,
            "split": split,
            "protocol": {
                "dataset": cfg.get("data", {}).get("dataset"),
                "root": cfg.get("data", {}).get("root"),
                "ppg_channels": ds.ppg_channels,
                "ecg_channels": ds.ecg_channels,
                "fs": ds.fs,
                "signal_length": ds.signal_length,
                "split": "subject-wise 22/5/5",
                "paper_adaptation": True,
            },
            "model": evaluate_all(
                pred, target, ds.fs, ds.ecg_channels, subject_ids,
                ppg=ds._x, ppg_names=ds.ppg_channels,
            ),
            "efficiency": {
                "params_M": sum(p.numel() for p in model.model.parameters()) / 1e6
                if method == "rddm" else sum(
                    p.numel() for p in model.g_ecg.parameters()
                ) / 1e6,
                "inference_time_s": elapsed,
                "rtf": elapsed / (len(ds) * ds.signal_length / ds.fs),
                "device": str(device),
                "sampling_steps": steps if method == "rddm" else None,
            },
        }
        with (out_dir / f"eval_{split}.json").open("w", encoding="utf-8") as f:
            json.dump(result, f, ensure_ascii=False, indent=2, default=float)
        np.savez_compressed(
            out_dir / f"pred_{split}.npz",
            prediction=pred.astype(np.float32),
            target=target.astype(np.float32),
            ppg=ds._x.astype(np.float32),
            subject_id=subject_ids,
        )
        macro = result["model"]["waveform"]["macro"]
        phys = result["model"]["physiology"]
        print(
            f"[{method}/{split}] RMSE={macro['rmse/macro']['mean']:.4f} "
            f"PCC={macro['pcc/macro']['mean']:.4f} "
            f"HRerr={phys['hr_err_bpm']['mean']:.3f} "
            f"R-F1={phys['rpeak_f1']['mean']:.4f}"
        )


def main():
    args = parse_args()
    cfg = build_config(args)
    set_seed(int(cfg.get("seed", 42)))
    device = resolve_device(str(cfg.get("device", "auto")))
    data_cfg = cfg.get("data", {})
    data_root = Path(data_cfg.get("root", ""))
    if not data_root.is_absolute():
        data_root = PROJECT_ROOT / data_root
    if not data_root.exists():
        raise FileNotFoundError(f"Data root does not exist: {data_root}")
    cfg.setdefault("data", {})["root"] = str(data_root)
    loaders = create_dataloaders(
        dataset=data_cfg.get("dataset", "senssmarttech"),
        root=data_root,
        batch_size=int(data_cfg.get("batch_size", 8)),
        num_workers=int(data_cfg.get("num_workers", 0)),
        ppg_channel=data_cfg.get("ppg_channel"),
        ecg_lead=data_cfg.get("ecg_lead"),
    )
    if "train" not in loaders or "val" not in loaders or "test" not in loaders:
        raise RuntimeError("train/val/test splits are required for a comparison run")

    method = args.method
    suffix = f"_seed{int(cfg.get('seed', 42))}"
    default_name = f"senssmarttech_{method}_repro_20ep{suffix}"
    name = str(cfg.get("experiment", {}).get("name") or default_name)
    if args.output is not None:
        out_dir = args.output
    else:
        out_dir = PROJECT_ROOT / "runs" / name
    out_dir.mkdir(parents=True, exist_ok=True)
    cfg["output_dir"] = str(out_dir)
    cfg["experiment"] = {**cfg.get("experiment", {}), "name": name}
    save_config(cfg, out_dir / "config.yaml")
    print(f"Method={method}, device={device}, output={out_dir}")
    print(
        f"Data train/val/test={len(loaders['train'].dataset)}/"
        f"{len(loaders['val'].dataset)}/{len(loaders['test'].dataset)}; "
        f"channels={loaders['train'].dataset.num_ppg_channels}->"
        f"{loaders['train'].dataset.ecg_leads}; "
        f"fs={loaders['train'].dataset.fs}; T={loaders['train'].dataset.signal_length}"
    )

    if method == "cardiogan":
        trainer = CardioGANTrainer(cfg, loaders, device, out_dir)
    else:
        trainer = RDDMTrainer(cfg, loaders, device, out_dir)

    epochs = int(cfg.get("training", {}).get("epochs", 20))
    save_every = int(cfg.get("training", {}).get("save_every", 5))
    for epoch in range(1, epochs + 1):
        train_metrics = trainer.train_epoch(loaders["train"])
        val_metric = trainer.validation_l1(loaders["val"]) if method == "cardiogan" else trainer.validation_loss(loaders["val"])
        row = {"epoch": epoch, "val_metric": val_metric, **train_metrics}
        trainer.history.append(row)
        print(f"epoch={epoch:03d} val={val_metric:.6f} " + " ".join(
            f"{k}={v:.5f}" for k, v in train_metrics.items()
        ))
        if val_metric < trainer.best_val:
            trainer.best_val = val_metric
            trainer.save(out_dir / "best.pth", epoch)
        if epoch % save_every == 0 or epoch == epochs:
            trainer.save(out_dir / f"epoch_{epoch:03d}.pth", epoch)
        with (out_dir / "training_history.json").open("w", encoding="utf-8") as f:
            json.dump(trainer.history, f, ensure_ascii=False, indent=2, default=float)

    trainer.load(out_dir / "best.pth")
    evaluate_method(method, trainer, loaders, out_dir, device, cfg, int(cfg.get("seed", 42)))
    print(f"Finished: {out_dir}")


if __name__ == "__main__":
    main()
