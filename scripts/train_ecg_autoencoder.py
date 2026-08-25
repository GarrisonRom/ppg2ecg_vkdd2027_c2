#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""Train and evaluate an ECG->ECG autoencoder diagnostic.

The diagnostic intentionally bypasses PPG, VAE, Flow, GRL, IRM, and activity
labels.  It answers one engineering question: can the target-side temporal
encoder/decoder reproduce sharp ECG complexes on the same preprocessed target
distribution?
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import numpy as np
import torch
import torch.nn.functional as F
from tqdm import tqdm

PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

from src.data import create_dataloaders, create_dataset
from src.evaluation.metrics import evaluate_all
from src.models import build_decoder, build_encoder
from src.utils import set_seed
from src.utils.config import load_config, save_config
from scripts.plot_ecg_comparisons import plot_split


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=Path, required=True)
    parser.add_argument("--epochs", type=int, default=None)
    parser.add_argument("--device", choices=("auto", "cuda", "cpu"), default=None)
    return parser.parse_args()


def resolve_path(path_value: str | Path) -> Path:
    path = Path(path_value)
    return path if path.is_absolute() else PROJECT_ROOT / path


def build_models(config: dict, dataset, device: torch.device):
    model_cfg = config["model"]
    encoder = build_encoder(
        model_cfg["encoder"],
        signal_length=dataset.signal_length,
        latent_dim=int(model_cfg.get("latent_dim", 128)),
        ppg_channels=dataset.ecg_leads,
    ).to(device)
    decoder = build_decoder(
        model_cfg["decoder"],
        signal_length=dataset.signal_length,
        latent_dim=int(model_cfg.get("latent_dim", 128)),
        ecg_leads=dataset.ecg_leads,
    ).to(device)
    return encoder, decoder


def run_epoch(
    encoder,
    decoder,
    loader,
    device: torch.device,
    optimizer=None,
    max_grad_norm: float = 1.0,
) -> float:
    training = optimizer is not None
    encoder.train(training)
    decoder.train(training)
    total = 0.0
    count = 0

    iterator = tqdm(loader, desc="Train" if training else "Validation", leave=False)
    for batch in iterator:
        ecg = batch["ecg"].to(device, non_blocking=True)
        if training:
            optimizer.zero_grad(set_to_none=True)
        encoded = encoder(ecg)
        pred = decoder(encoded)
        loss = F.l1_loss(pred, ecg)
        if training:
            loss.backward()
            torch.nn.utils.clip_grad_norm_(
                list(encoder.parameters()) + list(decoder.parameters()),
                max_grad_norm,
            )
            optimizer.step()
        total += float(loss.detach().item()) * ecg.size(0)
        count += ecg.size(0)
        iterator.set_postfix(loss=f"{loss.item():.4f}")
    return total / max(count, 1)


@torch.no_grad()
def predict(encoder, decoder, dataset, device: torch.device, batch_size: int) -> np.ndarray:
    encoder.eval()
    decoder.eval()
    preds = []
    for start in range(0, len(dataset), batch_size):
        ecg = torch.from_numpy(dataset._y[start:start + batch_size]).float().to(device)
        preds.append(decoder(encoder(ecg)).cpu().numpy())
    return np.concatenate(preds, axis=0)


def evaluate_split(config: dict, dataset, pred: np.ndarray, run_dir: Path, split: str):
    subject_ids = dataset.metadata["subject_id"].to_numpy()
    result = {
        "run": str(run_dir),
        "split": split,
        "input_modality": "ecg",
        "model": evaluate_all(
            pred,
            dataset._y,
            dataset.fs,
            dataset.ecg_channels,
            subject_ids,
            ppg=dataset._x,
            ppg_names=dataset.ppg_channels,
            distribution=False,
        ),
    }
    with (run_dir / f"eval_{split}.json").open("w", encoding="utf-8") as handle:
        json.dump(result, handle, ensure_ascii=False, indent=2, default=float)
    return result


def main() -> None:
    args = parse_args()
    config = load_config(args.config)
    config["data"]["root"] = str(resolve_path(config["data"]["root"]))
    if args.epochs is not None:
        config.setdefault("training", {})["epochs"] = args.epochs
    if args.device is not None:
        config["device"] = args.device

    set_seed(int(config.get("seed", 42)))
    device_name = config.get("device", "auto")
    device = torch.device(
        "cuda" if device_name == "auto" and torch.cuda.is_available()
        else "cpu" if device_name == "auto" else device_name
    )
    data_cfg = config["data"]
    loaders = create_dataloaders(
        data_cfg["dataset"],
        data_cfg["root"],
        batch_size=int(data_cfg.get("batch_size", 32)),
        num_workers=int(data_cfg.get("num_workers", 4)),
    )
    train_ds = loaders["train"].dataset
    encoder, decoder = build_models(config, train_ds, device)

    run_dir = PROJECT_ROOT / "runs" / config["experiment"]["name"]
    run_dir.mkdir(parents=True, exist_ok=True)
    config["output_dir"] = str(run_dir)
    save_config(config, run_dir / "config.yaml")

    optimizer = torch.optim.AdamW(
        list(encoder.parameters()) + list(decoder.parameters()),
        lr=float(config["training"].get("lr", 1e-3)),
        weight_decay=float(config["training"].get("weight_decay", 1e-5)),
    )
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
        optimizer,
        T_max=int(config["training"].get("epochs", 20)),
        eta_min=float(config["training"].get("min_lr", 1e-5)),
    )

    history = []
    best_val = float("inf")
    best_epoch = 0
    epochs = int(config["training"].get("epochs", 20))
    print(f"Training ECG autoencoder on {device}; windows={len(train_ds)}")
    for epoch in range(epochs):
        train_loss = run_epoch(
            encoder,
            decoder,
            loaders["train"],
            device,
            optimizer=optimizer,
            max_grad_norm=float(config["training"].get("max_grad_norm", 1.0)),
        )
        with torch.no_grad():
            val_loss = run_epoch(encoder, decoder, loaders["val"], device)
        scheduler.step()
        record = {
            "epoch": epoch + 1,
            "train_l1": train_loss,
            "val_l1": val_loss,
            "lr": optimizer.param_groups[0]["lr"],
        }
        history.append(record)
        (run_dir / "training_history.json").write_text(
            json.dumps(history, ensure_ascii=False, indent=2), encoding="utf-8",
        )
        print(f"Epoch {epoch + 1}/{epochs}: train_l1={train_loss:.5f} val_l1={val_loss:.5f}")

        state = {
            "epoch": epoch + 1,
            "encoder": encoder.state_dict(),
            "decoder": decoder.state_dict(),
            "optimizer": optimizer.state_dict(),
            "scheduler": scheduler.state_dict(),
            "config": config,
            "history": history,
            "input_modality": "ecg",
        }
        if val_loss < best_val:
            best_val = val_loss
            best_epoch = epoch + 1
            torch.save(state, run_dir / "best.pth")
        if (epoch + 1) % int(config["training"].get("save_every", 5)) == 0:
            torch.save(state, run_dir / f"epoch_{epoch + 1}.pth")

    torch.save(state, run_dir / "final.pth")
    with (run_dir / "summary.json").open("w", encoding="utf-8") as handle:
        json.dump({"best_epoch": best_epoch, "best_val_l1": best_val}, handle, indent=2)

    checkpoint = torch.load(run_dir / "best.pth", map_location=device, weights_only=False)
    encoder.load_state_dict(checkpoint["encoder"])
    decoder.load_state_dict(checkpoint["decoder"])
    encoder.eval()
    decoder.eval()

    for split in ("train", "test"):
        dataset = create_dataset(data_cfg["dataset"], data_cfg["root"], split=split)
        pred = predict(encoder, decoder, dataset, device, int(data_cfg.get("batch_size", 32)))
        result = evaluate_split(config, dataset, pred, run_dir, split)
        macro = result["model"]["waveform"]["macro"]
        phys = result["model"]["physiology"]
        print(
            f"[{split}] rmse={macro['rmse/macro']['mean']:.4f} "
            f"pcc={macro['pcc/macro']['mean']:.4f} "
            f"rpeak_f1={phys['rpeak_f1']['mean']:.4f}"
        )
        plot_split(
            split, pred, dataset._y, dataset, dataset.fs, "best.pth", run_dir / "figures",
        )
        plot_split(
            split, pred, dataset._y, dataset, dataset.fs, "best.pth", run_dir / "figures",
            detail_seconds=2.5, suffix="_detail",
        )
    print(f"Saved diagnostic run to {run_dir}; best_epoch={best_epoch}")


if __name__ == "__main__":
    main()
