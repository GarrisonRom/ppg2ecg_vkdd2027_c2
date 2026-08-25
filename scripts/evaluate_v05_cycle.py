#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""Evaluate the bidirectional v0.5 ECG<->PPG cycle model.

This reports the auxiliary reverse path separately from the deployable
PPG->ECG metrics in ``scripts/evaluate.py``:

    direct: ECG -> PPG
    cycle:  PPG -> ECG_hat -> PPG

The activity label is used only for post-hoc stratification.
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np
import torch

PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

from src.data import create_dataset
from src.models import build_decoder, build_encoder
from src.utils.config import load_config


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--run", type=Path, required=True)
    parser.add_argument("--checkpoint", default="best.pth")
    parser.add_argument("--split", choices=("train", "val", "test"), default="test")
    parser.add_argument("--batch-size", type=int, default=32)
    return parser.parse_args()


def resolve_path(path_value: str | Path) -> Path:
    path = Path(path_value)
    return path if path.is_absolute() else PROJECT_ROOT / path


def build_cycle_models(config: dict, dataset, device: torch.device):
    model_cfg = config["model"]
    cycle_cfg = model_cfg.get("cycle_consistency", {}) or {}
    if not cycle_cfg.get("enabled", False):
        raise ValueError("The run configuration does not enable cycle_consistency")

    latent_dim = int(model_cfg.get("latent_dim", 128))
    ppg_channels = dataset.num_ppg_channels
    main_encoder = build_encoder(
        model_cfg["encoder"], dataset.signal_length, latent_dim,
        ppg_channels=ppg_channels,
    ).to(device)
    main_decoder = build_decoder(
        model_cfg["decoder"], dataset.signal_length, latent_dim,
        ecg_leads=dataset.ecg_leads,
    ).to(device)
    reverse_encoder = build_encoder(
        cycle_cfg.get("reverse_encoder") or model_cfg["encoder"],
        dataset.signal_length, latent_dim,
        ppg_channels=dataset.ecg_leads,
    ).to(device)
    reverse_decoder = build_decoder(
        cycle_cfg.get("reverse_decoder") or model_cfg["decoder"],
        dataset.signal_length, latent_dim,
        ecg_leads=ppg_channels,
    ).to(device)
    return main_encoder, main_decoder, reverse_encoder, reverse_decoder


def subject_summary(values: np.ndarray, subjects: np.ndarray) -> dict[str, float | int]:
    per_subject = [values[subjects == sid].mean() for sid in np.unique(subjects)]
    return {
        "mean": float(np.mean(per_subject)),
        "std": float(np.std(per_subject, ddof=1)) if len(per_subject) > 1 else 0.0,
        "n_subjects": int(len(per_subject)),
    }


def summarize_pair(pred: np.ndarray, target: np.ndarray, subjects: np.ndarray) -> dict:
    error = pred - target
    l1 = np.mean(np.abs(error), axis=(1, 2))
    mse = np.mean(error ** 2, axis=(1, 2))
    return {
        "l1": subject_summary(l1, subjects),
        "mse": subject_summary(mse, subjects),
    }


@torch.no_grad()
def predict(models, dataset, device: torch.device, batch_size: int):
    main_encoder, main_decoder, reverse_encoder, reverse_decoder = models
    ppg = torch.from_numpy(dataset._x).float()
    ecg = torch.from_numpy(dataset._y).float()
    generated_ecg = []
    direct_ppg = []
    cycle_ppg = []
    for start in range(0, len(dataset), batch_size):
        ppg_batch = ppg[start:start + batch_size].to(device)
        ecg_batch = ecg[start:start + batch_size].to(device)
        ecg_hat = main_decoder(main_encoder(ppg_batch))
        direct_hat = reverse_decoder(reverse_encoder(ecg_batch))
        cycle_hat = reverse_decoder(reverse_encoder(ecg_hat))
        generated_ecg.append(ecg_hat.cpu().numpy())
        direct_ppg.append(direct_hat.cpu().numpy())
        cycle_ppg.append(cycle_hat.cpu().numpy())
    return (
        np.concatenate(generated_ecg),
        np.concatenate(direct_ppg),
        np.concatenate(cycle_ppg),
    )


def main() -> None:
    args = parse_args()
    run_dir = resolve_path(args.run)
    config = load_config(run_dir / "config.yaml")
    data_root = resolve_path(config["data"]["root"])
    dataset = create_dataset(config["data"]["dataset"], data_root, split=args.split)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    models = build_cycle_models(config, dataset, device)

    checkpoint_path = run_dir / args.checkpoint
    checkpoint = torch.load(checkpoint_path, map_location=device, weights_only=False)
    for model, name in zip(
        models,
        ("encoder", "decoder", "reverse_encoder", "reverse_decoder"),
    ):
        if name not in checkpoint:
            raise KeyError(f"{checkpoint_path} does not contain {name} weights")
        model.load_state_dict(checkpoint[name])
        model.eval()

    ecg_hat, ppg_direct, ppg_cycle = predict(models, dataset, device, args.batch_size)
    subjects = dataset.metadata["subject_id"].to_numpy()
    activities = dataset.metadata["activity"].astype(str).to_numpy()
    result = {
        "run": str(run_dir),
        "checkpoint": args.checkpoint,
        "split": args.split,
        "device": str(device),
        "direct_ecg_to_ppg": summarize_pair(ppg_direct, dataset._x, subjects),
        "cycle_ppg_to_ecg_to_ppg": summarize_pair(ppg_cycle, dataset._x, subjects),
        "activity": {},
    }
    result["cycle_improvement_l1"] = float(
        result["direct_ecg_to_ppg"]["l1"]["mean"]
        - result["cycle_ppg_to_ecg_to_ppg"]["l1"]["mean"]
    )
    for activity in sorted(np.unique(activities)):
        mask = activities == activity
        result["activity"][activity] = {
            "n_windows": int(mask.sum()),
            "direct_ecg_to_ppg": summarize_pair(ppg_direct[mask], dataset._x[mask], subjects[mask]),
            "cycle_ppg_to_ecg_to_ppg": summarize_pair(ppg_cycle[mask], dataset._x[mask], subjects[mask]),
        }

    output_path = run_dir / f"cycle_eval_{args.split}.json"
    output_path.write_text(
        json.dumps(result, ensure_ascii=False, indent=2), encoding="utf-8",
    )
    print(json.dumps(result, ensure_ascii=False, indent=2))
    print(f"Saved cycle evaluation to {output_path}")


if __name__ == "__main__":
    main()
