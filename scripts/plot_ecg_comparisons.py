#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""Plot true versus generated ECG for train/test activity groups.

The script creates two matched figures:

* ``train_ecg_true_vs_generated``: training windows, split into A/B activity;
* ``test_ecg_true_vs_generated``: held-out test windows, split into A/B activity.

For each split/state, the plotted window is the sample whose per-window MSE is
closest to that group's median. This gives a deterministic, representative
qualitative example without selecting the best-looking waveform.
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import torch

PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

from src.data import create_dataset
from src.models import build_decoder, build_encoder


TRUE_COLOR = "#4C72B0"
PRED_COLOR = "#DD8452"
ACTIVITY_ORDER = ("B", "A")
ACTIVITY_LABELS = {
    "A": "A: after activity / exercise",
    "B": "B: before activity / rest",
}

plt.rcParams.update({
    "font.family": "sans-serif",
    "font.sans-serif": ["Arial", "Helvetica", "DejaVu Sans"],
    "font.size": 9,
    "axes.labelsize": 10,
    "axes.titlesize": 11,
    "legend.fontsize": 8.5,
    "figure.titlesize": 13,
    "axes.linewidth": 0.9,
    "grid.linewidth": 0.5,
    "lines.linewidth": 1.8,
    "pdf.fonttype": 42,
    "ps.fonttype": 42,
    "svg.fonttype": "none",
    "axes.spines.top": False,
    "axes.spines.right": False,
    "savefig.facecolor": "white",
})


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--run", type=Path, required=True,
                        help="run directory containing config.yaml and checkpoint")
    parser.add_argument("--checkpoint", default="best.pth")
    parser.add_argument("--output", type=Path, default=None,
                        help="output directory (default: <run>/figures)")
    parser.add_argument("--batch-size", type=int, default=64)
    return parser.parse_args()


def _resolve_root(path_value: str | Path) -> Path:
    path = Path(path_value)
    return path if path.is_absolute() else PROJECT_ROOT / path


def load_model(run_dir: Path, checkpoint_name: str):
    config_path = run_dir / "config.yaml"
    if not config_path.exists():
        raise FileNotFoundError(f"Missing run config: {config_path}")

    import yaml

    with config_path.open("r", encoding="utf-8") as handle:
        config = yaml.safe_load(handle) or {}
    data_cfg = config["data"]
    data_root = _resolve_root(data_cfg["root"])
    reference_ds = create_dataset(data_cfg["dataset"], data_root, split="test")

    configured_ppg = data_cfg.get("ppg_channel")
    ppg_channels = 1 if configured_ppg else reference_ds.num_ppg_channels
    model_cfg = config["model"]
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    encoder = build_encoder(
        model_cfg["encoder"],
        signal_length=reference_ds.signal_length,
        latent_dim=model_cfg.get("latent_dim", 128),
        ppg_channels=ppg_channels,
    ).to(device)
    decoder = build_decoder(
        model_cfg["decoder"],
        signal_length=reference_ds.signal_length,
        latent_dim=model_cfg.get("latent_dim", 128),
        ecg_leads=reference_ds.ecg_leads,
    ).to(device)

    checkpoint_path = run_dir / checkpoint_name
    checkpoint = torch.load(checkpoint_path, map_location=device, weights_only=False)
    encoder.load_state_dict(checkpoint["encoder"])
    decoder.load_state_dict(checkpoint["decoder"])
    encoder.eval()
    decoder.eval()
    return config, data_root, reference_ds, encoder, decoder, device, checkpoint_path


@torch.no_grad()
def predict_dataset(encoder, decoder, dataset, ppg_channel: str | None,
                    device: torch.device, batch_size: int) -> np.ndarray:
    ppg = dataset._x
    if ppg_channel:
        if ppg_channel not in dataset.ppg_channels:
            raise ValueError(f"Unknown PPG channel {ppg_channel!r}")
        index = dataset.ppg_channels.index(ppg_channel)
        ppg = ppg[:, index:index + 1]

    predictions = []
    for start in range(0, len(ppg), batch_size):
        batch = torch.from_numpy(ppg[start:start + batch_size]).float().to(device)
        predictions.append(decoder(encoder(batch)).cpu().numpy())
    return np.concatenate(predictions, axis=0)


def choose_representative(pred: np.ndarray, target: np.ndarray,
                          activities: np.ndarray, state: str) -> int | None:
    indices = np.flatnonzero(activities == state)
    if indices.size == 0:
        return None
    errors = ((pred[indices] - target[indices]) ** 2).mean(axis=(1, 2))
    median_error = float(np.median(errors))
    return int(indices[np.argmin(np.abs(errors - median_error))])


def _metadata_value(metadata, index: int, column: str, fallback: str = "") -> str:
    if metadata is None or column not in metadata.columns:
        return fallback
    return str(metadata.iloc[index][column])


def plot_split(split: str, pred: np.ndarray, target: np.ndarray,
               dataset, fs: int, checkpoint_name: str, output_dir: Path) -> dict:
    metadata = dataset.metadata
    activities = (
        metadata["activity"].astype(str).to_numpy()
        if metadata is not None and "activity" in metadata.columns
        else np.full(len(dataset), "unknown", dtype=object)
    )
    lead_names = list(dataset.ecg_channels)
    time = np.arange(dataset.signal_length) / float(fs)
    selected: dict[str, dict] = {}

    fig, axes = plt.subplots(
        len(ACTIVITY_ORDER), len(lead_names),
        figsize=(4.4 * len(lead_names), 5.8),
        sharex=True,
        squeeze=False,
    )
    handles = None
    for row, state in enumerate(ACTIVITY_ORDER):
        index = choose_representative(pred, target, activities, state)
        if index is None:
            for axis in axes[row]:
                axis.set_visible(False)
            continue

        group_errors = ((pred[activities == state] - target[activities == state]) ** 2).mean(axis=(1, 2))
        selected[state] = {
            "index": index,
            "activity": state,
            "activity_label": ACTIVITY_LABELS.get(state, state),
            "subject_id": _metadata_value(metadata, index, "subject_id"),
            "record_id": _metadata_value(metadata, index, "record_id"),
            "start_sec": _metadata_value(metadata, index, "start_sec"),
            "window_mse": float(((pred[index] - target[index]) ** 2).mean()),
            "group_median_mse": float(np.median(group_errors)),
        }

        for col, lead_name in enumerate(lead_names):
            axis = axes[row, col]
            true_line, = axis.plot(
                time, target[index, col], color=TRUE_COLOR, linestyle="-",
                linewidth=1.25, label="True ECG",
            )
            pred_line, = axis.plot(
                time, pred[index, col], color=PRED_COLOR, linestyle="--",
                linewidth=1.15, alpha=0.95, label="Generated ECG",
            )
            if handles is None:
                handles = (true_line, pred_line)
            axis.set_title(lead_name)
            axis.grid(True, color="#D9D9D9", linewidth=0.45, alpha=0.65)
            axis.spines["top"].set_visible(False)
            axis.spines["right"].set_visible(False)
            if col == 0:
                axis.set_ylabel(ACTIVITY_LABELS.get(state, state), fontsize=10)
            if row == len(ACTIVITY_ORDER) - 1:
                axis.set_xlabel("Time (s)")

        axes[row, 0].text(
            0.01, 0.98,
            f"subject={selected[state]['subject_id']}  "
            f"start={selected[state]['start_sec']} s  "
            f"MSE={selected[state]['window_mse']:.3f}",
            transform=axes[row, 0].transAxes,
            va="top", ha="left", fontsize=8.5,
            bbox={"facecolor": "white", "alpha": 0.78, "edgecolor": "none", "pad": 2.5},
        )

    fig.suptitle(
        f"SensSmartTech {split} ECG reconstruction | {checkpoint_name}\n"
        "Representative windows selected by group-median MSE",
        y=1.02,
    )
    if handles is not None:
        fig.legend(handles, ["True ECG", "Generated ECG"],
                   loc="upper center", bbox_to_anchor=(0.5, 0.965),
                   ncol=2, frameon=False)
    fig.tight_layout(rect=(0, 0, 1, 0.91))

    output_dir.mkdir(parents=True, exist_ok=True)
    base_name = f"{split.lower()}_ecg_true_vs_generated"
    for extension, kwargs in (("png", {"dpi": 300}), ("pdf", {}), ("svg", {})):
        fig.savefig(output_dir / f"{base_name}.{extension}",
                    bbox_inches="tight", **kwargs)
    plt.close(fig)
    return selected


def main() -> None:
    args = parse_args()
    run_dir = args.run if args.run.is_absolute() else PROJECT_ROOT / args.run
    output_dir = args.output or run_dir / "figures"
    output_dir = output_dir if output_dir.is_absolute() else PROJECT_ROOT / output_dir

    config, data_root, reference_ds, encoder, decoder, device, checkpoint_path = load_model(
        run_dir, args.checkpoint,
    )
    ppg_channel = config["data"].get("ppg_channel")
    manifest = {
        "run": str(run_dir),
        "checkpoint": str(checkpoint_path),
        "dataset": config["data"]["dataset"],
        "seed": config.get("seed"),
        "data_root": str(data_root),
        "device": str(device),
        "fs_hz": int(reference_ds.fs),
        "activity_labels": ACTIVITY_LABELS,
        "figures": {},
    }

    for split in ("train", "test"):
        dataset = create_dataset(config["data"]["dataset"], data_root, split=split)
        pred = predict_dataset(encoder, decoder, dataset, ppg_channel, device, args.batch_size)
        selected = plot_split(
            split, pred, dataset._y, dataset, dataset.fs,
            args.checkpoint, output_dir,
        )
        manifest["figures"][split] = selected
        print(f"[{split}] windows={len(dataset)} selected={selected}")

    with (output_dir / "representatives.json").open("w", encoding="utf-8") as handle:
        json.dump(manifest, handle, ensure_ascii=False, indent=2)
    print(f"Figures saved to: {output_dir}")


if __name__ == "__main__":
    main()
