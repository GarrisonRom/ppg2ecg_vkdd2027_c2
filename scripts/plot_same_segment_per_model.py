#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""Draw one high-resolution Lead-II figure per model on shared segments.

The target window and short crop are selected once per protocol group, then
reused unchanged for every model in that group. Each output figure contains
only one model, with train/test rows and true/generated traces.
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
from scipy.signal import find_peaks

PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT))

from scripts.evaluate import load_run, predict  # noqa: E402


MAIN_DEFAULTS = [
    ("runs/senssmarttech_vae_flow_adv_irm_20ep_seed42", "v0.2 VAE+Flow+GRL"),
    ("runs/senssmarttech_v052_multiband_frozen_cycle_20ep_seed42", "v0.52 Multi-band"),
    ("runs/senssmarttech_v061_vae_multiband_transfer_latent128_20ep_seed42", "v0.61 Latent-128"),
    ("runs/senssmarttech_v064_vae_multiband_latent256_transfer_20ep_seed42", "v0.64 Latent-256"),
]
PAPER_DEFAULTS = [
    ("paper_repro/runs/senssmarttech_1to1_128hz_seed42/cardiogan", "CardioGAN"),
    ("paper_repro/runs/senssmarttech_1to1_128hz_seed42/rddm", "RDDM"),
    ("paper_repro/runs/senssmarttech_recent_1to1_128hz_seed42/qrs_transattn", "QRS-TransAttn"),
    ("paper_repro/runs/senssmarttech_recent_1to1_128hz_seed42/p2e_wgan", "P2E-WGAN"),
    ("paper_repro/runs/senssmarttech_recent_1to1_128hz_seed42/li2024_lightweight", "Li 2024 lightweight"),
]
COLORS = {
    "v0.2 VAE+Flow+GRL": "#3568A8",
    "v0.52 Multi-band": "#D95F02",
    "v0.61 Latent-128": "#2A9D8F",
    "v0.64 Latent-256": "#7B2CBF",
    "CardioGAN": "#3568A8",
    "RDDM": "#D95F02",
    "QRS-TransAttn": "#2A9D8F",
    "P2E-WGAN": "#7B2CBF",
    "Li 2024 lightweight": "#C77DFF",
}
TRUE_COLOR = "#222222"

plt.rcParams.update({
    "font.family": "sans-serif",
    "font.sans-serif": ["Arial", "Helvetica", "DejaVu Sans"],
    "font.size": 11,
    "axes.labelsize": 12,
    "axes.titlesize": 13,
    "xtick.labelsize": 10,
    "ytick.labelsize": 10,
    "legend.fontsize": 10,
    "figure.titlesize": 16,
    "axes.linewidth": 0.9,
    "grid.linewidth": 0.55,
    "lines.linewidth": 2.0,
    "pdf.fonttype": 42,
    "ps.fonttype": 42,
    "svg.fonttype": "none",
    "axes.spines.top": False,
    "axes.spines.right": False,
    "savefig.facecolor": "white",
})


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--group", choices=["main", "paper", "both"], default="both")
    parser.add_argument("--output", type=Path, default=PROJECT_ROOT / "results/same_segment_per_model")
    parser.add_argument("--checkpoint", default="best.pth")
    parser.add_argument("--segment-seconds", type=float, default=2.5)
    parser.add_argument("--batch-size", type=int, default=64)
    parser.add_argument("--dpi", type=int, default=600)
    parser.add_argument("--index", type=int, default=None,
                        help="fixed shared window index; default uses median mean MSE")
    return parser.parse_args()


def absolute(path: str | Path) -> Path:
    value = Path(path)
    return value if value.is_absolute() else PROJECT_ROOT / value


def peak_center(signal: np.ndarray, fs: float) -> int:
    peaks, props = find_peaks(signal, distance=max(1, int(0.30 * fs)),
                              prominence=max(float(signal.std() * 0.55), 1e-4))
    if len(peaks):
        return int(peaks[int(np.argmax(props["prominences"]))])
    return int(np.argmax(np.abs(signal - np.median(signal))))


def crop(signal: np.ndarray, fs: float, seconds: float) -> tuple[int, int, int]:
    size = min(len(signal), max(8, int(round(seconds * fs))))
    center = peak_center(signal, fs)
    start = max(0, center - size // 2)
    stop = min(len(signal), start + size)
    start = max(0, stop - size)
    return start, stop, center


def choose_shared_index(target: np.ndarray, preds: dict[str, np.ndarray], lead: int,
                        fixed: int | None) -> int:
    if fixed is not None:
        if fixed < 0 or fixed >= len(target):
            raise IndexError(f"index {fixed} outside [0, {len(target)})")
        return fixed
    errors = np.mean([
        np.mean((prediction[:, lead] - target[:, lead]) ** 2, axis=1)
        for prediction in preds.values()
    ], axis=0)
    return int(np.argmin(np.abs(errors - np.median(errors))))


def save_triplet(fig, base: Path, dpi: int) -> list[str]:
    base.parent.mkdir(parents=True, exist_ok=True)
    paths = []
    for suffix, kwargs in ((".png", {"dpi": dpi}), (".pdf", {}), (".svg", {})):
        path = base.with_suffix(suffix)
        fig.savefig(path, bbox_inches="tight", **kwargs)
        paths.append(str(path))
    return paths


def load_predictions(group: str, entries: list[tuple[Path, str]], checkpoint: str,
                     batch_size: int) -> tuple[dict[str, dict[str, np.ndarray]], dict[str, np.ndarray],
                                               dict[str, np.ndarray], float, int]:
    predictions = {"train": {}, "test": {}}
    targets: dict[str, np.ndarray] = {}
    subjects: dict[str, np.ndarray] = {}
    fs: float | None = None
    lead_idx: int | None = None
    for split in ("train", "test"):
        for run_dir, label in entries:
            if group == "paper":
                data = np.load(run_dir / f"pred_{split}.npz")
                prediction, target = data["prediction"], data["target"]
                subject = data["subject_id"]
                current_fs, current_lead = 128.0, 0
            else:
                config, dataset, encoder, decoder, latent_flow, device = load_run(
                    run_dir, checkpoint, split=split,
                )
                ppg_channel = config["data"].get("ppg_channel")
                ppg_idx = dataset.ppg_channels.index(ppg_channel) if ppg_channel else None
                flow_steps = int((config.get("model", {}).get("cardio_align", {}) or {}).get(
                    "integration_steps", 8,
                ))
                prediction, _ = predict(
                    encoder, decoder, dataset._x, ppg_idx, device, fs=dataset.fs,
                    batch_size=batch_size, latent_flow=latent_flow, flow_steps=flow_steps,
                )
                target = dataset._y
                subject = (dataset.metadata["subject_id"].to_numpy()
                           if dataset.metadata is not None else np.arange(len(target)))
                current_fs = float(dataset.fs)
                current_lead = dataset.ecg_channels.index("II")
            predictions[split][label] = prediction
            if split not in targets:
                targets[split], subjects[split] = target, subject
            elif not np.allclose(targets[split], target, atol=1e-6, rtol=1e-6):
                raise ValueError(f"target mismatch in {group}/{split}; use aligned predictions")
            fs = current_fs if fs is None else fs
            lead_idx = current_lead if lead_idx is None else lead_idx
            if fs != current_fs or lead_idx != current_lead:
                raise ValueError("inconsistent sampling rate or Lead-II index")
    return predictions, targets, subjects, float(fs), int(lead_idx)


def plot_model(group: str, label: str, predictions: dict[str, np.ndarray], targets: dict[str, np.ndarray],
               subjects: dict[str, np.ndarray], fs: float, lead: int, indices: dict[str, int],
               output_dir: Path, segment_seconds: float, dpi: int) -> dict:
    fig, axes = plt.subplots(2, 1, figsize=(10.5, 4.6), sharex=False)
    selected = {}
    for row, split in enumerate(("train", "test")):
        index = indices[split]
        target = targets[split][index, lead]
        start, stop, center = crop(target, fs, segment_seconds)
        x = np.arange(start, stop) / fs
        ax = axes[row]
        ax.plot(x - x[0], target[start:stop], color=TRUE_COLOR, lw=2.4, label="True ECG")
        ax.plot(x - x[0], predictions[split][index, lead, start:stop],
                color=COLORS.get(label, "#D95F02"), lw=2.0, label="Generated ECG")
        ax.axvline((center - start) / fs, color="#777777", lw=0.9, ls=":")
        ax.set_title(f"{split.capitalize()} set | same window index={index} | Lead II",
                     loc="left", fontweight="bold")
        ax.set_xlabel("Time within common segment (s)")
        ax.set_ylabel("Normalized amplitude")
        ax.grid(True, alpha=0.24)
        mse = float(np.mean((predictions[split][index, lead] - target) ** 2))
        ax.text(0.995, 0.95, f"subject={subjects[split][index]}  MSE={mse:.3f}",
                transform=ax.transAxes, ha="right", va="top", fontsize=9,
                bbox={"facecolor": "white", "alpha": 0.82, "edgecolor": "#D9D9D9", "pad": 3.0})
        selected[split] = {
            "index": index,
            "subject": int(subjects[split][index]),
            "crop_start_s": float(x[0]),
            "crop_stop_s": float(x[-1]),
            "mse": mse,
        }
    handles, labels = axes[0].get_legend_handles_labels()
    fig.legend(handles, labels, loc="upper center", bbox_to_anchor=(0.5, 0.965),
               ncol=2, frameon=False)
    protocol = "main 4->4 | 250 Hz" if group == "main" else "paper 1->1 | 128 Hz"
    fig.suptitle(f"{label} | Lead II | {protocol}", y=1.02, fontweight="bold")
    fig.tight_layout(rect=(0, 0, 1, 0.88), h_pad=1.7)
    safe = label.lower().replace(" ", "_").replace("+", "plus").replace(".", "")
    outputs = save_triplet(fig, output_dir / f"{group}_{safe}_same_segment_leadII", dpi)
    plt.close(fig)
    return {"label": label, "outputs": outputs, "selected": selected}


def main() -> None:
    args = parse_args()
    if args.segment_seconds <= 0:
        raise ValueError("--segment-seconds must be positive")
    output_dir = args.output if args.output.is_absolute() else PROJECT_ROOT / args.output
    groups = {}
    if args.group in ("main", "both"):
        groups["main"] = [(absolute(path), label) for path, label in MAIN_DEFAULTS]
    if args.group in ("paper", "both"):
        groups["paper"] = [(absolute(path), label) for path, label in PAPER_DEFAULTS]
    manifest = {
        "one_model_per_figure": True,
        "segment_seconds": args.segment_seconds,
        "selection": "shared window per split; centered crop around target R peak",
        "dpi": args.dpi,
        "groups": {},
    }
    for group, entries in groups.items():
        for path, _ in entries:
            if not path.exists():
                raise FileNotFoundError(path)
        predictions, targets, subjects, fs, lead = load_predictions(
            group, entries, args.checkpoint, args.batch_size,
        )
        indices = {
            split: choose_shared_index(targets[split], predictions[split], lead, args.index)
            for split in ("train", "test")
        }
        group_out = output_dir / group
        manifest["groups"][group] = {
            "fs": fs,
            "lead": "II",
            "shared_indices": indices,
            "models": {},
        }
        for _, label in entries:
            model_predictions = {
                split: predictions[split][label]
                for split in ("train", "test")
            }
            result = plot_model(
                group, label, model_predictions,
                targets, subjects, fs, lead, indices, group_out, args.segment_seconds, args.dpi,
            )
            manifest["groups"][group]["models"][label] = result
            print(f"[{group}/{label}] {result['outputs']}")
    output_dir.mkdir(parents=True, exist_ok=True)
    with (output_dir / "manifest.json").open("w", encoding="utf-8") as handle:
        json.dump(manifest, handle, ensure_ascii=False, indent=2, default=float)
    with (output_dir / "STYLE_BRIEF.md").open("w", encoding="utf-8") as handle:
        handle.write(
            "# One-Model-Per-Figure Lead-II Brief\n\n"
            "- Every model has its own figure; train/test use the same target window within a protocol group.\n"
            "- The common crop is 2.5 seconds by default and centered on the target R peak.\n"
            "- True ECG is dark gray; generated ECG is model-specific color.\n"
            "- Main and paper protocols are kept separate.\n"
            "- Exports include 600-DPI PNG, PDF, and SVG.\n"
        )
    print(f"Figures saved to: {output_dir}")


if __name__ == "__main__":
    main()
