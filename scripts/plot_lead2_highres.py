#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""Render paper-style single-lead Lead II ECG comparison figures.

For each requested run and split, the script creates one figure containing
only Lead II.  Rest (B) and after-activity (A) are shown in separate rows;
each row has the full window and a data-derived QRS zoom.  The representative
window is selected by the median Lead-II MSE within its activity group, so the
plot is reproducible and is not hand-picked for appearance.
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
from scipy.signal import find_peaks

PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT))

from scripts.evaluate import load_run, predict  # noqa: E402
from src.data import create_dataset  # noqa: E402


TRUE_COLOR = "#3568A8"
GENERATED_COLOR = "#D96B27"
ACTIVITY_ORDER = ("B", "A")
ACTIVITY_LABELS = {
    "A": "A: after activity",
    "B": "B: rest",
}

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
    parser.add_argument(
        "--run", action="append", required=True,
        help="run directory; repeat for multiple models",
    )
    parser.add_argument("--checkpoint", default="best.pth")
    parser.add_argument("--output", type=Path, default=None)
    parser.add_argument("--batch-size", type=int, default=64)
    parser.add_argument("--zoom-seconds", type=float, default=1.6)
    parser.add_argument("--dpi", type=int, default=600)
    return parser.parse_args()


def run_name(run_dir: Path) -> str:
    name = run_dir.name
    replacements = {
        "senssmarttech_vae_flow_adv_irm_20ep_seed42": "v0.2 VAE+Flow+GRL",
        "senssmarttech_v052_multiband_frozen_cycle_20ep_seed42": "v0.52 Multi-band",
        "senssmarttech_v064_vae_multiband_latent256_transfer_20ep_seed42": "v0.64 Latent-256",
    }
    return replacements.get(name, name)


def lead_ii_index(dataset) -> int:
    for candidate in ("II", "lead_II", "ii"):
        if candidate in dataset.ecg_channels:
            return dataset.ecg_channels.index(candidate)
    raise ValueError(f"Lead II not found in ECG channels: {dataset.ecg_channels}")


def representative_indices(pred: np.ndarray, target: np.ndarray, dataset, lead_idx: int) -> dict[str, int]:
    metadata = dataset.metadata
    if metadata is None or "activity" not in metadata.columns:
        errors = ((pred[:, lead_idx] - target[:, lead_idx]) ** 2).mean(axis=1)
        median = float(np.median(errors))
        return {"all": int(np.argmin(np.abs(errors - median)))}
    activities = metadata["activity"].astype(str).to_numpy()
    selected: dict[str, int] = {}
    for state in ACTIVITY_ORDER:
        indices = np.flatnonzero(activities == state)
        if not len(indices):
            continue
        errors = ((pred[indices, lead_idx] - target[indices, lead_idx]) ** 2).mean(axis=1)
        median = float(np.median(errors))
        selected[state] = int(indices[np.argmin(np.abs(errors - median))])
    return selected


def qrs_center(signal: np.ndarray, fs: float) -> int:
    distance = max(1, int(0.30 * fs))
    prominence = max(float(np.std(signal) * 0.55), 1e-4)
    peaks, props = find_peaks(signal, distance=distance, prominence=prominence)
    if len(peaks):
        prominences = props.get("prominences", np.ones(len(peaks)))
        return int(peaks[int(np.argmax(prominences))])
    return int(np.argmax(np.abs(signal - np.median(signal))))


def metadata_text(dataset, index: int) -> str:
    if dataset.metadata is None:
        return ""
    row = dataset.metadata.iloc[index]
    parts = []
    for key in ("subject_id", "record_id", "start_sec"):
        if key in row.index:
            parts.append(f"{key.replace('_', ' ')}={row[key]}")
    return "  ".join(parts)


def save_triplet(fig, output_base: Path, dpi: int) -> list[str]:
    output_base.parent.mkdir(parents=True, exist_ok=True)
    outputs = []
    for suffix, kwargs in ((".png", {"dpi": dpi}), (".pdf", {}), (".svg", {})):
        path = output_base.with_suffix(suffix)
        fig.savefig(path, bbox_inches="tight", **kwargs)
        outputs.append(str(path))
    return outputs


def plot_one_split(
    model_label: str,
    split: str,
    pred: np.ndarray,
    target: np.ndarray,
    dataset,
    output_dir: Path,
    checkpoint: str,
    zoom_seconds: float,
    dpi: int,
) -> dict:
    fs = float(dataset.fs)
    lead_idx = lead_ii_index(dataset)
    selected = representative_indices(pred, target, dataset, lead_idx)
    time = np.arange(target.shape[-1]) / fs
    zoom_samples = max(8, int(round(zoom_seconds * fs)))
    has_groups = any(state in selected for state in ACTIVITY_ORDER)
    rows = [state for state in ACTIVITY_ORDER if state in selected] if has_groups else ["all"]
    fig, axes = plt.subplots(
        len(rows), 2, figsize=(15.5, 4.0 * len(rows)),
        squeeze=False, gridspec_kw={"width_ratios": [2.4, 1.0]},
    )
    handles = None
    manifest_rows = []
    for row_idx, state in enumerate(rows):
        index = selected[state]
        true = target[index, lead_idx]
        generated = pred[index, lead_idx]
        center = qrs_center(true, fs)
        start = max(0, center - zoom_samples // 2)
        stop = min(len(time), start + zoom_samples)
        start = max(0, stop - zoom_samples)
        label = ACTIVITY_LABELS.get(state, "representative")
        row_info = {
            "state": state,
            "index": index,
            "metadata": metadata_text(dataset, index),
            "lead": "II",
            "mse": float(np.mean((generated - true) ** 2)),
            "zoom_start_s": float(time[start]),
            "zoom_stop_s": float(time[max(start, stop - 1)]),
        }
        manifest_rows.append(row_info)

        ax_full, ax_zoom = axes[row_idx]
        true_line, = ax_full.plot(time, true, color=TRUE_COLOR, lw=2.1, label="True ECG")
        gen_line, = ax_full.plot(time, generated, color=GENERATED_COLOR, lw=1.8,
                                 ls="--", label="Generated ECG")
        handles = handles or (true_line, gen_line)
        ax_full.set_title(f"{label} | full 8 s", loc="left", fontweight="bold")
        ax_full.set_ylabel("Normalized amplitude")
        ax_full.grid(True, alpha=0.24)
        ax_full.text(
            0.995, 0.96,
            f"{metadata_text(dataset, index)}\nMSE={row_info['mse']:.3f}",
            transform=ax_full.transAxes, ha="right", va="top", fontsize=9,
            bbox={"facecolor": "white", "alpha": 0.82, "edgecolor": "#D9D9D9", "pad": 3.0},
        )

        zoom_time = time[start:stop]
        ax_zoom.plot(zoom_time, true[start:stop], color=TRUE_COLOR, lw=2.4)
        ax_zoom.plot(zoom_time, generated[start:stop], color=GENERATED_COLOR, lw=2.0, ls="--")
        ax_zoom.set_title(f"QRS zoom | {zoom_seconds:.1f} s", loc="left", fontweight="bold")
        ax_zoom.set_xlabel("Time (s)")
        ax_zoom.set_ylabel("Normalized amplitude")
        ax_zoom.grid(True, alpha=0.24)
        ax_zoom.axvline(time[center], color="#777777", lw=0.9, ls=":", alpha=0.8)
        ax_full.set_xlabel("Time (s)")

    if handles:
        fig.legend(handles, ["True ECG", "Generated ECG"], loc="upper center",
                   bbox_to_anchor=(0.5, 1.005), ncol=2, frameon=False)
    fig.suptitle(
        f"{model_label} | {split} set | Lead II\n"
        "Representative window selected by within-state median Lead-II MSE",
        y=1.04, fontweight="bold",
    )
    fig.tight_layout(rect=(0, 0, 1, 0.93), h_pad=2.2, w_pad=2.4)
    base = output_dir / f"{split.lower()}_leadII_true_vs_generated_highres"
    outputs = save_triplet(fig, base, dpi)
    plt.close(fig)
    return {"outputs": outputs, "selected": manifest_rows}


def main() -> None:
    args = parse_args()
    output_root = args.output or (PROJECT_ROOT / "results" / "leadII_highres")
    if not output_root.is_absolute():
        output_root = PROJECT_ROOT / output_root
    manifest = {
        "lead": "II",
        "checkpoint": args.checkpoint,
        "dpi": args.dpi,
        "zoom_seconds": args.zoom_seconds,
        "selection": "within-state median Lead-II MSE",
        "models": {},
    }

    for run_value in args.run:
        run_dir = Path(run_value)
        if not run_dir.is_absolute():
            run_dir = PROJECT_ROOT / run_dir
        config, _, encoder, decoder, latent_flow, device = load_run(
            run_dir, args.checkpoint, split="test",
        )
        model_label = run_name(run_dir)
        model_key = run_dir.name
        model_output = output_root / model_key
        manifest["models"][model_key] = {
            "label": model_label,
            "run": str(run_dir),
            "checkpoint": str(run_dir / args.checkpoint),
            "device": str(device),
            "splits": {},
        }
        flow_steps = int((config.get("model", {}).get("cardio_align", {}) or {}).get(
            "integration_steps", 8,
        ))
        for split in ("train", "test"):
            dataset = create_dataset(
                config["data"]["dataset"], config["data"]["root"], split=split,
                ppg_channel=config["data"].get("ppg_channel"),
                ecg_lead=config["data"].get("ecg_lead"),
            )
            pred, efficiency = predict(
                encoder, decoder, dataset._x, None, device, fs=dataset.fs,
                batch_size=args.batch_size, latent_flow=latent_flow,
                flow_steps=flow_steps,
            )
            result = plot_one_split(
                model_label, split, pred, dataset._y, dataset, model_output,
                args.checkpoint, args.zoom_seconds, args.dpi,
            )
            result["efficiency"] = efficiency
            manifest["models"][model_key]["splits"][split] = result
            print(f"[{model_key}/{split}] selected={result['selected']}")

    output_root.mkdir(parents=True, exist_ok=True)
    with (output_root / "manifest.json").open("w", encoding="utf-8") as handle:
        json.dump(manifest, handle, ensure_ascii=False, indent=2, default=float)
    with (output_root / "STYLE_BRIEF.md").open("w", encoding="utf-8") as handle:
        handle.write(
            "# Lead-II High-Resolution Figure Brief\n\n"
            "- Profile: technical-neutral; immediate draft because no venue was specified.\n"
            "- Domain/audience: PPG-to-ECG machine-learning research and technical review.\n"
            "- Encoding: true ECG blue solid; generated ECG orange dashed.\n"
            "- Layout: one Lead-II figure per model and split; rest/activity rows; full-window and QRS zoom columns.\n"
            "- Transformations: no smoothing or amplitude alteration; representative samples are selected by within-state median Lead-II MSE.\n"
            "- Outputs: 600-DPI PNG plus PDF/SVG vector exports.\n"
        )
    print(f"Figures saved to: {output_root}")


if __name__ == "__main__":
    main()
