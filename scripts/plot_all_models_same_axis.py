#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""Overlay every available model on the same Lead-II time axis.

The main and paper adaptations are trained with different rates and window
lengths.  For this qualitative figure they are paired by the same raw
SensSmartTech ``record_id`` and window start time, then plotted on a common
250-Hz grid.  This makes the display comparable without treating the two
training protocols as identical metric experiments.
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D
import numpy as np
import pandas as pd
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
    "CardioGAN": "#1D4E89",
    "RDDM": "#E76F51",
    "QRS-TransAttn": "#21867A",
    "P2E-WGAN": "#6A1B9A",
    "Li 2024 lightweight": "#B565D9",
}
TRUE_COLOR = "#222222"
MAIN_FS = 250.0
PAPER_FS = 128.0
PAPER_STRIDE_SECONDS = 2.0
PAPER_WINDOW_SECONDS = 4.0

plt.rcParams.update({
    "font.family": "sans-serif",
    "font.sans-serif": ["Arial", "Helvetica", "DejaVu Sans"],
    "font.size": 11,
    "axes.labelsize": 12,
    "axes.titlesize": 13,
    "xtick.labelsize": 10,
    "ytick.labelsize": 10,
    "legend.fontsize": 9.2,
    "figure.titlesize": 16,
    "axes.linewidth": 0.9,
    "grid.linewidth": 0.55,
    "lines.linewidth": 1.65,
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
        "--output", type=Path,
        default=PROJECT_ROOT / "results/all_models_same_axis",
    )
    parser.add_argument("--checkpoint", default="best.pth")
    parser.add_argument("--segment-seconds", type=float, default=2.5)
    parser.add_argument("--batch-size", type=int, default=64)
    parser.add_argument("--dpi", type=int, default=600)
    return parser.parse_args()


def absolute(path: str | Path) -> Path:
    value = Path(path)
    return value if value.is_absolute() else PROJECT_ROOT / value


def peak_center(signal: np.ndarray, fs: float) -> int:
    peaks, props = find_peaks(
        signal,
        distance=max(1, int(0.30 * fs)),
        prominence=max(float(signal.std() * 0.55), 1e-4),
    )
    if len(peaks):
        return int(peaks[int(np.argmax(props["prominences"]))])
    return int(np.argmax(np.abs(signal - np.median(signal))))


def crop_around_peak(signal: np.ndarray, fs: float, seconds: float) -> tuple[int, int, int]:
    size = min(len(signal), max(8, int(round(seconds * fs))))
    center = peak_center(signal, fs)
    start = max(0, center - size // 2)
    stop = min(len(signal), start + size)
    start = max(0, stop - size)
    return start, stop, center


def paper_local_metadata(data_path: Path, split: str) -> tuple[np.ndarray, pd.DataFrame]:
    """Return paper target arrays' record/start metadata in prediction order."""
    data = np.load(data_path, allow_pickle=False)
    split_subjects = data["train_subjects"] if split == "train" else data["test_subjects"]
    global_indices = np.flatnonzero(np.isin(data["subject_id"], split_subjects))
    if len(global_indices) == 0:
        raise ValueError(f"paper protocol has no {split} windows")

    record_ids = data["record_id"][global_indices].astype(str)
    starts = np.zeros(len(record_ids), dtype=np.float64)
    counts: dict[str, int] = {}
    for pos, record in enumerate(record_ids):
        starts[pos] = counts.get(record, 0) * PAPER_STRIDE_SECONDS
        counts[record] = counts.get(record, 0) + 1
    metadata = pd.DataFrame({
        "record_id": record_ids,
        "start_sec": starts,
        "subject_id": data["subject_id"][global_indices].astype(np.int64),
        "global_index": global_indices,
    })
    return global_indices, metadata


def load_main_predictions(entries: list[tuple[Path, str]], split: str, checkpoint: str,
                          batch_size: int) -> tuple[dict[str, np.ndarray], np.ndarray, pd.DataFrame]:
    predictions: dict[str, np.ndarray] = {}
    targets: np.ndarray | None = None
    metadata: pd.DataFrame | None = None
    lead_idx: int | None = None
    for run_dir, label in entries:
        config, dataset, encoder, decoder, latent_flow, device = load_run(
            run_dir, checkpoint, split=split,
        )
        ppg_channel = config["data"].get("ppg_channel")
        ppg_idx = dataset.ppg_channels.index(ppg_channel) if ppg_channel else None
        flow_steps = int((config.get("model", {}).get("cardio_align", {}) or {}).get(
            "integration_steps", 8,
        ))
        pred, _ = predict(
            encoder, decoder, dataset._x, ppg_idx, device, fs=dataset.fs,
            batch_size=batch_size, latent_flow=latent_flow, flow_steps=flow_steps,
        )
        # Keep the comparison strictly on Lead II, matching the paper
        # protocol's single output lead.
        predictions[label] = pred[:, dataset.ecg_channels.index("II")]
        if targets is None:
            targets = dataset._y
            metadata = dataset.metadata.copy()
            lead_idx = dataset.ecg_channels.index("II")
        elif not np.allclose(targets, dataset._y, atol=1e-6, rtol=1e-6):
            raise ValueError(f"main target mismatch for {split}")
    assert targets is not None and metadata is not None and lead_idx is not None
    return predictions, targets[:, lead_idx], metadata


def load_paper_predictions(entries: list[tuple[Path, str]], split: str, data_path: Path,
                           global_indices: np.ndarray) -> tuple[dict[str, np.ndarray], np.ndarray]:
    predictions: dict[str, np.ndarray] = {}
    target: np.ndarray | None = None
    for run_dir, label in entries:
        data = np.load(run_dir / f"pred_{split}.npz", allow_pickle=False)
        predictions[label] = data["prediction"][:, 0]
        current_target = data["target"][:, 0]
        if target is None:
            target = current_target
        elif not np.allclose(target, current_target, atol=1e-6, rtol=1e-6):
            raise ValueError(f"paper target mismatch for {split}")
    assert target is not None
    source = np.load(data_path, allow_pickle=False)["y"][global_indices, 0]
    if not np.allclose(target, source, atol=1e-5, rtol=1e-5):
        raise ValueError(f"paper prediction target is not aligned with protocol data for {split}")
    return predictions, target


def choose_shared_pair(main_target: np.ndarray, main_meta: pd.DataFrame,
                       paper_meta: pd.DataFrame, segment_seconds: float) -> dict:
    paper_lookup = {
        (str(row.record_id), round(float(row.start_sec), 6)): int(row.Index)
        for row in paper_meta.itertuples()
    }
    candidates: list[dict] = []
    shared_samples = int(round(PAPER_WINDOW_SECONDS * MAIN_FS))
    for main_index, row in main_meta.iterrows():
        key = (str(row["record_id"]), round(float(row["start_sec"]), 6))
        paper_index = paper_lookup.get(key)
        if paper_index is None:
            continue
        signal = main_target[main_index, :shared_samples]
        start, stop, center = crop_around_peak(signal, MAIN_FS, segment_seconds)
        prominence = float(np.max(signal) - np.median(signal))
        center_distance = abs(center - shared_samples / 2.0) / MAIN_FS
        # Prefer a clear R peak near the centre so both protocols share the crop.
        score = prominence - 0.35 * center_distance
        candidates.append({
            "main_index": int(main_index),
            "paper_index": int(paper_index),
            "record_id": str(row["record_id"]),
            "start_sec": float(row["start_sec"]),
            "crop_start": int(start),
            "crop_stop": int(stop),
            "peak_index": int(center),
            "score": score,
        })
    if not candidates:
        raise RuntimeError("No common record/start-time window exists between protocols")
    return max(candidates, key=lambda item: item["score"])


def normalize_for_display(signal: np.ndarray, reference: np.ndarray) -> np.ndarray:
    """Center each curve and use the reference's robust range as amplitude unit."""
    scale = float(np.percentile(reference, 95) - np.percentile(reference, 5))
    if not np.isfinite(scale) or scale < 1e-8:
        scale = float(np.std(reference)) or 1.0
    return (signal - np.median(signal)) / scale


def resample(signal: np.ndarray, source_fs: float, target_time: np.ndarray) -> np.ndarray:
    source_time = np.arange(len(signal), dtype=np.float64) / source_fs
    return np.interp(target_time, source_time, signal)


def save_triplet(fig, base: Path, dpi: int) -> list[str]:
    base.parent.mkdir(parents=True, exist_ok=True)
    outputs: list[str] = []
    for suffix, kwargs in ((".png", {"dpi": dpi}), (".pdf", {}), (".svg", {})):
        path = base.with_suffix(suffix)
        fig.savefig(path, bbox_inches="tight", **kwargs)
        outputs.append(str(path))
    return outputs


def plot(args: argparse.Namespace) -> dict:
    main_entries = [(absolute(path), label) for path, label in MAIN_DEFAULTS]
    paper_entries = [(absolute(path), label) for path, label in PAPER_DEFAULTS]
    paper_data_path = absolute(
        "paper_repro/runs/senssmarttech_1to1_128hz_seed42/paper_protocol_data.npz"
    )
    output = args.output if args.output.is_absolute() else PROJECT_ROOT / args.output

    for path, _ in main_entries + paper_entries:
        if not path.exists():
            raise FileNotFoundError(path)
    if not paper_data_path.exists():
        raise FileNotFoundError(paper_data_path)

    loaded: dict[str, dict] = {}
    for split in ("train", "test"):
        main_pred, main_target, main_meta = load_main_predictions(
            main_entries, split, args.checkpoint, args.batch_size,
        )
        global_indices, paper_meta = paper_local_metadata(paper_data_path, split)
        paper_pred, paper_target = load_paper_predictions(
            paper_entries, split, paper_data_path, global_indices,
        )
        selected = choose_shared_pair(main_target, main_meta, paper_meta, args.segment_seconds)

        start, stop = selected["crop_start"], selected["crop_stop"]
        x = np.arange(stop - start, dtype=np.float64) / MAIN_FS
        reference = main_target[selected["main_index"], start:stop]
        traces: dict[str, np.ndarray] = {"True ECG": normalize_for_display(reference, reference)}
        for label, values in main_pred.items():
            traces[label] = normalize_for_display(
                values[selected["main_index"], start:stop], reference,
            )

        paper_start = start / MAIN_FS
        paper_time = paper_start + x
        paper_reference = resample(
            paper_target[selected["paper_index"]], PAPER_FS, paper_time,
        )
        for label, values in paper_pred.items():
            curve = resample(values[selected["paper_index"]], PAPER_FS, paper_time)
            traces[label] = normalize_for_display(curve, reference)

        selected["target_shape_mse_after_independent_scaling"] = float(
            np.mean(
                (normalize_for_display(paper_reference, paper_reference)
                 - normalize_for_display(reference, reference)) ** 2
            )
        )
        loaded[split] = {
            "x": x,
            "traces": traces,
            "selected": selected,
            "subject_id": int(main_meta.iloc[selected["main_index"]]["subject_id"]),
            "activity": str(main_meta.iloc[selected["main_index"]].get("activity", "")),
        }

    global_limit = max(
        float(np.max(np.abs(trace)))
        for split in loaded.values()
        for trace in split["traces"].values()
    )
    global_limit = max(1.0, global_limit * 1.08)
    fig, axes = plt.subplots(2, 1, figsize=(14.0, 6.2), sharex=True, sharey=True)
    main_labels = [label for _, label in main_entries]
    paper_labels = [label for _, label in paper_entries]
    for row, split in enumerate(("train", "test")):
        item = loaded[split]
        ax = axes[row]
        ax.plot(item["x"], item["traces"]["True ECG"], color=TRUE_COLOR, lw=2.6, zorder=5)
        for label in main_labels:
            ax.plot(item["x"], item["traces"][label], color=COLORS[label], lw=1.65,
                    solid_capstyle="round")
        for label in paper_labels:
            ax.plot(item["x"], item["traces"][label], color=COLORS[label], lw=1.5,
                    ls="--", solid_capstyle="round")
        ax.axvline(
            (item["selected"]["peak_index"] - item["selected"]["crop_start"]) / MAIN_FS,
            color="#777777", lw=0.9, ls=":", alpha=0.85,
        )
        selected = item["selected"]
        ax.set_title(
            f"{split.capitalize()} | {selected['record_id']} | start={selected['start_sec']:.1f} s "
            f"| subject={item['subject_id']} | Lead II",
            loc="left", fontweight="bold",
        )
        ax.set_ylabel("Relative normalized amplitude")
        ax.grid(True, alpha=0.24)
        ax.set_ylim(-global_limit, global_limit)
        ax.text(
            0.995, 0.94,
            f"common crop={args.segment_seconds:.1f} s | same raw record/start",
            transform=ax.transAxes, ha="right", va="top", fontsize=8.8,
            bbox={"facecolor": "white", "alpha": 0.84, "edgecolor": "#D9D9D9", "pad": 3.0},
        )
    axes[-1].set_xlabel("Time within common segment (s)")

    handles = [Line2D([0], [0], color=TRUE_COLOR, lw=2.6, label="True ECG")]
    handles += [Line2D([0], [0], color=COLORS[label], lw=1.8, label=label) for label in main_labels]
    handles += [Line2D([0], [0], color=COLORS[label], lw=1.8, ls="--", label=label) for label in paper_labels]
    handles += [
        Line2D([0], [0], color="#555555", lw=1.8, label="Main protocol (solid)"),
        Line2D([0], [0], color="#555555", lw=1.8, ls="--", label="Paper protocol (dashed)"),
    ]
    fig.legend(handles=handles, loc="upper center", bbox_to_anchor=(0.5, 0.995),
               ncol=5, frameon=False, columnspacing=1.2, handlelength=2.4)
    fig.suptitle(
        "All PPG2ECG models on one common Lead-II time axis",
        y=1.045, fontweight="bold",
    )
    fig.text(
        0.5, 0.006,
        "Same raw record/start time; paper outputs resampled 128 -> 250 Hz. "
        "Each trace is baseline-centered and scaled by the common target range for visual comparison.",
        ha="center", va="bottom", fontsize=8.8, color="#555555",
    )
    fig.tight_layout(rect=(0, 0.035, 1, 0.84), h_pad=1.7)
    outputs = save_triplet(fig, output / "all_models_same_axis_leadII", args.dpi)
    plt.close(fig)

    manifest = {
        "one_common_axis": True,
        "lead": "II",
        "segment_seconds": args.segment_seconds,
        "plot_fs_hz": MAIN_FS,
        "main_protocol": "4->4, 250 Hz, 8 s windows",
        "paper_protocol": "1->1, 128 Hz, 4 s windows",
        "alignment": "same raw record_id and same window start_sec; paper trace resampled to 250 Hz",
        "display_transform": "baseline-center each trace; divide by common main-target 5-95 percentile range",
        "outputs": outputs,
        "splits": {
            split: {
                "selected": item["selected"],
                "subject_id": item["subject_id"],
                "activity": item["activity"],
            }
            for split, item in loaded.items()
        },
        "models": main_labels + paper_labels,
        "note": "Qualitative cross-protocol overlay; use each protocol's metrics for numerical claims.",
    }
    output.mkdir(parents=True, exist_ok=True)
    (output / "manifest.json").write_text(
        json.dumps(manifest, ensure_ascii=False, indent=2, default=float), encoding="utf-8",
    )
    (output / "STYLE_BRIEF.md").write_text(
        "# All-Models One-Axis Lead-II Brief\n\n"
        "- Every model is drawn on one shared x/y coordinate system.\n"
        "- Train/test use the same raw record and start time within each split.\n"
        "- The paper-protocol output is interpolated from 128 Hz to the 250-Hz display grid.\n"
        "- Solid lines are the main 4->4 protocol; dashed lines are the paper 1->1 adaptation.\n"
        "- Curves are baseline-centered and scaled by the common target range; this is a qualitative display transform.\n"
        "- Numerical comparisons remain protocol-specific.\n"
        "- Exports include 600-DPI PNG, PDF, and SVG.\n",
        encoding="utf-8",
    )
    return manifest


def main() -> None:
    args = parse_args()
    if args.segment_seconds <= 0 or args.segment_seconds > PAPER_WINDOW_SECONDS:
        raise ValueError(f"segment seconds must be in (0, {PAPER_WINDOW_SECONDS}]")
    manifest = plot(args)
    print(json.dumps(manifest, ensure_ascii=False, indent=2, default=float))


if __name__ == "__main__":
    main()
