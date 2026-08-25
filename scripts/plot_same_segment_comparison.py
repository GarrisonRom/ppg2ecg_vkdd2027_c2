#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""Compare all methods on the same short Lead-II signal segment.

Two protocol groups are rendered separately:

* ``main``: the project's 4-PPG -> 4-ECG, 250 Hz, 8-second runs;
* ``paper``: the 1-PPG -> Lead-II, 128 Hz, 4-second paper adaptations.

Within each group, every method uses the identical target window and identical
time crop. The crop is centered on a prominent target R peak and defaults to
2.5 seconds so QRS details remain legible at normal viewing size.
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


TRUE_COLOR = "#222222"
MAIN_COLORS = {
    "v0.2 VAE+Flow+GRL": "#3568A8",
    "v0.52 Multi-band": "#D95F02",
    "v0.61 Latent-128": "#2A9D8F",
    "v0.64 Latent-256": "#7B2CBF",
}
PAPER_COLORS = {
    "CardioGAN": "#3568A8",
    "RDDM": "#D95F02",
    "QRS-TransAttn": "#2A9D8F",
    "P2E-WGAN": "#7B2CBF",
    "Li 2024 lightweight": "#C77DFF",
}

plt.rcParams.update({
    "font.family": "sans-serif",
    "font.sans-serif": ["Arial", "Helvetica", "DejaVu Sans"],
    "font.size": 11,
    "axes.labelsize": 12,
    "axes.titlesize": 13,
    "xtick.labelsize": 10,
    "ytick.labelsize": 10,
    "legend.fontsize": 9.5,
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


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--group", choices=["main", "paper", "both"], default="both")
    parser.add_argument("--output", type=Path, default=PROJECT_ROOT / "results/same_segment_comparison")
    parser.add_argument("--checkpoint", default="best.pth")
    parser.add_argument("--segment-seconds", type=float, default=2.5,
                        help="short common crop length; default 2.5 seconds")
    parser.add_argument("--batch-size", type=int, default=64)
    parser.add_argument("--dpi", type=int, default=600)
    parser.add_argument("--index", type=int, default=None,
                        help="fixed target window index; default selects median-MSE window")
    return parser.parse_args()


def as_abs(path: str | Path) -> Path:
    path = Path(path)
    return path if path.is_absolute() else PROJECT_ROOT / path


def r_peak_center(signal: np.ndarray, fs: float) -> int:
    distance = max(1, int(0.30 * fs))
    prominence = max(float(np.std(signal) * 0.55), 1e-4)
    peaks, props = find_peaks(signal, distance=distance, prominence=prominence)
    if len(peaks):
        return int(peaks[int(np.argmax(props["prominences"]))])
    return int(np.argmax(np.abs(signal - np.median(signal))))


def crop_around_peak(signal: np.ndarray, fs: float, seconds: float) -> tuple[int, int, int]:
    n = min(len(signal), max(8, int(round(seconds * fs))))
    center = r_peak_center(signal, fs)
    start = max(0, center - n // 2)
    stop = min(len(signal), start + n)
    start = max(0, stop - n)
    return start, stop, center


def choose_index(target: np.ndarray, predictions: dict[str, np.ndarray], lead_idx: int,
                 fixed: int | None) -> int:
    if fixed is not None:
        if fixed < 0 or fixed >= len(target):
            raise IndexError(f"index {fixed} outside [0, {len(target)})")
        return fixed
    errors = []
    for index in range(len(target)):
        per_model = [
            float(np.mean((prediction[index, lead_idx] - target[index, lead_idx]) ** 2))
            for prediction in predictions.values()
        ]
        errors.append(float(np.mean(per_model)))
    median = float(np.median(errors))
    return int(np.argmin(np.abs(np.asarray(errors) - median)))


def save_triplet(fig, base: Path, dpi: int) -> list[str]:
    base.parent.mkdir(parents=True, exist_ok=True)
    outputs = []
    for suffix, kwargs in ((".png", {"dpi": dpi}), (".pdf", {}), (".svg", {})):
        path = base.with_suffix(suffix)
        fig.savefig(path, bbox_inches="tight", **kwargs)
        outputs.append(str(path))
    return outputs


def plot_group(group: str, entries: list[tuple[Path, str]], output_dir: Path,
               args: argparse.Namespace) -> dict:
    predictions: dict[str, dict[str, np.ndarray]] = {"train": {}, "test": {}}
    targets: dict[str, np.ndarray] = {}
    subjects: dict[str, np.ndarray] = {}
    fs = None
    lead_idx = None
    for split in ("train", "test"):
        for run_dir, label in entries:
            if group == "paper":
                data = np.load(run_dir / f"pred_{split}.npz")
                pred = data["prediction"]
                target = data["target"]
                subject = data["subject_id"]
                current_fs = 128
                current_lead_idx = 0
            else:
                config, dataset, encoder, decoder, latent_flow, device = load_run(
                    run_dir, args.checkpoint, split=split,
                )
                ppg_channel = config["data"].get("ppg_channel")
                ppg_idx = dataset.ppg_channels.index(ppg_channel) if ppg_channel else None
                flow_steps = int((config.get("model", {}).get("cardio_align", {}) or {}).get(
                    "integration_steps", 8,
                ))
                pred, _ = predict(
                    encoder, decoder, dataset._x, ppg_idx, device, fs=dataset.fs,
                    batch_size=args.batch_size, latent_flow=latent_flow, flow_steps=flow_steps,
                )
                target = dataset._y
                subject = dataset.metadata["subject_id"].to_numpy() if dataset.metadata is not None else np.arange(len(target))
                current_fs = int(dataset.fs)
                current_lead_idx = dataset.ecg_channels.index("II")
            if label in predictions[split]:
                raise ValueError(f"duplicate method label: {label}")
            predictions[split][label] = pred
            if split not in targets:
                targets[split] = target
                subjects[split] = subject
            elif not np.allclose(targets[split], target, atol=1e-6, rtol=1e-6):
                raise ValueError(f"target mismatch for {group}/{split}; methods are not aligned")
            fs = fs or current_fs
            lead_idx = lead_idx if lead_idx is not None else current_lead_idx
            if fs != current_fs or lead_idx != current_lead_idx:
                raise ValueError("inconsistent sampling rate or Lead-II index")

    selected = {}
    for split in ("train", "test"):
        selected[split] = choose_index(targets[split], predictions[split], lead_idx, args.index)

    fig, axes = plt.subplots(2, 1, figsize=(12.0, 5.0), sharex=False)
    color_map = MAIN_COLORS if group == "main" else PAPER_COLORS
    for row, split in enumerate(("train", "test")):
        index = selected[split]
        target = targets[split][index, lead_idx]
        start, stop, center = crop_around_peak(target, fs, args.segment_seconds)
        x = np.arange(start, stop) / float(fs)
        ax = axes[row]
        true_line, = ax.plot(x - x[0], target[start:stop], color=TRUE_COLOR, lw=2.4, label="True ECG")
        for label, prediction in predictions[split].items():
            ax.plot(x - x[0], prediction[index, lead_idx, start:stop],
                    color=color_map.get(label, None), lw=1.8, label=label)
        ax.axvline((center - start) / float(fs), color="#777777", lw=0.9, ls=":", alpha=0.8)
        ax.set_title(f"{split.capitalize()} set | same window index={index} | Lead II", loc="left", fontweight="bold")
        ax.set_ylabel("Normalized amplitude")
        ax.set_xlabel("Time within common segment (s)")
        ax.grid(True, alpha=0.24)
        ax.text(0.995, 0.95,
                f"subject={subjects[split][index]}  crop={args.segment_seconds:.1f} s",
                transform=ax.transAxes, ha="right", va="top", fontsize=9,
                bbox={"facecolor": "white", "alpha": 0.82, "edgecolor": "#D9D9D9", "pad": 3.0})
    handles, labels = axes[0].get_legend_handles_labels()
    fig.legend(handles, labels, loc="upper center", bbox_to_anchor=(0.5, 1.005),
               ncol=min(4, len(labels)), frameon=False)
    protocol_text = "4->4 | 250 Hz | 8 s source" if group == "main" else "1->1 | 128 Hz | 4 s source"
    fig.suptitle(f"Same-segment Lead-II comparison | {group} protocol | {protocol_text}",
                 y=1.04, fontweight="bold")
    fig.tight_layout(rect=(0, 0, 1, 0.91), h_pad=1.7)
    outputs = save_triplet(fig, output_dir / f"{group}_same_segment_leadII", args.dpi)
    plt.close(fig)
    return {
        "group": group,
        "protocol": protocol_text,
        "lead": "II",
        "sampling_rate_hz": fs,
        "source_window_seconds": 8.0 if group == "main" else 4.0,
        "common_segment_seconds": args.segment_seconds,
        "selection": "same index across methods; index selected by median mean Lead-II MSE",
        "selected": selected,
        "outputs": outputs,
        "methods": [label for _, label in entries],
    }


def main() -> None:
    args = parse_args()
    if args.segment_seconds <= 0:
        raise ValueError("--segment-seconds must be positive")
    output_dir = args.output if args.output.is_absolute() else PROJECT_ROOT / args.output
    groups = {}
    if args.group in ("main", "both"):
        groups["main"] = [(as_abs(path), label) for path, label in MAIN_DEFAULTS]
    if args.group in ("paper", "both"):
        groups["paper"] = [(as_abs(path), label) for path, label in PAPER_DEFAULTS]
    manifest = {
        "same_segment": True,
        "selection": "common target window per split; crop centered on target R peak",
        "segment_seconds": args.segment_seconds,
        "dpi": args.dpi,
        "groups": {},
    }
    for group, entries in groups.items():
        for path, _ in entries:
            if not path.exists():
                raise FileNotFoundError(path)
        result = plot_group(group, entries, output_dir, args)
        manifest["groups"][group] = result
        print(f"[{group}] selected={result['selected']}")
        print(f"[{group}] outputs={result['outputs']}")
    output_dir.mkdir(parents=True, exist_ok=True)
    with (output_dir / "manifest.json").open("w", encoding="utf-8") as handle:
        json.dump(manifest, handle, ensure_ascii=False, indent=2, default=float)
    with (output_dir / "STYLE_BRIEF.md").open("w", encoding="utf-8") as handle:
        handle.write(
            "# Same-Segment Lead-II Comparison Brief\n\n"
            "- Same target window and same short crop are used for every method within each protocol group.\n"
            "- True ECG is dark gray; generated methods use stable color assignments.\n"
            "- The crop is 2.5 seconds by default and centered on a prominent target R peak.\n"
            "- Main and paper protocols are exported separately because their sampling rates and window definitions differ.\n"
            "- Exports include 600-DPI PNG, PDF, and SVG.\n"
        )
    print(f"Figures saved to: {output_dir}")


if __name__ == "__main__":
    main()
