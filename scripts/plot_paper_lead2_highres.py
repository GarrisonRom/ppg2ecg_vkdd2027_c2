#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""Export high-resolution single-Lead-II figures for paper adaptations."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from scipy.signal import find_peaks

PROJECT_ROOT = Path(__file__).resolve().parents[1]
TRUE_COLOR = "#222222"
GENERATED_COLOR = "#D95F02"

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
        "--run", type=Path,
        default=PROJECT_ROOT / "paper_repro/runs/senssmarttech_recent_1to1_128hz_seed42",
    )
    parser.add_argument("--output", type=Path, default=PROJECT_ROOT / "results/paper_leadII_highres")
    parser.add_argument("--method", action="append", default=None,
                        help="method directory; repeat or omit for all available methods")
    parser.add_argument("--zoom-seconds", type=float, default=1.6)
    parser.add_argument("--dpi", type=int, default=600)
    return parser.parse_args()


def label_for(name: str) -> str:
    return {
        "cardiogan": "CardioGAN",
        "rddm": "RDDM",
        "qrs_transattn": "QRS-TransAttn",
        "p2e_wgan": "P2E-WGAN",
        "li2024_lightweight": "Li 2024 lightweight",
    }.get(name, name)


def qrs_center(signal: np.ndarray) -> int:
    peaks, props = find_peaks(signal, distance=32, prominence=max(float(signal.std() * 0.55), 1e-4))
    if len(peaks):
        return int(peaks[int(np.argmax(props["prominences"]))])
    return int(np.argmax(np.abs(signal - np.median(signal))))


def save_triplet(fig, base: Path, dpi: int) -> list[str]:
    base.parent.mkdir(parents=True, exist_ok=True)
    outputs = []
    for ext, kwargs in ((".png", {"dpi": dpi}), (".pdf", {}), (".svg", {})):
        path = base.with_suffix(ext)
        fig.savefig(path, bbox_inches="tight", **kwargs)
        outputs.append(str(path))
    return outputs


def plot_method(method_dir: Path, output_dir: Path, zoom_seconds: float, dpi: int) -> dict:
    name = method_dir.name
    rows = []
    selected = {}
    for split in ("train", "test"):
        data = np.load(method_dir / f"pred_{split}.npz")
        pred = data["prediction"][:, 0]
        target = data["target"][:, 0]
        errors = ((pred - target) ** 2).mean(axis=1)
        index = int(np.argmin(np.abs(errors - np.median(errors))))
        rows.append((split, index, pred[index], target[index], float(errors[index])))
        selected[split] = {"index": index, "mse": float(errors[index])}

    fs = 128.0
    length = rows[0][2].shape[-1]
    time = np.arange(length) / fs
    zoom_samples = max(8, int(round(zoom_seconds * fs)))
    fig, axes = plt.subplots(2, 2, figsize=(15.5, 8.0),
                             gridspec_kw={"width_ratios": [2.4, 1.0]})
    handles = None
    for row, (split, index, pred, target, mse) in enumerate(rows):
        center = qrs_center(target)
        start = max(0, center - zoom_samples // 2)
        stop = min(length, start + zoom_samples)
        start = max(0, stop - zoom_samples)
        ax_full, ax_zoom = axes[row]
        true_line, = ax_full.plot(time, target, color=TRUE_COLOR, lw=2.2, label="True ECG")
        pred_line, = ax_full.plot(time, pred, color=GENERATED_COLOR, lw=1.9, ls="--", label="Generated ECG")
        handles = handles or (true_line, pred_line)
        ax_full.set_title(f"{split.capitalize()} set | full 4 s", loc="left", fontweight="bold")
        ax_full.set_xlabel("Time (s)")
        ax_full.set_ylabel("Normalized amplitude")
        ax_full.grid(True, alpha=0.24)
        ax_full.text(0.995, 0.96, f"window={index}\nMSE={mse:.3f}",
                     transform=ax_full.transAxes, ha="right", va="top", fontsize=9,
                     bbox={"facecolor": "white", "alpha": 0.82, "edgecolor": "#D9D9D9", "pad": 3.0})
        ax_zoom.plot(time[start:stop], target[start:stop], color=TRUE_COLOR, lw=2.4)
        ax_zoom.plot(time[start:stop], pred[start:stop], color=GENERATED_COLOR, lw=2.0, ls="--")
        ax_zoom.set_title(f"QRS zoom | {zoom_seconds:.1f} s", loc="left", fontweight="bold")
        ax_zoom.set_xlabel("Time (s)")
        ax_zoom.set_ylabel("Normalized amplitude")
        ax_zoom.grid(True, alpha=0.24)
        ax_zoom.axvline(time[center], color="#777777", lw=0.9, ls=":")
    fig.legend(handles, ["True ECG", "Generated ECG"], loc="upper center",
               bbox_to_anchor=(0.5, 1.005), ncol=2, frameon=False)
    fig.suptitle(f"{label_for(name)} | single Lead II | 128 Hz\n"
                 "Representative windows selected by median waveform MSE",
                 y=1.04, fontweight="bold")
    fig.tight_layout(rect=(0, 0, 1, 0.93), h_pad=2.2, w_pad=2.4)
    outputs = save_triplet(fig, output_dir / f"{name}_leadII_train_test_highres", dpi)
    plt.close(fig)
    return {"method": name, "label": label_for(name), "outputs": outputs, "selected": selected}


def main() -> None:
    args = parse_args()
    run_dir = args.run if args.run.is_absolute() else PROJECT_ROOT / args.run
    output_dir = args.output if args.output.is_absolute() else PROJECT_ROOT / args.output
    methods = args.method or [path.name for path in sorted(run_dir.iterdir())
                              if path.is_dir() and (path / "pred_test.npz").exists()]
    manifest = {
        "run": str(run_dir),
        "lead": "II",
        "sampling_rate_hz": 128,
        "dpi": args.dpi,
        "zoom_seconds": args.zoom_seconds,
        "selection": "median waveform MSE per split",
        "methods": {},
    }
    for method in methods:
        result = plot_method(run_dir / method, output_dir, args.zoom_seconds, args.dpi)
        manifest["methods"][method] = result
        print(f"[{method}] {result['outputs']}")
    output_dir.mkdir(parents=True, exist_ok=True)
    with (output_dir / "manifest.json").open("w", encoding="utf-8") as handle:
        json.dump(manifest, handle, ensure_ascii=False, indent=2)
    with (output_dir / "STYLE_BRIEF.md").open("w", encoding="utf-8") as handle:
        handle.write(
            "# Paper-adaptation Lead-II Figure Brief\n\n"
            "- Profile: technical-neutral; single-lead waveform inspection.\n"
            "- True ECG: dark gray solid; generated ECG: orange dashed.\n"
            "- Layout: train/test rows, full 4-second window and QRS zoom columns.\n"
            "- Selection: median waveform MSE, without hand-picking or smoothing.\n"
            "- Exports: 600-DPI PNG plus PDF/SVG.\n"
        )
    print(f"Figures saved to: {output_dir}")


if __name__ == "__main__":
    main()
