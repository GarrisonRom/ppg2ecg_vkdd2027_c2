#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""Plot training diagnostics for advanced PPG2ECG runs.

The input is the ``training_history.json`` written by ``PPG2ECGTrainer``.
The figure keeps reconstruction and adversarial diagnostics in separate
panels so a low discriminator accuracy cannot be mistaken for better ECG
reconstruction by itself.
"""

from __future__ import annotations

import argparse
import json
import math
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--run", type=Path, required=True)
    parser.add_argument("--output", type=Path, default=None)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    run_dir = args.run
    history_path = run_dir / "training_history.json"
    with history_path.open("r", encoding="utf-8") as handle:
        history = json.load(handle)
    if not history:
        raise ValueError(f"No epochs found in {history_path}")

    epochs = [int(row["epoch"]) for row in history]
    train = [row["train"] for row in history]
    val = [row["val"] for row in history]

    def series(rows, key):
        return [float(row.get(key, float("nan"))) for row in rows]

    def select_reconstruction_key():
        """Use the active reconstruction term instead of assuming MSE."""
        for key in ("mse", "l1", "qrs_weighted", "objective", "total"):
            values = series(train, key) + series(val, key)
            finite = [abs(value) for value in values if math.isfinite(value)]
            if finite and max(finite) > 1e-12:
                return key
        return "total"

    reconstruction_key = select_reconstruction_key()

    fig, axes = plt.subplots(2, 2, figsize=(10.5, 6.8), sharex=True)
    colors = {"train": "#2F6690", "val": "#D97745"}

    ax = axes[0, 0]
    ax.plot(epochs, series(train, reconstruction_key), color=colors["train"], label="Train")
    ax.plot(epochs, series(val, reconstruction_key), color=colors["val"], label="Validation")
    ax.set_title(f"Active reconstruction term: {reconstruction_key}")
    ax.set_ylabel("Loss (lower is better)")
    ax.legend(frameon=False)

    ax = axes[0, 1]
    ax.plot(epochs, series(train, "subject_loss"), color="#4C956C", label="Subject loss")
    random_ce = math.log(22.0)
    ax.axhline(random_ce, color="#666666", linestyle=":", linewidth=1.2,
               label="22-class random CE")
    ax.set_title("Subject discriminator loss")
    ax.set_ylabel("Cross-entropy")
    ax.legend(frameon=False)

    ax = axes[1, 0]
    ax.plot(epochs, series(train, "subject_acc"), color="#7B2CBF", label="Subject accuracy")
    ax.axhline(1.0 / 22.0, color="#666666", linestyle=":", linewidth=1.2,
               label="22-class random accuracy")
    ax.set_title("Subject prediction accuracy")
    ax.set_xlabel("Epoch")
    ax.set_ylabel("Accuracy")
    ax.set_ylim(bottom=0.0)
    ax.legend(frameon=False)

    ax = axes[1, 1]
    ax.plot(epochs, series(train, "grl_lambda"), color="#C44536", label="GRL lambda")
    ax2 = ax.twinx()
    ax2.plot(epochs, series(train, "irm_aux"), color="#E09F3E", label="V-REx auxiliary")
    ax.set_title("Disentanglement controls")
    ax.set_xlabel("Epoch")
    ax.set_ylabel("GRL coefficient", color="#C44536")
    ax2.set_ylabel("V-REx risk variance", color="#E09F3E")
    ax.grid(True, alpha=0.25)

    for row in axes:
        for axis in row:
            axis.spines["top"].set_visible(False)
            axis.spines["right"].set_visible(False)
            axis.grid(True, alpha=0.25)
    fig.suptitle("VAE + Flow + GRL + V-REx training diagnostics", y=0.995)
    fig.tight_layout()

    output_dir = args.output or (run_dir / "figures")
    output_dir.mkdir(parents=True, exist_ok=True)
    base = output_dir / "training_diagnostics"
    fig.savefig(base.with_suffix(".png"), dpi=300, bbox_inches="tight")
    fig.savefig(base.with_suffix(".pdf"), bbox_inches="tight")
    fig.savefig(base.with_suffix(".svg"), bbox_inches="tight")
    plt.close(fig)
    print(f"Saved diagnostics to {output_dir}")


if __name__ == "__main__":
    main()
