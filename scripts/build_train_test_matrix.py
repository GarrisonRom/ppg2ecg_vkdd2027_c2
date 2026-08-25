#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""Build one unified Train/Test metric matrix for all recorded models."""

from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path


PROJECT_ROOT = Path(__file__).resolve().parents[1]

MODEL_SPECS = [
    {
        "model": "v0.2",
        "protocol": "main 4->4 | 250 Hz | 8 s | 22/5/5",
        "kind": "main",
        "directory": "runs/senssmarttech_vae_flow_adv_irm_20ep_seed42",
    },
    {
        "model": "v0.52",
        "protocol": "main 4->4 | 250 Hz | 8 s | 22/5/5",
        "kind": "main",
        "directory": "runs/senssmarttech_v052_multiband_frozen_cycle_20ep_seed42",
    },
    {
        "model": "v0.61",
        "protocol": "main 4->4 | 250 Hz | 8 s | 22/5/5",
        "kind": "main",
        "directory": "runs/senssmarttech_v061_vae_multiband_transfer_latent128_20ep_seed42",
    },
    {
        "model": "v0.64",
        "protocol": "main 4->4 | 250 Hz | 8 s | 22/5/5",
        "kind": "main",
        "directory": "runs/senssmarttech_v064_vae_multiband_latent256_transfer_20ep_seed42",
    },
    {
        "model": "CardioGAN",
        "protocol": "paper 1->1 | 128 Hz | 4 s | 26/6",
        "kind": "paper",
        "directory": "paper_repro/runs/senssmarttech_1to1_128hz_seed42/cardiogan",
    },
    {
        "model": "RDDM",
        "protocol": "paper 1->1 | 128 Hz | 4 s | 26/6",
        "kind": "paper",
        "directory": "paper_repro/runs/senssmarttech_1to1_128hz_seed42/rddm",
    },
    {
        "model": "QRS-TransAttn",
        "protocol": "paper 1->1 | 128 Hz | 4 s | 26/6",
        "kind": "paper",
        "directory": "paper_repro/runs/senssmarttech_recent_1to1_128hz_seed42/qrs_transattn",
    },
    {
        "model": "P2E-WGAN",
        "protocol": "paper 1->1 | 128 Hz | 4 s | 26/6",
        "kind": "paper",
        "directory": "paper_repro/runs/senssmarttech_recent_1to1_128hz_seed42/p2e_wgan",
    },
    {
        "model": "Li 2024 lightweight",
        "protocol": "paper 1->1 | 128 Hz | 4 s | 26/6",
        "kind": "paper",
        "directory": "paper_repro/runs/senssmarttech_recent_1to1_128hz_seed42/li2024_lightweight",
    },
]

METRICS = [
    ("rmse", "RMSE", "waveform", "rmse/macro", "lower"),
    ("mae", "MAE", "waveform", "mae/macro", "lower"),
    ("pcc", "PCC", "waveform", "pcc/macro", "higher"),
    ("nrmse", "NRMSE", "waveform", "nrmse/macro", "lower"),
    ("snr_db", "SNR (dB)", "waveform", "snr_db/macro", "higher"),
    ("dtw", "DTW", "waveform", "dtw/macro", "lower"),
    ("hr_err_bpm", "HR err (bpm)", "physiology", "hr_err_bpm", "lower"),
    ("rpeak_f1", "R-peak F1", "physiology", "rpeak_f1", "higher"),
    ("rpeak_time_err_ms", "Peak err (ms)", "physiology", "rpeak_time_err_ms", "lower"),
    ("qrs_width_err_ms", "QRS err (ms)", "physiology", "qrs_width_err_ms", "lower"),
    ("rmssd_err_ms", "RMSSD err (ms)", "physiology", "rmssd_err_ms", "lower"),
    ("qrs_amp_ratio", "QRS amp ratio", "physiology", "qrs_amp_ratio", "near 1"),
    ("ptt_carotid_err_ms", "Carotid PTT err (ms)", "physiology", "ptt_carotid_err_s", "lower"),
    ("ptt_carotid_in_range", "Carotid PTT in range", "physiology", "ptt_carotid_in_range", "higher"),
    ("validity_rate", "Valid rate", "validity", "validity_rate", "higher"),
]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--markdown", type=Path,
        default=PROJECT_ROOT / "docs/train_test_all_models_matrix.md",
    )
    parser.add_argument(
        "--csv", type=Path,
        default=PROJECT_ROOT / "results/train_test_all_models_matrix.csv",
    )
    return parser.parse_args()


def read_json(path: Path) -> dict:
    return json.loads(path.read_text(encoding="utf-8"))


def nested_metric(container: dict, path: str) -> float:
    value: object = container
    for part in path.split("/"):
        value = value[part]  # type: ignore[index]
    if isinstance(value, dict) and "mean" in value:
        value = value["mean"]
    return float(value)


def mean_value(value: object) -> float:
    if isinstance(value, dict) and "mean" in value:
        value = value["mean"]
    return float(value)


def load_row(spec: dict, split: str) -> dict:
    directory = PROJECT_ROOT / spec["directory"]
    if spec["kind"] == "main":
        result = read_json(directory / f"eval_{split}.json")
        metrics = result["model"]
        params = float(result["efficiency"].get("params_M", float("nan")))
        n_windows = int(metrics.get("n_windows", 0))
        n_subjects = int(metrics.get("n_subjects", 0))
    else:
        result = read_json(directory / f"metrics_{split}.json")
        metrics = result["metrics"]
        params = float(result.get("params_m", result.get("params_M", float("nan"))))
        n_windows = int(metrics.get("n_windows", 0))
        n_subjects = int(metrics.get("n_subjects", 0))

    row = {
        "protocol": spec["protocol"],
        "model": spec["model"],
        "split": split,
        "params_M": params,
        "n_windows": n_windows,
        "n_subjects": n_subjects,
    }
    for key, _, section, path, _ in METRICS:
        if key == "validity_rate":
            row[key] = float(metrics["validity_rate"])
        elif key == "ptt_carotid_err_ms":
            row[key] = mean_value(metrics[section][path]) * 1000.0
        elif section == "waveform":
            row[key] = mean_value(metrics["waveform"]["macro"][path])
        else:
            row[key] = mean_value(metrics[section][path])
    return row


def format_value(value: float, key: str) -> str:
    if key in {"validity_rate", "ptt_carotid_in_range"}:
        return f"{value:.3f}"
    if key in {"rpeak_f1", "pcc", "qrs_amp_ratio"}:
        return f"{value:.3f}"
    if key in {"snr_db", "hr_err_bpm", "rpeak_time_err_ms", "qrs_width_err_ms", "rmssd_err_ms", "ptt_carotid_err_ms"}:
        return f"{value:.2f}"
    return f"{value:.3f}"


def markdown_text(value: object) -> str:
    return str(value).replace("|", "\\|").replace("\n", " ")


def write_csv(rows: list[dict], path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    fields = ["protocol", "model", "split", "params_M", "n_windows", "n_subjects"]
    fields += [item[0] for item in METRICS]
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields)
        writer.writeheader()
        writer.writerows(rows)


def write_markdown(rows: list[dict], path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    by_model: dict[str, dict[str, dict]] = {}
    order: list[str] = []
    for row in rows:
        by_model.setdefault(row["model"], {})[row["split"]] = row
        if row["model"] not in order:
            order.append(row["model"])

    compact_columns = [
        ("protocol", "Protocol"),
        ("model", "Model"),
        ("train_rmse_pcc", "Train RMSE / PCC"),
        ("test_rmse_pcc", "Test RMSE / PCC"),
        ("rmse_gap_pct", "RMSE gap"),
        ("test_hr_f1", "Test HR err / R-F1"),
        ("test_peak_qrs", "Test Peak / QRS ms"),
        ("test_rmssd", "Test RMSSD err ms"),
        ("test_snr_dtw", "Test SNR / DTW"),
        ("test_valid", "Test valid"),
    ]
    lines = [
        "# Unified Train/Test Matrix",
        "",
        "> Main protocol: 4 PPG -> 4 ECG, 250 Hz, 8 s, subject-wise 22/5/5. "
        "Paper protocol: 1 PPG -> Lead II, 128 Hz, 4 s, subject-wise 26/6. "
        "The two protocols are listed together for reference but are not one numerical leaderboard.",
        "",
        "## One-Table Summary",
        "",
        "| " + " | ".join(label for _, label in compact_columns) + " |",
        "|" + "|".join("---" for _ in compact_columns) + "|",
    ]
    for model in order:
        train = by_model[model]["train"]
        test = by_model[model]["test"]
        gap = (test["rmse"] - train["rmse"]) / abs(train["rmse"]) * 100.0
        values = [
            train["protocol"],
            model,
            f"{train['rmse']:.3f} / {train['pcc']:.3f}",
            f"{test['rmse']:.3f} / {test['pcc']:.3f}",
            f"{gap:+.1f}%",
            f"{test['hr_err_bpm']:.2f} / {test['rpeak_f1']:.3f}",
            f"{test['rpeak_time_err_ms']:.2f} / {test['qrs_width_err_ms']:.2f}",
            f"{test['rmssd_err_ms']:.2f}",
            f"{test['snr_db']:.2f} / {test['dtw']:.3f}",
            f"{test['validity_rate']:.3f}",
        ]
        lines.append("| " + " | ".join(markdown_text(value) for value in values) + " |")

    lines += [
        "",
        "RMSE/PCC: lower/higher is better. HR, Peak, QRS and RMSSD errors: lower is better. "
        "QRS amp ratio is best near 1. Valid rate is higher is better.",
        "",
        "## Full Long-Form Table",
        "",
        "The following rows preserve every recorded metric and place Train/Test beside each other for every model.",
        "",
    ]
    full_headers = ["Protocol", "Model", "Split", "Params M", "Windows", "Subjects"] + [item[1] for item in METRICS]
    lines.append("| " + " | ".join(full_headers) + " |")
    lines.append("|" + "|".join("---" for _ in full_headers) + "|")
    for row in rows:
        values = [
            row["protocol"], row["model"], row["split"], f"{row['params_M']:.3f}",
            str(row["n_windows"]), str(row["n_subjects"]),
        ]
        values += [format_value(row[item[0]], item[0]) for item in METRICS]
        lines.append("| " + " | ".join(markdown_text(value) for value in values) + " |")

    lines += [
        "",
        "## Reading the Matrix",
        "",
        "- Train is an in-sample fit reference, not a generalization score.",
        "- A small RMSE gap with near-zero PCC indicates underfitting, not strong invariance.",
        "- Cross-protocol absolute RMSE values must not be ranked directly because sampling, window length, channel count and normalization differ.",
        "- For SCD-oriented interpretation, prioritize Test R-peak F1, Peak error, QRS width error and HR/RMSSD errors over RMSE alone.",
        "",
        "Source files: `runs/*/eval_train.json`, `runs/*/eval_test.json`, and `paper_repro/runs/*/metrics_{train,test}.json`.",
        "",
    ]
    path.write_text("\n".join(lines), encoding="utf-8")


def main() -> None:
    args = parse_args()
    rows = []
    for spec in MODEL_SPECS:
        for split in ("train", "test"):
            rows.append(load_row(spec, split))
    write_csv(rows, args.csv if args.csv.is_absolute() else PROJECT_ROOT / args.csv)
    write_markdown(rows, args.markdown if args.markdown.is_absolute() else PROJECT_ROOT / args.markdown)
    print(f"Wrote {len(rows)} split rows")
    print(f"Markdown: {args.markdown}")
    print(f"CSV: {args.csv}")


if __name__ == "__main__":
    main()
