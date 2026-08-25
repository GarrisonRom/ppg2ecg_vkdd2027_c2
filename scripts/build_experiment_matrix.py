#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""Build a reproducible, multi-dimensional experiment comparison report.

The evaluator writes one JSON file per split.  This script deliberately reads
those files instead of duplicating metric calculations, then joins
train/validation/test results to expose checkpoint selection, subject-wise
generalization, and activity robustness.
"""

from __future__ import annotations

import argparse
import json
import math
from pathlib import Path


RUNS = [
    ("v0.1", "senssmarttech_baseline_20ep", "reference baseline"),
    ("v0.2", "senssmarttech_vae_flow_adv_irm_20ep_seed42", "representation baseline"),
    ("v0.3", "senssmarttech_v03_qrs_vae_flow_adv_irm_20ep_seed42", "negative loss control"),
    ("v0.4", "senssmarttech_v04_ppgflowecg_inspired_20ep_seed42", "latent alignment axis"),
    ("v0.41", "senssmarttech_v041_ecg_autoencoder_skip_20ep_seed42", "ECG capacity diagnostic"),
    ("v0.5", "senssmarttech_v05_bidirectional_cycle_20ep_seed42", "negative cycle control"),
    ("v0.51", "senssmarttech_v051_frozen_reverse_cycle_20ep_seed42", "cycle schedule control"),
    ("v0.52-hf4", "senssmarttech_v052_highfreq4_frozen_cycle_20ep_seed42", "high-band ablation"),
    ("v0.52", "senssmarttech_v052_multiband_frozen_cycle_20ep_seed42", "primary balanced baseline"),
    ("v0.53", "senssmarttech_v053_wavelet_frozen_cycle_20ep_seed42", "wavelet loss candidate"),
    ("v0.54", "senssmarttech_v054_wavelet_coeff_frozen_cycle_20ep_seed42", "negative wavelet decoder"),
    ("v0.55", "senssmarttech_v055_encoder_wide_wavelet_frozen_cycle_20ep_seed42", "encoder capacity ablation"),
    ("v0.56", "senssmarttech_v056_highfreq_decoder_wide_encoder_20ep_seed42", "high-frequency decoder ablation"),
    ("v0.57", "senssmarttech_v057_residual_highfreq_decoder_wide_encoder_20ep_seed42", "residual control"),
    ("v0.58", "senssmarttech_v058_multiband_qrs_amplitude_frozen_cycle_20ep_seed42", "over-weighted QRS control"),
    ("v0.59", "senssmarttech_v059_multiband_qrs_amplitude_lowweight_20ep_seed42", "low-weight QRS candidate"),
    ("v0.60", "senssmarttech_v060_multiband_high_qrs_amplitude_20ep_seed42", "high-band amplitude candidate"),
    ("v0.61", "senssmarttech_v061_vae_multiband_transfer_latent128_20ep_seed42", "VAE latent transfer + multi-band decoder"),
    ("v0.62", "senssmarttech_v062_vae_multiband_latent256_20ep_seed42", "latent capacity ablation (256)"),
    ("v0.63", "senssmarttech_v063_vae_multiband_qrs_peak_20ep_seed42", "VAE multi-band + light QRS/peak supervision"),
    ("v0.64", "senssmarttech_v064_vae_multiband_latent256_transfer_20ep_seed42", "latent 256 overlap-transfer ablation"),
    ("v0.67-control", "senssmarttech_v067_control_v061_protocol_20ep_seed42", "v0.61 protocol control"),
    ("v0.67-gated", "senssmarttech_v067_gated_highskip_bandloss_20ep_seed42", "gated high-resolution skip"),
    ("v0.67-residual", "senssmarttech_v067_residual_highskip_20ep_seed42", "zero-started residual high skip"),
    ("v0.68-PatchGAN", "senssmarttech_v068_patchgan_20ep_seed42", "conditional PatchGAN ablation"),
    ("CardioGAN*", "senssmarttech_cardiogan_repro_20ep_seed42", "paper-mechanism adaptation (GAN)"),
    ("RDDM*", "senssmarttech_rddm_repro_20ep_seed42", "paper-mechanism adaptation (diffusion)"),
]

DIAGNOSTIC_IDS = {"v0.41"}
PAPER_ADAPTATION_IDS = {"CardioGAN*", "RDDM*"}


def load_json(path: Path) -> dict | None:
    if not path.exists():
        return None
    with path.open("r", encoding="utf-8") as handle:
        return json.load(handle)


def nested(obj: dict | None, *keys: str):
    value = obj
    for key in keys:
        if not isinstance(value, dict) or key not in value:
            return None
        value = value[key]
    return value


def mean_std(obj: dict | None, *keys: str) -> tuple[float | None, float | None]:
    value = nested(obj, *keys)
    if not isinstance(value, dict) or "mean" not in value:
        return None, None
    mean = value.get("mean")
    std = value.get("std", 0.0)
    if not isinstance(mean, (int, float)) or not math.isfinite(float(mean)):
        return None, None
    return float(mean), float(std) if isinstance(std, (int, float)) else 0.0


def scalar(obj: dict | None, *keys: str) -> float | None:
    value = nested(obj, *keys)
    if isinstance(value, bool):
        return float(value)
    if isinstance(value, (int, float)) and math.isfinite(float(value)):
        return float(value)
    return None


def fmt(value: float | None, digits: int = 3) -> str:
    if value is None or not math.isfinite(value):
        return "n/a"
    return f"{value:.{digits}f}"


def fmt_stat(obj: dict | None, keys: tuple[str, ...], digits: int = 3,
             scale: float = 1.0) -> str:
    mean, std = mean_std(obj, *keys)
    if mean is None:
        return "n/a"
    return f"{mean * scale:.{digits}f} +/- {std * abs(scale):.{digits}f}"


def stat_mean(obj: dict | None, keys: tuple[str, ...], scale: float = 1.0) -> float | None:
    value, _ = mean_std(obj, *keys)
    return value * scale if value is not None else None


def gap(test: dict | None, train: dict | None,
        keys: tuple[str, ...], scale: float = 1.0) -> float | None:
    test_value = stat_mean(test, keys, scale)
    train_value = stat_mean(train, keys, scale)
    if test_value is None or train_value is None:
        return None
    return (test_value - train_value) / (abs(train_value) + 1e-8) * 100.0


def pair(obj: dict | None, activity: str, keys: tuple[str, ...],
         scale: float = 1.0) -> str:
    section = nested(obj, "model_by_activity", activity)
    return fmt_stat(section, keys, 2, scale)


def _summary_entry(value: float, std: float = 0.0, n_subjects: int = 0) -> dict:
    return {"mean": value, "std": std, "n_subjects": n_subjects}


def _v041_archived_model(split: str) -> dict:
    """Restore the valid ECG->ECG diagnostic summary.

    The generic PPG evaluator cannot evaluate v0.41 because that checkpoint
    consumes ECG as its input. Its correct values are archived in
    docs/v0.41_ecg_autoencoder_debug_20ep.md.
    """
    values = {
        "train": {"rmse": 0.0663, "mae": 0.0366, "pcc": 0.9971,
                  "hr": 1.12, "f1": 0.9840, "peak": None, "qrs": 1.39},
        "test": {"rmse": 0.1501, "mae": 0.0860, "pcc": 0.9832,
                 "hr": 1.63, "f1": 0.9879, "peak": 1.89, "qrs": 3.76},
    }[split]
    wave = {"macro": {
        "rmse/macro": _summary_entry(values["rmse"]),
        "mae/macro": _summary_entry(values["mae"]),
        "pcc/macro": _summary_entry(values["pcc"]),
    }}
    phys = {
        "hr_err_bpm": _summary_entry(values["hr"]),
        "rpeak_f1": _summary_entry(values["f1"]),
        "rpeak_time_err_ms": (_summary_entry(values["peak"])
                              if values["peak"] is not None else {}),
        "qrs_width_err_ms": _summary_entry(values["qrs"]),
    }
    return {"validity_rate": 1.0, "waveform": wave, "physiology": phys}


def training_last(run_dir: Path) -> dict:
    history = load_json(run_dir / "training_history.json")
    if not isinstance(history, list) or not history:
        return {}
    item = history[-1]
    if not isinstance(item, dict):
        return {}
    return {
        "train": item.get("train", {}) if isinstance(item.get("train"), dict) else {},
        "val": item.get("val", {}) if isinstance(item.get("val"), dict) else {},
    }


def row_data(runs_root: Path) -> list[dict]:
    rows = []
    for exp_id, run_name, role in RUNS:
        run_dir = runs_root / run_name
        test = load_json(run_dir / "eval_test.json")
        train = load_json(run_dir / "eval_train.json")
        if test is None:
            continue
        if exp_id == "v0.41":
            # The generic evaluator receives PPG for this ECG->ECG checkpoint
            # and would otherwise produce an invalid diagnostic summary.
            test["model"] = _v041_archived_model("test")
            test["model_by_activity"] = {}
            test["efficiency"] = {}
            if train is not None:
                train["model"] = _v041_archived_model("train")
        rows.append({
            "id": exp_id,
            "run": run_name,
            "role": role,
            "dir": run_dir,
            "test": test,
            "train": train,
            "val": load_json(run_dir / "eval_val.json"),
            "model": test.get("model", {}),
            "efficiency": test.get("efficiency", {}),
            "history": training_last(run_dir),
        })
    return rows


def table(headers: list[str], rows: list[list[str]]) -> str:
    lines = ["| " + " | ".join(headers) + " |",
             "|" + "|".join("---" for _ in headers) + "|"]
    lines.extend("| " + " | ".join(row) + " |" for row in rows)
    return "\n".join(lines)


def build_report(runs_root: Path) -> str:
    rows = row_data(runs_root)
    generated = "2026-08-24"
    lines = [
        "# SensSmartTech Full Experiment Results",
        "",
        f"> Generated from `eval_train.json`, `eval_val.json` when available, "
        f"and `eval_test.json` on {generated}. "
        "All values are subject-level mean +/- standard deviation.",
        "> The test split contains five unseen subjects; the train split contains "
        "22 seen subjects and the validation split contains five subjects. "
        "`v0.41` is an ECG->ECG diagnostic and is excluded from "
        "PPG2ECG rankings.",
        "",
        "## Protocol comparability",
        "",
        "The primary rows are 4 PPG -> 4 ECG under subject-wise 22/5/5. "
        "Train is an in-sample fit reference; it is not a generalization score.",
        "Existing Lead II values are output slices from four-PPG/four-ECG "
        "models, not retrained single-to-single models.",
        "Sample-wise/record-wise rows allow subject overlap and are leakage "
        "upper bounds, excluded from the main ranking. See "
        "`docs/evaluation_protocol_and_literature.md` for the channel/split "
        "matrix and published single-to-single references.",
        "",
        "## Metric registry",
        "",
        table(
            ["Layer", "Metric", "Unit", "Direction", "Recorded status", "Interpretation"],
            [
                ["Waveform", "RMSE / MAE", "normalized", "lower", "recorded", "global pointwise error"],
                ["Waveform", "PCC", "unitless", "higher", "recorded", "shape correlation"],
                ["Waveform", "NRMSE", "unitless", "lower", "recorded", "RMSE normalized by target std"],
                ["Waveform", "SNR", "dB", "higher", "recorded", "signal power relative to error"],
                ["Waveform", "DTW proxy", "normalized", "lower", "recorded", "multi-scale time alignment"],
                ["Waveform", "RMS / energy / peak ratio", "ratio", "near 1", "recorded", "amplitude calibration"],
                ["Physiology", "HR error", "bpm", "lower", "recorded", "rhythm transfer"],
                ["Physiology", "R-peak F1 / precision / recall", "score", "higher", "recorded", "beat detection quality"],
                ["Physiology", "R-peak timing error", "ms", "lower", "recorded", "temporal alignment"],
                ["Physiology", "QRS width error", "ms", "lower", "recorded", "QRS morphology"],
                ["Physiology", "QRS amplitude ratio", "ratio", "near 1", "recorded", "matched-beat spike amplitude"],
                ["Physiology", "RMSSD absolute / relative error", "ms / %", "lower", "recorded", "short-window HRV proxy"],
                ["Physiology", "Carotid / brachial PTT", "ms / % in range", "lower / higher", "recorded", "pulse-transit consistency"],
                ["Validity", "Valid generation rate", "%", "higher", "recorded", "finite signal and plausible HR"],
                ["Generalization", "Train-test gap", "%", "lower", "recorded", "seen-subject fit vs unseen-subject degradation"],
                ["Activity", "A/B subgroup metrics", "various", "compare", "recorded", "post-hoc motion robustness"],
                ["Efficiency", "Params / inference / RTF", "M / s / x", "lower", "recorded", "deployment cost"],
                ["Distribution", "1-NNA / Coverage", "score", "0.5 / higher", "optional", "only when `--distribution` is run"],
                ["Protocol", "Absolute synchrony / R-peak timing", "ms", "auxiliary", "recorded", "paired-window diagnostic, not the main generation ranking"],
                ["Not used", "LF/HF, FID/IS, counterfactual HR", "various", "n/a", "not recorded", "window length or protocol is insufficient"],
            ],
        ),
        "",
        "## Waveform comparison",
        "",
        table(
            ["ID", "Role", "RMSE", "MAE", "PCC", "NRMSE", "SNR (dB)", "DTW", "RMS ratio", "Peak ratio"],
            [[
                row["id"], row["role"],
                fmt_stat(row["model"], ("waveform", "macro", "rmse/macro"), 3),
                fmt_stat(row["model"], ("waveform", "macro", "mae/macro"), 3),
                fmt_stat(row["model"], ("waveform", "macro", "pcc/macro"), 3),
                fmt_stat(row["model"], ("waveform", "macro", "nrmse/macro"), 3),
                fmt_stat(row["model"], ("waveform", "macro", "snr_db/macro"), 2),
                fmt_stat(row["model"], ("waveform", "macro", "dtw/macro"), 3),
                fmt_stat(row["model"], ("waveform", "macro", "rms_ratio/macro"), 3),
                fmt_stat(row["model"], ("waveform", "macro", "peak_abs_ratio/macro"), 3),
            ] for row in rows],
        ),
        "",
        "## Physiological comparison",
        "",
        table(
            ["ID", "HR err (bpm)", "R F1", "Precision", "Recall", "Peak err (ms)", "QRS err (ms)", "QRS amp ratio", "RMSSD err (ms)", "RMSSD rel (%)", "Carotid PTT err (ms)", "Carotid in range (%)", "Brachial PTT err (ms)", "Brachial in range (%)", "Valid (%)"],
            [[
                row["id"],
                fmt_stat(row["model"], ("physiology", "hr_err_bpm"), 2),
                fmt_stat(row["model"], ("physiology", "rpeak_f1"), 3),
                fmt_stat(row["model"], ("physiology", "rpeak_precision"), 3),
                fmt_stat(row["model"], ("physiology", "rpeak_recall"), 3),
                fmt_stat(row["model"], ("physiology", "rpeak_time_err_ms"), 2),
                fmt_stat(row["model"], ("physiology", "qrs_width_err_ms"), 2),
                fmt_stat(row["model"], ("physiology", "qrs_amp_ratio"), 3),
                fmt_stat(row["model"], ("physiology", "rmssd_err_ms"), 2),
                fmt_stat(row["model"], ("physiology", "rmssd_rel_err_pct"), 1),
                fmt_stat(row["model"], ("physiology", "ptt_carotid_err_s"), 2, 1000.0),
                fmt_stat(row["model"], ("physiology", "ptt_carotid_in_range"), 1, 100.0),
                fmt_stat(row["model"], ("physiology", "ptt_brachial_err_s"), 2, 1000.0),
                fmt_stat(row["model"], ("physiology", "ptt_brachial_in_range"), 1, 100.0),
                fmt(scalar(row["model"], "validity_rate"), 1),
            ] for row in rows],
        ),
        "",
        "## Validation (checkpoint selection)",
        "",
        "> Validation metrics are shown only when `eval_val.json` has been generated. "
        "They are not final test results.",
        "",
        table(
            ["ID", "RMSE", "PCC", "HR err (bpm)", "R-peak F1", "QRS width (ms)"],
            [[
                row["id"],
                fmt_stat(row["val"].get("model", {}) if row.get("val") else None,
                         ("waveform", "macro", "rmse/macro"), 3),
                fmt_stat(row["val"].get("model", {}) if row.get("val") else None,
                         ("waveform", "macro", "pcc/macro"), 3),
                fmt_stat(row["val"].get("model", {}) if row.get("val") else None,
                         ("physiology", "hr_err_bpm"), 2),
                fmt_stat(row["val"].get("model", {}) if row.get("val") else None,
                         ("physiology", "rpeak_f1"), 3),
                fmt_stat(row["val"].get("model", {}) if row.get("val") else None,
                         ("physiology", "qrs_width_err_ms"), 2),
            ] for row in rows if row.get("val") is not None],
        ),
        "",
        "## Train-test generalization",
        "",
        "> Gap is `(test - train) / |train| * 100%`. It is a practical split gap, "
        "not a causal estimate; lower error gaps and smaller absolute morphology "
        "gaps indicate better transfer.",
        "",
        table(
            ["ID", "Train RMSE", "Test RMSE", "RMSE gap", "Train PCC", "Test PCC", "PCC gap", "Train HR err", "Test HR err", "HR gap", "Train F1", "Test F1", "F1 gap", "Train QRS", "Test QRS", "QRS gap"],
            [[
                row["id"],
                fmt_stat(row["train"].get("model", {}), ("waveform", "macro", "rmse/macro"), 3),
                fmt_stat(row["model"], ("waveform", "macro", "rmse/macro"), 3),
                f"{fmt(gap(row['test'].get('model', {}), row['train'].get('model', {}), ('waveform', 'macro', 'rmse/macro')), 1)}%",
                fmt_stat(row["train"].get("model", {}), ("waveform", "macro", "pcc/macro"), 3),
                fmt_stat(row["model"], ("waveform", "macro", "pcc/macro"), 3),
                f"{fmt(gap(row['test'].get('model', {}), row['train'].get('model', {}), ('waveform', 'macro', 'pcc/macro')), 1)}%",
                fmt_stat(row["train"].get("model", {}), ("physiology", "hr_err_bpm"), 2),
                fmt_stat(row["model"], ("physiology", "hr_err_bpm"), 2),
                f"{fmt(gap(row['test'].get('model', {}), row['train'].get('model', {}), ('physiology', 'hr_err_bpm')), 1)}%",
                fmt_stat(row["train"].get("model", {}), ("physiology", "rpeak_f1"), 3),
                fmt_stat(row["model"], ("physiology", "rpeak_f1"), 3),
                f"{fmt(gap(row['test'].get('model', {}), row['train'].get('model', {}), ('physiology', 'rpeak_f1')), 1)}%",
                fmt_stat(row["train"].get("model", {}), ("physiology", "qrs_width_err_ms"), 2),
                fmt_stat(row["model"], ("physiology", "qrs_width_err_ms"), 2),
                f"{fmt(gap(row['test'].get('model', {}), row['train'].get('model', {}), ('physiology', 'qrs_width_err_ms')), 1)}%",
            ] for row in rows],
        ),
        "",
        "## Activity robustness (post-hoc)",
        "",
        "> A/B activity labels are never model inputs. Each cell is `A / B` and is computed on the test split; differences are descriptive rather than a new training condition.",
        "",
        table(
            ["ID", "RMSE A / B", "PCC A / B", "HR err A / B (bpm)", "R F1 A / B", "QRS err A / B (ms)"],
            [[
                row["id"],
                "A: " + pair(row["test"], "A", ("waveform", "macro", "rmse/macro")) + "; B: " + pair(row["test"], "B", ("waveform", "macro", "rmse/macro")),
                "A: " + pair(row["test"], "A", ("waveform", "macro", "pcc/macro")) + "; B: " + pair(row["test"], "B", ("waveform", "macro", "pcc/macro")),
                "A: " + pair(row["test"], "A", ("physiology", "hr_err_bpm")) + "; B: " + pair(row["test"], "B", ("physiology", "hr_err_bpm")),
                "A: " + pair(row["test"], "A", ("physiology", "rpeak_f1")) + "; B: " + pair(row["test"], "B", ("physiology", "rpeak_f1")),
                "A: " + pair(row["test"], "A", ("physiology", "qrs_width_err_ms")) + "; B: " + pair(row["test"], "B", ("physiology", "qrs_width_err_ms")),
            ] for row in rows],
        ),
        "",
        "## Efficiency and training diagnostics",
        "",
        table(
            ["ID", "Params (M)", "Inference (s)", "RTF", "Device", "Final train subject acc", "Final train subject loss", "GRL lambda", "IRM aux", "KL content", "Flow loss", "PatchGAN D", "PatchGAN G"],
            [[
                row["id"],
                fmt(scalar(row["efficiency"], "params_M"), 3),
                fmt(scalar(row["efficiency"], "inference_time_s"), 3),
                fmt(scalar(row["efficiency"], "rtf"), 5),
                str(row["efficiency"].get("device", "n/a")),
                fmt(row["history"].get("train", {}).get("subject_acc"), 3),
                fmt(row["history"].get("train", {}).get("subject_loss"), 3),
                fmt(row["history"].get("train", {}).get("grl_lambda"), 3),
                fmt(row["history"].get("train", {}).get("irm_aux"), 4),
                fmt(row["history"].get("train", {}).get("kl_content"), 4),
                fmt(row["history"].get("train", {}).get("flow_nll"), 4),
                fmt(row["history"].get("train", {}).get("patchgan_d"), 4),
                fmt(row["history"].get("train", {}).get("patchgan_g"), 4),
            ] for row in rows],
        ),
        "",
        "## What is currently useful",
        "",
    ]

    deployable = [
        row for row in rows
        if row["id"] not in DIAGNOSTIC_IDS
        and row["id"] not in PAPER_ADAPTATION_IDS
    ]
    if deployable:
        def best(metric: tuple[str, ...], reverse: bool, label: str) -> str:
            valid = [(stat_mean(row["model"], metric), row["id"]) for row in deployable]
            valid = [(value, exp_id) for value, exp_id in valid if value is not None]
            if not valid:
                return f"- {label}: n/a"
            value, exp_id = (max if reverse else min)(valid)
            return f"- {label}: {exp_id} ({value:.3f})"

        lines.extend([
            best(("waveform", "macro", "rmse/macro"), False, "Best deployable RMSE"),
            best(("waveform", "macro", "pcc/macro"), True, "Best deployable PCC"),
            best(("physiology", "rpeak_f1"), True, "Best deployable R-peak F1"),
            best(("physiology", "qrs_width_err_ms"), False, "Best deployable QRS width error"),
            best(("physiology", "hr_err_bpm"), False, "Best deployable HR error"),
            "- Recommended reference: v0.52 remains the balanced baseline because it jointly preserves R-peak F1, QRS width, HR error, and global waveform quality.",
            "- v0.53 is the strongest wavelet-loss candidate for global morphology; v0.60 is the controlled high-band amplitude candidate.",
            "- v0.61 is the strongest VAE-to-multi-band decoder transplant: it retains high R-peak F1 while improving PCC over the original v0.2 path.",
            "- v0.62 is a useful negative capacity result: latent 256 lowers RMSE but worsens peak timing, QRS width, amplitude, and HR transfer.",
            "- v0.63 is a negative light-supervision control: small fused-output QRS-amplitude and differentiable peak-interval terms do not improve the held-out peak/QRS metrics under the current schedule.",
            "- v0.64 shows that overlap-transfer initialization rescues much of the latent-256 regression: RMSE, PCC, HR, and peak metrics improve over v0.62, but QRS width and R-peak F1 remain behind v0.61.",
            "- v0.67-gated is a small high-resolution skip gain over its corrected v0.61-protocol control: test RMSE/PCC/QRS width improve, but HR error is slightly worse; the zero-started residual follow-up does not reproduce the full gain.",
            "- v0.68-PatchGAN is a negative adversarial control: the discriminator saturates early and R-peak F1/QRS width collapse on both train and test despite competitive global RMSE.",
            "- v0.41 is a capacity/data diagnostic only: its ECG->ECG result cannot be compared as a PPG2ECG method.",
            "- CardioGAN* and RDDM* are retained as external paper-mechanism adaptations; they are excluded from the v0.x recommendation bullets because their channel mapping and paper protocol differ.",
        ])
    lines.extend([
        "",
        "## Not yet part of the main ranking",
        "",
        "- `LF/HF`: an 8-second window is too short for reliable HRV frequency-domain estimation.",
        "- `FID/IS`: these require a validated ECG feature encoder; raw 8000-dimensional nearest-neighbor scores are optional diagnostics only.",
        "- Counterfactual stability and adversary accuracy are recorded only when the relevant training/evaluation path exists; they are not silently inferred from waveform scores.",
        "- Absolute PPG-ECG/R-peak synchrony remains in the JSON files for paired reconstruction and PTT analysis, but is auxiliary for independent ECG generation or downstream diagnosis.",
        "",
        "## Reproduction",
        "",
        "```powershell",
        "D:\\Anaconda\\envs\\cuda126_env\\python.exe scripts\\evaluate.py --run <run> --split test",
        "D:\\Anaconda\\envs\\cuda126_env\\python.exe scripts\\evaluate.py --run <run> --split train",
        "D:\\Anaconda\\envs\\cuda126_env\\python.exe scripts\\evaluate.py --run <run> --split val",
        "D:\\Anaconda\\envs\\cuda126_env\\python.exe scripts\\build_experiment_matrix.py",
        "```",
    ])
    return "\n".join(lines) + "\n"


def main() -> None:
    parser = argparse.ArgumentParser(description="Build full experiment comparison markdown")
    parser.add_argument("--runs-root", type=Path, default=Path("runs"))
    parser.add_argument("--out", type=Path, default=Path("docs/experiment_results_full.md"))
    args = parser.parse_args()
    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(build_report(args.runs_root), encoding="utf-8")
    print(f"Wrote {args.out}")


if __name__ == "__main__":
    main()
