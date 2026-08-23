#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
Preprocess the SensSmartTech dataset for multi-lead PPG-to-ECG modeling.

Input (CSV mode, default):
    <root>/CSV/*_ecg.csv        columns: t, lead_I, lead_II, v3, v4
    <root>/CSV/*_ppg.csv        columns: t, carotid_880nm, carotid_660nm,
                                          brachial_880nm, brachial_660nm
    <root>/Demographics.csv     (optional)

Output:
    train.npz / val.npz / test.npz
    train_metadata.csv / val_metadata.csv / test_metadata.csv
    normalization.json
    split_subjects.json

Each NPZ contains:
    x: float32 [N, 4, T]  (PPG)
    y: float32 [N, 4, T]  (normalized ECG)
    ppg_channels
    ecg_channels
    fs
"""

from __future__ import annotations

import argparse
import json
import math
from pathlib import Path
from typing import Dict, Iterable, List, Tuple

import numpy as np
import pandas as pd
from scipy.signal import butter, sosfiltfilt
from tqdm import tqdm


ECG_CHANNELS = ["I", "II", "V3", "V4"]
PPG_CHANNELS = [
    "carotid_880nm",
    "carotid_660nm",
    "brachial_880nm",
    "brachial_660nm",
]

# CSV 列名别名：官方 CSV 用 lead_I / v3 等小写形式，规范化到脚本内部通道名。
ECG_COLUMN_ALIASES: Dict[str, List[str]] = {
    "I": ["I", "lead_I", "lead_i"],
    "II": ["II", "lead_II", "lead_ii"],
    "V3": ["V3", "v3"],
    "V4": ["V4", "v4"],
}
PPG_COLUMN_ALIASES: Dict[str, List[str]] = {
    name: [name] for name in PPG_CHANNELS
}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--root",
        type=Path,
        required=True,
        help="SensSmartTech root directory containing WFDB and Demographics.csv",
    )
    parser.add_argument(
        "--out",
        type=Path,
        required=True,
        help="Output directory",
    )
    parser.add_argument("--target-fs", type=int, default=250)
    parser.add_argument("--window-sec", type=float, default=8.0)
    parser.add_argument("--stride-sec", type=float, default=4.0)
    parser.add_argument(
        "--filter-mode",
        choices=["clean", "none"],
        default="clean",
        help=(
            "clean: PPG 0.3-12 Hz and ECG 0.5-45 Hz after resampling; "
            "none: no extra band-pass filtering"
        ),
    )
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument(
        "--clip-ppg",
        type=float,
        default=10.0,
        help="Clip robustly normalized PPG to +/- this value; <=0 disables clipping",
    )
    return parser.parse_args()


def locate_csv_dir(root: Path) -> Path:
    """Locate the CSV directory under the dataset root."""
    candidates = [
        root / "CSV",
        root / "data" / "CSV",
    ]
    for candidate in candidates:
        if candidate.is_dir():
            return candidate

    matches = [p for p in root.rglob("CSV") if p.is_dir()]
    if len(matches) == 1:
        return matches[0]
    if not matches:
        raise FileNotFoundError(f"Cannot find a CSV directory under: {root}")
    raise RuntimeError(
        "Multiple CSV directories found. Point --root to the dataset directory: "
        + ", ".join(str(p) for p in matches)
    )


def locate_demographics(root: Path) -> Path | None:
    candidates = [
        root / "Demographics.csv",
        root / "data" / "Demographics.csv",
    ]
    for candidate in candidates:
        if candidate.is_file():
            return candidate

    matches = list(root.rglob("Demographics.csv"))
    if len(matches) == 1:
        return matches[0]
    return None


def resolve_column(df_columns: List[str], wanted: str, aliases: List[str]) -> str:
    """Return the actual column name in df_columns matching wanted or an alias."""
    for name in [wanted] + list(aliases):
        if name in df_columns:
            return name
    raise ValueError(
        f"Column for channel '{wanted}' not found. "
        f"Tried {aliases}. Available columns: {df_columns}"
    )


def read_csv_signal(
    path: Path,
    wanted_channels: List[str],
    aliases: Dict[str, List[str]],
) -> Tuple[np.ndarray, np.ndarray, float]:
    """
    Read a SensSmartTech CSV signal file.

    Returns:
        t:    [N] float64 time vector in seconds.
        sig:  [N, C] float64 signal matrix ordered as `wanted_channels`.
        fs:   estimated sample rate in Hz, derived from the median dt.
    """
    df = pd.read_csv(path)
    cols = list(df.columns)

    t = df["t"].to_numpy(dtype=np.float64)
    col_names = [resolve_column(cols, w, aliases.get(w, [])) for w in wanted_channels]
    sig = df[col_names].to_numpy(dtype=np.float64)

    dt = np.diff(t)
    dt = dt[np.isfinite(dt) & (dt > 0)]
    if dt.size == 0:
        raise ValueError(f"No valid time deltas in {path}")
    fs = 1.0 / float(np.median(dt))
    return t, sig, fs


def align_and_resample(
    t_ppg: np.ndarray,
    ppg: np.ndarray,
    fs_ppg: float,
    t_ecg: np.ndarray,
    ecg: np.ndarray,
    fs_ecg: float,
    target_fs: int,
) -> Tuple[np.ndarray, np.ndarray]:
    """
    Resample PPG and ECG onto a common time grid at target_fs.

    Both signals may have different sample rates and start times; we align them
    to the overlapping time interval using linear interpolation. Linear
    interpolation is sufficient here because downstream band-pass filtering
    removes out-of-band mirror images introduced by resampling.
    """
    t_start = max(float(t_ppg[0]), float(t_ecg[0]))
    t_end = min(float(t_ppg[-1]), float(t_ecg[-1]))
    if t_end <= t_start:
        raise ValueError("PPG and ECG recordings do not overlap in time.")

    n_common = int(math.floor((t_end - t_start) * target_fs))
    if n_common <= 1:
        raise ValueError("Overlapping interval too short after resampling.")

    t_grid = t_start + np.arange(n_common, dtype=np.float64) / target_fs

    ppg_out = np.empty((n_common, ppg.shape[1]), dtype=np.float64)
    ecg_out = np.empty((n_common, ecg.shape[1]), dtype=np.float64)
    for c in range(ppg.shape[1]):
        ppg_out[:, c] = np.interp(t_grid, t_ppg, ppg[:, c])
    for c in range(ecg.shape[1]):
        ecg_out[:, c] = np.interp(t_grid, t_ecg, ecg[:, c])
    return ppg_out, ecg_out


def load_demographics(path: Path | None) -> Dict[str, dict]:
    """
    The first line of the official CSV is a table title; the second line is the
    actual header. Only the first 14 columns contain data.
    """
    if path is None:
        return {}

    df = pd.read_csv(path, header=1, usecols=range(14))
    df = df[df["File number"].notna()].copy()

    metadata: Dict[str, dict] = {}
    activity_col = "Before (B)  / after (A) activity"
    hr_col = "Median heart rate (bpm)"

    for _, row in df.iterrows():
        ecg_name = str(row["ECG"]).strip()
        metadata[ecg_name] = {
            "subject_id": int(row["Subject number"]),
            "gender": str(row["Gender"]).strip(),
            "age": float(row["Age (year)"]),
            "height_cm": float(row["Height (cm)"]),
            "weight_kg": float(row["Weight (kg)"]),
            "bmi": float(row["Body-mass index"]),
            "activity": str(row[activity_col]).strip(),
            "heart_rate_bpm": float(row[hr_col]),
        }
    return metadata


def bandpass(
    x: np.ndarray,
    fs: float,
    low_hz: float,
    high_hz: float,
    order: int = 4,
) -> np.ndarray:
    nyquist = 0.5 * fs
    if not (0 < low_hz < high_hz < nyquist):
        raise ValueError(
            f"Invalid band-pass range ({low_hz}, {high_hz}) for fs={fs}"
        )
    sos = butter(
        order,
        [low_hz / nyquist, high_hz / nyquist],
        btype="bandpass",
        output="sos",
    )
    return sosfiltfilt(sos, x, axis=0)


def robust_normalize_ppg(
    x: np.ndarray,
    clip_value: float,
) -> np.ndarray:
    """
    Normalize each PPG channel per 30-second recording. This removes sensor
    contact/gain scale while retaining temporal morphology.
    """
    median = np.nanmedian(x, axis=0, keepdims=True)
    q25 = np.nanpercentile(x, 25, axis=0, keepdims=True)
    q75 = np.nanpercentile(x, 75, axis=0, keepdims=True)
    scale = (q75 - q25) / 1.349

    fallback = np.nanstd(x, axis=0, keepdims=True)
    scale = np.where(scale > 1e-8, scale, fallback)
    scale = np.where(scale > 1e-8, scale, 1.0)

    z = (x - median) / scale
    if clip_value > 0:
        z = np.clip(z, -clip_value, clip_value)
    return z


def valid_window(ppg: np.ndarray, ecg: np.ndarray) -> bool:
    if not (np.isfinite(ppg).all() and np.isfinite(ecg).all()):
        return False

    # Reject flat or almost-flat channels.
    ppg_range = np.percentile(ppg, 99, axis=0) - np.percentile(ppg, 1, axis=0)
    ecg_range = np.percentile(ecg, 99, axis=0) - np.percentile(ecg, 1, axis=0)
    if np.any(ppg_range < 1e-6) or np.any(ecg_range < 1e-6):
        return False

    return True


def make_subject_splits(
    subject_ids: Iterable[int],
    seed: int,
) -> Tuple[Dict[int, str], dict]:
    subjects = np.array(sorted(set(int(x) for x in subject_ids)), dtype=int)
    if len(subjects) < 3:
        raise ValueError("At least three subjects are required for train/val/test.")

    rng = np.random.default_rng(seed)
    rng.shuffle(subjects)

    n_total = len(subjects)
    n_test = max(1, int(round(0.15 * n_total)))
    n_val = max(1, int(round(0.15 * n_total)))
    n_train = n_total - n_val - n_test
    if n_train < 1:
        raise ValueError("Not enough subjects after split.")

    train_subjects = sorted(subjects[:n_train].tolist())
    val_subjects = sorted(subjects[n_train:n_train + n_val].tolist())
    test_subjects = sorted(subjects[n_train + n_val:].tolist())

    split_map = {sid: "train" for sid in train_subjects}
    split_map.update({sid: "val" for sid in val_subjects})
    split_map.update({sid: "test" for sid in test_subjects})

    split_info = {
        "seed": seed,
        "train": train_subjects,
        "val": val_subjects,
        "test": test_subjects,
    }
    return split_map, split_info


def main() -> None:
    args = parse_args()
    args.root = args.root.expanduser().resolve()
    args.out = args.out.expanduser().resolve()
    args.out.mkdir(parents=True, exist_ok=True)

    csv_dir = locate_csv_dir(args.root)
    demographics_path = locate_demographics(args.root)
    demographics = load_demographics(demographics_path)

    ecg_files = sorted(csv_dir.glob("*_ecg.csv"))
    if not ecg_files:
        raise FileNotFoundError(f"No *_ecg.csv files found under {csv_dir}")

    records = []
    for ecg_file in ecg_files:
        ecg_name = ecg_file.stem  # e.g. "1_10-09-54_ecg"
        ppg_name = ecg_name.removesuffix("_ecg") + "_ppg"
        ppg_file = ecg_file.with_name(ppg_name + ".csv")

        if not ppg_file.is_file():
            print(f"[warning] Skip incomplete pair: {ecg_name}")
            continue

        if ecg_name in demographics:
            subject_id = int(demographics[ecg_name]["subject_id"])
        else:
            subject_id = int(ecg_name.split("_", 1)[0])

        records.append(
            {
                "subject_id": subject_id,
                "ecg_name": ecg_name,
                "ppg_name": ppg_name,
                "ecg_path": ecg_file,
                "ppg_path": ppg_file,
            }
        )

    if not records:
        raise RuntimeError("No complete ECG/PPG record pairs were found.")

    split_map, split_info = make_subject_splits(
        [r["subject_id"] for r in records],
        args.seed,
    )

    buffers = {
        split: {"x": [], "y": [], "meta": []}
        for split in ("train", "val", "test")
    }

    window_samples = int(round(args.window_sec * args.target_fs))
    stride_samples = int(round(args.stride_sec * args.target_fs))
    if window_samples <= 0 or stride_samples <= 0:
        raise ValueError("Window and stride must be positive.")

    skipped_records = 0
    rejected_windows = 0

    for item in tqdm(records, desc="Preprocessing records"):
        try:
            t_ecg, ecg, fs_ecg = read_csv_signal(
                item["ecg_path"], ECG_CHANNELS, ECG_COLUMN_ALIASES
            )
            t_ppg, ppg, fs_ppg = read_csv_signal(
                item["ppg_path"], PPG_CHANNELS, PPG_COLUMN_ALIASES
            )

            # Resample both signals onto a common time grid at target_fs.
            # This handles different sample rates (PPG ~100Hz, ECG ~500Hz)
            # and different start times in one step.
            ppg, ecg = align_and_resample(
                t_ppg, ppg, fs_ppg,
                t_ecg, ecg, fs_ecg,
                args.target_fs,
            )
            common_length = min(len(ecg), len(ppg))
            ecg = ecg[:common_length]
            ppg = ppg[:common_length]

            if args.filter_mode == "clean":
                # These are intentionally mild model-oriented filters.
                ppg = bandpass(ppg, args.target_fs, 0.3, 12.0)
                ecg = bandpass(ecg, args.target_fs, 0.5, 45.0)

            # PPG scale depends strongly on optical coupling and sensor contact.
            ppg = robust_normalize_ppg(ppg, args.clip_ppg)

            split = split_map[item["subject_id"]]
            record_meta = demographics.get(item["ecg_name"], {})

            for start in range(
                0,
                common_length - window_samples + 1,
                stride_samples,
            ):
                end = start + window_samples
                x_window = ppg[start:end]
                y_window = ecg[start:end]

                if not valid_window(x_window, y_window):
                    rejected_windows += 1
                    continue

                buffers[split]["x"].append(x_window.T.astype(np.float32))
                buffers[split]["y"].append(y_window.T.astype(np.float32))
                buffers[split]["meta"].append(
                    {
                        "subject_id": item["subject_id"],
                        "record_id": item["ecg_name"].removesuffix("_ecg"),
                        "ecg_record": item["ecg_name"],
                        "ppg_record": item["ppg_name"],
                        "start_sec": start / args.target_fs,
                        "end_sec": end / args.target_fs,
                        "activity": record_meta.get("activity", ""),
                        "heart_rate_bpm": record_meta.get("heart_rate_bpm", np.nan),
                        "gender": record_meta.get("gender", ""),
                        "age": record_meta.get("age", np.nan),
                    }
                )
        except Exception as exc:
            skipped_records += 1
            print(f"[warning] Skip {item['ecg_name']}: {exc}")

    for split in ("train", "val", "test"):
        if not buffers[split]["x"]:
            raise RuntimeError(f"No valid windows produced for split: {split}")

    # Compute target ECG statistics using TRAIN only. Do not normalize ECG
    # independently per window, because that would erase meaningful amplitude
    # differences across leads and recordings.
    train_y = np.stack(buffers["train"]["y"])
    ecg_mean = train_y.mean(axis=(0, 2), dtype=np.float64)
    ecg_std = train_y.std(axis=(0, 2), dtype=np.float64)
    ecg_std = np.where(ecg_std > 1e-8, ecg_std, 1.0)

    summary = {}
    for split in ("train", "val", "test"):
        x = np.stack(buffers[split]["x"]).astype(np.float32)
        y = np.stack(buffers[split]["y"]).astype(np.float32)
        y = (
            (y - ecg_mean[None, :, None]) /
            ecg_std[None, :, None]
        ).astype(np.float32)

        np.savez_compressed(
            args.out / f"{split}.npz",
            x=x,
            y=y,
            ppg_channels=np.asarray(PPG_CHANNELS),
            ecg_channels=np.asarray(ECG_CHANNELS),
            fs=np.asarray(args.target_fs),
        )
        pd.DataFrame(buffers[split]["meta"]).to_csv(
            args.out / f"{split}_metadata.csv",
            index=False,
        )

        summary[split] = {
            "windows": int(x.shape[0]),
            "x_shape": list(x.shape),
            "y_shape": list(y.shape),
        }

    normalization = {
        "target_fs": args.target_fs,
        "window_sec": args.window_sec,
        "stride_sec": args.stride_sec,
        "filter_mode": args.filter_mode,
        "ppg_normalization": "per-recording robust z-score using IQR/1.349",
        "ppg_clip": args.clip_ppg,
        "ecg_normalization": "per-lead mean/std computed from train split only",
        "ecg_mean_mV": ecg_mean.tolist(),
        "ecg_std_mV": ecg_std.tolist(),
        "ppg_channels": PPG_CHANNELS,
        "ecg_channels": ECG_CHANNELS,
    }

    with open(args.out / "normalization.json", "w", encoding="utf-8") as f:
        json.dump(normalization, f, ensure_ascii=False, indent=2)

    with open(args.out / "split_subjects.json", "w", encoding="utf-8") as f:
        json.dump(split_info, f, ensure_ascii=False, indent=2)

    with open(args.out / "summary.json", "w", encoding="utf-8") as f:
        json.dump(
            {
                "dataset_root": str(args.root),
                "csv_dir": str(csv_dir),
                "demographics": (
                    str(demographics_path) if demographics_path else None
                ),
                "skipped_records": skipped_records,
                "rejected_windows": rejected_windows,
                "splits": summary,
            },
            f,
            ensure_ascii=False,
            indent=2,
        )

    print("\nDone.")
    print(json.dumps(summary, ensure_ascii=False, indent=2))
    print(f"Output directory: {args.out}")


if __name__ == "__main__":
    main()
