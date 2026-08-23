#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
Preprocess the SensSmartTech dataset for multi-lead PPG-to-ECG modeling (v2).

Fixes over v1:
    - The CSV time column is in MILLISECONDS (v1 assumed seconds, which inflated
      every record 1000x and caused the OOM crash). The unit is now auto-detected.

Pipeline stages (decoupled so that re-splitting never re-runs preprocessing):

    1. preprocess  raw CSV -> per-subject truth store (interim/)
                   PPG: per-recording, per-channel robust z-score (split-independent)
                   ECG: filtered only, NOT normalized (stats depend on the split)
    2. split       windows_index.csv -> split JSONs (subjectwise / recordwise, seeded)
    3. materialize truth store + split + ecg-norm -> train/val/test caches

Split strategies:
    subjectwise   hold out entire subjects (main, clinically meaningful result)
    recordwise    hold out records while subjects may leak across splits
                  (sample-wise variant; window-level random splits are NOT offered
                  because 8s windows with 4s stride overlap by 50% -> leakage)

ECG normalization modes (both materialized, stats from train windows only):
    per-lead      per-lead mean/std; each lead ends up with std 1
    global        per-lead mean, single scalar std; preserves inter-lead
                  amplitude ratios (V3/V4 are larger than limb leads)

Usage:
    python scripts/preprocess_senssmarttech.py \
        --root data/raw/SensSmartTech \
        --out data/processed/senssmarttech \
        --splits subjectwise recordwise \
        --ecg-norms per-lead global \
        --seed 42
"""

from __future__ import annotations

import argparse
import json
import math
from pathlib import Path
from typing import Dict, List, Tuple

import numpy as np
import pandas as pd
from scipy.signal import butter, sosfiltfilt

ECG_CHANNELS = ["I", "II", "V3", "V4"]
PPG_CHANNELS = [
    "carotid_880nm",
    "carotid_660nm",
    "brachial_880nm",
    "brachial_660nm",
]

ECG_COLUMN_ALIASES: Dict[str, List[str]] = {
    "I": ["I", "lead_I", "lead_i"],
    "II": ["II", "lead_II", "lead_ii"],
    "V3": ["V3", "v3"],
    "V4": ["V4", "v4"],
}
PPG_COLUMN_ALIASES: Dict[str, List[str]] = {name: [name] for name in PPG_CHANNELS}

SPLIT_STRATEGIES = ("subjectwise", "recordwise")
ECG_NORM_MODES = ("per-lead", "global")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[1])
    parser.add_argument("--root", type=Path, required=True,
                        help="SensSmartTech root directory containing CSV/ and Demographics.csv")
    parser.add_argument("--out", type=Path, required=True,
                        help="Output directory for materialized split caches")
    parser.add_argument("--interim", type=Path, default=None,
                        help="Truth-store directory (default: <root>/../../interim/senssmarttech)")
    parser.add_argument("--target-fs", type=int, default=250)
    parser.add_argument("--window-sec", type=float, default=8.0)
    parser.add_argument("--stride-sec", type=float, default=4.0)
    parser.add_argument("--filter-mode", choices=["clean", "none"], default="clean",
                        help="clean: PPG 0.3-12 Hz and ECG 0.5-45 Hz after resampling")
    parser.add_argument("--clip-ppg", type=float, default=10.0,
                        help="Clip robustly normalized PPG to +/- this value; <=0 disables")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--splits", nargs="+", choices=SPLIT_STRATEGIES,
                        default=["subjectwise", "recordwise"])
    parser.add_argument("--ecg-norms", nargs="+", choices=ECG_NORM_MODES,
                        default=["per-lead", "global"])
    parser.add_argument("--stage", choices=["all", "preprocess", "split", "materialize"],
                        default="all")
    return parser.parse_args()


# --------------------------------------------------------------------------
# Raw loading
# --------------------------------------------------------------------------

def locate_csv_dir(root: Path) -> Path:
    for candidate in [root / "CSV", root / "data" / "CSV"]:
        if candidate.is_dir():
            return candidate
    matches = [p for p in root.rglob("CSV") if p.is_dir()]
    if len(matches) == 1:
        return matches[0]
    if not matches:
        raise FileNotFoundError(f"Cannot find a CSV directory under: {root}")
    raise RuntimeError("Multiple CSV directories found: " + ", ".join(map(str, matches)))


def locate_demographics(root: Path) -> Path | None:
    for candidate in [root / "Demographics.csv", root / "data" / "Demographics.csv"]:
        if candidate.is_file():
            return candidate
    matches = list(root.rglob("Demographics.csv"))
    return matches[0] if len(matches) == 1 else None


def infer_time_scale(t: np.ndarray) -> Tuple[float, float]:
    """Return (scale, median_dt) so that t_seconds = t / scale.

    The official CSVs store time in milliseconds (dt ~ 10 ms for PPG, ~2 ms for
    ECG). If the median dt is >= 0.5 it cannot be seconds for any cardiac- or
    pulse-rate signal, so it is treated as milliseconds.
    """
    dt = np.diff(t)
    dt = dt[np.isfinite(dt) & (dt > 0)]
    if dt.size == 0:
        raise ValueError("No valid time deltas.")
    med = float(np.median(dt))
    if med >= 0.5:
        return 1000.0, med
    return 1.0, med


def read_csv_signal(
    path: Path,
    wanted_channels: List[str],
    aliases: Dict[str, List[str]],
) -> Tuple[np.ndarray, np.ndarray, float]:
    """Read a SensSmartTech CSV. Returns (t_seconds [N], sig [N, C], fs_hz)."""
    df = pd.read_csv(path)
    cols = list(df.columns)

    t_raw = df["t"].to_numpy(dtype=np.float64)
    scale, _ = infer_time_scale(t_raw)
    t = t_raw / scale

    col_names = [resolve_column(cols, w, aliases.get(w, [])) for w in wanted_channels]
    sig = df[col_names].to_numpy(dtype=np.float64)

    fs = scale / np.median(np.diff(t))
    return t, sig, fs


def resolve_column(df_columns: List[str], wanted: str, aliases: List[str]) -> str:
    for name in [wanted] + list(aliases):
        if name in df_columns:
            return name
    raise ValueError(
        f"Column for channel '{wanted}' not found. "
        f"Tried {aliases}. Available columns: {df_columns}"
    )


def load_demographics(path: Path | None) -> Dict[str, dict]:
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


# --------------------------------------------------------------------------
# Signal processing
# --------------------------------------------------------------------------

def align_and_resample(
    t_ppg: np.ndarray, ppg: np.ndarray,
    t_ecg: np.ndarray, ecg: np.ndarray,
    target_fs: int,
) -> Tuple[np.ndarray, np.ndarray]:
    """Resample PPG and ECG onto a common time grid (seconds) at target_fs."""
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


def bandpass(x: np.ndarray, fs: float, low_hz: float, high_hz: float, order: int = 4) -> np.ndarray:
    nyquist = 0.5 * fs
    if not (0 < low_hz < high_hz < nyquist):
        raise ValueError(f"Invalid band-pass range ({low_hz}, {high_hz}) for fs={fs}")
    sos = butter(order, [low_hz / nyquist, high_hz / nyquist], btype="bandpass", output="sos")
    return sosfiltfilt(sos, x, axis=0)


def robust_normalize_ppg(x: np.ndarray, clip_value: float) -> np.ndarray:
    """Per-recording, per-channel robust z-score (IQR/1.349)."""
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
    ppg_range = np.percentile(ppg, 99, axis=0) - np.percentile(ppg, 1, axis=0)
    ecg_range = np.percentile(ecg, 99, axis=0) - np.percentile(ecg, 1, axis=0)
    if np.any(ppg_range < 1e-6) or np.any(ecg_range < 1e-6):
        return False
    return True


# --------------------------------------------------------------------------
# Stage 1: preprocess -> truth store
# --------------------------------------------------------------------------

def run_preprocess(args: argparse.Namespace, interim: Path) -> pd.DataFrame:
    csv_dir = locate_csv_dir(args.root)
    demographics = load_demographics(locate_demographics(args.root))
    interim.mkdir(parents=True, exist_ok=True)

    ecg_files = sorted(csv_dir.glob("*_ecg.csv"))
    if not ecg_files:
        raise FileNotFoundError(f"No *_ecg.csv files found under {csv_dir}")

    window_samples = int(round(args.window_sec * args.target_fs))
    stride_samples = int(round(args.stride_sec * args.target_fs))
    if window_samples <= 0 or stride_samples <= 0:
        raise ValueError("Window and stride must be positive.")

    records = []
    for ecg_file in ecg_files:
        ecg_name = ecg_file.stem
        ppg_file = ecg_file.with_name(ecg_name.removesuffix("_ecg") + "_ppg.csv")
        if not ppg_file.is_file():
            print(f"[warning] Skip incomplete pair: {ecg_name}")
            continue
        subject_id = (
            demographics[ecg_name]["subject_id"] if ecg_name in demographics
            else int(ecg_name.split("_", 1)[0])
        )
        records.append((subject_id, ecg_name, ecg_file, ppg_file))
    records.sort(key=lambda r: (r[0], r[1]))

    index_rows: List[dict] = []
    subject_buffers: Dict[int, dict] = {}
    skipped_records = 0
    rejected_windows = 0

    for subject_id, ecg_name, ecg_file, ppg_file in records:
        try:
            t_ecg, ecg, _ = read_csv_signal(ecg_file, ECG_CHANNELS, ECG_COLUMN_ALIASES)
            t_ppg, ppg, _ = read_csv_signal(ppg_file, PPG_CHANNELS, PPG_COLUMN_ALIASES)
            ppg, ecg = align_and_resample(t_ppg, ppg, t_ecg, ecg, args.target_fs)
            common_length = min(len(ecg), len(ppg))
            ecg = ecg[:common_length]
            ppg = ppg[:common_length]

            if args.filter_mode == "clean":
                ppg = bandpass(ppg, args.target_fs, 0.3, 12.0)
                ecg = bandpass(ecg, args.target_fs, 0.5, 45.0)

            ppg = robust_normalize_ppg(ppg, args.clip_ppg)

            meta = demographics.get(ecg_name, {})
            buf = subject_buffers.setdefault(subject_id, {"x": [], "y": [], "rows": []})

            for start in range(0, common_length - window_samples + 1, stride_samples):
                end = start + window_samples
                if not valid_window(ppg[start:end], ecg[start:end]):
                    rejected_windows += 1
                    continue
                local_idx = len(buf["x"])
                buf["x"].append(ppg[start:end].T.astype(np.float32))
                buf["y"].append(ecg[start:end].T.astype(np.float32))
                buf["rows"].append({
                    "subject_id": subject_id,
                    "record_id": ecg_name.removesuffix("_ecg"),
                    "ecg_record": ecg_name,
                    "ppg_record": ecg_name.removesuffix("_ecg") + "_ppg",
                    "start_sec": start / args.target_fs,
                    "end_sec": end / args.target_fs,
                    "activity": meta.get("activity", ""),
                    "heart_rate_bpm": meta.get("heart_rate_bpm", np.nan),
                    "gender": meta.get("gender", ""),
                    "age": meta.get("age", np.nan),
                    "local_idx": local_idx,
                })
        except Exception as exc:
            skipped_records += 1
            print(f"[warning] Skip {ecg_name}: {exc}")

    for subject_id, buf in subject_buffers.items():
        if not buf["x"]:
            continue
        np.savez_compressed(
            interim / f"subject_{subject_id:02d}.npz",
            x=np.stack(buf["x"]).astype(np.float32),
            y=np.stack(buf["y"]).astype(np.float32),
            fs=np.asarray(args.target_fs),
            ppg_channels=np.asarray(PPG_CHANNELS),
            ecg_channels=np.asarray(ECG_CHANNELS),
        )
        index_rows.extend(buf["rows"])

    index = pd.DataFrame(index_rows)
    index.to_csv(interim / "windows_index.csv", index=False)
    stats = {
        "records_total": len(records),
        "records_skipped": skipped_records,
        "windows_kept": len(index),
        "windows_rejected": rejected_windows,
        "subjects_with_windows": int(index["subject_id"].nunique()),
    }
    with open(interim / "preprocess_stats.json", "w", encoding="utf-8") as f:
        json.dump(stats, f, ensure_ascii=False, indent=2)
    print(f"[preprocess] {stats}")
    return index


# --------------------------------------------------------------------------
# Stage 2: split
# --------------------------------------------------------------------------

def make_split(ids: list, seed: int) -> dict:
    """Shuffle ids with the v1 algorithm so seed=42 reproduces the old split.

    ids may be subject ids (ints) or record ids (strings).
    """
    unique = sorted(set(ids))
    if len(unique) < 3:
        raise ValueError("At least three ids are required for train/val/test.")
    arr = np.array(unique)
    rng = np.random.default_rng(seed)
    rng.shuffle(arr)
    n_total = len(arr)
    n_test = max(1, int(round(0.15 * n_total)))
    n_val = max(1, int(round(0.15 * n_total)))
    n_train = n_total - n_val - n_test
    if n_train < 1:
        raise ValueError("Not enough ids after split.")
    return {
        "train": sorted(arr[:n_train].tolist()),
        "val": sorted(arr[n_train:n_train + n_val].tolist()),
        "test": sorted(arr[n_train + n_val:].tolist()),
    }


def run_split(args: argparse.Namespace, index: pd.DataFrame, interim: Path) -> dict:
    splits = {}
    for strategy in args.splits:
        key = "subject_id" if strategy == "subjectwise" else "record_id"
        vals = index[key].tolist()
        if strategy == "subjectwise":
            vals = [int(v) for v in vals]
        parts = make_split(vals, args.seed)
        info = {"strategy": strategy, "seed": args.seed, **parts}
        path = interim / f"split_{strategy}.json"
        with open(path, "w", encoding="utf-8") as f:
            json.dump(info, f, ensure_ascii=False, indent=2)
        splits[strategy] = info
        print(f"[split] {strategy}: train/val/test = "
              f"{len(parts['train'])}/{len(parts['val'])}/{len(parts['test'])} {key}s")
    return splits


# --------------------------------------------------------------------------
# Stage 3: materialize
# --------------------------------------------------------------------------

def load_truth_store(index: pd.DataFrame, interim: Path) -> Tuple[np.ndarray, np.ndarray]:
    """Load all windows in index order. Returns x [N,4,T], y_raw [N,4,T]."""
    subjects = sorted(index["subject_id"].unique())
    xs, ys, subject_slices = [], [], {}
    for sid in subjects:
        data = np.load(interim / f"subject_{int(sid):02d}.npz", allow_pickle=False)
        xs.append(data["x"])
        ys.append(data["y"])
        subject_slices[sid] = data["x"].shape[0]
    x = np.concatenate(xs, axis=0)
    y = np.concatenate(ys, axis=0)

    offsets = {}
    pos = 0
    for sid in subjects:
        offsets[sid] = pos
        pos += subject_slices[sid]

    global_idx = np.empty(len(index), dtype=np.int64)
    counters: Dict[int, int] = {}
    for i, row in enumerate(index.itertuples()):
        local = int(row.local_idx)
        counters[row.subject_id] = counters.get(row.subject_id, 0)
        global_idx[i] = offsets[row.subject_id] + local
    if np.unique(global_idx).size != global_idx.size:
        raise RuntimeError("Duplicate window indices detected.")
    return x[global_idx], y[global_idx]


def compute_ecg_stats(y_train: np.ndarray, mode: str) -> Tuple[np.ndarray, np.ndarray]:
    mean = y_train.mean(axis=(0, 2), dtype=np.float64)
    if mode == "per-lead":
        std = y_train.std(axis=(0, 2), dtype=np.float64)
    elif mode == "global":
        std = np.full_like(mean, float(y_train.std(dtype=np.float64)))
    else:
        raise ValueError(f"Unknown ECG norm mode: {mode}")
    std = np.where(std > 1e-8, std, 1.0)
    return mean, std


def run_materialize(args: argparse.Namespace, index: pd.DataFrame,
                    splits: dict, interim: Path) -> None:
    x, y_raw = load_truth_store(index, interim)
    args.out.mkdir(parents=True, exist_ok=True)

    summary = {"dataset_root": str(args.root), "seed": args.seed, "combos": {}}
    for strategy, info in splits.items():
        key = "subject_id" if strategy == "subjectwise" else "record_id"
        assign = {}
        for part in ("train", "val", "test"):
            for uid in info[part]:
                assign[uid] = part
        masks = {p: index[key].map(assign).to_numpy() == p for p in ("train", "val", "test")}
        if not all(masks[p].any() for p in masks):
            raise RuntimeError(f"Empty split for strategy {strategy}.")

        for mode in args.ecg_norms:
            combo_dir = args.out / f"{strategy}_{mode}"
            combo_dir.mkdir(parents=True, exist_ok=True)
            ecg_mean, ecg_std = compute_ecg_stats(y_raw[masks["train"]], mode)

            combo_summary = {}
            for part in ("train", "val", "test"):
                idx = masks[part]
                y = ((y_raw[idx] - ecg_mean[None, :, None]) / ecg_std[None, :, None]
                     ).astype(np.float32)
                np.savez_compressed(
                    combo_dir / f"{part}.npz",
                    x=x[idx].astype(np.float32),
                    y=y,
                    ppg_channels=np.asarray(PPG_CHANNELS),
                    ecg_channels=np.asarray(ECG_CHANNELS),
                    fs=np.asarray(args.target_fs),
                )
                index[idx].drop(columns=["local_idx"]).to_csv(
                    combo_dir / f"{part}_metadata.csv", index=False)
                combo_summary[part] = {
                    "windows": int(idx.sum()),
                    "subjects": int(index[idx]["subject_id"].nunique()),
                    "records": int(index[idx]["record_id"].nunique()),
                }

            normalization = {
                "split_strategy": strategy,
                "split_seed": args.seed,
                "target_fs": args.target_fs,
                "window_sec": args.window_sec,
                "stride_sec": args.stride_sec,
                "filter_mode": args.filter_mode,
                "ppg_normalization": "per-recording per-channel robust z-score (IQR/1.349)",
                "ppg_clip": args.clip_ppg,
                "ecg_normalization": {
                    "per-lead": "per-lead mean/std from train windows only",
                    "global": "per-lead mean, scalar std from train windows only "
                              "(preserves inter-lead amplitude ratios)",
                }[mode],
                "ecg_mean": ecg_mean.tolist(),
                "ecg_std": ecg_std.tolist(),
                "ppg_channels": PPG_CHANNELS,
                "ecg_channels": ECG_CHANNELS,
            }
            with open(combo_dir / "normalization.json", "w", encoding="utf-8") as f:
                json.dump(normalization, f, ensure_ascii=False, indent=2)
            with open(combo_dir / "split.json", "w", encoding="utf-8") as f:
                json.dump(info, f, ensure_ascii=False, indent=2)
            summary["combos"][f"{strategy}_{mode}"] = combo_summary
            print(f"[materialize] {strategy}_{mode}: "
                  + json.dumps(combo_summary, ensure_ascii=False))

    with open(args.out / "summary.json", "w", encoding="utf-8") as f:
        json.dump(summary, f, ensure_ascii=False, indent=2)


def main() -> None:
    args = parse_args()
    args.root = args.root.expanduser().resolve()
    args.out = args.out.expanduser().resolve()
    interim = (
        args.interim.expanduser().resolve() if args.interim
        else (args.root / ".." / ".." / "interim" / "senssmarttech").resolve()
    )
    print(f"[config] root={args.root}\n[config] interim={interim}\n[config] out={args.out}")

    index_path = interim / "windows_index.csv"
    if args.stage in ("all", "preprocess"):
        index = run_preprocess(args, interim)
    else:
        index = pd.read_csv(index_path)

    if args.stage in ("all", "split"):
        splits = run_split(args, index, interim)
    else:
        splits = {
            s: json.loads((interim / f"split_{s}.json").read_text(encoding="utf-8"))
            for s in args.splits
        }

    if args.stage in ("all", "materialize"):
        run_materialize(args, index, splits, interim)

    print("\nDone.")


if __name__ == "__main__":
    main()
