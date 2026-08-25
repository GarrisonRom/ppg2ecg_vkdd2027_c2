#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""Paper-protocol PPG->ECG comparison in one reproducible entry point.

This script is deliberately self-contained at the experiment level. It builds
a SensSmartTech adaptation of the common CardioGAN/RDDM signal protocol from
raw CSV files, trains both methods, and evaluates them with the project's
single metric implementation.

The original paper datasets are not shipped here. The resulting numbers are
therefore a protocol adaptation, not an exact dataset reproduction.
"""

from __future__ import annotations

import argparse
import json
import math
import sys
import time
from pathlib import Path

import numpy as np
import pandas as pd
import torch
import torch.nn.functional as F
from scipy.signal import butter, sosfiltfilt
from torch import nn
from torch.optim import Adam
from torch.utils.data import DataLoader, Dataset
from tqdm import tqdm

PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT))

from src.evaluation.metrics import evaluate_all  # noqa: E402
from src.models.paper_baselines import (  # noqa: E402
    AttentionUNet1D,
    PatchDiscriminator1D,
    RDDMCore,
    SpectrogramDiscriminator,
)
from src.utils import set_seed  # noqa: E402


PPG_NAME = "carotid_880nm"
ECG_NAME = "II"
WINDOW_SEC = 4.0
TARGET_FS = 128
STRIDE_SEC = 2.0


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--method", choices=["cardiogan", "rddm", "both"], default="both")
    parser.add_argument("--raw-root", type=Path, default=PROJECT_ROOT / "data/raw/SensSmartTech")
    parser.add_argument("--output", type=Path, default=PROJECT_ROOT / "paper_repro/runs/senssmarttech_1to1_128hz_seed42")
    parser.add_argument("--epochs", type=int, default=20)
    parser.add_argument("--batch-size", type=int, default=32)
    parser.add_argument("--sampling-steps", type=int, default=10)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--device", choices=["auto", "cuda", "cpu"], default="auto")
    parser.add_argument("--rebuild-data", action="store_true")
    return parser.parse_args()


def resolve_device(value: str) -> torch.device:
    if value == "auto":
        return torch.device("cuda" if torch.cuda.is_available() else "cpu")
    return torch.device(value)


def infer_time_scale(time_values: np.ndarray) -> float:
    dt = np.diff(time_values)
    dt = dt[np.isfinite(dt) & (dt > 0)]
    if not len(dt):
        raise ValueError("No positive time differences")
    return 1000.0 if float(np.median(dt)) >= 0.5 else 1.0


def read_channel(path: Path, channel: str) -> tuple[np.ndarray, np.ndarray]:
    frame = pd.read_csv(path)
    aliases = {
        "II": ("II", "lead_II", "lead_ii", "ii"),
        "I": ("I", "lead_I", "lead_i", "i"),
        "V3": ("V3", "v3", "lead_V3", "lead_v3"),
        "V4": ("V4", "v4", "lead_V4", "lead_v4"),
    }
    candidates = aliases.get(channel, (channel,))
    column = next((name for name in candidates if name in frame.columns), None)
    if "t" not in frame or column is None:
        raise ValueError(f"{path.name} must contain t and a column for {channel}")
    raw_t = frame["t"].to_numpy(dtype=np.float64)
    t = raw_t / infer_time_scale(raw_t)
    signal = frame[column].to_numpy(dtype=np.float64)
    good = np.isfinite(t) & np.isfinite(signal)
    t, signal = t[good], signal[good]
    order = np.argsort(t)
    return t[order], signal[order]


def resample_pair(ppg_path: Path, ecg_path: Path, fs: int) -> tuple[np.ndarray, np.ndarray]:
    t_ppg, ppg = read_channel(ppg_path, PPG_NAME)
    t_ecg, ecg = read_channel(ecg_path, ECG_NAME)
    start, end = max(t_ppg[0], t_ecg[0]), min(t_ppg[-1], t_ecg[-1])
    if end <= start:
        raise ValueError("PPG and ECG do not overlap")
    n = int(math.floor((end - start) * fs))
    grid = start + np.arange(n, dtype=np.float64) / fs
    ppg = np.interp(grid, t_ppg, ppg)
    ecg = np.interp(grid, t_ecg, ecg)
    return ppg, ecg


def bandpass(signal: np.ndarray, fs: int, low: float, high: float) -> np.ndarray:
    sos = butter(4, [low / (fs / 2), high / (fs / 2)], btype="bandpass", output="sos")
    return sosfiltfilt(sos, signal)


def minmax_11(signal: np.ndarray) -> np.ndarray:
    lo, hi = float(np.nanmin(signal)), float(np.nanmax(signal))
    if not np.isfinite(lo) or not np.isfinite(hi) or hi - lo < 1e-8:
        raise ValueError("constant or invalid signal")
    return (2.0 * (signal - lo) / (hi - lo) - 1.0).clip(-1.0, 1.0)


def subject_from_name(name: str) -> int:
    try:
        return int(name.split("_", 1)[0])
    except (IndexError, ValueError) as exc:
        raise ValueError(f"cannot infer subject from {name}") from exc


def split_subjects(subjects: list[int], seed: int) -> dict[str, list[int]]:
    values = np.asarray(sorted(set(subjects)), dtype=np.int64)
    rng = np.random.default_rng(seed)
    rng.shuffle(values)
    n_test = max(1, int(round(0.20 * len(values))))
    return {"train": sorted(values[:-n_test].tolist()), "test": sorted(values[-n_test:].tolist())}


def prepare_data(raw_root: Path, data_path: Path, seed: int, rebuild: bool) -> dict:
    """Prepare 1->1, 4 s, 128 Hz, per-recording [-1,1] windows."""
    if data_path.exists() and not rebuild:
        with np.load(data_path, allow_pickle=False) as data:
            arrays = {key: data[key] for key in data.files}
        return {key: arrays[key] for key in arrays}

    csv_dir = raw_root / "CSV"
    if not csv_dir.is_dir():
        raise FileNotFoundError(f"CSV directory not found: {csv_dir}")
    window = int(round(WINDOW_SEC * TARGET_FS))
    stride = int(round(STRIDE_SEC * TARGET_FS))
    xs: list[np.ndarray] = []
    ys: list[np.ndarray] = []
    subjects: list[int] = []
    records: list[str] = []
    for ecg_path in sorted(csv_dir.glob("*_ecg.csv")):
        record = ecg_path.name.removesuffix("_ecg.csv")
        ppg_path = csv_dir / f"{record}_ppg.csv"
        if not ppg_path.exists():
            continue
        try:
            ppg, ecg = resample_pair(ppg_path, ecg_path, TARGET_FS)
            ppg = bandpass(ppg, TARGET_FS, 0.3, 12.0)
            ecg = bandpass(ecg, TARGET_FS, 0.5, 45.0)
            # CardioGAN/RDDM use bounded signal representations. We normalize
            # each recording independently, matching the public-paper style.
            ppg, ecg = minmax_11(ppg), minmax_11(ecg)
            for start in range(0, min(len(ppg), len(ecg)) - window + 1, stride):
                stop = start + window
                if not (np.isfinite(ppg[start:stop]).all() and np.isfinite(ecg[start:stop]).all()):
                    continue
                xs.append(ppg[start:stop][None, :].astype(np.float32))
                ys.append(ecg[start:stop][None, :].astype(np.float32))
                subjects.append(subject_from_name(record))
                records.append(record)
        except Exception as exc:
            print(f"[prepare] skip {record}: {exc}")

    if not xs:
        raise RuntimeError("No valid paper-protocol windows were prepared")
    subject_split = split_subjects(subjects, seed)
    data_path.parent.mkdir(parents=True, exist_ok=True)
    arrays = {
        "x": np.stack(xs),
        "y": np.stack(ys),
        "subject_id": np.asarray(subjects, dtype=np.int64),
        "record_id": np.asarray(records),
        "train_subjects": np.asarray(subject_split["train"], dtype=np.int64),
        "test_subjects": np.asarray(subject_split["test"], dtype=np.int64),
    }
    np.savez_compressed(data_path, **arrays)
    with data_path.with_suffix(".json").open("w", encoding="utf-8") as handle:
        json.dump({
            "dataset": "SensSmartTech",
            "mapping": f"{PPG_NAME} -> {ECG_NAME}",
            "fs": TARGET_FS,
            "window_sec": WINDOW_SEC,
            "window_samples": window,
            "stride_sec": STRIDE_SEC,
            "normalization": "per-recording min-max to [-1, 1]",
            "split": "subject-wise 80/20",
            "train_subjects": subject_split["train"],
            "test_subjects": subject_split["test"],
            "windows": len(xs),
        }, handle, ensure_ascii=False, indent=2)
    print(f"[prepare] windows={len(xs)} subjects={len(set(subjects))} saved={data_path}")
    return arrays


class WindowDataset(Dataset):
    def __init__(self, arrays: dict, indices: np.ndarray):
        self.x = arrays["x"][indices]
        self.y = arrays["y"][indices]
        self.subject_id = arrays["subject_id"][indices]

    def __len__(self) -> int:
        return len(self.x)

    def __getitem__(self, index: int) -> dict[str, torch.Tensor]:
        return {
            "ppg": torch.from_numpy(self.x[index]).float(),
            "ecg": torch.from_numpy(self.y[index]).float(),
            "subject_id": torch.tensor(int(self.subject_id[index])),
        }


def make_loaders(arrays: dict, batch_size: int) -> tuple[dict[str, DataLoader], dict[str, np.ndarray]]:
    subjects = arrays["subject_id"]
    train_subjects, test_subjects = arrays["train_subjects"], arrays["test_subjects"]
    indices = {
        "train": np.flatnonzero(np.isin(subjects, train_subjects)),
        "test": np.flatnonzero(np.isin(subjects, test_subjects)),
    }
    datasets = {key: WindowDataset(arrays, value) for key, value in indices.items()}
    loaders = {
        "train": DataLoader(datasets["train"], batch_size=batch_size, shuffle=True, num_workers=0),
        "test": DataLoader(datasets["test"], batch_size=batch_size, shuffle=False, num_workers=0),
    }
    return loaders, indices


def make_eval_loaders(datasets: dict[str, WindowDataset], batch_size: int) -> dict[str, DataLoader]:
    """Evaluation loaders preserve the array order for metric alignment."""
    return {
        split: DataLoader(dataset, batch_size=batch_size, shuffle=False, num_workers=0)
        for split, dataset in datasets.items()
    }


def gan_loss(logits: torch.Tensor, real: bool) -> torch.Tensor:
    target = torch.ones_like(logits) if real else torch.zeros_like(logits)
    return F.mse_loss(logits, target)


class CardioGANRunner:
    def __init__(self, device: torch.device, output: Path):
        self.device, self.output = device, output
        self.g_ecg = AttentionUNet1D(1, 1, base_channels=16, output_activation="tanh").to(device)
        self.g_ppg = AttentionUNet1D(1, 1, base_channels=16, output_activation="tanh").to(device)
        self.d_ecg_time = PatchDiscriminator1D(1, 16).to(device)
        self.d_ppg_time = PatchDiscriminator1D(1, 16).to(device)
        self.d_ecg_freq = SpectrogramDiscriminator(1, 8).to(device)
        self.d_ppg_freq = SpectrogramDiscriminator(1, 8).to(device)
        self.opt_g = Adam(list(self.g_ecg.parameters()) + list(self.g_ppg.parameters()), lr=1e-4, betas=(0.5, 0.999))
        self.opt_d = Adam(
            list(self.d_ecg_time.parameters()) + list(self.d_ppg_time.parameters())
            + list(self.d_ecg_freq.parameters()) + list(self.d_ppg_freq.parameters()),
            lr=1e-4, betas=(0.5, 0.999),
        )

    def train(self, loader: DataLoader, epochs: int) -> list[dict]:
        history = []
        for epoch in range(1, epochs + 1):
            totals = {"g_total": 0.0, "d_total": 0.0, "cycle": 0.0}
            for batch in tqdm(loader, desc=f"CardioGAN {epoch:03d}", leave=False):
                ppg = batch["ppg"].to(self.device)
                ecg = batch["ecg"].to(self.device)
                ppg_real = ppg[torch.randperm(len(ppg), device=self.device)]
                ecg_real = ecg[torch.randperm(len(ecg), device=self.device)]
                with torch.no_grad():
                    fake_ecg, fake_ppg = self.g_ecg(ppg_real), self.g_ppg(ecg_real)
                d_loss = 0.5 * (
                    gan_loss(self.d_ecg_time(ecg_real), True) + gan_loss(self.d_ecg_time(fake_ecg.detach()), False)
                    + gan_loss(self.d_ppg_time(ppg_real), True) + gan_loss(self.d_ppg_time(fake_ppg.detach()), False)
                    + gan_loss(self.d_ecg_freq(ecg_real), True) + gan_loss(self.d_ecg_freq(fake_ecg.detach()), False)
                    + gan_loss(self.d_ppg_freq(ppg_real), True) + gan_loss(self.d_ppg_freq(fake_ppg.detach()), False)
                )
                self.opt_d.zero_grad(set_to_none=True); d_loss.backward(); self.opt_d.step()
                fake_ecg, fake_ppg = self.g_ecg(ppg_real), self.g_ppg(ecg_real)
                rec_ppg, rec_ecg = self.g_ppg(fake_ecg), self.g_ecg(fake_ppg)
                adv_time = gan_loss(self.d_ecg_time(fake_ecg), True) + gan_loss(self.d_ppg_time(fake_ppg), True)
                adv_freq = gan_loss(self.d_ecg_freq(fake_ecg), True) + gan_loss(self.d_ppg_freq(fake_ppg), True)
                cycle = F.l1_loss(rec_ppg, ppg_real) + F.l1_loss(rec_ecg, ecg_real)
                g_loss = 3.0 * adv_time + adv_freq + 30.0 * cycle
                self.opt_g.zero_grad(set_to_none=True); g_loss.backward(); self.opt_g.step()
                totals["g_total"] += float(g_loss.detach()); totals["d_total"] += float(d_loss.detach()); totals["cycle"] += float(cycle.detach())
            row = {"epoch": epoch, **{key: value / max(1, len(loader)) for key, value in totals.items()}}
            history.append(row); print(row)
        return history

    @torch.no_grad()
    def predict(self, loader: DataLoader) -> np.ndarray:
        self.g_ecg.eval()
        return np.concatenate([self.g_ecg(batch["ppg"].to(self.device)).cpu().numpy() for batch in loader], axis=0)

    def save(self, path: Path) -> None:
        torch.save({"generator_ecg": self.g_ecg.state_dict(), "generator_ppg": self.g_ppg.state_dict()}, path)


class RDDMRunner:
    def __init__(self, device: torch.device, output: Path, sampling_steps: int):
        self.device, self.output, self.sampling_steps = device, output, sampling_steps
        self.model = RDDMCore(1, 1, timesteps=1000, beta_end=0.2, base_channels=16,
                              roi_gamma=32, roi_threshold=1.5, lambda_roi=100.0, lambda_global=1.0).to(device)
        self.optimizer = Adam(self.model.parameters(), lr=1e-4)

    def train(self, loader: DataLoader, epochs: int) -> list[dict]:
        history = []
        for epoch in range(1, epochs + 1):
            totals = {"total": 0.0, "roi": 0.0, "global": 0.0}
            self.model.train()
            for batch in tqdm(loader, desc=f"RDDM {epoch:03d}", leave=False):
                ppg, ecg = batch["ppg"].to(self.device), batch["ecg"].to(self.device)
                losses = self.model.training_loss(ppg, ecg)
                self.optimizer.zero_grad(set_to_none=True); losses["total"].backward(); self.optimizer.step()
                for key in totals:
                    totals[key] += float(losses[key].detach())
            row = {"epoch": epoch, **{key: value / max(1, len(loader)) for key, value in totals.items()}}
            history.append(row); print(row)
        return history

    @torch.no_grad()
    def predict(self, loader: DataLoader) -> np.ndarray:
        self.model.eval()
        return np.concatenate([
            self.model.sample(batch["ppg"].to(self.device), steps=self.sampling_steps).cpu().numpy()
            for batch in tqdm(loader, desc="RDDM sampling", leave=False)
        ], axis=0)

    def save(self, path: Path) -> None:
        torch.save({"model": self.model.state_dict()}, path)


def evaluate_method(name: str, runner, loaders: dict[str, DataLoader], datasets: dict[str, WindowDataset],
                    output: Path, device: torch.device, sampling_steps: int) -> dict:
    method_dir = output / name
    method_dir.mkdir(parents=True, exist_ok=True)
    results = {"method": name, "protocol": {
        "dataset": "SensSmartTech adaptation",
        "mapping": f"{PPG_NAME} -> {ECG_NAME}",
        "fs": TARGET_FS, "window_sec": WINDOW_SEC, "normalization": "per-recording min-max [-1,1]",
        "split": "subject-wise 80/20", "paper_dataset_exact": False,
    }, "splits": {}}
    runner.save(method_dir / "checkpoint_final.pth")
    for split in ("train", "test"):
        start = time.perf_counter()
        pred = runner.predict(loaders[split])
        elapsed = time.perf_counter() - start
        target = datasets[split].y
        ppg = datasets[split].x
        subject_id = datasets[split].subject_id
        metrics = evaluate_all(pred, target, TARGET_FS, [ECG_NAME], subject_id,
                               r_lead_idx=0, ppg=ppg, ppg_names=[PPG_NAME])
        result = {"method": name, "split": split, "metrics": metrics,
                  "inference_time_s": elapsed, "params_M": sum(p.numel() for p in runner.g_ecg.parameters()) / 1e6
                  if name == "cardiogan" else sum(p.numel() for p in runner.model.parameters()) / 1e6}
        if name == "rddm":
            result["sampling_steps"] = sampling_steps
        results["splits"][split] = result
        np.savez_compressed(method_dir / f"pred_{split}.npz", prediction=pred, target=target, ppg=ppg, subject_id=subject_id)
        with (method_dir / f"metrics_{split}.json").open("w", encoding="utf-8") as handle:
            json.dump(result, handle, ensure_ascii=False, indent=2, default=float)
        macro = metrics["waveform"]["macro"]
        phys = metrics["physiology"]
        print(f"[{name}/{split}] RMSE={macro['rmse/macro']['mean']:.4f} PCC={macro['pcc/macro']['mean']:.4f} "
              f"HRerr={phys['hr_err_bpm']['mean']:.3f} R-F1={phys['rpeak_f1']['mean']:.4f}")
    with (method_dir / "metrics.json").open("w", encoding="utf-8") as handle:
        json.dump(results, handle, ensure_ascii=False, indent=2, default=float)
    return results


def main() -> None:
    args = parse_args()
    set_seed(args.seed)
    device = resolve_device(args.device)
    args.output.mkdir(parents=True, exist_ok=True)
    arrays = prepare_data(args.raw_root.resolve(), args.output / "paper_protocol_data.npz", args.seed, args.rebuild_data)
    loaders, indices = make_loaders(arrays, args.batch_size)
    datasets = {key: loaders[key].dataset for key in loaders}
    eval_loaders = make_eval_loaders(datasets, args.batch_size)
    protocol = {
        "paper_protocol_adaptation": True,
        "mapping": f"{PPG_NAME} -> {ECG_NAME}", "fs": TARGET_FS,
        "window_sec": WINDOW_SEC, "stride_sec": STRIDE_SEC,
        "normalization": "per-recording min-max to [-1, 1]",
        "split": "subject-wise 80/20", "train_windows": int(len(indices["train"])),
        "test_windows": int(len(indices["test"])), "device": str(device),
        "epochs": args.epochs, "seed": args.seed,
    }
    with (args.output / "protocol.json").open("w", encoding="utf-8") as handle:
        json.dump(protocol, handle, ensure_ascii=False, indent=2)
    methods = [args.method] if args.method != "both" else ["cardiogan", "rddm"]
    all_results = {}
    for method in methods:
        set_seed(args.seed)
        if method == "cardiogan":
            runner = CardioGANRunner(device, args.output / method)
        else:
            runner = RDDMRunner(device, args.output / method, args.sampling_steps)
        history = runner.train(loaders["train"], args.epochs)
        method_dir = args.output / method
        method_dir.mkdir(parents=True, exist_ok=True)
        with (method_dir / "training_history.json").open("w", encoding="utf-8") as handle:
            json.dump(history, handle, ensure_ascii=False, indent=2, default=float)
        all_results[method] = evaluate_method(method, runner, eval_loaders, datasets, args.output, device, args.sampling_steps)
    with (args.output / "comparison.json").open("w", encoding="utf-8") as handle:
        json.dump({"protocol": protocol, "results": all_results}, handle, ensure_ascii=False, indent=2, default=float)
    def comparison_row(method: str, split: str, result: dict) -> dict:
        metrics = result["splits"][split]["metrics"]
        wave = metrics["waveform"]["macro"]
        phys = metrics["physiology"]
        return {
            "method": method,
            "split": split,
            "rmse": wave["rmse/macro"]["mean"],
            "mae": wave["mae/macro"]["mean"],
            "pcc": wave["pcc/macro"]["mean"],
            "snr_db": wave["snr_db/macro"]["mean"],
            "nrmse": wave["nrmse/macro"]["mean"],
            "hr_error_bpm": phys["hr_err_bpm"]["mean"],
            "rmssd_error_ms": phys["rmssd_err_ms"]["mean"],
            "rpeak_f1": phys["rpeak_f1"]["mean"],
            "rpeak_time_error_ms": phys["rpeak_time_err_ms"]["mean"],
            "qrs_width_error_ms": phys["qrs_width_err_ms"]["mean"],
            "qrs_amplitude_abs_error": phys["qrs_amp_abs_err"]["mean"],
            "validity_rate": metrics["validity_rate"],
            "inference_time_s": result["splits"][split]["inference_time_s"],
        }

    rows = [
        comparison_row(method, split, result)
        for method, result in all_results.items()
        for split in ("train", "test")
    ]
    comparison = pd.DataFrame(rows)
    comparison[comparison["split"] == "train"].drop(columns="split").to_csv(
        args.output / "comparison_train.csv", index=False,
    )
    comparison[comparison["split"] == "test"].drop(columns="split").to_csv(
        args.output / "comparison_test.csv", index=False,
    )
    comparison.to_csv(args.output / "comparison_all.csv", index=False)
    print(f"Finished paper-protocol comparison: {args.output}")


if __name__ == "__main__":
    main()
