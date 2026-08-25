#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""Reproduce recent/strong PPG-to-ECG mechanisms on a common local protocol.

The source papers use different public datasets.  This script therefore keeps
the paper mechanisms, but evaluates every method on the cached SensSmartTech
1 -> 1 adaptation used by ``reproduce_compare.py``:

    carotid_880nm -> Lead II, 128 Hz, 4 s, subject-wise 80/20, seed 42.

Methods:
  * ``qrs_transattn``: QRS-TransAttn-style attention CNN with QRS weighting.
  * ``p2e_wgan``: P2E-WGAN-style paired conditional WGAN-GP with sample loss.
  * ``li2024_lightweight``: compact multi-kernel attention/residual network
    inspired by the 2024 lightweight PPG2ECG work.

These are protocol adaptations, not the numerical results reported by the
papers.  All evaluation goes through ``src.evaluation.metrics.evaluate_all``.
"""

from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import torch
import torch.nn.functional as F
from torch import autograd, nn
from torch.optim import Adam
from torch.utils.data import DataLoader
from tqdm import tqdm

PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT))
sys.path.insert(0, str(Path(__file__).resolve().parent))

from reproduce_compare import (  # noqa: E402
    ECG_NAME,
    PPG_NAME,
    TARGET_FS,
    make_eval_loaders,
    make_loaders,
    prepare_data,
)
from src.evaluation.metrics import evaluate_all  # noqa: E402
from src.models.paper_baselines import (  # noqa: E402
    ConditionalPatchDiscriminator1D,
    LightweightPPG2ECG,
    P2EWGANGenerator,
    QRSTransAttnNet,
)
from src.utils import set_seed  # noqa: E402


PAPER_INFO = {
    "qrs_transattn": {
        "paper": "Reconstructing QRS Complex From PPG by Transformed Attentional Neural Networks",
        "citation": "Chiu et al., IEEE Sensors Journal, 2020, DOI: 10.1109/JSEN.2020.3000344",
        "code": "https://github.com/james77777778/ppg2ecg-pytorch",
        "mechanism": "temporal/channel attention CNN encoder-decoder + explicit QRS-weighted objective",
    },
    "p2e_wgan": {
        "paper": "P2E-WGAN: ECG Waveform Synthesis from PPG with Conditional Wasserstein GANs",
        "citation": "Vo et al., ACM SAC, 2021, DOI: 10.1145/3412841.3441979",
        "code": "https://github.com/khuongav/P2E-WGAN-ecg-ppg-reconstruction",
        "mechanism": "paired 1D U-Net generator + conditional WGAN-GP critic + sample waveform loss",
    },
    "li2024_lightweight": {
        "paper": "Inferring Electrocardiography From Optical Sensing Using Lightweight Neural Network",
        "citation": "Li et al., IEEE Transactions on Artificial Intelligence, 2024, DOI: 10.1109/TAI.2024.3400749",
        "code": "https://github.com/AnaLovesToCod3/reproducible-ppg-to-ecg-reconstruction",
        "mechanism": "grouped multi-kernel temporal convolution + channel/temporal attention + residual reconstruction",
        "code_note": "independent reproducible implementation; no author repository was located",
    },
}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--method",
        choices=["qrs_transattn", "p2e_wgan", "li2024_lightweight", "all"],
        default="all",
    )
    parser.add_argument("--raw-root", type=Path, default=PROJECT_ROOT / "data/raw/SensSmartTech")
    parser.add_argument(
        "--data-path",
        type=Path,
        default=PROJECT_ROOT / "paper_repro/runs/senssmarttech_1to1_128hz_seed42/paper_protocol_data.npz",
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=PROJECT_ROOT / "paper_repro/runs/senssmarttech_recent_1to1_128hz_seed42",
    )
    parser.add_argument("--epochs", type=int, default=20)
    parser.add_argument("--batch-size", type=int, default=32)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--device", choices=["auto", "cuda", "cpu"], default="auto")
    parser.add_argument("--ncritic", type=int, default=1, help="P2E-WGAN critic updates per generator update")
    parser.add_argument("--rebuild-data", action="store_true")
    return parser.parse_args()


def resolve_device(value: str) -> torch.device:
    if value == "auto":
        return torch.device("cuda" if torch.cuda.is_available() else "cpu")
    return torch.device(value)


def qrs_mask(target: torch.Tensor, kernel_size: int = 15) -> torch.Tensor:
    """Create a target-derived QRS ROI without external beat annotations."""
    lead = target[:, :1]
    kernel = max(7, int(kernel_size) | 1)
    baseline_kernel = max(kernel * 5 + 1, 31)
    if baseline_kernel % 2 == 0:
        baseline_kernel += 1
    baseline = F.avg_pool1d(lead, baseline_kernel, stride=1, padding=baseline_kernel // 2)
    energy = (lead - baseline).abs()
    scale = energy.flatten(1).std(dim=1).clamp_min(1e-4)[:, None, None]
    seeds = (energy > 1.25 * scale).float()
    mask = F.max_pool1d(seeds, kernel, stride=1, padding=kernel // 2)
    return mask.expand(-1, target.size(1), -1).clamp(0.0, 1.0)


def weighted_qrs_l1(pred: torch.Tensor, target: torch.Tensor, weight: float = 8.0) -> torch.Tensor:
    mask = qrs_mask(target)
    weights = 1.0 + float(weight) * mask
    return (weights * (pred - target).abs()).mean()


def gradient_penalty(
    critic: nn.Module,
    condition: torch.Tensor,
    real: torch.Tensor,
    fake: torch.Tensor,
) -> torch.Tensor:
    alpha = torch.rand(real.size(0), 1, 1, device=real.device, dtype=real.dtype)
    mixed = (alpha * real + (1.0 - alpha) * fake).requires_grad_(True)
    score = critic(condition, mixed)
    grad = autograd.grad(
        outputs=score,
        inputs=mixed,
        grad_outputs=torch.ones_like(score),
        create_graph=True,
        retain_graph=True,
        only_inputs=True,
    )[0]
    return ((grad.flatten(1).norm(2, dim=1) - 1.0) ** 2).mean()


class DeterministicRunner:
    def __init__(self, model: nn.Module, device: torch.device, name: str):
        self.model, self.device, self.name = model.to(device), device, name
        self.optimizer = Adam(self.model.parameters(), lr=1e-4, betas=(0.9, 0.999))

    def train(self, loader: DataLoader, epochs: int) -> list[dict]:
        history: list[dict] = []
        for epoch in range(1, epochs + 1):
            self.model.train()
            totals = {"total": 0.0, "global_l1": 0.0, "qrs_l1": 0.0}
            for batch in tqdm(loader, desc=f"{self.name} {epoch:03d}", leave=False):
                ppg, ecg = batch["ppg"].to(self.device), batch["ecg"].to(self.device)
                pred = self.model(ppg)
                global_l1 = F.l1_loss(pred, ecg)
                qrs_l1 = weighted_qrs_l1(pred, ecg)
                total = global_l1 + 1.0 * qrs_l1
                self.optimizer.zero_grad(set_to_none=True)
                total.backward()
                self.optimizer.step()
                totals["total"] += float(total.detach())
                totals["global_l1"] += float(global_l1.detach())
                totals["qrs_l1"] += float(qrs_l1.detach())
            row = {"epoch": epoch, **{k: v / max(1, len(loader)) for k, v in totals.items()}}
            history.append(row)
            print(row)
        return history

    @torch.no_grad()
    def predict(self, loader: DataLoader) -> np.ndarray:
        self.model.eval()
        return np.concatenate([self.model(batch["ppg"].to(self.device)).cpu().numpy() for batch in loader], axis=0)

    def save(self, path: Path) -> None:
        torch.save({"model": self.model.state_dict()}, path)


class P2EWGANRunner:
    def __init__(self, device: torch.device, ncritic: int):
        self.device = device
        self.ncritic = max(1, int(ncritic))
        self.generator = P2EWGANGenerator(1, 1, base_channels=16).to(device)
        self.critic = ConditionalPatchDiscriminator1D(1, 1, base_channels=16).to(device)
        self.opt_g = Adam(self.generator.parameters(), lr=2e-4, betas=(0.5, 0.9))
        self.opt_d = Adam(self.critic.parameters(), lr=2e-4, betas=(0.5, 0.9))

    def train(self, loader: DataLoader, epochs: int) -> list[dict]:
        history: list[dict] = []
        for epoch in range(1, epochs + 1):
            self.generator.train(); self.critic.train()
            totals = {"g_total": 0.0, "d_total": 0.0, "wasserstein": 0.0, "sample_mse": 0.0, "qrs_l1": 0.0}
            for batch in tqdm(loader, desc=f"p2e_wgan {epoch:03d}", leave=False):
                ppg, ecg = batch["ppg"].to(self.device), batch["ecg"].to(self.device)
                d_total = 0.0
                for _ in range(self.ncritic):
                    with torch.no_grad():
                        fake = self.generator(ppg)
                    real_score = self.critic(ppg, ecg).mean()
                    fake_score = self.critic(ppg, fake).mean()
                    gp = gradient_penalty(self.critic, ppg, ecg, fake)
                    d_loss = fake_score - real_score + 10.0 * gp
                    self.opt_d.zero_grad(set_to_none=True)
                    d_loss.backward()
                    self.opt_d.step()
                    d_total += float(d_loss.detach())

                fake = self.generator(ppg)
                wasserstein = -self.critic(ppg, fake).mean()
                sample_mse = F.mse_loss(fake, ecg)
                qrs_l1 = weighted_qrs_l1(fake, ecg)
                g_loss = wasserstein + 50.0 * sample_mse + 0.5 * qrs_l1
                self.opt_g.zero_grad(set_to_none=True)
                g_loss.backward()
                self.opt_g.step()
                totals["g_total"] += float(g_loss.detach())
                totals["d_total"] += d_total / self.ncritic
                totals["wasserstein"] += float(wasserstein.detach())
                totals["sample_mse"] += float(sample_mse.detach())
                totals["qrs_l1"] += float(qrs_l1.detach())
            row = {"epoch": epoch, **{k: v / max(1, len(loader)) for k, v in totals.items()}}
            history.append(row)
            print(row)
        return history

    @torch.no_grad()
    def predict(self, loader: DataLoader) -> np.ndarray:
        self.generator.eval()
        return np.concatenate([self.generator(batch["ppg"].to(self.device)).cpu().numpy() for batch in loader], axis=0)

    def save(self, path: Path) -> None:
        torch.save({"generator": self.generator.state_dict(), "critic": self.critic.state_dict()}, path)


def params_m(runner) -> float:
    if isinstance(runner, P2EWGANRunner):
        modules = (runner.generator, runner.critic)
    else:
        modules = (runner.model,)
    return float(sum(p.numel() for m in modules for p in m.parameters()) / 1e6)


def evaluate_method(name: str, runner, loaders: dict[str, DataLoader], output: Path) -> dict:
    method_dir = output / name
    method_dir.mkdir(parents=True, exist_ok=True)
    runner.save(method_dir / "checkpoint_final.pth")
    result = {"method": name, "paper": PAPER_INFO[name], "protocol": {
        "dataset": "SensSmartTech adaptation",
        "mapping": f"{PPG_NAME} -> {ECG_NAME}",
        "fs": TARGET_FS,
        "window_sec": 4.0,
        "normalization": "per-recording min-max to [-1, 1]",
        "split": "subject-wise 80/20",
        "paper_dataset_exact": False,
    }, "params_m": params_m(runner), "splits": {}}
    for split in ("train", "test"):
        start = time.perf_counter()
        pred = runner.predict(loaders[split])
        elapsed = time.perf_counter() - start
        dataset = loaders[split].dataset
        target = dataset.y
        ppg = dataset.x
        subject_id = dataset.subject_id
        metrics = evaluate_all(pred, target, TARGET_FS, [ECG_NAME], subject_id,
                               r_lead_idx=0, ppg=ppg, ppg_names=[PPG_NAME])
        row = {"method": name, "split": split, "metrics": metrics,
               "inference_time_s": elapsed, "params_m": result["params_m"]}
        result["splits"][split] = row
        np.savez_compressed(method_dir / f"pred_{split}.npz", prediction=pred,
                            target=target, ppg=ppg, subject_id=subject_id)
        with (method_dir / f"metrics_{split}.json").open("w", encoding="utf-8") as handle:
            json.dump(row, handle, ensure_ascii=False, indent=2, default=float)
        wave = metrics["waveform"]["macro"]
        phys = metrics["physiology"]
        print(f"[{name}/{split}] RMSE={wave['rmse/macro']['mean']:.4f} PCC={wave['pcc/macro']['mean']:.4f} "
              f"HRerr={phys['hr_err_bpm']['mean']:.3f} R-F1={phys['rpeak_f1']['mean']:.4f}")
    with (method_dir / "metrics.json").open("w", encoding="utf-8") as handle:
        json.dump(result, handle, ensure_ascii=False, indent=2, default=float)
    return result


def save_plot(result: dict, output: Path) -> None:
    """Save compact train/test qualitative panels for every method."""
    name = result["method"]
    fig, axes = plt.subplots(2, 1, figsize=(10, 4.2), sharex=False)
    for ax, split in zip(axes, ("train", "test")):
        data = np.load(output / name / f"pred_{split}.npz")
        target = data["target"][0, 0]
        pred = data["prediction"][0, 0]
        n = min(len(target), 512)
        ax.plot(target[:n], color="#222222", lw=1.0, label="real")
        ax.plot(pred[:n], color="#d95f02", lw=0.9, label="generated")
        ax.set_title(f"{name} | {split}", fontsize=9)
        ax.grid(alpha=0.2)
        ax.legend(loc="upper right", frameon=False, fontsize=8)
    fig.tight_layout()
    fig.savefig(output / name / "qualitative_train_test.png", dpi=180)
    plt.close(fig)


def comparison_row(name: str, split: str, result: dict) -> dict:
    metrics = result["splits"][split]["metrics"]
    wave, phys = metrics["waveform"]["macro"], metrics["physiology"]
    return {
        "method": name,
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
        "params_m": result["params_m"],
    }


def main() -> None:
    args = parse_args()
    set_seed(args.seed)
    device = resolve_device(args.device)
    args.output.mkdir(parents=True, exist_ok=True)
    data_path = args.data_path.resolve()
    if args.rebuild_data:
        data_path = args.output / "paper_protocol_data.npz"
    arrays = prepare_data(args.raw_root.resolve(), data_path, args.seed, args.rebuild_data)
    loaders, _ = make_loaders(arrays, args.batch_size)
    eval_loaders = make_eval_loaders({k: v.dataset for k, v in loaders.items()}, args.batch_size)
    methods = [args.method] if args.method != "all" else ["qrs_transattn", "p2e_wgan", "li2024_lightweight"]
    all_results: dict = {}
    protocol = {
        "dataset": "SensSmartTech",
        "mapping": f"{PPG_NAME} -> {ECG_NAME}",
        "fs": TARGET_FS,
        "window_sec": 4.0,
        "split": "subject-wise 80/20",
        "seed": args.seed,
        "epochs": args.epochs,
        "device": str(device),
        "data_path": str(data_path),
        "uniform_evaluator": "src.evaluation.metrics.evaluate_all",
    }
    with (args.output / "protocol.json").open("w", encoding="utf-8") as handle:
        json.dump(protocol, handle, ensure_ascii=False, indent=2)
    for name in methods:
        set_seed(args.seed)
        if name == "qrs_transattn":
            runner = DeterministicRunner(QRSTransAttnNet(1, 1, base_channels=16), device, name)
        elif name == "li2024_lightweight":
            runner = DeterministicRunner(LightweightPPG2ECG(1, 1, width=32), device, name)
        else:
            runner = P2EWGANRunner(device, args.ncritic)
        history = runner.train(loaders["train"], args.epochs)
        method_dir = args.output / name
        method_dir.mkdir(parents=True, exist_ok=True)
        with (method_dir / "training_history.json").open("w", encoding="utf-8") as handle:
            json.dump(history, handle, ensure_ascii=False, indent=2, default=float)
        result = evaluate_method(name, runner, eval_loaders, args.output)
        save_plot(result, args.output)
        all_results[name] = result
    with (args.output / "comparison.json").open("w", encoding="utf-8") as handle:
        json.dump({"protocol": protocol, "results": all_results}, handle, ensure_ascii=False, indent=2, default=float)
    rows = [comparison_row(name, split, result) for name, result in all_results.items() for split in ("train", "test")]
    comparison = pd.DataFrame(rows)
    comparison[comparison["split"] == "train"].to_csv(args.output / "comparison_train.csv", index=False)
    comparison[comparison["split"] == "test"].to_csv(args.output / "comparison_test.csv", index=False)
    comparison.to_csv(args.output / "comparison_all.csv", index=False)
    print(f"Finished recent paper adaptations: {args.output}")


if __name__ == "__main__":
    main()
