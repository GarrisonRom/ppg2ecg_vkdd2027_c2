#!/usr/bin/env python
"""Isolate v0.54 encoder/decoder capacity with a target-side overfit test.

This is a diagnostic only: real ECG is used as the input to the same encoder
and wavelet decoder, so no result from this script is a deployable PPG2ECG
model. It answers whether the coefficient heads and IDWT can fit high-
frequency target coefficients when the input contains the ECG information.
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import torch

PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

from src.data import create_dataset
from src.models import ReconstructionLoss, build_decoder, build_encoder


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--data", type=Path,
        default=PROJECT_ROOT / "data/processed/SensSmartTech/subjectwise_per-lead",
    )
    parser.add_argument("--batch-size", type=int, default=32)
    parser.add_argument("--steps", type=int, default=150)
    parser.add_argument("--lr", type=float, default=1e-3)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--device", default="cuda")
    parser.add_argument(
        "--input", choices=("ecg", "ppg"), default="ecg",
        help="ECG isolates model capacity; PPG tests the cross-modal path",
    )
    return parser.parse_args()


def build_criterion() -> ReconstructionLoss:
    return ReconstructionLoss(
        weights={
            "l1": 1.0,
            "derivative": 0.05,
            "haar_wavelet": 0.20,
            "haar_qrs": 0.15,
            "peak_interval": 0.10,
        },
        ecg_leads=4,
        sample_rate=250.0,
        wavelet_config={
            "haar_levels": 4,
            "haar_qrs_levels": [2, 3, 4],
            "haar_detail_weights": [0.15, 1.0, 1.0, 0.5],
            "peak_window_ms": 100.0,
            "search_radius_ms": 80.0,
            "peak_threshold": 0.25,
            "softmax_temperature": 0.05,
            "max_peaks": 16,
            "lead_index": 1,
        },
    )


def summarize(
    encoder: torch.nn.Module,
    decoder: torch.nn.Module,
    criterion: ReconstructionLoss,
    source: torch.Tensor,
    target: torch.Tensor,
) -> tuple[dict[str, float], list[tuple[float, float]]]:
    encoder.eval()
    decoder.eval()
    with torch.no_grad():
        encoded = encoder(source)
        output = decoder(encoded, return_coeffs=True)
        losses = criterion(output["fused"], target, pred_coeffs=output["coefficients"])
        target_coeffs = criterion.haar_wavelet.transform(target)
        coeff_stats = []
        for prediction, reference in zip(
            output["coefficients"]["details"], target_coeffs["details"],
        ):
            coeff_stats.append((float(prediction.std()), float(reference.std())))
    return {
        "total": float(losses["total"]),
        "l1": float(losses["l1"]),
        "haar_wavelet": float(losses["haar_wavelet"]),
        "haar_qrs": float(losses["haar_qrs"]),
        "peak_interval": float(losses["peak_interval"]),
    }, coeff_stats


def main() -> None:
    args = parse_args()
    torch.manual_seed(args.seed)
    if args.device == "cuda" and not torch.cuda.is_available():
        raise RuntimeError("CUDA was requested but is not available")
    device = torch.device(args.device)
    dataset = create_dataset("senssmarttech", args.data, split="train")
    target = torch.from_numpy(dataset._y[:args.batch_size]).float().to(device)
    source = target if args.input == "ecg" else torch.from_numpy(
        dataset._x[:args.batch_size],
    ).float().to(device)

    encoder = build_encoder(
        {
            "name": "baseline_encoder",
            "base_channels": 32,
            "dropout": 0.0,
        },
        signal_length=dataset.signal_length,
        latent_dim=128,
            ppg_channels=source.size(1),
    ).to(device)
    decoder = build_decoder(
        {
            "name": "wavelet_decoder",
            "base_channels": 32,
            "dropout": 0.0,
            "levels": 4,
        },
        signal_length=dataset.signal_length,
        latent_dim=128,
        ecg_leads=dataset.ecg_leads,
    ).to(device)
    criterion = build_criterion().to(device)
    optimizer = torch.optim.AdamW(
        list(encoder.parameters()) + list(decoder.parameters()),
        lr=args.lr,
        weight_decay=1e-5,
    )

    initial, initial_coeffs = summarize(encoder, decoder, criterion, source, target)
    print(
        f"device={device} input={args.input} source={tuple(source.shape)} "
        f"target={tuple(target.shape)} steps={args.steps}"
    )
    print(f"initial={initial}")
    print(f"initial_detail_std={initial_coeffs}")

    encoder.train()
    decoder.train()
    for step in range(1, args.steps + 1):
        optimizer.zero_grad(set_to_none=True)
        encoded = encoder(source)
        output = decoder(encoded, return_coeffs=True)
        losses = criterion(output["fused"], target, pred_coeffs=output["coefficients"])
        losses["total"].backward()
        torch.nn.utils.clip_grad_norm_(
            list(encoder.parameters()) + list(decoder.parameters()), 5.0,
        )
        optimizer.step()
        if step in {1, 25, 50, 100, args.steps}:
            print(
                f"step={step} total={losses['total'].item():.5f} "
                f"haar={losses['haar_wavelet'].item():.5f} "
                f"qrs={losses['haar_qrs'].item():.5f} "
                f"peak={losses['peak_interval'].item():.5f}"
            )

    final, final_coeffs = summarize(encoder, decoder, criterion, source, target)
    print(f"final={final}")
    print(f"final_detail_std={final_coeffs}")


if __name__ == "__main__":
    main()
