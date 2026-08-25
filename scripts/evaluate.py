#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""评估入口: 对某个 run 的 checkpoint 计算多层指标, 附朴素基线锚点。

用法:
  python scripts/evaluate.py --run runs/baseline
  python scripts/evaluate.py --run runs/baseline --checkpoint final.pth
  python scripts/evaluate.py --run runs/baseline --baselines --distribution

产出 (保存到 <run>/eval_<split>.json):
  L1 波形:    MSE/RMSE/MAE/PCC/DTW/SNR (宏观 + 逐导联)
  L2 生理:    HR误差 / R峰F1+时序误差 / RMSSD误差 / QRS宽度误差 / PTT(分部位)
              生成有效率 / HR传递斜率
  L4 分布:    1-NNA / Coverage (可选)
  L5 效率:    参数量 / 推理时延 / RTF
  基线锚点:   mean_template (均值波形) / zero (平凡下限) / delayed_ppg (无形态线性代理)

注: LF/HF 频域 HRV 未实现 —— 8s 窗不满足其 >=2min 统计效度要求。
"""

from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

import numpy as np
import torch

from src.data import create_dataset
from src.evaluation.metrics import evaluate_all
from src.models import build_decoder, build_encoder, build_latent_flow
from src.utils.config import load_config


def parse_args():
    parser = argparse.ArgumentParser(description="PPG2ECG 多层评估")
    parser.add_argument("--run", type=Path, required=True, help="run 目录 (含 config.yaml)")
    parser.add_argument("--checkpoint", type=str, default="best.pth")
    parser.add_argument("--split", type=str, default="test", choices=["train", "val", "test"])
    parser.add_argument("--baselines", action="store_true", help="附带三个朴素基线")
    parser.add_argument("--distribution", action="store_true",
                        help="计算 1-NNA/Coverage (O(N²·D))")
    parser.add_argument("--batch-size", type=int, default=64)
    return parser.parse_args()


def load_run(run_dir: Path, checkpoint: str, split: str):
    """加载 run 配置、数据集与模型, 返回 (config, dataset, model, device)。"""
    config = load_config(run_dir / "config.yaml")
    data_cfg = config["data"]

    # Keep the configured channel protocol explicit.  ``None`` preserves the
    # historical four-channel/four-lead evaluation; selecting both options
    # makes a strict single-PPG -> single-ECG run possible without changing the
    # cached files.
    dataset = create_dataset(
        data_cfg["dataset"],
        data_cfg["root"],
        split=split,
        ppg_channel=data_cfg.get("ppg_channel"),
        ecg_lead=data_cfg.get("ecg_lead"),
    )
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    model_cfg = config["model"]
    model_ppg_channels = dataset.num_ppg_channels
    encoder = build_encoder(
        model_cfg.get("encoder"),
        signal_length=dataset.signal_length,
        latent_dim=model_cfg.get("latent_dim", 128),
        ppg_channels=model_ppg_channels,
    ).to(device)
    decoder = build_decoder(
        model_cfg.get("decoder"),
        signal_length=dataset.signal_length,
        latent_dim=model_cfg.get("latent_dim", 128),
        ecg_leads=dataset.ecg_leads,
    ).to(device)

    latent_flow = None
    if model_cfg.get("latent_flow") is not None:
        latent_flow = build_latent_flow(
            model_cfg["latent_flow"],
            latent_dim=model_cfg.get("latent_dim", 128),
        ).to(device)

    ckpt = torch.load(run_dir / checkpoint, map_location=device, weights_only=False)
    encoder.load_state_dict(ckpt["encoder"])
    decoder.load_state_dict(ckpt["decoder"])
    if latent_flow is not None:
        if "latent_flow" not in ckpt:
            raise KeyError(f"Checkpoint {checkpoint} does not contain latent_flow weights")
        latent_flow.load_state_dict(ckpt["latent_flow"])
    encoder.eval()
    decoder.eval()
    if latent_flow is not None:
        latent_flow.eval()
    return config, dataset, encoder, decoder, latent_flow, device


@torch.no_grad()
def predict(encoder, decoder, ppg: np.ndarray, ppg_channel_idx: int | None,
            device, fs: float, batch_size: int = 64,
            latent_flow=None, flow_steps: int = 8) -> tuple[np.ndarray, dict]:
    """批量推理。ppg: [N, C_p, T] (全通道), 返回 (pred [N, C_e, T], 效率信息)。"""
    if ppg_channel_idx is None:
        xs = torch.from_numpy(ppg).float()
    else:
        xs = torch.from_numpy(ppg[:, ppg_channel_idx: ppg_channel_idx + 1]).float()
    preds = []
    n_params = sum(p.numel() for p in encoder.parameters()) + \
               sum(p.numel() for p in decoder.parameters())
    if latent_flow is not None:
        n_params += sum(p.numel() for p in latent_flow.parameters())

    t0 = time.perf_counter()
    for i in range(0, len(xs), batch_size):
        batch = xs[i: i + batch_size].to(device)
        encoded = encoder(batch)
        if latent_flow is not None:
            latent = encoded["mu"] if isinstance(encoded, dict) else encoded
            latent = latent_flow.integrate(latent, steps=flow_steps)
            preds.append(decoder(latent).cpu().numpy())
        else:
            preds.append(decoder(encoded).cpu().numpy())
    elapsed = time.perf_counter() - t0
    pred = np.concatenate(preds, axis=0)

    signal_seconds = len(xs) * xs.shape[-1] / fs
    return pred, {
        "params_M": n_params / 1e6,
        "inference_time_s": elapsed,
        "rtf": elapsed / signal_seconds,  # < 1 即快于实时
        "device": str(device),
    }


# ----------------------------------------------------------------------
# 朴素基线 (锚点, 不是竞争方法)
# ----------------------------------------------------------------------

def baseline_mean_template(train: np.ndarray, n_test: int) -> np.ndarray:
    """训练集逐导联均值波形 (演示 'MSE 尚可 + PCC 崩' 的平均波形现象)。"""
    template = train.mean(axis=0)  # [C, T]
    return np.tile(template[None], (n_test, 1, 1))


def baseline_zero(n_test: int, c: int, t: int) -> np.ndarray:
    """零预测器: RMSE = 目标 std (归一化域 ≈ 1), 平凡下限。"""
    return np.zeros((n_test, c, t), dtype=np.float32)


def baseline_delayed_ppg(train_ppg: np.ndarray, train_ecg: np.ndarray,
                         test_ppg: np.ndarray, ppg_ch: int,
                         fs: float = 250.0) -> np.ndarray:
    """逐导联拟合 (lag, a, b): pred_j(t) = a_j·ppg(t-lag_j)+b_j。

    无形态信息的线性代理 —— PCC 锚点: 连它都打不过说明模型没学到东西。
    """
    n, c, t = train_ecg.shape
    x_full = train_ppg[:, ppg_ch]  # [N, T]
    lags = np.arange(0, int(0.4 * fs), int(0.01 * fs))  # 0-400ms, 10ms 步进

    transforms = []
    for j in range(c):
        best = (np.inf, 0, 0.0, 0.0)
        y = train_ecg[:, j].ravel()
        for lag in lags:
            xs = np.roll(x_full, lag, axis=1).ravel()
            a = np.cov(xs, y)[0, 1] / (np.var(xs) + 1e-8)
            b = y.mean() - a * xs.mean()
            mse = np.mean((a * xs + b - y) ** 2)
            if mse < best[0]:
                best = (mse, lag, a, b)
        transforms.append(best[1:])
    pred = np.empty((test_ppg.shape[0], c, t), dtype=np.float32)
    for j, (lag, a, b) in enumerate(transforms):
        xs = np.roll(test_ppg[:, ppg_ch], lag, axis=1)
        pred[:, j] = a * xs + b
    return pred


# ----------------------------------------------------------------------
# 汇总输出
# ----------------------------------------------------------------------

def flatten_summary(summary: dict, prefix: str = "") -> dict[str, float]:
    """evaluate_all 输出 -> {'metric': mean} 扁平字典 (供打印对比表)。"""
    flat: dict[str, float] = {}
    for k, v in summary.items():
        key = f"{prefix}{k}"
        if isinstance(v, dict) and "mean" in v:
            flat[key] = v["mean"]
        elif isinstance(v, dict):
            flat.update(flatten_summary(v, prefix=f"{key}/"))
        elif isinstance(v, (int, float)) and not isinstance(v, bool):
            flat[key] = float(v)
    return flat


def main():
    args = parse_args()
    config, dataset, encoder, decoder, latent_flow, device = load_run(
        args.run, args.checkpoint, args.split)

    # The dataset already applies the configured channel selection.
    ppg_full = dataset._x
    ecg_true = dataset._y
    fs = dataset.fs
    ch_names = dataset.ppg_channels
    ppg_channel = config["data"].get("ppg_channel")
    ppg_idx = ch_names.index(ppg_channel) if ppg_channel in ch_names else None
    subject_ids = dataset.metadata["subject_id"].to_numpy()

    print(f"[eval] run={args.run.name} split={args.split} "
          f"N={len(dataset)} subjects={len(np.unique(subject_ids))} fs={fs}")

    # 模型预测 + 效率
    flow_steps = int((config.get("model", {}).get("cardio_align", {}) or {}).get(
        "integration_steps", 8,
    ))
    pred, efficiency = predict(
        encoder, decoder, ppg_full, ppg_idx, device,
        fs=fs, batch_size=args.batch_size,
        latent_flow=latent_flow, flow_steps=flow_steps,
    )
    efficiency["fs"] = fs  # 记录

    results = {
        "run": str(args.run),
        "checkpoint": args.checkpoint,
        "split": args.split,
        "normalization_domain": str(config["data"]["root"]),
        "model": evaluate_all(
            pred, ecg_true, fs, dataset.ecg_channels, subject_ids,
            ppg=ppg_full, ppg_names=ch_names, distribution=args.distribution,
        ),
        "efficiency": efficiency,
    }

    # A/B 不作为模型输入，但保留分状态结果用于泛化分析。
    if dataset.metadata is not None and "activity" in dataset.metadata.columns:
        by_activity = {}
        activities = dataset.metadata["activity"].astype(str).to_numpy()
        for activity in sorted(np.unique(activities)):
            mask = activities == activity
            if not np.any(mask):
                continue
            by_activity[activity] = evaluate_all(
                pred[mask], ecg_true[mask], fs, dataset.ecg_channels,
                subject_ids[mask], ppg=ppg_full[mask], ppg_names=ch_names,
                distribution=False,
            )
        results["model_by_activity"] = by_activity

    # 朴素基线锚点
    if args.baselines:
        train_ds = create_dataset(config["data"]["dataset"],
                                  config["data"]["root"], split="train")
        train_ppg, train_ecg = train_ds._x, train_ds._y

        mean_tpl = baseline_mean_template(train_ecg, len(dataset))
        results["baseline_mean_template"] = evaluate_all(
            mean_tpl, ecg_true, fs, dataset.ecg_channels, subject_ids,
            ppg=ppg_full, ppg_names=ch_names)

        zero = baseline_zero(len(dataset), ecg_true.shape[1], ecg_true.shape[2])
        results["baseline_zero"] = evaluate_all(
            zero, ecg_true, fs, dataset.ecg_channels, subject_ids)

        delayed_idx = 0 if ppg_idx is None else ppg_idx
        delayed = baseline_delayed_ppg(train_ppg, train_ecg, ppg_full, delayed_idx, fs)
        results["baseline_delayed_ppg"] = evaluate_all(
            delayed, ecg_true, fs, dataset.ecg_channels, subject_ids,
            ppg=ppg_full, ppg_names=ch_names)

    # 保存
    out_path = args.run / f"eval_{args.split}.json"
    with open(out_path, "w", encoding="utf-8") as f:
        json.dump(results, f, ensure_ascii=False, indent=2, default=float)

    # 打印对比表
    KEY_METRICS = [
        "waveform/macro/rmse/macro", "waveform/macro/mae/macro",
        "waveform/macro/pcc/macro", "waveform/macro/snr_db/macro",
        "physiology/hr_err_bpm", "physiology/rpeak_f1",
        "physiology/rpeak_time_err_ms", "physiology/qrs_width_err_ms",
        "physiology/rmssd_err_ms", "physiology/hr_slope",
    ]
    methods = [k for k in results if isinstance(results[k], dict)
               and "waveform" in results[k]]
    print(f"\n{'metric':<28s}" + "".join(f"{m.replace('baseline_', 'b_'):>22s}" for m in methods))
    flats = {m: flatten_summary(results[m]) for m in methods}
    for key in KEY_METRICS:
        row = f"{key:<28s}"
        for m in methods:
            v = flats[m].get(key, float("nan"))
            row += f"{v:>22.4f}"
        print(row)
    if "validity_rate" in results["model"]:
        print(f"{'validity_rate':<28s}" +
              "".join(f"{results[m].get('validity_rate', float('nan')):>22.4f}"
                      for m in methods))
    print(f"\n效率: params={efficiency['params_M']:.2f}M, "
          f"inference={efficiency['inference_time_s']:.2f}s, "
          f"RTF={efficiency['rtf']:.4f} ({efficiency['device']})")
    print(f"结果已保存: {out_path}")


if __name__ == "__main__":
    main()
