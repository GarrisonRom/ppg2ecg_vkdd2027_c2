"""多层评估指标体系 (PPG→ECG 重建)。

层级 (按讨论定稿的矩阵, 已修正):
  L1 波形重建:  MSE / RMSE / MAE / PCC (逐导联) / DTW近似 / SNR
  L2 生理一致:  HR误差 / R峰F1+时序误差 / RMSSD误差 / QRS宽度误差 /
                PTT误差(分部位) / 生成有效率 / HR传递斜率
  L3 泛化对比:  OOD gap 等由 evaluate_all 的多组结果相减得到 (见 ood_gap)
  L4 分布质量:  1-NNA / Coverage (无需特征编码器; FID/IS 暂缓)
  L5 效率:      在 scripts/evaluate.py 中测量 (时延/RTF/参数量)

明确不实现的: LF/HF —— 频域 HRV 需要 >=2min 窗口, 本数据集 8s/30s 记录
不满足统计效度 (详见 paper limitation)。

聚合协议: 逐窗口计算 -> 被试内平均 -> 跨被试 mean±std (n = 被试数)。
所有生理指标基于 lead II (SensSmartTech 的 I/II/V3/V4 中 R 波最显著)。
"""

from __future__ import annotations

from typing import Optional

import numpy as np
import pandas as pd
from scipy.signal import find_peaks, peak_widths

EPS = 1e-8
HR_RANGE = (40.0, 180.0)          # 有效心率范围 (bpm)
PEAK_MATCH_TOL_SEC = 0.15         # R 峰匹配容差
PTT_SITE_RANGES = {               # R峰->PPG峰 的生理合理区间 (秒)
    "carotid": (0.08, 0.20),
    "brachial": (0.15, 0.30),
}


# ======================================================================
# 信号处理原语
# ======================================================================

def _centered(x: np.ndarray, win: int) -> np.ndarray:
    """去基线 (滑动均值), win 为样本数。"""
    win = max(1, int(win))
    kernel = np.ones(win) / win
    return x - np.convolve(x, kernel, mode="same")


def detect_r_peaks(sig: np.ndarray, fs: float) -> np.ndarray:
    """单导联 R 峰检测 (信号已带通+归一化)。

    Returns: R 峰样本索引; 无效信号返回空数组。
    """
    sig = np.asarray(sig, dtype=np.float64)
    if sig.size < int(0.5 * fs) or not np.isfinite(sig).all():
        return np.array([], dtype=int)
    x = _centered(sig, int(0.25 * fs))
    std = x.std()
    if std < 1e-6:
        return np.array([], dtype=int)
    min_dist = int(60.0 / HR_RANGE[1] * fs)  # 最大心率对应的 RR
    peaks, _ = find_peaks(x, distance=min_dist, prominence=0.4 * std)
    # 剔除 RR < 60/max_hr 的相邻峰 (find_peaks 的 distance 已保证)
    return peaks


def match_peaks(pred: np.ndarray, true: np.ndarray, tol: int) -> tuple[int, int, int, list[float]]:
    """贪心最近邻匹配两组峰。

    Returns: (TP, FP, FN, 匹配对的 |Δt| 列表, 单位样本)
    """
    if len(pred) == 0 or len(true) == 0:
        return 0, len(pred), len(true), []
    used = np.zeros(len(true), dtype=bool)
    errors: list[float] = []
    true_f = true.astype(np.float64)
    for p in pred:
        d = np.abs(true_f - p)
        d[used] = np.inf
        j = int(np.argmin(d))
        if d[j] <= tol:
            used[j] = True
            errors.append(float(d[j]))
    tp = int(used.sum())
    return tp, len(pred) - tp, len(true) - tp, errors


def hr_from_peaks(peaks: np.ndarray, fs: float) -> float:
    """心率 (bpm), 由 RR 中位数估计; 峰数不足返回 NaN。"""
    if len(peaks) < 2:
        return float("nan")
    rr = np.diff(peaks) / fs
    return float(60.0 / np.median(rr))


def rmssd_from_peaks(peaks: np.ndarray, fs: float) -> float:
    """RMSSD (ms); 至少 3 个峰, 否则 NaN。"""
    if len(peaks) < 3:
        return float("nan")
    rr_ms = np.diff(peaks) / fs * 1000.0
    return float(np.sqrt(np.mean(np.diff(rr_ms) ** 2)))


def qrs_width_ms(sig: np.ndarray, peaks: np.ndarray, fs: float) -> float:
    """QRS 宽度代理 (半峰高宽度, ms); 峰数不足返回 NaN。"""
    if len(peaks) < 1:
        return float("nan")
    x = _centered(sig, int(0.25 * fs))
    widths = peak_widths(x, peaks, rel_height=0.5)[0]
    widths = widths[(widths > 0.02 * fs) & (widths < 0.3 * fs)]  # 20-300ms 内才合理
    if len(widths) == 0:
        return float("nan")
    return float(np.median(widths) / fs * 1000.0)


def ptt_from_peaks(r_peaks: np.ndarray, ppg_sig: np.ndarray, fs: float,
                   max_lag: float = 0.6) -> float:
    """平均 PTT (s): 每个 R 峰到其后最近 PPG 收缩峰的延迟。"""
    if len(r_peaks) < 1:
        return float("nan")
    ppg = np.asarray(ppg_sig, dtype=np.float64)
    x = _centered(ppg, int(0.5 * fs))
    if x.std() < 1e-6:
        return float("nan")
    p_peaks, _ = find_peaks(x, distance=int(0.3 * fs), prominence=0.4 * x.std())
    if len(p_peaks) == 0:
        return float("nan")
    ptts = []
    for r in r_peaks:
        later = p_peaks[p_peaks >= r]
        if len(later) == 0 or later[0] - r > max_lag * fs:
            continue
        ptts.append((later[0] - r) / fs)
    return float(np.mean(ptts)) if len(ptts) >= 2 else float("nan")


# ======================================================================
# L1 波形重建 (逐导联 + 宏观)
# ======================================================================

def waveform_metrics_1v1(pred: np.ndarray, true: np.ndarray) -> dict[str, float]:
    """单导联 [T] 波形指标。"""
    err = pred - true
    mse = float(np.mean(err ** 2))
    rmse = float(np.sqrt(mse))
    mae = float(np.mean(np.abs(err)))
    snr = float(10.0 * np.log10(np.mean(true ** 2) / (np.mean(err ** 2) + EPS)))
    pc = pred - pred.mean()
    tc = true - true.mean()
    denom = np.sqrt(float(pc @ pc) * float(tc @ tc))
    pcc = float(pc @ tc / denom) if denom > EPS else 0.0
    pcc = float(np.clip(pcc, -1.0, 1.0))
    # DTW 近似: 多尺度 L1
    dtw = mae
    n_scales = 1
    for k in (5, 15, 50):
        if pred.size >= k:
            n = pred.size // k
            dtw += float(np.mean(np.abs(
                pred[: n * k].reshape(n, k).mean(1) - true[: n * k].reshape(n, k).mean(1))))
            n_scales += 1
    return {"mse": mse, "rmse": rmse, "mae": mae, "pcc": pcc,
            "snr_db": snr, "dtw": dtw / n_scales}


def evaluate_waveform(pred: np.ndarray, target: np.ndarray,
                      lead_names: list[str]) -> dict:
    """逐导联波形指标, 输出 [N, C] 级中间结果。

    Args:
        pred, target: [N, C, T]
    Returns:
        {"per_window": DataFrame(每行一个窗口, 每导联各指标列),
         "lead_names": lead_names}
    """
    n, c, _ = pred.shape
    rows = []
    for i in range(n):
        row: dict = {"window": i}
        for j, name in enumerate(lead_names):
            m = waveform_metrics_1v1(pred[i, j], target[i, j])
            for k, v in m.items():
                row[f"{k}/{name}"] = v
        # 宏观 = 各导联平均
        for key in ("mse", "rmse", "mae", "pcc", "snr_db", "dtw"):
            row[f"{key}/macro"] = float(np.mean(
                [row[f"{key}/{name}"] for name in lead_names]))
        rows.append(row)
    return {"per_window": pd.DataFrame(rows), "lead_names": lead_names}


# ======================================================================
# L2 生理一致 (lead II 基准, 含 PPG 交互)
# ======================================================================

def window_physiology(pred_win: np.ndarray, true_win: np.ndarray, fs: float,
                      ppg_win: Optional[np.ndarray] = None,
                      ppg_names: Optional[list[str]] = None,
                      r_lead_idx: int = 1) -> dict[str, float]:
    """单个窗口的生理指标。

    Args:
        pred_win, true_win: [C, T] (归一化 ECG)
        ppg_win: [C_p, T] 或 None (PTT 需要)
    """
    tol = int(PEAK_MATCH_TOL_SEC * fs)
    true_sig = true_win[r_lead_idx]
    pred_sig = pred_win[r_lead_idx]

    out: dict[str, float] = {}

    # --- 有效性 (对生成信号) ---
    finite = bool(np.isfinite(pred_win).all())
    pred_peaks = detect_r_peaks(pred_sig, fs)
    hr_pred = hr_from_peaks(pred_peaks, fs)
    valid = finite and len(pred_peaks) >= 2 and HR_RANGE[0] <= hr_pred <= HR_RANGE[1]
    out["valid"] = float(valid)
    out["n_r_peaks_pred"] = float(len(pred_peaks))

    # --- R 峰检测质量 (生成 vs 真实) ---
    true_peaks = detect_r_peaks(true_sig, fs)
    tp, fp, fn, errs = match_peaks(pred_peaks, true_peaks, tol)
    precision = tp / (tp + fp) if tp + fp > 0 else 0.0
    recall = tp / (tp + fn) if tp + fn > 0 else 0.0
    out["rpeak_f1"] = (2 * precision * recall / (precision + recall)
                       if precision + recall > 0 else 0.0)
    out["rpeak_time_err_ms"] = (float(np.mean(errs) / fs * 1000.0)
                                if errs else float("nan"))
    out["rpeak_precision"] = precision
    out["rpeak_recall"] = recall

    # --- HR ---
    hr_true = hr_from_peaks(true_peaks, fs)
    out["hr_pred"] = hr_pred
    out["hr_true"] = hr_true
    out["hr_err_bpm"] = abs(hr_pred - hr_true) if np.isfinite(hr_pred) else float("nan")

    # --- HRV (超短时) ---
    out["rmssd_err_ms"] = abs(rmssd_from_peaks(pred_peaks, fs)
                              - rmssd_from_peaks(true_peaks, fs))

    # --- QRS 宽度 ---
    out["qrs_width_err_ms"] = abs(qrs_width_ms(pred_sig, pred_peaks, fs)
                                  - qrs_width_ms(true_sig, true_peaks, fs))

    # --- PTT (分部位; 需要 PPG 输入) ---
    if ppg_win is not None and ppg_names is not None:
        for site in PTT_SITE_RANGES:
            chans = [i for i, nm in enumerate(ppg_names) if site in nm]
            if not chans:
                continue
            lo, hi = PTT_SITE_RANGES[site]
            ptt_true = ptt_from_peaks(true_peaks, ppg_win[chans[0]], fs)
            ptt_pred = ptt_from_peaks(pred_peaks, ppg_win[chans[0]], fs)
            out[f"ptt_{site}_pred_s"] = ptt_pred
            out[f"ptt_{site}_err_s"] = (abs(ptt_pred - ptt_true)
                                        if np.isfinite(ptt_pred) and np.isfinite(ptt_true)
                                        else float("nan"))
            out[f"ptt_{site}_in_range"] = float(
                np.isfinite(ptt_pred) and lo <= ptt_pred <= hi)
    return out


def evaluate_physiology(pred: np.ndarray, target: np.ndarray, fs: float,
                        r_lead_idx: int = 1,
                        ppg: Optional[np.ndarray] = None,
                        ppg_names: Optional[list[str]] = None) -> pd.DataFrame:
    """所有窗口的生理指标, 每行一个窗口。pred/target: [N, C, T]。"""
    rows = []
    for i in range(pred.shape[0]):
        row = window_physiology(
            pred[i], target[i], fs,
            ppg_win=(ppg[i] if ppg is not None else None),
            ppg_names=ppg_names, r_lead_idx=r_lead_idx,
        )
        row["window"] = i
        rows.append(row)
    return pd.DataFrame(rows)


def hr_transfer_slope(df: pd.DataFrame) -> dict[str, float]:
    """HR 传递函数: hr_pred ~ a*hr_true + b (跨窗口回归, 因果式证据)。"""
    m = df[["hr_pred", "hr_true"]].dropna()
    m = m[m["hr_pred"] != 0]
    if len(m) < 10:
        return {"hr_slope": float("nan"), "hr_intercept": float("nan"),
                "hr_transfer_r": float("nan")}
    x, y = m["hr_true"].to_numpy(), m["hr_pred"].to_numpy()
    slope, intercept = np.polyfit(x, y, 1)
    if x.std() < EPS or y.std() < EPS:
        r = float("nan")
    else:
        r = float(np.corrcoef(x, y)[0, 1])
    return {"hr_slope": float(slope), "hr_intercept": float(intercept),
            "hr_transfer_r": r}


# ======================================================================
# L4 分布质量 (1-NNA / Coverage, 无需特征编码器)
# ======================================================================

def evaluate_distribution(pred: np.ndarray, target: np.ndarray) -> dict[str, float]:
    """1-NNA 与 Coverage (原始信号空间欧氏距离)。

    1-NNA: 最近邻类别投票的准确率, 50% = 真假不可分 (理想)。
    Coverage: 真实样本中, 距最近生成样本 < 距最近其他真实样本 的比例。
    """
    x = target.reshape(len(target), -1).astype(np.float32)
    y = pred.reshape(len(pred), -1).astype(np.float32)

    def nn_dist(a: np.ndarray, b: np.ndarray) -> np.ndarray:
        # a: [M, D], b: [K, D] -> a 中每行到 b 的最小距离
        d2 = ((a[:, None, :] - b[None, :, :]) ** 2).sum(-1)
        return np.sqrt(d2.min(axis=1))

    # 真实->真实 (排除自身)
    xx = np.sqrt(((x[:, None, :] - x[None, :, :]) ** 2).sum(-1))
    np.fill_diagonal(xx, np.inf)
    d_x_to_x = xx.min(axis=1)
    d_x_to_y = nn_dist(x, y)
    yy = np.sqrt(((y[:, None, :] - y[None, :, :]) ** 2).sum(-1))
    np.fill_diagonal(yy, np.inf)
    d_y_to_y = yy.min(axis=1)
    d_y_to_x = nn_dist(y, x)

    real_correct = (d_x_to_x < d_x_to_y).mean()
    fake_correct = (d_y_to_y < d_y_to_x).mean()
    nna_acc = float((real_correct + fake_correct) / 2.0)
    coverage = float((d_x_to_y < d_x_to_x).mean())
    return {"1nna_acc": nna_acc, "coverage": coverage}


# ======================================================================
# 聚合: 窗口 -> 被试 -> mean±std
# ======================================================================

def aggregate_by_subject(df: pd.DataFrame, subject_ids: np.ndarray) -> dict:
    """逐窗口指标按被试平均, 再跨被试 mean±std。

    Returns: {metric: {"mean": m, "std": s, "n_subjects": n, "n_valid": k}}
    """
    df = df.copy()
    df["subject_id"] = np.asarray(subject_ids)[: len(df)]
    by_subj = df.groupby("subject_id").mean(numeric_only=True)
    summary = {}
    for col in by_subj.columns:
        vals = by_subj[col].dropna()
        summary[col] = {
            "mean": float(vals.mean()) if len(vals) else float("nan"),
            "std": float(vals.std(ddof=1)) if len(vals) > 1 else 0.0,
            "n_subjects": int(len(vals)),
        }
    return summary


# ======================================================================
# L3 泛化对比工具
# ======================================================================

def ood_gap(summary_in: dict, summary_out: dict,
            metric: str = "rmse/macro") -> dict[str, float]:
    """跨域泛化差距: (Out - In) / In。

    Args:
        summary_in:  被试内/recordwise 的 aggregate_by_subject 输出
        summary_out: subjectwise (跨被试) 的 aggregate_by_subject 输出
    """
    if metric not in summary_in or metric not in summary_out:
        raise KeyError(f"指标 {metric} 不在两个 summary 中")
    m_in = summary_in[metric]["mean"]
    m_out = summary_out[metric]["mean"]
    return {
        "metric": metric,
        "in_subject": m_in,
        "out_subject": m_out,
        "gap": (m_out - m_in) / (abs(m_in) + EPS),
    }


# ======================================================================
# 总入口
# ======================================================================

def evaluate_all(pred: np.ndarray, target: np.ndarray, fs: float,
                 lead_names: list[str], subject_ids: np.ndarray,
                 r_lead_idx: int = 1,
                 ppg: Optional[np.ndarray] = None,
                 ppg_names: Optional[list[str]] = None,
                 distribution: bool = False) -> dict:
    """计算 L1/L2 (+可选 L4) 全部指标并按被试聚合。

    Args:
        pred, target: [N, C, T] 归一化 ECG
        fs: 采样率 (Hz)
        lead_names: 导联名列表
        subject_ids: [N] 每窗口被试编号
        ppg: [N, C_p, T] PPG 输入 (PTT 需要; 可 None)
        ppg_names: PPG 通道名
        distribution: 是否计算 1-NNA/Coverage (O(N²·D), N 大时较慢)
    """
    if pred.shape != target.shape:
        raise ValueError(f"pred {pred.shape} 与 target {target.shape} 形状不符")
    n = pred.shape[0]
    if len(subject_ids) != n:
        raise ValueError("subject_ids 长度与窗口数不符")

    wave = evaluate_waveform(pred, target, lead_names)
    phys = evaluate_physiology(pred, target, fs, r_lead_idx=r_lead_idx,
                               ppg=ppg, ppg_names=ppg_names)
    phys_slope = hr_transfer_slope(phys)

    wave_summary = aggregate_by_subject(wave["per_window"], subject_ids)
    phys_summary = aggregate_by_subject(phys, subject_ids)

    # 生成有效率 (窗口级, 不做被试平均 —— 直接报告比例)
    valid_rate = float(phys["valid"].mean())

    result = {
        "n_windows": int(n),
        "n_subjects": int(len(np.unique(subject_ids))),
        "validity_rate": valid_rate,
        "waveform": {
            "macro": {k: v for k, v in wave_summary.items()
                      if k.endswith("/macro")},
            "per_lead": {name: {k: v for k, v in wave_summary.items()
                                if k.endswith(f"/{name}")}
                         for name in lead_names},
        },
        "physiology": {**phys_summary, **phys_slope},
    }
    if distribution:
        result["distribution"] = evaluate_distribution(pred, target)
    return result
