"""评估指标模块正确性测试: 恒等预测必须完美, 均值模板必须拉胯, 加噪预测居中。"""

import json
import sys
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

from src.evaluation.metrics import (
    evaluate_all,
    evaluate_distribution,
    hr_transfer_slope,
    ood_gap,
    aggregate_by_subject,
)

OUT = ROOT / "data" / "processed" / "SensSmartTech" / "subjectwise_per-lead"
data = np.load(OUT / "test.npz", allow_pickle=False)
y = data["y"]          # [N, C, T] ECG
x = data["x"]          # [N, C_p, T] PPG
LEADS = [str(c) for c in data["ecg_channels"]]
PPG_NAMES = [str(c) for c in data["ppg_channels"]]
FS = int(data["fs"])

import pandas as pd
meta = pd.read_csv(OUT / "test_metadata.csv")
sids = meta["subject_id"].to_numpy()
N = len(y)

report = []


def log(msg=""):
    print(msg)
    report.append(str(msg))


failures = []


def check(name, cond, detail=""):
    status = "PASS" if cond else "FAIL"
    log(f"  [{status}] {name}" + (f" — {detail}" if detail else ""))
    if not cond:
        failures.append(name)


log("=" * 60)
log("评估指标模块正确性测试")
log(f"test 集: {N} 窗, {len(set(sids))} 被试, fs={FS}Hz, leads={LEADS}")
log("=" * 60)

# ---------- 1. 恒等预测: 所有指标必须完美 ----------
log("\n[1] 恒等预测 (pred == true)")
res_id = evaluate_all(y, y, FS, LEADS, sids, ppg=x, ppg_names=PPG_NAMES)
w = res_id["waveform"]["macro"]
check("rmse ≈ 0", w["rmse/macro"]["mean"] < 1e-6, f"{w['rmse/macro']['mean']:.2e}")
check("pcc ≈ 1", abs(w["pcc/macro"]["mean"] - 1) < 1e-4,
      f"{w['pcc/macro']['mean']:.6f}")
check("snr > 60 dB (EPS 地板 ≈78)", w["snr_db/macro"]["mean"] > 60,
      f"{w['snr_db/macro']['mean']:.1f}")
p = res_id["physiology"]
check("hr_err ≈ 0", p["hr_err_bpm"]["mean"] < 1e-9, f"{p['hr_err_bpm']['mean']:.2e}")
check("rpeak_f1 ≈ 1", p["rpeak_f1"]["mean"] > 0.99, f"{p['rpeak_f1']['mean']:.4f}")
check("rpeak_time_err ≈ 0", p["rpeak_time_err_ms"]["mean"] < 1e-9)
check("qrs_width_err ≈ 0", p["qrs_width_err_ms"]["mean"] < 1e-9)
check("rmssd_err ≈ 0", p["rmssd_err_ms"]["mean"] < 1e-9)
check("validity_rate = 1", res_id["validity_rate"] == 1.0)
check("hr_slope ≈ 1", abs(res_id["physiology"]["hr_slope"] - 1) < 0.01,
      f"{res_id['physiology']['hr_slope']:.4f}")
for site in ("carotid", "brachial"):
    k = f"ptt_{site}_err_s"
    if k in p:
        check(f"ptt_{site}_err ≈ 0", p[k]["mean"] < 1e-9,
              f"{p[k]['mean']:.2e}")

# ---------- 2. 真实 HR 分布合理性 ----------
log("\n[2] 真实数据心率范围 (生理效度)")
hr_true_med = np.nanmedian(res_id["physiology"]["hr_true"]["mean"])
check("test 集真实 HR 在 50-120 bpm", 50 < hr_true_med < 120,
      f"被试级 HR 中位数 {hr_true_med:.1f} bpm")
ptt_c = res_id["physiology"].get("ptt_carotid_pred_s", {}).get("mean", float("nan"))
ptt_b = res_id["physiology"].get("ptt_brachial_pred_s", {}).get("mean", float("nan"))
check("真实 PTT 颈动脉在宽生理区间 (50-350 ms)", 0.05 < ptt_c < 0.35,
      f"{ptt_c*1000:.0f} ms (文献参考 80-200, 本数据集自定义传感器偏长)")
check("真实 PTT 肱动脉在宽生理区间 (100-350 ms)", 0.10 < ptt_b < 0.35,
      f"{ptt_b*1000:.0f} ms (文献参考 150-300)")
check("PTT 颈动脉 < 肱动脉 (解剖学)", ptt_c < ptt_b,
      f"{ptt_c*1000:.0f} < {ptt_b*1000:.0f} ms")

# ---------- 3. 均值模板: 必须呈现 'RMSE 尚可 + PCC 崩' ----------
log("\n[3] 均值模板基线 (平均波形现象)")
train = np.load(OUT / "train.npz", allow_pickle=False)
template = np.tile(train["y"].mean(axis=0)[None], (N, 1, 1)).astype(np.float32)
res_tpl = evaluate_all(template, y, FS, LEADS, sids, ppg=x, ppg_names=PPG_NAMES)
wt = res_tpl["waveform"]["macro"]
check("模板 rmse 明显大于 0 (>0.3)", wt["rmse/macro"]["mean"] > 0.3,
      f"{wt['rmse/macro']['mean']:.3f}")
check("模板 pcc 崩塌 (<0.3)", wt["pcc/macro"]["mean"] < 0.3,
      f"{wt['pcc/macro']['mean']:.3f}")
check("模板 HR 误差大 (>5 bpm)", res_tpl["physiology"]["hr_err_bpm"]["mean"] > 5,
      f"{res_tpl['physiology']['hr_err_bpm']['mean']:.1f} bpm")
check("模板 R峰F1 低 (<0.8, 无个体节律)", res_tpl["physiology"]["rpeak_f1"]["mean"] < 0.8,
      f"{res_tpl['physiology']['rpeak_f1']['mean']:.3f}")
check("模板 HR 斜率远离 1", abs(res_tpl["physiology"].get("hr_slope", 0) - 1) > 0.3,
      f"slope={res_tpl['physiology'].get('hr_slope', float('nan')):.3f}")

# ---------- 4. 加噪预测: 居中行为 ----------
log("\n[4] 加噪预测 (中间强度)")
rng = np.random.default_rng(0)
noisy = (y + rng.normal(0, 0.3, y.shape)).astype(np.float32)
res_noise = evaluate_all(noisy, y, FS, LEADS, sids)
check("噪声 rmse ≈ 0.3", abs(res_noise["waveform"]["macro"]["rmse/macro"]["mean"] - 0.3) < 0.05,
      f"{res_noise['waveform']['macro']['rmse/macro']['mean']:.3f}")
check("噪声 pcc 居中 (0.5-0.95)", 0.5 < res_noise["waveform"]["macro"]["pcc/macro"]["mean"] < 0.95,
      f"{res_noise['waveform']['macro']['pcc/macro']['mean']:.3f}")

# ---------- 5. 分布指标行为 ----------
log("\n[5] 1-NNA / Coverage 行为")
dist_identity = evaluate_distribution(y[:80], y[:80])
check("恒等分布: coverage ≈ 1", dist_identity["coverage"] > 0.95,
      f"{dist_identity['coverage']:.3f}")
# 配对加噪: 生成物紧贴自身条件输入 -> 应完美可分 (1nna≈0) 且全覆盖
dist_paired = evaluate_distribution(noisy[:80], y[:80])
check("配对加噪: 1nna ≈ 0 (可分, 行为正确)", dist_paired["1nna_acc"] < 0.1,
      f"{dist_paired['1nna_acc']:.3f}")
check("配对加噪: coverage ≈ 1", dist_paired["coverage"] > 0.95,
      f"{dist_paired['coverage']:.3f}")
# 不相交子集 + 微噪: 两分布来自同总体 -> 1nna 应接近 0.5 (不可分)
# (注: 同集合置换无法去配对 —— 置换后仍含每个样本自身副本)
tiny = (y[100:180] + rng.normal(0, 0.05, (80, *y.shape[1:]))).astype(np.float32)
dist_unpaired = evaluate_distribution(tiny, y[:80])
check("不相交微噪: 1nna 接近 0.5 (不可分)", 0.3 < dist_unpaired["1nna_acc"] < 0.7,
      f"{dist_unpaired['1nna_acc']:.3f}")

# ---------- 6. OOD gap 工具 ----------
log("\n[6] OOD gap 工具")
gap = ood_gap({"rmse/macro": {"mean": 0.3}}, {"rmse/macro": {"mean": 0.6}})
check("OOD gap 计算 ≈ 1.0", abs(gap["gap"] - 1.0) < 1e-6,
      f"{gap['gap']:.6f}")

# ---------- 7. 逐导联分解存在性 ----------
log("\n[7] 逐导联分解")
for lead in LEADS:
    present = any(k.endswith(f"/{lead}") for k in res_id["waveform"]["per_lead"][lead])
    check(f"per_lead[{lead}] 含波形指标", present and
          res_id["waveform"]["per_lead"][lead] != {})

log(f"\n{'='*60}")
log("总结论: " + ("全部通过" if not failures else f"{len(failures)} 项失败: {failures}"))
log(f"{'='*60}")

(ROOT / "data_check" / "metrics_check_report.txt").write_text(
    "\n".join(report) + "\n", encoding="utf-8")
print("\n报告已保存: data_check/metrics_check_report.txt")
