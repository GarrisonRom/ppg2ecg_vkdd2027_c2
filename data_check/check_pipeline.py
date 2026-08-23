"""验证 v2 预处理管线产物：划分复现性、形状、归一化语义、防泄漏。"""

import json
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parent.parent
OUT = ROOT / "data" / "processed" / "SensSmartTech"
INTERIM = ROOT / "data" / "interim" / "senssmarttech"
CHECK = ROOT / "data_check"

report = []


def log(msg=""):
    print(msg)
    report.append(str(msg))


log("=" * 60)
log("v2 预处理管线产物验证报告")
log("=" * 60)

failures = []


def check(name, cond, detail=""):
    status = "PASS" if cond else "FAIL"
    log(f"  [{status}] {name}" + (f" — {detail}" if detail else ""))
    if not cond:
        failures.append(name)


# ---------- 1. subjectwise seed=42 划分复现旧文件 ----------
log("\n[1] 划分复现性 (seed=42 vs 旧 split_subjects.json)")
old_split = json.loads((ROOT / "data" / "processed" / "SensSmartTech" / "split_subjects.json").read_text()) \
    if (OUT / ".." / "split_subjects.json").exists() else None
new_split = json.loads((OUT / "subjectwise_per-lead" / "split.json").read_text())
if old_split is not None:
    check("train 划分一致", old_split["train"] == new_split["train"])
    check("val 划分一致", old_split["val"] == new_split["val"])
    check("test 划分一致", old_split["test"] == new_split["test"])
else:
    log("  [skip] 旧 split_subjects.json 不存在")

index = pd.read_csv(INTERIM / "windows_index.csv")
log(f"  windows_index: {len(index)} 窗, {index['subject_id'].nunique()} 被试, "
    f"{index['record_id'].nunique()} 记录")

# ---------- 2. 四套缓存形状与数值 ----------
log("\n[2] NPZ 缓存形状与数值")
for combo in ["subjectwise_per-lead", "subjectwise_global",
              "recordwise_per-lead", "recordwise_global"]:
    d = OUT / combo
    shapes_ok, nan_ok, finite_ok = True, True, True
    n_windows = 0
    for part in ["train", "val", "test"]:
        data = np.load(d / f"{part}.npz", allow_pickle=False)
        x, y = data["x"], data["y"]
        n_windows += x.shape[0]
        shapes_ok &= (x.shape == y.shape and x.shape[1] == 4 and x.shape[2] == 2000)
        nan_ok &= bool(np.isfinite(x).all() and np.isfinite(y).all())
    check(f"{combo}: 共 {n_windows} 窗, 形状 [N,4,2000] 一致", shapes_ok)
    check(f"{combo}: 全部有限值 (无 NaN/Inf)", nan_ok)

# ---------- 3. 归一化语义 ----------
log("\n[3] ECG 归一化语义")
for combo in ["subjectwise_per-lead", "subjectwise_global"]:
    d = OUT / combo
    tr = np.load(d / "train.npz", allow_pickle=False)
    y = tr["y"]  # [N, 4, T]
    per_lead_std = y.std(axis=(0, 2))
    if "per-lead" in combo:
        check(f"{combo}: 各导联 std≈1 ({np.round(per_lead_std, 3).tolist()})",
              bool(np.allclose(per_lead_std, 1.0, atol=0.05)))
    else:
        ratios = per_lead_std / per_lead_std[0]
        spread = ratios.max() / ratios.min()
        check(f"{combo}: 导联幅度比保留 (std 比值 {np.round(ratios, 2).tolist()})",
              spread > 1.3)  # V3/V4 幅度应显著大于肢体导联

ppg_std = np.load(OUT / "subjectwise_per-lead" / "train.npz")["x"].std(axis=(0, 2))
check(f"PPG 逐通道归一化生效 (各通道 std {np.round(ppg_std, 2).tolist()})",
      bool(np.all(ppg_std > 0.1) and ppg_std.max() / ppg_std.min() < 5.0))

# ---------- 4. 防泄漏 ----------
log("\n[4] 防泄漏检查")
for combo, key in [("subjectwise_per-lead", "subject_id"),
                   ("recordwise_per-lead", "record_id")]:
    d = OUT / combo
    tr = set(pd.read_csv(d / "train_metadata.csv")[key])
    va = set(pd.read_csv(d / "val_metadata.csv")[key])
    te = set(pd.read_csv(d / "test_metadata.csv")[key])
    name = "被试" if key == "subject_id" else "记录"
    check(f"{combo}: train/val/test 无 {name} 交集",
          not (tr & va) and not (tr & te) and not (va & te))

# recordwise: 被试跨 split 出现是预期行为 (sample-wise 语义)
d = OUT / "recordwise_per-lead"
tr_s = set(pd.read_csv(d / "train_metadata.csv")["subject_id"])
te_s = set(pd.read_csv(d / "test_metadata.csv")["subject_id"])
log(f"  [info] recordwise 中 test 被试出现在 train 的有 {len(tr_s & te_s)}/"
    f"{len(te_s)} 人 (sample-wise 预期行为)")

# ---------- 5. 窗口重叠泄漏自查 ----------
log("\n[5] 窗口来源完整性")
idx = index.sort_values(["subject_id", "record_id", "start_sec"])
total = sum(len(pd.read_csv(OUT / c / f"{p}_metadata.csv"))
            for c in ["subjectwise_per-lead"] for p in ["train", "val", "test"])
check(f"各 split metadata 窗口总数 = windows_index ({total} = {len(index)})",
      total == len(index))

log(f"\n{'='*60}")
log("总结论: " + ("全部通过" if not failures else f"{len(failures)} 项失败: {failures}"))
log(f"{'='*60}")

(CHECK / "pipeline_check_report.txt").write_text("\n".join(report) + "\n", encoding="utf-8")
print(f"\n报告已保存: data_check/pipeline_check_report.txt")
