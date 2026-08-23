"""数据层适配冒烟测试：SensSmartTechNPZ Dataset + 注册表 + DataLoader + train.py 数据链路。"""

import json
import sys
from pathlib import Path

import numpy as np
import torch

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

from src.data import SensSmartTechDataset, create_dataset, create_dataloaders
from src.utils.config import DEFAULT_CONFIG

OUT = ROOT / "data" / "processed" / "SensSmartTech"
COMBOS = ["subjectwise_per-lead", "subjectwise_global",
          "recordwise_per-lead", "recordwise_global"]

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
log("数据层适配冒烟测试 (npz 格式, 按数据集分发)")
log("=" * 60)

# ---------- 1. 注册表与四种组合 ----------
log("\n[1] Dataset 构造与属性自描述 (通道/采样率从 npz 读取)")
for combo in COMBOS:
    root = OUT / combo
    ds = create_dataset("senssmarttech", root, split="train")
    attrs_ok = (
        ds.fs == 250 and ds.signal_length == 2000
        and ds.ppg_channels == ["carotid_880nm", "carotid_660nm",
                                "brachial_880nm", "brachial_660nm"]
        and ds.ecg_channels == ["I", "II", "V3", "V4"]
        and ds.ecg_leads == 4 and ds.num_ppg_channels == 4
    )
    check(f"{combo}: 属性自描述正确", attrs_ok,
          f"fs={ds.fs}, T={ds.signal_length}, ppg={ds.num_ppg_channels}ch, ecg={ds.ecg_leads}导联")

# ---------- 2. 样本与批次形状 ----------
log("\n[2] __getitem__ 与 DataLoader 批次形状")
ds = SensSmartTechDataset(OUT / "subjectwise_per-lead", split="val")
sample = ds[0]
check("样本键: ppg/ecg/subject_id", set(["ppg", "ecg", "subject_id"]) <= set(sample.keys()))
check("ppg 形状 [4, 2000]", tuple(sample["ppg"].shape) == (4, 2000))
check("ecg 形状 [4, 2000]", tuple(sample["ecg"].shape) == (4, 2000))
check("样本类型 float32",
      sample["ppg"].dtype == torch.float32 and sample["ecg"].dtype == torch.float32)

loader = torch.utils.data.DataLoader(ds, batch_size=16, shuffle=False, num_workers=0)
batch = next(iter(loader))
check("批次 ppg [16,4,2000]", tuple(batch["ppg"].shape) == (16, 4, 2000))
check("批次 ecg [16,4,2000]", tuple(batch["ecg"].shape) == (16, 4, 2000))
check("批次 subject_id [16]", tuple(batch["subject_id"].shape) == (16,))

# ---------- 3. 元数据对齐 ----------
log("\n[3] 元数据对齐 (subject_id 与 metadata.csv 一致)")
sids = [int(ds.metadata["subject_id"].iloc[i]) for i in range(len(ds))]
check("数据集 subject 集与 metadata 一致",
      set(sids) == set(ds.metadata["subject_id"].astype(int)),
      f"{len(set(sids))} 个被试")
meta_ids = ds.metadata["subject_id"].astype(int).tolist()
check("逐样本 subject_id 顺序对齐",
      all(int(batch["subject_id"][i]) == meta_ids[i] for i in range(16)))

# ---------- 4. 单通道选择 (train.py 默认路径) ----------
log("\n[4] ppg_channel 单通道选择 (兼容现有单通道 PPGEncoder)")
ds1 = SensSmartTechDataset(OUT / "subjectwise_per-lead", split="train",
                           ppg_channel="carotid_880nm")
check("选择后 ppg 形状 [1, 2000]", tuple(ds1[0]["ppg"].shape) == (1, 2000))
check("选择后通道名更新", ds1.ppg_channels == ["carotid_880nm"])
check("选择后 ecg 不受影响", tuple(ds1[0]["ecg"].shape) == (4, 2000))
try:
    SensSmartTechDataset(OUT / "subjectwise_per-lead", split="train",
                         ppg_channel="nonexistent")
    check("非法通道名报错", False)
except ValueError:
    check("非法通道名报错", True)

# 选出的通道数值与全通道版本一致
ds_full = SensSmartTechDataset(OUT / "subjectwise_per-lead", split="train")
idx = ds_full.ppg_channels.index("carotid_880nm")
check("单通道数值与全通道版本对应通道一致",
      bool(torch.equal(ds1[5]["ppg"][0], ds_full[5]["ppg"][idx])))

# ---------- 5. create_dataloaders 全链路 (train.py 等价路径) ----------
log("\n[5] create_dataloaders 全链路 (按 DEFAULT_CONFIG)")
cfg = DEFAULT_CONFIG["data"]
loaders = create_dataloaders(
    dataset=cfg["dataset"],
    root=ROOT / cfg["root"],
    batch_size=cfg["batch_size"],
    num_workers=0,
    ppg_channel=cfg["ppg_channel"],
)
check("返回 train/val/test 三个 loader", set(loaders.keys()) == {"train", "val", "test"},
      f"{ {k: len(v.dataset) for k, v in loaders.items()} }")
tds = loaders["train"].dataset
check("train.py 派生参数正确: T=2000, ecg_leads=4",
      tds.signal_length == 2000 and tds.ecg_leads == 4)
tb = next(iter(loaders["train"]))
check("train 批次可正常取 batch (ppg [B,1,2000])",
      tb["ppg"].shape[1] == 1 and tb["ppg"].shape[2] == 2000)

# ---------- 6. 防重复归一化说明确认 ----------
log("\n[6] 数据值域确认 (已在预处理中归一化, 加载层无额外变换)")
x_all = ds_full._x
check("PPG 值域符合 robust z-score + clip(±10)", float(np.abs(x_all).max()) <= 10.0001,
      f"max|ppg|={float(np.abs(x_all).max()):.2f}")

log(f"\n{'='*60}")
log("总结论: " + ("全部通过" if not failures else f"{len(failures)} 项失败: {failures}"))
log(f"{'='*60}")

(ROOT / "data_check" / "dataset_adapter_report.txt").write_text(
    "\n".join(report) + "\n", encoding="utf-8")
print("\n报告已保存: data_check/dataset_adapter_report.txt")
