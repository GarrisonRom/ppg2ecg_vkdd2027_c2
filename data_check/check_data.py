"""SensSmartTech 数据可用性快速检验：文件完整性、采样率、信号质量、与预处理元数据一致性。"""

import json
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parent
DATA = ROOT.parent / "data"
RAW = DATA / "raw" / "SensSmartTech"
OUT = ROOT

report = []


def log(msg=""):
    print(msg)
    report.append(str(msg))


log("=" * 60)
log("SensSmartTech 数据可用性检验报告")
log("=" * 60)

# ---------- 1. 文件完整性 ----------
csv_dir = RAW / "CSV"
files = sorted(csv_dir.glob("*.csv"))
modalities = {}
for f in files:
    mod = f.stem.rsplit("_", 1)[-1]
    modalities[mod] = modalities.get(mod, 0) + 1

subjects = {f.stem.split("_")[0] for f in files}
log(f"\n[1] 文件完整性")
log(f"    CSV 总文件数: {len(files)}")
for m, n in sorted(modalities.items()):
    log(f"    {m}: {n} 个文件")
log(f"    受试者数: {len(subjects)}")

demog = pd.read_csv(RAW / "Demographics.csv", skiprows=1)
log(f"    Demographics 记录数: {len(demog)} (活动前后 B/A)")

# ---------- 2. 单条记录信号检验 ----------
rec = "10_18-12-33"
ppg = pd.read_csv(csv_dir / f"{rec}_ppg.csv")
ecg = pd.read_csv(csv_dir / f"{rec}_ecg.csv")

log(f"\n[2] 示例记录 {rec}")
log(f"    PPG 列: {list(ppg.columns)}")
log(f"    ECG 列: {list(ecg.columns)}")

ppg_dt = np.diff(ppg["t"].values) / 1000.0  # t 单位为 ms
ecg_dt = np.diff(ecg["t"].values) / 1000.0
ppg_fs = 1.0 / np.median(ppg_dt)
ecg_fs = 1.0 / np.median(ecg_dt)
ppg_dur = (ppg["t"].iloc[-1] - ppg["t"].iloc[0]) / 1000.0
ecg_dur = (ecg["t"].iloc[-1] - ecg["t"].iloc[0]) / 1000.0
log(f"    PPG 采样率: {ppg_fs:.1f} Hz, 时长 {ppg_dur:.1f} s, 样本数 {len(ppg)}")
log(f"    ECG 采样率: {ecg_fs:.1f} Hz, 时长 {ecg_dur:.1f} s, 样本数 {len(ecg)}")

for name, df in [("PPG", ppg), ("ECG", ecg)]:
    nan_n = int(df.isna().sum().sum())
    ch_stats = df.drop(columns="t").agg(["mean", "std"]).T
    log(f"    {name} NaN 总数: {nan_n}")
    for ch, row in ch_stats.iterrows():
        log(f"      {ch:>16s}: mean={row['mean']:12.1f}  std={row['std']:10.1f}")

# ---------- 3. 与预处理元数据一致性 ----------
meta = pd.read_csv(DATA / "processed" / "SensSmartTech" / "train_metadata.csv")
splits = json.loads((DATA / "processed" / "SensSmartTech" / "split_subjects.json").read_text())
norm = json.loads((DATA / "processed" / "SensSmartTech" / "normalization.json").read_text())

n_win = (meta["record_id"] == rec).sum()
expected = int((ppg_dur - norm["window_sec"]) / norm["stride_sec"]) + 1
log(f"\n[3] 预处理元数据一致性")
log(f"    metadata 窗口总数: {len(meta)} (覆盖 {meta['subject_id'].nunique()} 名受试者)")
log(f"    记录 {rec} 的窗口数: {n_win} (按窗长{norm['window_sec']}s/步长{norm['stride_sec']}s估计约 {expected})")
log(f"    划分 train/val/test = {len(splits['train'])}/{len(splits['val'])}/{len(splits['test'])} 人")
log(f"    通道配置: PPG {norm['ppg_channels']}")
log(f"             ECG {norm['ecg_channels']}")
meta_subjects = set(meta["subject_id"].unique())
log(f"    metadata 受试者与 train 划分一致: {meta_subjects == set(splits['train'])}")

# ---------- 4. 心率合理性 ----------
hr = demog["Median heart rate (bpm)"].dropna()
ok = ((hr >= 40) & (hr <= 180)).mean()
log(f"\n[4] 生理合理性")
log(f"    心率范围 {hr.min():.1f}-{hr.max():.1f} bpm, 40-180 bpm 内占比 {ok*100:.1f}%")

# ---------- 4b. ECG 去直流后的形态检验 ----------
from scipy.signal import find_peaks

lead2 = ecg["lead_II"].values.astype(float)
lead2_ac = lead2 - lead2.mean()
win = int(ecg_fs * 0.2)
kernel = np.ones(win) / win
lead2_hp = lead2_ac - np.convolve(lead2_ac, kernel, mode="same")
peaks, _ = find_peaks(lead2_hp, distance=int(ecg_fs * 0.4), prominence=lead2_hp.std() * 0.5)
rr_ms = np.diff(ecg["t"].values[peaks])
hr_ecg = 60_000.0 / np.median(rr_ms) if len(rr_ms) else float("nan")
hr_ref = demog.loc[demog["PPG"] == f"{rec}_ppg", "Median heart rate (bpm)"].iloc[0]
log(f"\n[4b] ECG 形态检验 (lead II 去直流后 R 波检测)")
log(f"    检出 R 波数: {len(peaks)}, ECG 估计心率: {hr_ecg:.1f} bpm")
log(f"    Demographics 参考心率: {hr_ref:.1f} bpm, 偏差 {abs(hr_ecg - hr_ref):.1f} bpm")

# ---------- 4c. 多记录抽检: ECG 心率 vs Demographics 参考值 ----------
rng = np.random.default_rng(0)
all_records = demog["ECG"].dropna().unique()
sample_recs = rng.choice(all_records, size=min(20, len(all_records)), replace=False)
devs = []
for r in sample_recs:
    try:
        e = pd.read_csv(csv_dir / f"{r}.csv")
    except FileNotFoundError:
        continue
    sig = e["lead_II"].values.astype(float)
    fs = 1000.0 / np.median(np.diff(e["t"].values))
    sig = sig - sig.mean()
    hp = sig - np.convolve(sig, np.ones(int(fs * 0.2)) / int(fs * 0.2), mode="same")
    pk, _ = find_peaks(hp, distance=int(fs * 0.4), prominence=hp.std() * 0.5)
    if len(pk) < 3:
        continue
    hr = 60_000.0 / np.median(np.diff(e["t"].values[pk]))
    ref = demog.loc[demog["ECG"] == r, "Median heart rate (bpm)"].iloc[0]
    devs.append(abs(hr - ref))
devs = np.array(devs)
log(f"\n[4c] 多记录抽检 ({len(devs)}/{len(sample_recs)} 条有效)")
log(f"    ECG 心率与参考值偏差: 中位数 {np.median(devs):.1f} bpm, 最大 {devs.max():.1f} bpm")
log(f"    偏差 <=5 bpm 的记录占比: {(devs <= 5).mean()*100:.0f}%")

# ---------- 5. 绘图 ----------
sec = 10
fig, axes = plt.subplots(2, 1, figsize=(7.0, 4.2), sharex=True)
t_p = ppg["t"].values / 1000.0
mask = t_p <= t_p[0] + sec
for ch in ppg.columns[1:]:
    axes[0].plot(t_p[mask], ppg[ch].values[mask], label=ch, lw=0.8)
axes[0].set_ylabel("PPG (a.u.)")
axes[0].legend(fontsize=6, ncol=4, loc="upper right", frameon=False)
axes[0].set_title(f"SensSmartTech sample: subject {rec.split('_')[0]}, first {sec}s", fontsize=9)

t_e = ecg["t"].values / 1000.0
me = t_e <= t_e[0] + sec
from scipy.signal import butter, filtfilt

b, a = butter(3, [0.5 / (ecg_fs / 2), 40 / (ecg_fs / 2)], btype="band")
for ch in ecg.columns[1:]:
    sig = filtfilt(b, a, ecg[ch].values.astype(float))
    axes[1].plot(t_e[me], sig[me], label=ch, lw=0.8)
axes[1].set_ylabel("ECG (a.u., 0.5-40 Hz)")
axes[1].set_xlabel("time (s)")
axes[1].legend(fontsize=6, ncol=4, loc="upper right", frameon=False)
for ax in axes:
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    ax.tick_params(labelsize=7)
fig.tight_layout()
fig.savefig(OUT / "sample_signals.png", dpi=200)
plt.close(fig)
log(f"\n[5] 已保存示例信号图: sample_signals.png")

(OUT / "report.txt").write_text("\n".join(report) + "\n", encoding="utf-8")
log(f"\n报告已保存: report.txt")
log("\n结论: 数据可用" if nan_n == 0 and ok > 0.95 else "\n结论: 存在需注意的问题，见上文")
