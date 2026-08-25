# PPG2ECG: Photoplethysmogram to 12-Lead ECG Reconstruction

> 基于 PPG 信号重建 12 导联 ECG 波形，并用于心脏疾病识别

## 项目架构

### 四阶段 Pipeline

```
Stage 1: 数据准备
    MIMIC-IV 跨模块匹配 → 信号提取 → 窗口分割 → 患者级划分

Stage 2: PPG→12导联ECG 重建 (核心)
    Baseline: 4路 PPG → 1D 时序 Res-Encoder → ECG Decoder
    Advanced: latent alignment / GRL / VAE / Flow 等按实验逐步加入

Stage 3: 心脏疾病识别
    Path A: 重建ECG → 独立分类器
    Path B: 端到端联合训练
    Path C: 跨模态对比学习 (推荐)

Stage 4: 评估
    重建质量 (MSE/DTW/CC)
    + 分类性能 (敏感性>95%)
    + 临床验证
```

## 运行环境

### 本地环境 (已配置)

| 组件 | 版本 |
|------|------|
| **Conda 环境** | `cuda126_env` |
| **Python** | 3.10.18 |
| **PyTorch** | 2.6.0+cu126 |
| **CUDA** | 12.6 |
| **GPU** | NVIDIA GeForce RTX 4060 Ti (16GB) |
| **numpy** | 1.26.4 |

### 快速激活

```powershell
# 方式1: 使用激活脚本 (推荐)
.\activate.ps1

# 方式2: 手动激活
conda activate cuda126_env
$env:PYTHONPATH = "."  # 项目根目录
```

### 从零创建环境

```bash
# 使用 conda 创建新环境
conda env create -f configs/environment.yml
conda activate ppg2ecg

# 或使用 pip (在已有 conda 环境中)
pip install -r configs/requirements.txt
```

## 数据集

| 数据集 | 类型 | 规模 | 用途 | 获取难度 | 状态 |
|--------|------|------|------|----------|------|
| **MIMIC-IV** | 需PhysioNet认证 | ~80万条ECG | 核心训练 | ★★★ | 待下载 |
| **MIMIC-III-Ext-PPG** | 需PhysioNet认证 | 630万段 | 预训练 | ★★★ | 待下载 |
| **VitalDB** | 公开 | 6,388例 | 验证 | ★ | 元数据已下载 |
| **BIDMC** | 公开 | 53例 | 基准验证 | ★ | 待下载 |

### 数据下载

```bash
# 1. 激活环境
conda activate cuda126_env

# 2. 下载公开数据集
python scripts/download_vitaldb.py    # VitalDB 元数据
python scripts/download_bidmc.py      # BIDMC (53例)

# 3. (PhysioNet认证后) 下载MIMIC数据
python scripts/download_mimiciv.py    # MIMIC-IV
python scripts/download_mimic3_ext.py # MIMIC-III Ext

# 4. 跨模块匹配 (MIMIC-IV)
python scripts/match_mimic_modules.py

# 5. 预处理
python scripts/preprocess_all.py
```

### 数据目录结构

```
data/
├── raw/                 # 原始下载数据
│   ├── vitaldb/         # VitalDB (cases.csv.gz ✓)
│   ├── bidmc/           # BIDMC
│   └── mimiciv/         # MIMIC-IV
├── interim/             # 中间处理 (匹配结果等)
├── processed/           # 最终训练数据 (HDF5)
├── external/            # 外部验证数据
└── README.md
```

## 快速开始

### 1. 训练 PPG→ECG 重建模型

```bash
# 激活环境
conda activate cuda126_env
$env:PYTHONPATH = "."

# 使用默认配置训练
python -m src.train

# 明确运行 SensSmartTech 四路 PPG baseline
python -m src.train --config configs/exp_baseline.yaml

# 对 train/test 两个 split 按 A/B 状态绘制真实 ECG 与生成 ECG
python scripts/plot_ecg_comparisons.py --run runs/senssmarttech_baseline_20ep --checkpoint best.pth

# 本次 20 epoch 参数归档位置（目录名包含数据集、划分、seed 和 epoch）
# checkpoints/SensSmartTech_subjectwise-per-lead_baseline_seed42_20ep/

# 只检查 baseline 的数据流和反向传播
python data_check/check_baseline.py

# 自定义配置
python -m src.train --config configs/default.yaml --epochs 50 --lr 2e-4

# 启用潜空间扩散模型
python -m src.train --use_diffusion
```

### 2. 训练参数说明

| 参数 | 默认值 | 说明 |
|------|--------|------|
| `--config` | 内置默认 | YAML 配置文件路径 |
| `--epochs` | 100 | 训练轮数 |
| `--batch_size` | 32 | 批大小 |
| `--lr` | 1e-4 | 学习率 |
| `--data` | data/processed/ppg2ecg.h5 | HDF5 数据文件 |
| `--output` | checkpoints | 输出目录 |
| `--seed` | 42 | 随机种子 |
| `--device` | auto | 计算设备 (auto/cuda/cpu) |
| `--use_diffusion` | False | 启用潜空间扩散 |

## 项目结构

```
ppg2ecg/
├── configs/                 # 配置文件
│   ├── environment.yml      # Conda 环境定义
│   ├── requirements.txt     # pip 依赖
│   └── default.yaml         # 训练默认配置
├── data/                    # 数据目录
│   ├── raw/                 # 原始数据
│   ├── interim/             # 中间处理
│   ├── processed/           # 训练数据 (HDF5)
│   └── external/            # 外部验证
├── scripts/                 # 数据下载与预处理脚本
│   ├── setup_project.py     # 环境一键设置
│   ├── download_vitaldb.py  # VitalDB 下载
│   ├── download_bidmc.py    # BIDMC 下载
│   ├── download_mimiciv.py  # MIMIC-IV 下载
│   ├── download_mimic3_ext.py # MIMIC-III 下载
│   ├── match_mimic_modules.py # MIMIC 跨模块匹配
│   └── preprocess_all.py    # 数据预处理
├── src/                     # 源代码
│   ├── models/              # 模型定义
│   │   ├── ppg_encoder.py   # PPG 编码器 (频域+时域+CrossAttn)
│   │   ├── ecg_decoder.py   # 多尺度 ECG 解码器
│   │   ├── latent_diffusion.py # 潜空间扩散模型 (DDIM)
│   │   ├── classifier.py    # 心脏疾病分类器
│   │   └── losses.py        # 复合损失函数
│   ├── data/                # 数据加载
│   │   ├── dataset.py       # PyTorch Dataset
│   │   └── transforms.py    # 信号变换与增强
│   ├── training/            # 训练逻辑
│   │   └── trainer.py       # PPG2ECG 训练器
│   ├── evaluation/          # 评估工具
│   │   ├── metrics.py       # 重建与分类指标
│   │   └── visualize.py     # 结果可视化
│   ├── utils/               # 工具函数
│   │   ├── config.py        # 配置管理
│   │   ├── logger.py        # 日志记录
│   │   └── seed.py          # 随机种子
│   └── train.py             # 训练入口
├── notebooks/               # 探索性分析
├── models/                  # 保存的模型
├── checkpoints/             # 训练检查点
├── results/                 # 实验结果
├── docs/                    # 文档
├── activate.ps1             # 环境激活脚本 (PowerShell)
├── activate.bat             # 环境激活脚本 (CMD)
├── pyproject.toml           # 项目配置
└── .gitignore
```

## 核心模型

### Stage 2: PPG→ECG 重建

- **Baseline 编码器** (`src/models/baseline.py`): 四路 PPG 的 1D Res-Encoder，保留时序 latent 与 skip features
- **Baseline 解码器** (`src/models/baseline.py`): 插值上采样 + 1D 残差块，输出四导联 ECG
- **旧版高级编码器** (`src/models/ppg_encoder.py`): 频域分支 (STFT→CNN) + 时域分支 (1D-CNN) + Cross-Attention 融合
- **潜空间扩散** (`src/models/latent_diffusion.py`): DDIM 采样，支持 linear/cosine 噪声调度
- **ECG 解码器** (`src/models/ecg_decoder.py`): 多尺度转置卷积上采样 + 残差块
- **损失函数** (`src/models/losses.py`): 可组合的 MSE/L1、目标侧 QRS 加权、QRS
  峰值/RMS 幅度、导数、STFT、频带和小波系数损失

Baseline 配置默认使用全部 4 路 PPG、仅 MSE 重建损失，不把 A/B 运动状态作为模型输入。

### v0.1 SensSmartTech baseline archive

本次可复现归档的架构、训练设置、指标和 checkpoint 清单见
[`docs/v0.1_baseline.md`](docs/v0.1_baseline.md)。模型权重位于
`checkpoints/SensSmartTech_subjectwise-per-lead_baseline_seed42_20ep/`，并使用
Git LFS 管理；A/B 仅用于分状态评估，未作为条件输入。

### v0.2 VAE + Flow + adversarial disentanglement

实验配置位于 `configs/exp_vae_flow_adv_irm.yaml`。该版本使用 VAE
content/style latent、conditional affine Flow、GRL subject discriminator 和
subject-wise V-REx 辅助项；A/B 仍不作为模型输入。架构、20 epoch 结果和
Subject Discriminator 饱和诊断见
[`docs/v0.2_vae_flow_adv_irm.md`](docs/v0.2_vae_flow_adv_irm.md)。

### v0.3 QRS-aware reconstruction objective

v0.3 keeps the v0.2 VAE + Flow + GRL + V-REx architecture and adds global L1,
target-only QRS-region L1, first-difference, and low-weight multi-scale STFT
losses. It is a controlled 20-epoch experiment rather than a claimed
replacement: global RMSE/MAE improve, but R-peak F1 and QRS width error worsen.
The full record and detail figures are in
[`docs/v0.3_qrs_aware_20ep.md`](docs/v0.3_qrs_aware_20ep.md) and
`runs/senssmarttech_v03_qrs_vae_flow_adv_irm_20ep_seed42/`.

### v0.4 PPGFlowECG-inspired paired latent Flow

`configs/exp_v04_ppgflowecg_inspired.yaml` is a controlled prototype of the
PPGFlowECG direction: paired PPG/ECG posterior alignment, cross-modal ECG
reconstruction, and a temporal rectified Flow in latent space. GRL/IRM and the
v0.3 QRS/STFT additions are intentionally disabled in this first isolation
experiment. The 20-epoch archive, test/train metrics, activity-only analysis,
and compact train/test detail figures are in
[`docs/v0.4_ppgflowecg_inspired_20ep.md`](docs/v0.4_ppgflowecg_inspired_20ep.md)
and `runs/senssmarttech_v04_ppgflowecg_inspired_20ep_seed42/`.

### v0.41 ECG autoencoder debug control

v0.41 is a target-side capacity diagnostic, not a PPG2ECG result. It trains a
pure ECG-to-ECG autoencoder with the existing high-resolution skip features,
global L1, and no VAE/Flow/GRL/IRM terms. It reconstructs sharp QRS complexes
on the training set (RMSE `0.0663`, PCC `0.9971`, R-peak F1 `0.9840`), which
rules out a general preprocessing or basic Decoder failure. Details and
figures are archived in
[`docs/v0.41_ecg_autoencoder_debug_20ep.md`](docs/v0.41_ecg_autoencoder_debug_20ep.md)
and `runs/senssmarttech_v041_ecg_autoencoder_skip_20ep_seed42/`.

### v0.5 Bidirectional cycle prototype

`configs/exp_v05_bidirectional_cycle.yaml` adds a separately supervised
`ECG->PPG` branch and the cycle `PPG->ECG_hat->PPG`. The first 20-epoch trial
keeps VAE/Flow/GRL/IRM and QRS/STFT terms disabled to isolate this hypothesis.
Global waveform metrics improve modestly over v0.1, but R-peak F1 falls to
`0.1972` on test. The cycle PPG L1 (`0.2172`) is much lower than the direct
reverse L1 (`0.7617`) without corresponding ECG quality, exposing joint-network
collusion. The complete negative/control result and figures are archived in
[`docs/v0.5_bidirectional_cycle_20ep.md`](docs/v0.5_bidirectional_cycle_20ep.md)
and `runs/senssmarttech_v05_bidirectional_cycle_20ep_seed42/`.

### v0.51 Frozen reverse-branch cycle experiment

`configs/exp_v051_frozen_reverse_cycle.yaml` separates the learned-forward-map
experiment into two stages: ten epochs of direct ECG->PPG pretraining followed
by ten epochs of frozen reverse-branch cycle training. On the test split,
R-peak F1 improves to `0.6018` and QRS width error to `34.03 ms`, compared with
`0.1972` and `67.63 ms` for v0.5; HR error increases to `14.73 bpm`, so rhythm
transfer is still unresolved. The complete metrics, reverse/cycle diagnostics,
and compact figures are archived in
[`docs/v0.51_frozen_reverse_cycle_20ep.md`](docs/v0.51_frozen_reverse_cycle_20ep.md)
and `runs/senssmarttech_v051_frozen_reverse_cycle_20ep_seed42/`.

### v0.52 Multi-band ECG generation

`configs/exp_v052_multiband_frozen_cycle.yaml` adds three time-domain decoder
heads projected with fixed FFT masks: `0-0.5 Hz` baseline, `0.5-10 Hz`
morphology/rhythm, and `10-40 Hz` QRS detail. Per-band energy normalization and
relative weights (`0.5/1.0/2.0`) keep the high-frequency branch from being
dominated by global L1. The v0.51 frozen ECG->PPG cycle schedule is retained.
On the test split, v0.52 reaches R-peak F1 `0.8920`, QRS width error `8.86 ms`,
and HR error `8.44 bpm`; global RMSE is `0.8552`. Details and figures are
archived in
[`docs/v0.52_multiband_frozen_cycle_20ep.md`](docs/v0.52_multiband_frozen_cycle_20ep.md)
and `runs/senssmarttech_v052_multiband_frozen_cycle_20ep_seed42/`.

### v0.52-highfreq4 controlled ablation

`configs/exp_v052_highfreq4_frozen_cycle.yaml` keeps the v0.52 architecture
and frozen reverse-cycle schedule but raises only the normalized QRS-band
(`10-40 Hz`) relative loss from `2.0` to `4.0`. On the test split, RMSE moves
from `0.8552` to `0.8485` and PCC from `0.3144` to `0.3175`, while R-peak F1
falls from `0.8920` to `0.8555` and QRS width error rises from `8.86 ms` to
`14.64 ms`. The higher weight is therefore retained as an ablation, not as the
default. Full metrics, cycle diagnostics, checkpoints, and compact figures are
archived in
[`docs/v0.52_highfreq4_frozen_cycle_20ep.md`](docs/v0.52_highfreq4_frozen_cycle_20ep.md)
and `runs/senssmarttech_v052_highfreq4_frozen_cycle_20ep_seed42/`.

### v0.53 Wavelet-supervised QRS generation

`configs/exp_v053_wavelet_frozen_cycle.yaml` replaces the global FFT band loss
with a differentiable Symlet-4 stationary-wavelet objective. Detail levels
`D2-D4` supervise time-localized QRS energy while the main decoder still emits
an ordinary time-domain ECG tensor. On the test split, RMSE improves to `0.8295`,
PCC to `0.3497`, and QRS width error to `9.65 ms`; R-peak F1 is `0.8598`, so
peak timing remains the next problem rather than a reason to increase the loss
weight again. The full archive and compact A/B figures are in
[`docs/v0.53_wavelet_frozen_cycle_20ep.md`](docs/v0.53_wavelet_frozen_cycle_20ep.md)
and `runs/senssmarttech_v053_wavelet_frozen_cycle_20ep_seed42/`.

### v0.54 Direct wavelet-coefficient decoder

`configs/exp_v054_wavelet_coeff_frozen_cycle.yaml` replaces the ordinary
time-domain ECG decoder with five predicted Haar coefficient fields followed
by an exact fixed IDWT. It adds direct coefficient, time-localized QRS
envelope, and differentiable peak-interval losses while retaining the v0.53
frozen reverse-cycle schedule. The 20-epoch result improves test RMSE to
`0.8196`, but R-peak F1 falls to `0.2199` and QRS width error rises to
`54.75 ms`; the experiment is archived as a controlled negative result in
[`docs/v0.54_wavelet_coeff_frozen_cycle_20ep.md`](docs/v0.54_wavelet_coeff_frozen_cycle_20ep.md)
and `runs/senssmarttech_v054_wavelet_coeff_frozen_cycle_20ep_seed42/`.

### v0.55 Wider Encoder ablation

`configs/exp_v055_encoder_wide_wavelet_frozen_cycle.yaml` widens the PPG
Encoder (`base_channels=64`, latent `256`) while retaining the v0.54 direct
Haar-coefficient Decoder and losses. Learned 1x1 projections connect the wider
Encoder skips to the fixed Decoder width. Test R-peak F1 improves from
`0.2199` to `0.3471` and peak-time error from `89.22 ms` to `71.41 ms`, while
QRS width error remains about `55.69 ms` and HR error worsens to `12.64 bpm`.
This supports an Encoder bottleneck but motivates a separate high-resolution
Decoder ablation. The archive is in
[`docs/v0.55_encoder_wide_wavelet_frozen_cycle_20ep.md`](docs/v0.55_encoder_wide_wavelet_frozen_cycle_20ep.md)
and `runs/senssmarttech_v055_encoder_wide_wavelet_frozen_cycle_20ep_seed42/`.

### v0.56 High-frequency Decoder branches

`configs/exp_v056_highfreq_decoder_wide_encoder.yaml` keeps the v0.55 wide
Encoder and widens only the direct Haar Decoder's D1/D2 coefficient branches.
Two no-normalization local residual blocks and a `1.15` high-frequency gain
make short QRS excursions easier to retain without changing the exact IDWT
reconstruction path. On the final test checkpoint, R-peak F1 improves from
`0.3994` to `0.6565` and QRS width error from `56.49 ms` to `31.12 ms`; global
PCC is slightly lower (`0.3442`), and absolute ECG amplitude remains
conservative. The complete archive and compact A/B figures are in
[`docs/v0.56_highfreq_decoder_wide_encoder_20ep.md`](docs/v0.56_highfreq_decoder_wide_encoder_20ep.md)
and `runs/senssmarttech_v056_highfreq_decoder_wide_encoder_20ep_seed42/`.

### v0.57 Residual high-frequency control

`configs/exp_v057_residual_highfreq_decoder_wide_encoder.yaml` keeps the
original D1/D2 heads as a base path and adds zero-started local residual heads.
This protects broad morphology, but the ten-epoch frozen-cycle stage is not
enough for the residual branch to recover v0.56's peak quality: test R-peak F1
falls to `0.4282` and QRS width error rises to `61.67 ms`. It is archived as a
negative control in
[`docs/v0.57_residual_highfreq_decoder_wide_encoder_20ep.md`](docs/v0.57_residual_highfreq_decoder_wide_encoder_20ep.md)
and `runs/senssmarttech_v057_residual_highfreq_decoder_wide_encoder_20ep_seed42/`.

### v0.58 Explicit QRS/amplitude supervision

`configs/exp_v058_multiband_qrs_amplitude_frozen_cycle.yaml` adds a target-only
QRS mask, QRS-region L1, and explicit local peak/RMS amplitude supervision to
the v0.52 fused ECG. The stronger weights over-regularize the waveform: the
test `best.pth` reaches RMSE `0.9187`, R-peak F1 `0.7362`, and QRS width error
`20.96 ms`. It is archived as a negative control in
[`docs/v0.58_multiband_qrs_amplitude_20ep.md`](docs/v0.58_multiband_qrs_amplitude_20ep.md)
and `runs/senssmarttech_v058_multiband_qrs_amplitude_frozen_cycle_20ep_seed42/`.

### v0.59 Low-weight QRS/amplitude supervision

`configs/exp_v059_multiband_qrs_amplitude_lowweight.yaml` lowers the explicit
terms to `0.10/0.20` while keeping the v0.52 multi-band decoder. On test,
`best.pth` reaches RMSE `0.9043`, PCC `0.2850`, R-peak F1 `0.8929`, and QRS
width error `9.37 ms`. Local spikes are more visible, but global waveform
quality is below v0.52, so this remains a detail-focused candidate rather than
the main baseline. See
[`docs/v0.59_multiband_qrs_amplitude_lowweight_20ep.md`](docs/v0.59_multiband_qrs_amplitude_lowweight_20ep.md)
and `runs/senssmarttech_v059_multiband_qrs_amplitude_lowweight_20ep_seed42/`.

### v0.60 High-band QRS amplitude calibration

`configs/exp_v060_multiband_high_qrs_amplitude.yaml` applies the explicit
peak/RMS term only to the projected `10-40 Hz` high-frequency branch, using the
full ECG only to locate target QRS windows. This preserves the low/mid
frequency morphology path. The test `best.pth` reaches RMSE `0.8856`, PCC
`0.2598`, HR error `8.11 bpm`, R-peak F1 `0.8926`, and QRS width error
`9.27 ms`. It avoids v0.58's collapse and is the preferred follow-up for
high-frequency calibration, but v0.52 remains the balanced primary baseline.
The archive is in
[`docs/v0.60_multiband_high_qrs_amplitude_20ep.md`](docs/v0.60_multiband_high_qrs_amplitude_20ep.md)
and `runs/senssmarttech_v060_multiband_high_qrs_amplitude_20ep_seed42/`.

### v0.61 VAE latent transfer + multi-band decoder

v0.61 grafts the v0.52 multi-band decoder onto the v0.2 VAE latent. The VAE
exposes three temporal skip features, uses posterior means during a five-epoch
decoder warm-up, and then jointly fine-tunes with GRL/V-REx. On test it reaches
RMSE `0.8696`, PCC `0.3415`, R-peak F1 `0.9266`, and QRS width error
`11.85 ms`. The detailed archive is in
[`docs/v0.61_vae_multiband_transfer_latent128_20ep.md`](docs/v0.61_vae_multiband_transfer_latent128_20ep.md)
and `runs/senssmarttech_v061_vae_multiband_transfer_latent128_20ep_seed42/`.

### v0.62 Latent capacity ablation (256)

v0.62 doubles the VAE temporal latent from 128 to 256 channels while keeping
the v0.61 decoder, losses, schedule, and split fixed. RMSE improves to `0.8443`,
but PCC (`0.2805`), R-peak F1 (`0.8119`), and QRS width error (`32.10 ms`)
worsen. The larger latent is therefore recorded as a capacity ablation, not
adopted as the main baseline. See
[`docs/v0.62_vae_multiband_latent256_20ep.md`](docs/v0.62_vae_multiband_latent256_20ep.md)
and `runs/senssmarttech_v062_vae_multiband_latent256_20ep_seed42/`.

### v0.63 Light QRS/peak supervision control

v0.63 keeps the v0.61 VAE plus multi-band decoder and adds low-weight
target-only QRS amplitude and differentiable peak-interval terms. The test
result is RMSE `0.8569`, PCC `0.3291`, R-peak F1 `0.9188`, and QRS width error
`13.76 ms`; it does not improve the v0.61 physiology metrics, so it is kept as
a controlled negative loss result. See
[`docs/v0.63_vae_multiband_qrs_peak_20ep.md`](docs/v0.63_vae_multiband_qrs_peak_20ep.md)
and `runs/senssmarttech_v063_vae_multiband_qrs_peak_20ep_seed42/`.

### v0.64 Latent-256 overlap-transfer ablation

v0.64 repeats v0.62 with a widened latent initialized by overlapping weights
from the trained v0.2 encoder. The test result improves to RMSE `0.8323`, PCC
`0.3597`, HR error `9.03 bpm`, and R-peak F1 `0.9010`, but QRS width remains
`18.16 ms`, behind v0.61. This supports transfer initialization as necessary
for a fair wider-latent test, but does not make latent 256 the main model. See
[`docs/v0.64_vae_multiband_latent256_transfer_20ep.md`](docs/v0.64_vae_multiband_latent256_transfer_20ep.md)
and `runs/senssmarttech_v064_vae_multiband_latent256_transfer_20ep_seed42/`.

### v0.67/v0.68 skip and PatchGAN follow-ups

v0.67 compares a corrected full-resolution PPG skip in the high-frequency
branch with a zero-started residual gate. The formal gated run improves test
RMSE/PCC/QRS width over its protocol control, while the residual gate remains
small (`alpha=0.0245`) but does not improve held-out peak metrics. The protocol
and artifacts are in [`docs/v0_67_protocol_experiment.md`](docs/v0_67_protocol_experiment.md).

v0.68 adds an optional conditional 1-D PatchGAN with a separate discriminator
optimizer. It saturates early and collapses R-peak/QRS metrics on both train and
test, so it is retained as a negative adversarial control rather than enabled
by default. See [`docs/v0_68_patchgan_20ep.md`](docs/v0_68_patchgan_20ep.md) and
the version-level committee archive
[`docs/v0.68_committee.md`](docs/v0.68_committee.md).

### Experiment matrix and selection guide

The standardized test-set comparison for v0.1-v0.68, including architecture
components, checkpoint policy, waveform metrics, physiology metrics, and the
recommended next ablations, is maintained in
[`docs/experiment_matrix.md`](docs/experiment_matrix.md). The current main
baseline is v0.52; v0.53 is the strongest wavelet-loss candidate and v0.60 is
the high-band amplitude-supervision candidate. The expanded multi-dimensional
report, including amplitude ratios, PTT, train/test gaps, A/B robustness, and
efficiency is in
[`docs/experiment_results_full.md`](docs/experiment_results_full.md), generated
by `scripts/build_experiment_matrix.py`.

The evaluation protocol, single-to-single versus multi-to-multi comparison,
sample-wise/record-wise leakage note, and the v0.2 comparison with published
PPG-to-ECG reports are documented in
[`docs/evaluation_protocol_and_literature.md`](docs/evaluation_protocol_and_literature.md).
The controlled 20-epoch CardioGAN and RDDM paper-mechanism adaptations are
archived in
[`docs/paper_reproductions_20ep.md`](docs/paper_reproductions_20ep.md), with
their checkpoints and predictions under `runs/`.
The main split is subject-wise `22/5/5` (train/validation/test subjects). Train
metrics are in-sample fit, validation selects the checkpoint, and test metrics
are the cross-subject result. Absolute PPG-ECG/R-peak synchrony is retained only
as an auxiliary paired-reconstruction diagnostic, not the primary ranking
criterion. The current Lead II values are slices from four-PPG/four-ECG models,
not retrained single-PPG/single-ECG results; setting `data.ecg_lead: II` is
supported for a future strict 1-to-1 run.

For leakage auditing, `configs/exp_recordwise_20ep.yaml` and
`runs/senssmarttech_recordwise_baseline_20ep_seed42/` contain a deliberately
record-wise baseline. It allows subjects to overlap across splits and is
therefore an upper bound only, excluded from all subject-wise rankings.

### Stage 3: 疾病识别

- **分类器** (`src/models/classifier.py`): 1D-CNN + Transformer 编码器
- **Path C (推荐)**: 跨模态对比学习 (`ContrastiveHead` + `ContrastiveLoss`)

## 评估指标

| 层级 | 当前记录 | 说明 |
|------|------|------|
| 波形重建 | MSE, RMSE, MAE, PCC, NRMSE, DTW proxy, SNR | 全局误差、形态相关性、时间对齐和信噪比 |
| 生理一致 | HR error, R-peak F1/precision/recall, timing error, QRS width/amplitude, RMSSD, carotid/brachial PTT | 节律、QRS 形态、短窗 HRV 和脉搏传导 |
| 泛化/状态 | train-test gap, A/B activity subgroup | A/B 只做事后分组，不作为模型输入 |
| 工程效率 | 参数量, inference time, RTF | 部署成本和实时性 |
| 可选分布 | 1-NNA, Coverage | 运行 `evaluate.py --distribution` 时记录；FID/IS 暂缓 |

完整的多维结果（含 mean +/- std、逐导联扩展、训练/测试差距和所有版本
横向比较）见 [`docs/experiment_results_full.md`](docs/experiment_results_full.md)。
8 秒窗口不用于 LF/HF 频域 HRV，因其统计窗口不足。

## 前置条件

1. **PhysioNet 注册 + CITI 培训** (~1周) — 仅 MIMIC 数据需要
2. **计算环境**: RTX 4060 Ti 16GB (已配置) 或 A100/V100
3. **存储**: >5TB (完整 MIMIC-IV)

## 参考文献

- MIMIC-IV: https://physionet.org/content/mimiciv/
- MIMIC-IV-ECG: https://physionet.org/content/mimic-iv-ecg/
- VitalDB: https://vitaldb.net/
- BIDMC: https://physionet.org/content/bidmc/
- Latent Diffusion: Rombach et al., CVPR 2022

## License

本项目代码遵循 MIT License。
数据使用需遵守各数据集的授权协议 (PhysioNet Credentialed Health Data License)。
