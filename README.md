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
- **损失函数** (`src/models/losses.py`): λ₁·MSE + λ₂·DTW + λ₃·Freq + λ₄·Perceptual

Baseline 配置默认使用全部 4 路 PPG、仅 MSE 重建损失，不把 A/B 运动状态作为模型输入。

### v0.1 SensSmartTech baseline archive

本次可复现归档的架构、训练设置、指标和 checkpoint 清单见
[`docs/v0.1_baseline.md`](docs/v0.1_baseline.md)。模型权重位于
`checkpoints/SensSmartTech_subjectwise-per-lead_baseline_seed42_20ep/`，并使用
Git LFS 管理；A/B 仅用于分状态评估，未作为条件输入。

### Stage 3: 疾病识别

- **分类器** (`src/models/classifier.py`): 1D-CNN + Transformer 编码器
- **Path C (推荐)**: 跨模态对比学习 (`ContrastiveHead` + `ContrastiveLoss`)

## 评估指标

| 类型 | 指标 | 目标 | 实现位置 |
|------|------|------|----------|
| 重建质量 | MSE, RMSE, CC, DTW, SNR | MSE < 0.10 | `src/evaluation/metrics.py` |
| 分类性能 | AUC, 敏感性, 特异性, F1 | VT/VF 敏感性 > 95% | `src/evaluation/metrics.py` |
| 临床验证 | 医生盲法评审 | 一致性 > 80% | - |

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
