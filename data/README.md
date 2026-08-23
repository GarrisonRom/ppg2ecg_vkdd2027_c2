# PPG2ECG 项目环境配置

## 数据集说明

### 1. VitalDB (公开数据集，可直接下载)
- **规模**: 6,388例手术患者
- **内容**: PPG波形 + 部分ECG参考
- **采样率**: 500Hz (可降采样至125Hz)
- **用途**: 场景多样性验证
- **下载**: 运行 `python scripts/download_vitaldb.py`

### 2. BIDMC PPG and ECG Dataset (公开数据集，可直接下载)
- **规模**: 53例ICU患者
- **内容**: 同步PPG + ECG记录
- **采样率**: 125Hz
- **用途**: 小规模验证基准
- **下载**: 运行 `python scripts/download_bidmc.py`

### 3. MIMIC-IV (需PhysioNet认证)
- **规模**: ~80万条ECG + ICU波形
- **内容**: 同步PPG + 12导联ECG
- **用途**: 核心训练数据
- **获取步骤**:
  1. 注册 PhysioNet 账号: https://physionet.org/register/
  2. 完成 CITI 培训 (人体受试者研究): https://physionet.org/about/citi-course/
  3. 申请 MIMIC-IV 数据使用权限: https://physionet.org/content/mimiciv/
  4. 申请 MIMIC-IV-ECG 子集: https://physionet.org/content/mimic-iv-ecg/
  5. 获得批准后，使用 `scripts/download_mimiciv.py` 下载

### 4. MIMIC-III-Ext-PPG (需PhysioNet认证)
- **规模**: 630万段，单导联ECG + PPG
- **内容**: 扩展PPG波形匹配
- **用途**: PPG→ECG重建预训练
- **获取**: 随MIMIC-III数据申请，运行 `scripts/download_mimic3_ext.py`

---

## 数据匹配策略 (MIMIC-IV)

MIMIC-IV的ECG和PPG记录**时间戳不同步**，需要手动匹配：

```
匹配键: subject_id + hadm_id
时间容差: ±24小时
信号时长: 10秒标准ECG窗口
```

匹配脚本: `scripts/match_mimic_modules.py`

---

## 目录结构

```
ppg2ecg/
├── data/
│   ├── raw/              # 原始下载数据
│   ├── external/         # 外部验证数据集
│   ├── interim/          # 中间处理数据
│   └── processed/        # 最终训练数据
├── scripts/              # 数据下载与预处理脚本
├── src/                  # 源代码
├── configs/              # 配置文件
├── notebooks/            # 探索性分析
├── models/               # 保存的模型
├── checkpoints/          # 训练检查点
├── results/              # 实验结果
└── docs/                 # 文档
```

---

## 计算环境建议

| 组件 | 最低配置 | 推荐配置 |
|------|---------|---------|
| GPU | V100 × 2 | A100 × 2 |
| 内存 | 64GB | 128GB |
| 存储 | 2TB SSD | 4TB NVMe SSD |
| Python | 3.9+ | 3.10+ |
| PyTorch | 2.0+ | 2.1+ |

---

## 快速启动

```bash
# 1. 创建环境
conda env create -f configs/environment.yml
conda activate ppg2ecg

# 2. 下载公开数据集
python scripts/download_vitaldb.py
python scripts/download_bidmc.py

# 3. (PhysioNet认证后)下载MIMIC数据
# 编辑 scripts/.physionet_credentials 填入用户名密码
python scripts/download_mimiciv.py
python scripts/download_mimic3_ext.py

# 4. 数据预处理
python scripts/preprocess_all.py
```
