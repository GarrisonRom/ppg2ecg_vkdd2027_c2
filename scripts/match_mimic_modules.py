#!/usr/bin/env python3
"""
MIMIC-IV 跨模块数据匹配脚本

核心任务: 将 MIMIC-IV-ECG (12导联ECG) 与 MIMIC-IV-Waveform (PPG) 进行匹配

匹配策略:
1. 使用 subject_id + hadm_id 关联患者和入院记录
2. 时间容差: ±24小时 (ECG和PPG记录的时间戳)
3. 优先匹配同一次住院期间(hadm_id)内的记录
4. 输出: 匹配清单 (matched_pairs.csv)

数据结构:
- ECG记录: record_id, subject_id, hadm_id, ecg_time, filepath
- PPG记录: subject_id, hadm_id, ppg_start_time, ppg_duration, filepath
- 匹配结果: subject_id, hadm_id, ecg_time, ppg_time, ecg_path, ppg_path

"""

import pandas as pd
import numpy as np
from pathlib import Path
from datetime import timedelta
from tqdm import tqdm

# 配置
RAW_DIR = Path(__file__).parent.parent / "data" / "raw" / "mimiciv"
INTERIM_DIR = Path(__file__).parent.parent / "data" / "interim"

# 匹配参数
TIME_TOLERANCE = timedelta(hours=24)  # 时间容差


def load_ecg_metadata():
    """加载 ECG 记录元数据"""
    ecg_meta_path = RAW_DIR / "ecg" / "record_list.csv"
    
    if not ecg_meta_path.exists():
        print(f"错误: 未找到 ECG 元数据文件: {ecg_meta_path}")
        print("请先下载 MIMIC-IV-ECG 数据集")
        return None
    
    print("加载 ECG 元数据...")
    df = pd.read_csv(ecg_meta_path)
    print(f"  共 {len(df)} 条 ECG 记录")
    return df


def load_ppg_metadata():
    """加载 PPG (Waveform) 记录元数据"""
    # Waveform 数据没有直接的 record_list，需要从文件结构中解析
    ppg_dir = RAW_DIR / "waveform"
    
    if not ppg_dir.exists():
        print(f"错误: 未找到 Waveform 目录: {ppg_dir}")
        print("请先下载 MIMIC-IV Waveform 数据集")
        return None
    
    print("扫描 Waveform 目录结构...")
    # 这里需要根据实际文件结构实现
    # MIMIC-IV-Waveform 的文件结构通常是: p00/p000020/...
    
    # 创建示例数据结构
    records = []
    for patient_dir in sorted(ppg_dir.glob("p*"))[:100]:  # 限制前100个患者目录
        for record_dir in patient_dir.glob("*"):
            if record_dir.is_dir():
                records.append({
                    "subject_id": patient_dir.name[1:],  # 去掉 'p' 前缀
                    "record_dir": str(record_dir.relative_to(ppg_dir)),
                })
    
    df = pd.DataFrame(records)
    print(f"  共 {len(df)} 条 Waveform 记录 (样本)")
    return df


def load_clinical_admissions():
    """加载临床入院记录 (用于 hadm_id 匹配)"""
    admissions_path = RAW_DIR / "clinical" / "hosp" / "admissions.csv"
    
    if not admissions_path.exists():
        print(f"警告: 未找到 admissions.csv: {admissions_path}")
        return None
    
    print("加载入院记录...")
    df = pd.read_csv(admissions_path)
    print(f"  共 {len(df)} 条入院记录")
    return df


def match_ecg_to_ppg(ecg_df, ppg_df, admissions_df=None):
    """
    匹配 ECG 和 PPG 记录
    
    算法:
    1. 按 subject_id 分组
    2. 对于每个患者，找到时间相近的 ECG 和 PPG 记录
    3. 使用 hadm_id 进一步过滤同一住院期间的记录
    """
    print("\n开始匹配 ECG ↔ PPG...")
    
    matched_pairs = []
    
    # 按 subject_id 分组处理
    ecg_by_subject = ecg_df.groupby("subject_id")
    
    for subject_id, ecg_group in tqdm(ecg_by_subject, desc="匹配进度"):
        # 找到该患者的 PPG 记录
        ppg_records = ppg_df[ppg_df["subject_id"] == subject_id]
        
        if len(ppg_records) == 0:
            continue
        
        for _, ecg_row in ecg_group.iterrows():
            # 时间匹配逻辑 (需要根据实际数据格式调整)
            # 这里使用伪代码，实际需要解析ECG和PPG的时间戳
            
            ecg_time = pd.to_datetime(ecg_row.get("ecg_time", "NaT"))
            
            for _, ppg_row in ppg_records.iterrows():
                ppg_time = pd.to_datetime(ppg_row.get("ppg_time", "NaT"))
                
                if pd.isna(ecg_time) or pd.isna(ppg_time):
                    continue
                
                # 检查时间容差
                time_diff = abs(ecg_time - ppg_time)
                if time_diff <= TIME_TOLERANCE:
                    matched_pairs.append({
                        "subject_id": subject_id,
                        "hadm_id": ecg_row.get("hadm_id"),
                        "ecg_time": ecg_time,
                        "ppg_time": ppg_time,
                        "time_diff_hours": time_diff.total_seconds() / 3600,
                        "ecg_path": ecg_row.get("filepath", ""),
                        "ppg_path": ppg_row.get("record_dir", ""),
                    })
    
    matched_df = pd.DataFrame(matched_pairs)
    print(f"\n匹配完成: 共 {len(matched_df)} 对匹配记录")
    
    return matched_df


def save_matches(matched_df):
    """保存匹配结果"""
    INTERIM_DIR.mkdir(parents=True, exist_ok=True)
    output_path = INTERIM_DIR / "mimiciv_matched_pairs.csv"
    
    matched_df.to_csv(output_path, index=False)
    print(f"\n匹配结果已保存: {output_path}")
    
    # 统计信息
    print("\n" + "=" * 60)
    print("匹配统计")
    print("=" * 60)
    print(f"  总匹配对数: {len(matched_df)}")
    print(f"  唯一患者数: {matched_df['subject_id'].nunique()}")
    print(f"  平均时间差: {matched_df['time_diff_hours'].mean():.2f} 小时")
    print(f"  中位时间差: {matched_df['time_diff_hours'].median():.2f} 小时")


def main():
    print("=" * 60)
    print("MIMIC-IV 跨模块数据匹配")
    print("=" * 60)
    print()
    
    # 检查数据是否存在
    if not RAW_DIR.exists():
        print(f"错误: 数据目录不存在: {RAW_DIR}")
        print("请先运行 download_mimiciv.py 下载数据")
        return
    
    # 加载数据
    ecg_df = load_ecg_metadata()
    ppg_df = load_ppg_metadata()
    admissions_df = load_clinical_admissions()
    
    if ecg_df is None or ppg_df is None:
        print("\n数据加载失败，无法继续匹配")
        return
    
    # 执行匹配
    matched_df = match_ecg_to_ppg(ecg_df, ppg_df, admissions_df)
    
    # 保存结果
    save_matches(matched_df)
    
    print("\n" + "=" * 60)
    print("下一步")
    print("=" * 60)
    print("""
1. 检查匹配结果: data/interim/mimiciv_matched_pairs.csv
2. 运行数据预处理: python scripts/preprocess_mimiciv.py
3. 提取信号窗口 (2秒, 125Hz)
4. 划分训练/验证/测试集
""")


if __name__ == "__main__":
    main()
