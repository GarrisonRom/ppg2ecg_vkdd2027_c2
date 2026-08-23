#!/usr/bin/env python3
"""
BIDMC PPG and ECG Dataset 下载脚本
数据集: BIDMC PPG and ECG Dataset
规模: 53例ICU患者，约20分钟同步记录
采样率: 125Hz
用途: 小规模验证基准

这是一个公开数据集，无需PhysioNet认证即可下载。
数据格式: WFDB (PhysioNet标准格式)

参考: https://physionet.org/content/bidmc/
"""

import os
import sys
import urllib.request
import tarfile
from pathlib import Path

# 数据集配置
DATASET_NAME = "bidmc"
RAW_DIR = Path(__file__).parent.parent / "data" / "raw" / DATASET_NAME

# BIDMC 数据集下载链接 (PhysioNet 直接下载)
BIDMC_URL = "https://physionet.org/files/bidmc/1.0.0/"

# 文件列表 (从PhysioNet页面获取)
BIDMC_FILES = [
    "bidmc_csv.zip",  # CSV格式数据
]

def ensure_dir(path: Path):
    path.mkdir(parents=True, exist_ok=True)
    return path

def download_file(url: str, dest: Path, desc: str = ""):
    """下载单个文件，显示进度"""
    if dest.exists():
        print(f"[SKIP] {desc or dest.name} 已存在")
        return True
    
    print(f"[DOWNLOAD] {desc or dest.name}")
    print(f"  URL: {url}")
    
    try:
        def report_progress(block_num, block_size, total_size):
            downloaded = block_num * block_size
            if total_size > 0:
                percent = min(100, downloaded * 100 / total_size)
                print(f"\r  进度: {percent:.1f}% ({downloaded/1024/1024:.1f}/{total_size/1024/1024:.1f} MB)", end="")
        
        urllib.request.urlretrieve(url, dest, reporthook=report_progress)
        print()  # 换行
        size = dest.stat().st_size
        print(f"  完成 ({size/1024/1024:.2f} MB)")
        return True
    except Exception as e:
        print(f"\n  错误: {e}")
        return False

def extract_zip(zip_path: Path, dest_dir: Path):
    """解压zip文件"""
    print(f"[EXTRACT] 解压 {zip_path.name}")
    import zipfile
    with zipfile.ZipFile(zip_path, 'r') as zf:
        zf.extractall(dest_dir)
    print(f"  完成")

def download_bidmc():
    """下载BIDMC数据集"""
    ensure_dir(RAW_DIR)
    
    print("=" * 60)
    print("BIDMC PPG and ECG Dataset 下载")
    print("=" * 60)
    print(f"目标目录: {RAW_DIR}")
    print()
    
    # 尝试通过 WFDB 工具下载 (推荐方式)
    print("[方法] 尝试使用 WFDB 工具下载...")
    print("WFDB 是 PhysioNet 的标准工具包")
    
    try:
        import wfdb
        print("WFDB 已安装，开始下载...")
        
        # 下载所有记录
        record_list = wfdb.io.get_record_list('bidmc')
        print(f"发现 {len(record_list)} 条记录")
        
        for rec in record_list[:5]:  # 先下载前5条测试
            print(f"  下载记录: {rec}")
            wfdb.io.dl_database('bidmc', RAW_DIR, records=[rec])
        
        print(f"\n前5条记录下载完成。完整下载请修改脚本中的 [:5] 限制。")
        
    except ImportError:
        print("WFDB 未安装，尝试直接下载CSV版本...")
        print()
        
        # 备选: 直接下载CSV zip
        csv_url = "https://physionet.org/files/bidmc/1.0.0/bidmc_csv.zip"
        csv_path = RAW_DIR / "bidmc_csv.zip"
        
        if download_file(csv_url, csv_path, "BIDMC CSV数据"):
            extract_zip(csv_path, RAW_DIR)
            print("\n[完成] BIDMC CSV数据已准备就绪")
        else:
            print("\n[错误] 下载失败，请尝试手动下载:")
            print(f"  1. 访问: https://physionet.org/content/bidmc/")
            print(f"  2. 点击 'Download the ZIP file'")
            print(f"  3. 解压到: {RAW_DIR}")

def main():
    download_bidmc()
    
    print("\n" + "=" * 60)
    print("BIDMC 数据集信息")
    print("=" * 60)
    print("""
数据集概况:
- 患者数: 53例ICU患者
- 采样率: 125Hz
- 信号: PPG (Pleth) + ECG (多导联)
- 时长: 每例约20分钟
- 格式: WFDB 或 CSV

数据文件说明:
- *.dat: 波形数据文件
- *.hea: 头文件 (信号信息)
- *.atr: 标注文件 (如适用)

使用方式:
  import wfdb
  record = wfdb.rdrecord('data/raw/bidmc/bidmc_01')
  ppg = record.p_signal[:, 0]  # PPG信号
  ecg = record.p_signal[:, 1]  # ECG信号
""")

if __name__ == "__main__":
    main()
