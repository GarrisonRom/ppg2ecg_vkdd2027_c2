#!/usr/bin/env python3
"""
VitalDB 数据集下载脚本
数据集: VitalDB (Vital Signs Database)
规模: 6,388例手术患者
用途: PPG→ECG重建的场景多样性验证

下载内容:
- 病例列表 (cases.csv)
- PPG波形数据
- 部分同步ECG参考

参考: https://vitaldb.net/
"""

import os
import sys
import urllib.request
import zipfile
from pathlib import Path

# 数据集配置
DATASET_NAME = "vitaldb"
RAW_DIR = Path(__file__).parent.parent / "data" / "raw" / DATASET_NAME
CHUNK_SIZE = 8192

# VitalDB 资源链接
VITALDB_URLS = {
    "cases": "https://api.vitaldb.net/cases",  # 病例元数据
    "tracks": "https://api.vitaldb.net/tracks",  # 波形track列表
}

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
    print(f"  Dest: {dest}")
    
    try:
        urllib.request.urlretrieve(url, dest)
        size = dest.stat().st_size
        print(f"  完成 ({size/1024/1024:.2f} MB)")
        return True
    except Exception as e:
        print(f"  错误: {e}")
        return False

def download_vitaldb_cases():
    """下载病例元数据"""
    ensure_dir(RAW_DIR)
    
    print("=" * 60)
    print("VitalDB 数据集下载")
    print("=" * 60)
    
    # 下载病例列表
    cases_path = RAW_DIR / "cases.csv"
    download_file(VITALDB_URLS["cases"], cases_path, "病例列表 (cases.csv)")
    
    # 下载track列表
    tracks_path = RAW_DIR / "tracks.csv"
    download_file(VITALDB_URLS["tracks"], tracks_path, "波形Track列表 (tracks.csv)")
    
    print("\n" + "=" * 60)
    print("病例元数据下载完成")
    print("=" * 60)
    print(f"\n注意: VitalDB的波形数据需要通过 vitaldb Python 包")
    print(f"或在线API按需下载。请运行以下命令安装工具包:")
    print(f"  pip install vitaldb")
    print(f"\n然后使用 notebooks/vitaldb_explore.ipynb 按需下载波形")

def main():
    download_vitaldb_cases()
    
    print("\n" + "=" * 60)
    print("VitalDB 数据获取指南")
    print("=" * 60)
    print("""
由于 VitalDB 波形数据量巨大（~2TB），建议按需下载：

方法1: 使用 vitaldb Python 包 (推荐)
  import vitaldb
  cases = vitaldb.find_cases(['ECG', 'PPG'])  # 找同时有ECG和PPG的病例
  for caseid in cases:
      vf = vitaldb.VitalFile(caseid, ['SNUADC/ECG_II', 'SNUADC/PLETH'])
      vf.to_wfdb(f'data/raw/vitaldb/case_{caseid}')

方法2: 使用批量下载脚本
  见 scripts/download_vitaldb_waveforms.py (待创建)

方法3: 手动下载
  访问 https://vitaldb.net/dataset 下载个案
""")

if __name__ == "__main__":
    main()
