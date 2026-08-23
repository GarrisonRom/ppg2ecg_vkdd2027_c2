#!/usr/bin/env python3
"""
MIMIC-III Extended PPG Dataset 下载脚本

⚠️ 重要: 需要 PhysioNet 认证！

数据集: MIMIC-III Waveform Database Matched Subset
规模: 630万段 PPG + 单导联ECG
用途: PPG→ECG 重建预训练

注意: 这是 MIMIC-III 的扩展波形数据，与 MIMIC-IV 不同。
MIMIC-III 数据集本身也需要 PhysioNet 权限。

参考: https://physionet.org/content/mimiciii/
"""

import os
import sys
from pathlib import Path
from getpass import getpass

RAW_DIR = Path(__file__).parent.parent / "data" / "raw" / "mimic3"

def main():
    print("=" * 60)
    print("MIMIC-III Extended PPG 数据集下载")
    print("=" * 60)
    print()
    print("数据集信息:")
    print("  - 名称: MIMIC-III Waveform Database")
    print("  - 规模: 630万段波形")
    print("  - 内容: PPG + 单导联ECG")
    print("  - 用途: 预训练 PPG→ECG 重建模型")
    print()
    print("⚠️  需要 PhysioNet 认证 (同 MIMIC-IV)")
    print()
    
    # 检查认证
    username = os.environ.get("PHYSIONET_USER")
    password = os.environ.get("PHYSIONET_PASS")
    
    if not username:
        print("未找到认证信息")
        print("请设置环境变量 PHYSIONET_USER 和 PHYSIONET_PASS")
        print("或创建 scripts/.physionet_credentials 文件")
        print()
        print("下载指南:")
        print("  1. 访问: https://physionet.org/content/mimiciii/")
        print("  2. 使用 WFDB 工具下载:")
        print("     wfdb.io.dl_database('mimic3wdb', 'data/raw/mimic3/')")
        return
    
    print("认证信息已配置")
    print()
    print("=" * 60)
    print("下载方法")
    print("=" * 60)
    print("""
方法1: 使用 WFDB Python 工具包 (推荐)
  import wfdb
  wfdb.io.dl_database('mimic3wdb', 'data/raw/mimic3/', 
                      user='your_username', password='your_password')

方法2: 使用 wget (命令行)
  wget -r -N -c -np --user YOUR_USER --password YOUR_PASS \\
       -P data/raw/mimic3/ \\
       https://physionet.org/files/mimic3wdb/1.0/

方法3: 使用 PhysioNet 网页界面
  访问: https://physionet.org/content/mimic3wdb/
  点击 "Download" 按钮

注意: 完整数据集非常大 (~数TB)，建议:
  - 先下载一小部分进行测试
  - 使用按需下载策略
  - 或使用 Google Cloud / AWS 上的镜像 (如可用)
""")

if __name__ == "__main__":
    main()
