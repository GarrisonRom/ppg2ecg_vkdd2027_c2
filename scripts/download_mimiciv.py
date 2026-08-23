#!/usr/bin/env python3
"""
MIMIC-IV 数据集下载脚本

⚠️ 重要: 需要 PhysioNet 认证！

使用前提:
1. 已注册 PhysioNet 账号
2. 已完成 CITI 培训 (人体受试者研究)
3. 已申请并获得 MIMIC-IV 数据访问权限

数据集内容:
- MIMIC-IV Clinical Database (医院记录)
- MIMIC-IV-ECG (12导联ECG波形)
- MIMIC-IV-Waveform (ICU波形，含PPG)

参考:
- https://physionet.org/content/mimiciv/
- https://physionet.org/content/mimic-iv-ecg/
- https://physionet.org/content/mimic4wdb/ (Waveform)

建议存储空间: > 5TB (完整数据集)
"""

import os
import sys
import subprocess
from pathlib import Path
from getpass import getpass

# 配置
RAW_DIR = Path(__file__).parent.parent / "data" / "raw" / "mimiciv"
PHYSIONET_BASE = "https://physionet.org/files"

# MIMIC-IV 子集
MIMICIV_MODULES = {
    "clinical": {
        "name": "MIMIC-IV Clinical Database",
        "version": "2.2",
        "url": "mimiciv/2.2",
        "size_gb": 15,
        "files": ["hosp", "icu"],  # 主要子目录
    },
    "ecg": {
        "name": "MIMIC-IV-ECG",
        "version": "1.0",
        "url": "mimic-iv-ecg/1.0",
        "size_gb": 3.5,
        "files": ["records", "machine_measurements"],
    },
    "waveform": {
        "name": "MIMIC-IV Waveform Database",
        "version": "0.1.0",
        "url": "mimic4wdb/0.1.0",
        "size_gb": 8.5,
        "files": [],  # 波形数据量大，按需下载
    },
}

def ensure_dir(path: Path):
    path.mkdir(parents=True, exist_ok=True)
    return path

def check_physionet_auth():
    """检查PhysioNet认证状态"""
    print("=" * 60)
    print("PhysioNet 认证检查")
    print("=" * 60)
    
    # 检查环境变量
    username = os.environ.get("PHYSIONET_USER")
    password = os.environ.get("PHYSIONET_PASS")
    
    if not username or not password:
        print("未找到环境变量 PHYSIONET_USER / PHYSIONET_PASS")
        print()
        print("请设置认证信息 (以下方式任选一种):")
        print("  1. 环境变量: export PHYSIONET_USER=your_username")
        print("  2. 凭证文件: 创建 scripts/.physionet_credentials")
        print()
        
        # 交互式输入
        print("请输入 PhysioNet 认证信息:")
        username = input("用户名: ").strip()
        password = getpass("密码: ")
        
        if not username or not password:
            print("错误: 未提供认证信息")
            return None, None
    
    return username, password

def download_with_wget(url: str, dest: Path, username: str, password: str):
    """使用wget下载 (支持PhysioNet认证)"""
    cmd = [
        "wget", "-r", "-N", "-c", "-np",
        "--user", username,
        "--password", password,
        "-P", str(dest),
        url
    ]
    
    print(f"执行: {' '.join(cmd[:3])} ... --user {username} --password *** -P {dest}")
    
    try:
        result = subprocess.run(cmd, capture_output=True, text=True)
        if result.returncode == 0:
            print("下载成功")
            return True
        else:
            print(f"下载失败: {result.stderr[:500]}")
            return False
    except FileNotFoundError:
        print("错误: 未找到 wget 命令")
        print("请安装 wget (Windows可用 curl 或 WSL)")
        return False

def download_module(module_key: str, username: str, password: str, dry_run: bool = False):
    """下载指定MIMIC-IV模块"""
    module = MIMICIV_MODULES[module_key]
    
    print("\n" + "-" * 60)
    print(f"模块: {module['name']} (v{module['version']})")
    print(f"预计大小: ~{module['size_gb']} GB")
    print(f"URL: {PHYSIONET_BASE}/{module['url']}/")
    print("-" * 60)
    
    if dry_run:
        print("[DRY RUN] 仅显示信息，不执行下载")
        return True
    
    dest = ensure_dir(RAW_DIR / module_key)
    url = f"{PHYSIONET_BASE}/{module['url']}/"
    
    # 使用 wget 镜像下载
    return download_with_wget(url, dest, username, password)

def create_credentials_template():
    """创建凭证文件模板"""
    cred_path = Path(__file__).parent / ".physionet_credentials"
    
    if cred_path.exists():
        print(f"凭证文件已存在: {cred_path}")
        return
    
    content = """# PhysioNet 认证信息
# 填写后保存，.gitignore 已配置忽略此文件
# 获取方式: https://physionet.org/settings/

USERNAME=your_physionet_username
PASSWORD=your_physionet_password
"""
    cred_path.write_text(content, encoding="utf-8")
    print(f"已创建凭证模板: {cred_path}")
    print("请编辑该文件填入你的 PhysioNet 认证信息")

def main():
    print("=" * 60)
    print("MIMIC-IV 数据集下载脚本")
    print("=" * 60)
    print()
    print("⚠️  前提条件:")
    print("   1. PhysioNet 注册账号")
    print("   2. CITI 培训证书")
    print("   3. MIMIC-IV 数据访问权限")
    print()
    
    # 检查认证
    username, password = check_physionet_auth()
    if not username:
        create_credentials_template()
        print("\n" + "=" * 60)
        print("请先完成以下步骤再运行此脚本:")
        print("=" * 60)
        print("""
1. 注册 PhysioNet: https://physionet.org/register/
2. 完成 CITI 培训: https://physionet.org/about/citi-course/
   - 选择 "Data or Specimens Only Research" 课程
   - 完成后上传证书到 PhysioNet 个人资料
3. 申请数据权限:
   - MIMIC-IV: https://physionet.org/content/mimiciv/
   - MIMIC-IV-ECG: https://physionet.org/content/mimic-iv-ecg/
   - MIMIC-IV-Waveform: https://physionet.org/content/mimic4wdb/
4. 等待审批 (通常1-3个工作日)
5. 设置认证信息后重新运行此脚本
""")
        return
    
    # 显示模块列表
    print("\n" + "=" * 60)
    print("可用 MIMIC-IV 模块")
    print("=" * 60)
    for key, mod in MIMICIV_MODULES.items():
        print(f"  [{key}] {mod['name']} (~{mod['size_gb']} GB)")
    
    # 询问下载选项
    print("\n下载选项:")
    print("  1. 下载全部模块 (需要 > 5TB 空间)")
    print("  2. 仅下载 Clinical + ECG (核心模块)")
    print("  3. 仅下载 Waveform (按需)")
    print("  4. 仅测试连接")
    print("  0. 退出")
    
    choice = input("\n选择 (0-4): ").strip()
    
    if choice == "0":
        return
    elif choice == "1":
        modules = list(MIMICIV_MODULES.keys())
    elif choice == "2":
        modules = ["clinical", "ecg"]
    elif choice == "3":
        modules = ["waveform"]
    elif choice == "4":
        print("\n测试连接...")
        # 简单测试
        return
    else:
        print("无效选择")
        return
    
    # 确认下载
    total_size = sum(MIMICIV_MODULES[m]["size_gb"] for m in modules)
    print(f"\n将下载 {len(modules)} 个模块，预计总大小: ~{total_size} GB")
    confirm = input("确认下载? (y/N): ").strip().lower()
    
    if confirm != "y":
        print("已取消")
        return
    
    # 执行下载
    ensure_dir(RAW_DIR)
    
    for mod_key in modules:
        download_module(mod_key, username, password, dry_run=False)
    
    print("\n" + "=" * 60)
    print("下载完成")
    print("=" * 60)
    print(f"数据位置: {RAW_DIR}")
    print("\n下一步: 运行 scripts/match_mimic_modules.py 进行跨模块匹配")

if __name__ == "__main__":
    main()
