#!/usr/bin/env python3
"""
PPG2ECG 项目环境设置脚本

一键完成:
1. 检查系统环境 (Python, CUDA, GPU)
2. 创建 Conda 环境
3. 安装依赖
4. 验证安装
5. 创建项目目录结构
"""

import os
import sys
import subprocess
import shutil
from pathlib import Path

PROJECT_ROOT = Path(__file__).parent.parent


def print_section(title):
    print("\n" + "=" * 60)
    print(f"  {title}")
    print("=" * 60)


def run_command(cmd, check=True):
    """运行 shell 命令"""
    print(f"$ {' '.join(cmd)}")
    result = subprocess.run(cmd, capture_output=True, text=True)
    if result.stdout:
        print(result.stdout.strip())
    if result.stderr and result.returncode != 0:
        print(f"stderr: {result.stderr.strip()}", file=sys.stderr)
    if check and result.returncode != 0:
        print(f"命令失败 (exit {result.returncode})")
    return result


def check_python():
    """检查 Python 版本"""
    print_section("Python 环境检查")
    
    version = sys.version_info
    print(f"Python 版本: {version.major}.{version.minor}.{version.micro}")
    
    if version.major < 3 or (version.major == 3 and version.minor < 9):
        print("警告: 建议 Python 3.9+")
        return False
    
    print("✓ Python 版本符合要求")
    return True


def check_conda():
    """检查 Conda 是否安装"""
    print_section("Conda 检查")
    
    conda_path = shutil.which("conda")
    if conda_path:
        print(f"✓ Conda 已安装: {conda_path}")
        result = run_command(["conda", "--version"], check=False)
        return True
    else:
        print("✗ Conda 未找到")
        print("请安装 Miniconda: https://docs.conda.io/en/latest/miniconda.html")
        return False


def check_cuda():
    """检查 CUDA 可用性"""
    print_section("CUDA / GPU 检查")
    
    # 检查 nvidia-smi
    nvidia_smi = shutil.which("nvidia-smi")
    if nvidia_smi:
        print("✓ nvidia-smi 可用")
        run_command(["nvidia-smi"], check=False)
    else:
        print("⚠ nvidia-smi 未找到 (可能是 CPU 环境或无 NVIDIA GPU)")
    
    # 检查 PyTorch CUDA
    try:
        import torch
        cuda_available = torch.cuda.is_available()
        print(f"PyTorch CUDA 可用: {cuda_available}")
        if cuda_available:
            print(f"  CUDA 版本: {torch.version.cuda}")
            print(f"  GPU 数量: {torch.cuda.device_count()}")
            for i in range(torch.cuda.device_count()):
                print(f"  GPU {i}: {torch.cuda.get_device_name(i)}")
        return cuda_available
    except ImportError:
        print("PyTorch 未安装，将在环境创建后检查")
        return None


def create_conda_env():
    """创建 Conda 环境"""
    print_section("创建 Conda 环境")
    
    env_file = PROJECT_ROOT / "configs" / "environment.yml"
    if not env_file.exists():
        print(f"错误: 环境文件不存在: {env_file}")
        return False
    
    print(f"使用环境文件: {env_file}")
    
    # 检查环境是否已存在
    result = run_command(["conda", "env", "list"], check=False)
    if "ppg2ecg" in result.stdout:
        print("环境 'ppg2ecg' 已存在")
        update = input("是否更新? (y/N): ").strip().lower()
        if update == "y":
            run_command([
                "conda", "env", "update", 
                "-f", str(env_file), 
                "--prune"
            ])
        return True
    
    # 创建环境
    run_command([
        "conda", "env", "create",
        "-f", str(env_file)
    ])
    
    print("\n✓ Conda 环境 'ppg2ecg' 已创建")
    print("激活环境: conda activate ppg2ecg")
    return True


def verify_installation():
    """验证关键包安装"""
    print_section("验证安装")
    
    packages = [
        ("torch", "PyTorch"),
        ("numpy", "NumPy"),
        ("pandas", "Pandas"),
        ("scipy", "SciPy"),
        ("wfdb", "WFDB"),
        ("sklearn", "scikit-learn"),
        ("h5py", "HDF5"),
        ("tqdm", "tqdm"),
    ]
    
    all_ok = True
    for module, name in packages:
        try:
            __import__(module)
            print(f"  ✓ {name}")
        except ImportError:
            print(f"  ✗ {name} - 未安装")
            all_ok = False
    
    return all_ok


def setup_directories():
    """确保目录结构完整"""
    print_section("目录结构")
    
    dirs = [
        "data/raw",
        "data/processed",
        "data/interim",
        "data/external",
        "scripts",
        "src",
        "configs",
        "notebooks",
        "models",
        "checkpoints",
        "results",
        "docs",
    ]
    
    for d in dirs:
        path = PROJECT_ROOT / d
        path.mkdir(parents=True, exist_ok=True)
        print(f"  ✓ {d}")


def create_physionet_credentials_template():
    """创建 PhysioNet 凭证模板"""
    cred_path = PROJECT_ROOT / "scripts" / ".physionet_credentials"
    
    if cred_path.exists():
        return
    
    content = """# PhysioNet 认证信息
# 获取方式: https://physionet.org/settings/
# 不要将此文件提交到 Git!

USERNAME=your_physionet_username
PASSWORD=your_physionet_password
"""
    cred_path.write_text(content, encoding="utf-8")
    print(f"  ✓ 创建凭证模板: {cred_path}")


def print_next_steps():
    """打印后续步骤"""
    print_section("后续步骤")
    print("""
1. 激活环境:
   conda activate ppg2ecg

2. 申请数据集 (PhysioNet):
   - 注册: https://physionet.org/register/
   - CITI培训: https://physionet.org/about/citi-course/
   - 申请 MIMIC-IV: https://physionet.org/content/mimiciv/
   - 申请 MIMIC-IV-ECG: https://physionet.org/content/mimic-iv-ecg/

3. 配置 PhysioNet 凭证:
   编辑 scripts/.physionet_credentials

4. 下载数据集:
   python scripts/download_vitaldb.py    # 公开数据
   python scripts/download_bidmc.py      # 公开数据
   python scripts/download_mimiciv.py    # 需认证
   python scripts/download_mimic3_ext.py # 需认证

5. 预处理:
   python scripts/match_mimic_modules.py  # MIMIC跨模块匹配
   python scripts/preprocess_all.py       # 数据预处理

6. 开始训练:
   python src/train.py
""")


def main():
    print("=" * 60)
    print("PPG2ECG 项目环境设置")
    print("=" * 60)
    print(f"项目路径: {PROJECT_ROOT}")
    
    # 检查环境
    python_ok = check_python()
    conda_ok = check_conda()
    cuda_ok = check_cuda()
    
    # 创建目录
    setup_directories()
    create_physionet_credentials_template()
    
    # 创建 Conda 环境 (如果 conda 可用)
    if conda_ok:
        env_ok = create_conda_env()
    else:
        print("\n跳过 Conda 环境创建 (Conda 未安装)")
        print("使用 pip 安装依赖: pip install -r configs/requirements.txt")
    
    # 打印后续步骤
    print_next_steps()
    
    print("\n" + "=" * 60)
    print("设置完成!")
    print("=" * 60)


if __name__ == "__main__":
    main()
