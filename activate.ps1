# PPG2ECG 环境激活脚本 (PowerShell)
# 用法: .\activate.ps1

$PROJECT_ROOT = Split-Path -Parent $MyInvocation.MyCommand.Path
$env:PYTHONPATH = $PROJECT_ROOT
$env:CONDA_DEFAULT_ENV = "cuda126_env"

# 激活 conda 环境
conda activate cuda126_env

Write-Host "========================================" -ForegroundColor Cyan
Write-Host "  PPG2ECG 环境已激活" -ForegroundColor Green
Write-Host "========================================" -ForegroundColor Cyan
Write-Host "  Python:  $($env:CONDA_PYTHON_EXE)"
Write-Host "  Project: $PROJECT_ROOT"
Write-Host "  PYTHONPATH: $env:PYTHONPATH"
Write-Host ""
Write-Host "  快速开始:" -ForegroundColor Yellow
Write-Host "    python -m src.train --help"
Write-Host "    python scripts/download_vitaldb.py"
Write-Host "========================================" -ForegroundColor Cyan
