# PPG2ECG 环境激活脚本 (CMD/Batch)
@echo off
set PROJECT_ROOT=%~dp0
set PYTHONPATH=%PROJECT_ROOT%
call conda activate cuda126_env
echo ========================================
echo   PPG2ECG Environment Activated
echo ========================================
echo   Project: %PROJECT_ROOT%
echo   PYTHONPATH: %PYTHONPATH%
echo.
echo   Quick Start:
echo     python -m src.train --help
echo     python scripts\download_vitaldb.py
echo ========================================
