"""配置管理: YAML 文件加载/保存 + 命令行覆盖。"""

from __future__ import annotations

import json
from pathlib import Path

import yaml


def load_config(config_path: str | Path, overrides: dict | None = None) -> dict:
    """加载 YAML 配置文件，支持命令行覆盖。

    Args:
        config_path: YAML 配置文件路径
        overrides: 覆盖配置项 (优先级最高)
    Returns:
        合并后的配置字典
    """
    config_path = Path(config_path)
    if not config_path.exists():
        raise FileNotFoundError(f"Config file not found: {config_path}")

    with open(config_path, "r", encoding="utf-8") as f:
        config = yaml.safe_load(f) or {}

    if overrides:
        config = _deep_merge(config, overrides)

    return config


def save_config(config: dict, save_path: str | Path):
    """保存配置到 YAML 文件。"""
    save_path = Path(save_path)
    save_path.parent.mkdir(parents=True, exist_ok=True)
    with open(save_path, "w", encoding="utf-8") as f:
        yaml.dump(config, f, default_flow_style=False, allow_unicode=True)


def _deep_merge(base: dict, override: dict) -> dict:
    """深度合并两个字典。"""
    result = base.copy()
    for k, v in override.items():
        if k in result and isinstance(result[k], dict) and isinstance(v, dict):
            result[k] = _deep_merge(result[k], v)
        else:
            result[k] = v
    return result


# 默认配置 (与 configs/default.yaml 保持一致; YAML 文件优先生效)
DEFAULT_CONFIG = {
    "experiment": {"name": "baseline"},
    # 数据 (按数据集分发; fs/窗口长度/通道由数据集实例自带)
    "data": {
        "dataset": "senssmarttech",
        # 预处理组合目录: {subjectwise|recordwise}_{per-lead|global}
        "root": "data/processed/SensSmartTech/subjectwise_per-lead",
        # baseline 使用全部 4 路 PPG; 单通道旧模型可显式填写通道名
        "ppg_channel": None,
        "batch_size": 32,
        "num_workers": 4,
    },
    # 模型 (模块按注册表构建; name 字段必填)
    "model": {
        "latent_dim": 128,
        "encoder": {"name": "baseline_encoder", "base_channels": 32, "dropout": 0.1},
        "decoder": {"name": "baseline_decoder", "base_channels": 32, "dropout": 0.0},
        "diffusion": None,
        "loss": {"weights": {"mse": 1.0, "dtw": 0.0, "freq": 0.0, "perceptual": 0.0}},
    },
    # 训练
    "training": {
        "epochs": 100,
        "lr": 1e-4,
        "weight_decay": 1e-5,
        "min_lr": 1e-6,
        "max_grad_norm": 1.0,
        "diff_weight": 0.1,
        "save_every": 10,
    },
    # 输出与复现 (output_dir 由 train.py 自动生成为 runs/<experiment.name>)
    "seed": 42,
    "device": "auto",
}
