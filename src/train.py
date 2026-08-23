#!/usr/bin/env python3
"""PPG2ECG 训练入口 (配置驱动)。

一个实验 = 一个 YAML 配置文件。输出自动隔离到 runs/<experiment_name>/,
并在其中保存配置快照与训练日志, 便于刷点对比。

用法:
  python -m src.train --config configs/default.yaml
  python -m src.train --config configs/default.yaml --epochs 5 --seed 1
  python -m src.train --config configs/exp_diffusion.yaml --name my_run

配置结构 (configs/default.yaml 有完整注释):
  experiment.name    实验名 (决定输出目录 runs/<name>)
  data               数据集选择 (dataset/root/ppg_channel) 与 DataLoader
  model              模块选择: encoder/decoder/diffusion + loss 模块权重
  training           优化器/调度/轮数
"""

from __future__ import annotations

import argparse
import json
import sys
from datetime import datetime
from pathlib import Path

PROJECT_ROOT = Path(__file__).parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

import torch

from src.data import create_dataloaders
from src.training import PPG2ECGTrainer
from src.utils import get_logger, set_seed
from src.utils.config import DEFAULT_CONFIG, load_config, save_config


def parse_args():
    parser = argparse.ArgumentParser(description="PPG2ECG Training (config-driven)")
    parser.add_argument(
        "--config", type=str, default=None,
        help="YAML 配置文件路径 (默认使用内置 DEFAULT_CONFIG)",
    )
    parser.add_argument("--name", type=str, default=None,
                        help="覆盖实验名 (默认取 experiment.name 或配置文件名)")
    parser.add_argument("--epochs", type=int, default=None, help="训练轮数")
    parser.add_argument("--batch_size", type=int, default=None, help="批大小")
    parser.add_argument("--lr", type=float, default=None, help="学习率")
    parser.add_argument("--data", type=str, default=None, help="覆盖数据根目录")
    parser.add_argument("--output", type=str, default=None, help="覆盖输出根目录 (默认 runs/)")
    parser.add_argument("--seed", type=int, default=None, help="随机种子")
    parser.add_argument(
        "--device", type=str, default="auto", choices=["auto", "cuda", "cpu"],
        help="计算设备",
    )
    return parser.parse_args()


def build_config(args) -> dict:
    """构建完整实验配置: DEFAULT_CONFIG <- YAML <- 命令行覆盖。"""
    config = DEFAULT_CONFIG.copy()

    if args.config:
        file_config = load_config(args.config)
        config = _deep_merge(config, file_config)
        if config.get("experiment", {}).get("name") in (None, "default"):
            config.setdefault("experiment", {})["name"] = Path(args.config).stem

    if args.name is not None:
        config.setdefault("experiment", {})["name"] = args.name
    if args.epochs is not None:
        config.setdefault("training", {})["epochs"] = args.epochs
    if args.batch_size is not None:
        config.setdefault("data", {})["batch_size"] = args.batch_size
    if args.lr is not None:
        config.setdefault("training", {})["lr"] = args.lr
    if args.data is not None:
        config.setdefault("data", {})["root"] = args.data
    if args.seed is not None:
        config["seed"] = args.seed

    return config


def _deep_merge(base: dict, override: dict) -> dict:
    result = base.copy()
    for k, v in override.items():
        if k in result and isinstance(result[k], dict) and isinstance(v, dict):
            result[k] = _deep_merge(result[k], v)
        else:
            result[k] = v
    return result


def resolve_device(device_str: str) -> torch.device:
    if device_str == "auto":
        return torch.device("cuda" if torch.cuda.is_available() else "cpu")
    return torch.device(device_str)


def main():
    args = parse_args()
    config = build_config(args)

    # 实验目录隔离: runs/<name>[_seed<N>]
    run_root = Path(args.output) if args.output else (PROJECT_ROOT / "runs")
    exp_name = config.get("experiment", {}).get("name") or "unnamed"
    if args.seed is not None:
        exp_name = f"{exp_name}_seed{args.seed}"
    output_dir = run_root / exp_name
    config["output_dir"] = str(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    # 配置快照 + 日志 (实验可复现的最低要求)
    save_config(config, output_dir / "config.yaml")
    logger = get_logger("ppg2ecg", log_file=output_dir / "train.log")

    logger.info("=" * 60)
    logger.info(f"Experiment: {exp_name}")
    logger.info("=" * 60)
    logger.info(f"Config: {json.dumps(config, ensure_ascii=False, default=str)}")

    set_seed(config["seed"])
    device = resolve_device(args.device or config.get("device", "auto"))
    logger.info(f"Device: {device}")
    if device.type == "cuda":
        logger.info(f"GPU: {torch.cuda.get_device_name(0)}")

    # 数据 (按数据集分发; 通道/采样率/窗口长度由数据集实例自带)
    data_config = config.get("data", {})
    dataset_name = data_config.get("dataset", "senssmarttech")
    data_root = data_config.get("root")

    if data_root is None or not Path(data_root).exists():
        logger.error(f"数据目录不存在: {data_root}")
        logger.error("请先运行预处理: python scripts/preprocess_senssmarttech.py --root data/raw/SensSmartTech --out <dir>")
        sys.exit(1)

    logger.info(f"加载数据: {dataset_name} @ {data_root}")
    dataloaders = create_dataloaders(
        dataset=dataset_name,
        root=data_root,
        batch_size=data_config.get("batch_size", 32),
        num_workers=data_config.get("num_workers", 4),
        ppg_channel=data_config.get("ppg_channel", None),
    )

    if "train" not in dataloaders:
        logger.error("未找到训练数据划分")
        sys.exit(1)

    train_dataset = dataloaders["train"].dataset
    logger.info(
        f"数据集属性: fs={train_dataset.fs}Hz, T={train_dataset.signal_length}, "
        f"ppg={train_dataset.ppg_channels}, ecg={train_dataset.ecg_channels}"
    )
    logger.info(f"Train/Val/Test windows: "
                + "/".join(str(len(dl.dataset)) for dl in
                           (dataloaders.get(s) for s in ("train", "val", "test"))
                           if dl is not None))

    # 训练 (模型模块按注册表从 config.model 构建)
    trainer = PPG2ECGTrainer(
        config=config,
        train_loader=dataloaders["train"],
        val_loader=dataloaders.get("val", dataloaders["train"]),
        device=device,
        signal_length=train_dataset.signal_length,
        ecg_leads=train_dataset.ecg_leads,
        ppg_channels=train_dataset.num_ppg_channels,
    )

    num_epochs = config.get("training", {}).get("epochs", 100)
    trainer.train(num_epochs=num_epochs)

    logger.info("训练完成!")
    logger.info(f"产物目录: {output_dir}")


if __name__ == "__main__":
    main()
