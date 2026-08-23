"""日志记录工具。"""

from __future__ import annotations

import logging
import sys
from pathlib import Path


def get_logger(
    name: str = "ppg2ecg",
    log_file: str | Path | None = None,
    level: int = logging.INFO,
) -> logging.Logger:
    """获取配置好的 logger。

    Args:
        name: logger 名称
        log_file: 日志文件路径 (None 则只输出到控制台)
        level: 日志级别
    Returns:
        logging.Logger
    """
    logger = logging.getLogger(name)

    if logger.handlers:
        return logger  # 已配置

    logger.setLevel(level)
    formatter = logging.Formatter(
        "[%(asctime)s] [%(levelname)s] %(name)s: %(message)s",
        datefmt="%Y-%m-%d %H:%M:%S",
    )

    # 控制台输出
    console = logging.StreamHandler(sys.stdout)
    console.setFormatter(formatter)
    logger.addHandler(console)

    # 文件输出
    if log_file:
        log_file = Path(log_file)
        log_file.parent.mkdir(parents=True, exist_ok=True)
        file_handler = logging.FileHandler(log_file, encoding="utf-8")
        file_handler.setFormatter(formatter)
        logger.addHandler(file_handler)

    return logger
