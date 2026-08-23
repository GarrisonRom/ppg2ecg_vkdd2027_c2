"""可复现性工具: 设置随机种子。"""

from __future__ import annotations

import os
import random

import numpy as np
import torch


def set_seed(seed: int = 42, deterministic: bool = True):
    """设置随机种子以确保可复现性。

    Args:
        seed: 随机种子
        deterministic: 是否使用确定性算法 (可能略慢)
    """
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)

    os.environ["PYTHONHASHSEED"] = str(seed)

    if deterministic:
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False
        # PyTorch 2.0+
        try:
            torch.use_deterministic_algorithms(True, warn_only=True)
        except AttributeError:
            pass
