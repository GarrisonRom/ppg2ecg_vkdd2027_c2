"""数据加载与预处理模块。

子模块:
  - dataset: 按数据集适配的 PyTorch Dataset (注册表分发)
  - transforms: 信号变换与增强
"""

from .dataset import (
    BasePPGECGDataset,
    SensSmartTechDataset,
    create_dataset,
    create_dataloaders,
    register_dataset,
)

__all__ = [
    "BasePPGECGDataset",
    "SensSmartTechDataset",
    "create_dataset",
    "create_dataloaders",
    "register_dataset",
]
