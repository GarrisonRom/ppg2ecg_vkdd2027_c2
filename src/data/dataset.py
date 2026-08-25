"""PPG-ECG 数据集层（按数据集适配，而非统一格式）。

设计:
  - BasePPGECGDataset: 基类，只约定 __getitem__ 返回
      {"ppg": Tensor [C_p, T], "ecg": Tensor [C_e, T], "subject_id": Tensor}
  - 每个数据集一个子类，自带:
      * 原生通道配置 (ppg_channels / ecg_channels)
      * 采样率 fs 与窗口长度 signal_length (从数据文件读取, 不硬编码)
      * 归一化语义说明 (哪些已在预处理中完成, 加载时不再重复)
  - 注册表 + create_dataset/create_dataloaders 按名称分发,
    新数据集 (VitalDB / BIDMC / MIMIC) 各写各的子类即可。

SensSmartTech (v2 预处理缓存, scripts/preprocess_senssmarttech.py 产物):
    <root>/train.npz|val.npz|test.npz   x: [N,4,T] PPG, y: [N,4,T] ECG
    <root>/<split>_metadata.csv         窗口级元数据 (subject/record/activity/...)
    <root>/normalization.json           PPG 已逐通道 per-recording robust z-score;
                                        ECG 已按 train 统计量归一化 (per-lead 或 global)
"""

from __future__ import annotations

from abc import ABC
from pathlib import Path
from typing import Type

import numpy as np
import pandas as pd
import torch
from torch.utils.data import BatchSampler, DataLoader, Dataset


class BasePPGECGDataset(ABC, Dataset):
    """按数据集适配的 PPG→ECG 数据集基类。"""

    name: str = "base"

    def __init__(self, transform=None):
        self.transform = transform
        self.ppg_channels: list[str] = []
        self.ecg_channels: list[str] = []
        self.fs: int = 0
        self.signal_length: int = 0

    @property
    def ecg_leads(self) -> int:
        return len(self.ecg_channels)

    @property
    def num_ppg_channels(self) -> int:
        return len(self.ppg_channels)

    def apply_transform(self, sample: dict) -> dict:
        if self.transform is not None:
            return self.transform(sample)
        return sample

    def __getitem__(self, idx: int) -> dict[str, torch.Tensor]:
        raise NotImplementedError


_DATASET_REGISTRY: dict[str, Type[BasePPGECGDataset]] = {}


def register_dataset(cls: Type[BasePPGECGDataset]) -> Type[BasePPGECGDataset]:
    _DATASET_REGISTRY[cls.name] = cls
    return cls


@register_dataset
class SensSmartTechDataset(BasePPGECGDataset):
    """SensSmartTech v2 预处理缓存。

    Args:
        root: 组合目录, 如 data/processed/SensSmartTech/subjectwise_per-lead
        split: train / val / test
        ppg_channel: 可选, 仅保留指定 PPG 通道 (兼容单通道编码器),
                     如 "carotid_880nm"; None 保留全部 4 通道。
        ecg_lead: 可选, 仅保留指定 ECG 导联 (例如 "II"); None 保留全部 4 导联。
        transform: 数据已在预处理中归一化, 默认 None 不再重复变换。
    """

    name = "senssmarttech"
    SPLITS = ("train", "val", "test")

    def __init__(
        self,
        root: str | Path,
        split: str = "train",
        ppg_channel: str | None = None,
        ecg_lead: str | None = None,
        transform=None,
    ):
        super().__init__(transform)
        if split not in self.SPLITS:
            raise ValueError(f"split must be one of {self.SPLITS}, got: {split}")
        root = Path(root)
        npz_path = root / f"{split}.npz"
        if not npz_path.exists():
            raise FileNotFoundError(
                f"{npz_path} 不存在。请先运行: "
                "python scripts/preprocess_senssmarttech.py --root data/raw/SensSmartTech --out <root>"
            )

        data = np.load(npz_path, allow_pickle=False)
        self._x = data["x"]  # [N, C_p, T]
        self._y = data["y"]  # [N, C_e, T]
        self.ppg_channels = [str(c) for c in data["ppg_channels"]]
        self.ecg_channels = [str(c) for c in data["ecg_channels"]]
        self.fs = int(data["fs"])
        self.signal_length = int(self._x.shape[2])

        if ppg_channel is not None:
            if ppg_channel not in self.ppg_channels:
                raise ValueError(
                    f"未知 PPG 通道 {ppg_channel!r}, 可选: {self.ppg_channels}"
                )
            idx = self.ppg_channels.index(ppg_channel)
            self._x = self._x[:, idx : idx + 1, :]
            self.ppg_channels = [self.ppg_channels[idx]]

        if ecg_lead is not None:
            if ecg_lead not in self.ecg_channels:
                raise ValueError(
                    f"未知 ECG 导联 {ecg_lead!r}, 可选: {self.ecg_channels}"
                )
            idx = self.ecg_channels.index(ecg_lead)
            self._y = self._y[:, idx : idx + 1, :]
            self.ecg_channels = [self.ecg_channels[idx]]

        meta_path = root / f"{split}_metadata.csv"
        if meta_path.exists():
            self.metadata = pd.read_csv(meta_path)
            if len(self.metadata) != len(self._x):
                raise ValueError(
                    f"metadata 行数 {len(self.metadata)} 与窗口数 {len(self._x)} 不一致"
                )
        else:
            self.metadata = None

    def __len__(self) -> int:
        return len(self._x)

    def __getitem__(self, idx: int) -> dict[str, torch.Tensor]:
        sample = {
            "ppg": torch.from_numpy(self._x[idx]).float(),
            "ecg": torch.from_numpy(self._y[idx]).float(),
        }
        if self.metadata is not None:
            sample["subject_id"] = torch.tensor(
                int(self.metadata["subject_id"].iloc[idx])
            )
        return self.apply_transform(sample)


class SubjectBalancedBatchSampler(BatchSampler):
    """Sample windows from several subjects in every training batch."""

    def __init__(
        self,
        dataset: SensSmartTechDataset,
        subjects_per_batch: int = 4,
        samples_per_subject: int = 4,
        batches_per_epoch: int | None = None,
    ):
        if dataset.metadata is None or "subject_id" not in dataset.metadata.columns:
            raise ValueError("subject-balanced sampling requires subject metadata")
        self.dataset = dataset
        self.subjects_per_batch = max(2, int(subjects_per_batch))
        self.samples_per_subject = max(1, int(samples_per_subject))
        self.batch_size = self.subjects_per_batch * self.samples_per_subject
        self.subject_to_indices = {
            int(subject): [int(i) for i in indices]
            for subject, indices in dataset.metadata.groupby("subject_id").groups.items()
        }
        if len(self.subject_to_indices) < self.subjects_per_batch:
            raise ValueError(
                f"need at least {self.subjects_per_batch} subjects, "
                f"got {len(self.subject_to_indices)}"
            )
        default_batches = len(dataset) // self.batch_size
        self.batches_per_epoch = max(1, int(batches_per_epoch or default_batches))

    def __iter__(self):
        subjects = list(self.subject_to_indices)
        for _ in range(self.batches_per_epoch):
            chosen = list(np.random.choice(
                subjects, size=self.subjects_per_batch, replace=False,
            ))
            batch: list[int] = []
            for subject in chosen:
                indices = self.subject_to_indices[int(subject)]
                selected = np.random.choice(
                    indices,
                    size=self.samples_per_subject,
                    replace=len(indices) < self.samples_per_subject,
                )
                batch.extend(int(index) for index in selected)
            np.random.shuffle(batch)
            yield batch

    def __len__(self) -> int:
        return self.batches_per_epoch


def create_dataset(name: str, root: str | Path, split: str = "train", **kwargs) -> BasePPGECGDataset:
    """按数据集名称创建 Dataset 实例。"""
    if name not in _DATASET_REGISTRY:
        raise KeyError(
            f"未注册的数据集 {name!r}。已注册: {sorted(_DATASET_REGISTRY)}"
        )
    return _DATASET_REGISTRY[name](root, split=split, **kwargs)


def create_dataloaders(
    dataset: str,
    root: str | Path,
    batch_size: int = 32,
    num_workers: int = 4,
    transform=None,
    subject_balanced: bool = False,
    subjects_per_batch: int = 4,
    samples_per_subject: int = 4,
    batches_per_epoch: int | None = None,
    **dataset_kwargs,
) -> dict[str, DataLoader]:
    """创建 train/val/test DataLoader (按数据集分发)。

    Returns:
        {"train": DataLoader, "val": DataLoader, "test": DataLoader}
        缺失的 split 会被跳过。每个 DataLoader.dataset 暴露
        fs / signal_length / ecg_leads / ppg_channels 等数据集原生属性。
    """
    dataloaders: dict[str, DataLoader] = {}
    for split in ("train", "val", "test"):
        try:
            ds = create_dataset(dataset, root, split=split, transform=transform, **dataset_kwargs)
        except FileNotFoundError:
            continue
        if split == "train" and subject_balanced:
            sampler = SubjectBalancedBatchSampler(
                ds,
                subjects_per_batch=subjects_per_batch,
                samples_per_subject=samples_per_subject,
                batches_per_epoch=batches_per_epoch,
            )
            dataloaders[split] = DataLoader(
                ds,
                batch_sampler=sampler,
                num_workers=num_workers,
                pin_memory=True,
            )
        else:
            dataloaders[split] = DataLoader(
                ds,
                batch_size=batch_size,
                shuffle=(split == "train"),
                num_workers=num_workers,
                pin_memory=True,
                drop_last=(split == "train"),
            )
    return dataloaders
