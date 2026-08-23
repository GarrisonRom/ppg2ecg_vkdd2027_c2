"""信号变换与数据增强。

所有变换接受并返回字典: {"ppg": Tensor, "ecg": Tensor, ...}
"""

from __future__ import annotations

import torch
import torch.nn.functional as F


class Compose:
    """组合多个变换。"""

    def __init__(self, transforms: list):
        self.transforms = transforms

    def __call__(self, sample: dict) -> dict:
        for t in self.transforms:
            sample = t(sample)
        return sample


class ZScoreNormalize:
    """Z-score 标准化 (逐样本)。"""

    def __call__(self, sample: dict) -> dict:
        ppg = sample["ppg"]
        ecg = sample["ecg"]

        # PPG: [L] -> 标准化
        sample["ppg"] = (ppg - ppg.mean()) / (ppg.std() + 1e-8)

        # ECG: [12, L] -> 逐导联标准化
        mean = ecg.mean(dim=-1, keepdim=True)
        std = ecg.std(dim=-1, keepdim=True)
        sample["ecg"] = (ecg - mean) / (std + 1e-8)

        return sample


class RandomCrop:
    """随机裁剪到指定长度。"""

    def __init__(self, length: int):
        self.length = length

    def __call__(self, sample: dict) -> dict:
        ppg = sample["ppg"]
        ecg = sample["ecg"]

        L = ppg.size(-1)
        if L <= self.length:
            # 如果不够长，补零
            pad = self.length - L
            sample["ppg"] = F.pad(ppg, (0, pad))
            sample["ecg"] = F.pad(ecg, (0, pad))
            return sample

        start = torch.randint(0, L - self.length + 1, (1,)).item()
        sample["ppg"] = ppg[..., start : start + self.length]
        sample["ecg"] = ecg[..., start : start + self.length]
        return sample


class RandomNoise:
    """随机高斯噪声增强 (仅训练时)。"""

    def __init__(self, ppg_noise_std: float = 0.01, ecg_noise_std: float = 0.005):
        self.ppg_noise_std = ppg_noise_std
        self.ecg_noise_std = ecg_noise_std

    def __call__(self, sample: dict) -> dict:
        if torch.rand(1).item() < 0.5:
            sample["ppg"] = sample["ppg"] + torch.randn_like(sample["ppg"]) * self.ppg_noise_std
            sample["ecg"] = sample["ecg"] + torch.randn_like(sample["ecg"]) * self.ecg_noise_std
        return sample


class RandomScale:
    """随机幅度缩放。"""

    def __init__(self, scale_range: tuple[float, float] = (0.9, 1.1)):
        self.scale_range = scale_range

    def __call__(self, sample: dict) -> dict:
        scale = torch.empty(1).uniform_(*self.scale_range).item()
        sample["ppg"] = sample["ppg"] * scale
        sample["ecg"] = sample["ecg"] * scale
        return sample


def get_train_transforms(signal_length: int = 1250) -> Compose:
    """获取训练集变换。"""
    return Compose([
        RandomCrop(signal_length),
        ZScoreNormalize(),
        RandomNoise(),
        RandomScale(),
    ])


def get_eval_transforms(signal_length: int = 1250) -> Compose:
    """获取验证/测试集变换。"""
    return Compose([
        RandomCrop(signal_length),
        ZScoreNormalize(),
    ])
