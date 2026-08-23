"""PPG2ECG 模型定义。

子模块:
  - ppg_encoder: PPG 信号编码器 (频域 + 时域 + Cross-Attention)
  - latent_diffusion: 潜空间扩散模型 (DDIM)
  - ecg_decoder: 多尺度 ECG 恢复解码器
  - classifier: 心脏疾病分类器
  - losses: 复合损失函数 (模块可选)
  - registry: 模块注册表, 按配置名称构建 encoder/decoder/diffusion
"""

from .ppg_encoder import PPGEncoder
from .baseline import BaselineECGDecoder, BaselinePPGEncoder
from .ecg_decoder import ECGDecoder
from .latent_diffusion import LatentDiffusion
from .classifier import DiseaseClassifier
from .losses import ReconstructionLoss
from .registry import (
    DECODER_REGISTRY,
    DIFFUSION_REGISTRY,
    ENCODER_REGISTRY,
    build_decoder,
    build_diffusion,
    build_encoder,
)

__all__ = [
    "PPGEncoder",
    "BaselinePPGEncoder",
    "BaselineECGDecoder",
    "ECGDecoder",
    "LatentDiffusion",
    "DiseaseClassifier",
    "ReconstructionLoss",
    "ENCODER_REGISTRY",
    "DECODER_REGISTRY",
    "DIFFUSION_REGISTRY",
    "build_encoder",
    "build_decoder",
    "build_diffusion",
]
