"""模型模块注册表：按配置名称构建 encoder / decoder / diffusion。

配置文件中 model 分节的每个模块是一个 {name: ..., <超参>: ...} 字典，
name 必须命中注册表；其余键原样作为构造参数传入对应类。

新增模块只需:
  1. 实现类 (签名中的可配置超参即构造参数)
  2. 在对应 REGISTRY 中登记一行
"""

from __future__ import annotations

from typing import Any, Callable

from .baseline import BaselineECGDecoder, BaselinePPGEncoder
from .ecg_decoder import ECGDecoder
from .latent_diffusion import LatentDiffusion
from .ppg_encoder import PPGEncoder

ENCODER_REGISTRY: dict[str, type] = {
    "ppg_encoder": PPGEncoder,
    "baseline_encoder": BaselinePPGEncoder,
}

DECODER_REGISTRY: dict[str, type] = {
    "ecg_decoder": ECGDecoder,
    "baseline_decoder": BaselineECGDecoder,
}

DIFFUSION_REGISTRY: dict[str, type] = {
    "latent_diffusion": LatentDiffusion,
}


def build_from_registry(
    registry: dict[str, type],
    module_name: str,
    cfg: dict[str, Any] | None,
    **forced_kwargs: Any,
):
    """从注册表构建模块。cfg 为 None 时返回 None。

    Args:
        registry: 名称 -> 类 的映射
        module_name: 报错信息中使用的模块类别名 (如 "encoder")
        cfg: {name: 注册名, ...其余为构造参数}; None 表示不启用
        forced_kwargs: 由外部事实决定的参数 (如 signal_length 来自数据集),
                       优先级高于 cfg 中的同名键。
    """
    if cfg is None:
        return None
    cfg = dict(cfg)
    name = cfg.pop("name", None)
    if name is None:
        raise ValueError(f"{module_name} 配置缺少 'name' 字段: {cfg}")
    if name not in registry:
        raise KeyError(
            f"未注册的 {module_name} {name!r}。可选: {sorted(registry)}"
        )
    cfg.update(forced_kwargs)
    return registry[name](**cfg)


def build_encoder(
    cfg: dict[str, Any] | None,
    signal_length: int,
    latent_dim: int,
    ppg_channels: int = 1,
):
    """构建 PPG 编码器。signal_length/latent_dim 由数据集与全局配置强制指定。"""
    cfg_name = cfg.get("name") if cfg else None
    forced = {"signal_length": signal_length, "latent_dim": latent_dim}
    # 旧版 ppg_encoder 保持单通道接口；新 baseline 动态接收数据集通道数。
    if cfg_name == "baseline_encoder":
        forced["ppg_channels"] = ppg_channels
    return build_from_registry(
        ENCODER_REGISTRY, "encoder", cfg,
        **forced,
    )


def build_decoder(
    cfg: dict[str, Any] | None,
    signal_length: int,
    latent_dim: int,
    ecg_leads: int,
):
    """构建 ECG 解码器。"""
    return build_from_registry(
        DECODER_REGISTRY, "decoder", cfg,
        signal_length=signal_length, latent_dim=latent_dim, ecg_leads=ecg_leads,
    )


def build_diffusion(
    cfg: dict[str, Any] | None,
    latent_dim: int,
):
    """构建潜空间扩散模块 (可选)。"""
    return build_from_registry(
        DIFFUSION_REGISTRY, "diffusion", cfg,
        latent_dim=latent_dim,
    )
