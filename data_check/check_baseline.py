"""验证 SensSmartTech 四路 PPG baseline 的数据流和反向传播。"""

from __future__ import annotations

import sys
from pathlib import Path

import torch

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

from src.data import create_dataloaders
from src.models import build_decoder, build_encoder
from src.utils.config import load_config


def main() -> None:
    config = load_config(ROOT / "configs" / "exp_baseline.yaml")
    data_cfg = config["data"]
    loaders = create_dataloaders(
        dataset=data_cfg["dataset"],
        root=ROOT / data_cfg["root"],
        batch_size=2,
        num_workers=0,
        ppg_channel=data_cfg.get("ppg_channel"),
    )
    dataset = loaders["train"].dataset
    batch = next(iter(loaders["train"]))

    model_cfg = config["model"]
    encoder = build_encoder(
        model_cfg["encoder"], dataset.signal_length,
        model_cfg["latent_dim"], dataset.num_ppg_channels,
    )
    decoder = build_decoder(
        model_cfg["decoder"], dataset.signal_length,
        model_cfg["latent_dim"], dataset.ecg_leads,
    )

    encoded = encoder(batch["ppg"])
    pred = decoder(encoded)
    loss = torch.nn.functional.mse_loss(pred, batch["ecg"])
    loss.backward()

    expected = (2, dataset.ecg_leads, dataset.signal_length)
    assert pred.shape == expected, f"输出形状 {tuple(pred.shape)} != {expected}"
    assert torch.isfinite(loss), "baseline loss 不是有限值"
    params = list(encoder.parameters()) + list(decoder.parameters())
    assert all(p.grad is not None and torch.isfinite(p.grad).all() for p in params)

    latent = encoded["latent"]
    print(f"PPG: {tuple(batch['ppg'].shape)}")
    print(f"latent: {tuple(latent.shape)}")
    print(f"ECG: {tuple(pred.shape)}")
    print(f"MSE: {loss.item():.6f}")
    print("baseline data-flow check: PASS")


if __name__ == "__main__":
    main()
