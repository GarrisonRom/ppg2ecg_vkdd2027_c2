"""端到端前向验证：真实数据批次 → PPGEncoder → ECGDecoder → ReconstructionLoss。"""

import sys
from pathlib import Path

import torch

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

from src.data import create_dataloaders
from src.models import ReconstructionLoss, build_decoder, build_encoder
from src.utils.config import load_config

config = load_config(ROOT / "configs" / "exp_baseline.yaml")
cfg = config["data"]
loaders = create_dataloaders(
    dataset=cfg["dataset"],
    root=ROOT / cfg["root"],
    batch_size=8,
    num_workers=0,
    ppg_channel=cfg["ppg_channel"],
)
ds = loaders["train"].dataset
print(f"数据: train={len(ds)} 窗, fs={ds.fs}Hz, T={ds.signal_length}, "
      f"ppg={ds.ppg_channels}, ecg={ds.ecg_channels}")

batch = next(iter(loaders["train"]))
ppg, ecg = batch["ppg"], batch["ecg"]
print(f"批次: ppg {tuple(ppg.shape)}, ecg {tuple(ecg.shape)}")

device = torch.device("cpu")
model_cfg = config["model"]
encoder = build_encoder(
    model_cfg["encoder"], signal_length=ds.signal_length,
    latent_dim=model_cfg["latent_dim"], ppg_channels=ds.num_ppg_channels,
).to(device)
decoder = build_decoder(
    model_cfg["decoder"], latent_dim=model_cfg["latent_dim"],
    ecg_leads=ds.ecg_leads, signal_length=ds.signal_length,
).to(device)
criterion = ReconstructionLoss(
    weights={"mse": 1.0},
    ecg_leads=ds.ecg_leads,
)

encoded = encoder(ppg)
latent = encoded["latent"] if isinstance(encoded, dict) else encoded
ecg_pred = decoder(encoded)
losses = criterion(ecg_pred, ecg)

print(f"latent: {tuple(latent.shape)}")
print(f"ecg_pred: {tuple(ecg_pred.shape)} (期望 [8, {ds.ecg_leads}, {ds.signal_length}])")
assert ecg_pred.shape == (8, ds.ecg_leads, ds.signal_length), "输出形状不符"
assert torch.isfinite(losses["total"]), "损失非有限值"

losses["total"].backward()
grad_ok = all(p.grad is not None and torch.isfinite(p.grad).all()
              for p in list(encoder.parameters())[:5])
print(f"损失: total={losses['total'].item():.4f}, 反向传播梯度有限: {grad_ok}")
print("\n端到端前向验证: 通过")
