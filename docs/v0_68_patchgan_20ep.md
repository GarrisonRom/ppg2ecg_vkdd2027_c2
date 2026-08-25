# v0.68 Conditional PatchGAN experiment

## Protocol

- Dataset: SensSmartTech
- Mapping: 4-channel PPG -> 4-lead ECG (`I/II/V3/V4`)
- Sampling: 250 Hz, 2000 samples (8 seconds)
- Subject split: 22 train / 5 validation / 5 test
- Seed: 42
- Activity A/B: metadata-only post-hoc groups; never a model input
- Backbone: v0.67 residual high-skip VAE/multi-band decoder path
- Training budget: 20 epochs, with the existing five-epoch encoder warm-up and
  early stopping patience 5

## PatchGAN design

The conditional discriminator receives the concatenated pair `[PPG, ECG]` and
returns a length-125 patch-logit map after four strided 1-D convolution blocks.
It is trained with hinge loss and its own AdamW optimizer (`2e-4`). The
generator sees only `0.02 * L_GAN`; the checkpoint is still selected by the
supervised reconstruction objective, not by the GAN loss. The discriminator and
optimizer states are saved in every checkpoint for auditability.

## Results (best checkpoint)

| Split | RMSE | PCC | HR error (bpm) | R-peak F1 | Peak error (ms) | QRS width error (ms) | RMSSD error (ms) |
|---|---:|---:|---:|---:|---:|---:|---:|
| Train | 0.8641 | 0.4319 | 10.8990 | 0.2957 | 60.3454 | 78.2932 | 65.0510 |
| Val | 1.0814 | 0.2931 | 8.9006 | 0.2130 | 75.1072 | 86.3134 | 60.7584 |
| Test | 0.8340 | 0.3331 | 11.5771 | 0.3234 | 65.6869 | 78.3057 | 79.3021 |

Training stopped at epoch 6; the best validation loss was `1.2727` at epoch 1.
The PatchGAN discriminator loss fell from `0.1431` in epoch 1 to `0.0144` in
epoch 6, while the generator adversarial loss grew from `2.11` to `3.27`.
This is discriminator saturation, not evidence of improved local morphology.

## Interpretation

The test RMSE (`0.8340`) is numerically competitive with the residual and
gated controls, but the physiological metrics collapse: R-peak F1 falls to
`0.3234` and QRS width error rises to `78.31 ms`. The same degradation is
already visible on the train split, so this run is not a generalization failure
or data leakage effect. With the current short-window, limited-subject setup,
the local discriminator rewards a waveform texture that is not aligned with
the R-peak/QRS structure needed for ECG monitoring.

The result is retained as a negative adversarial control. PatchGAN should not
be enabled in the main model until the discriminator is weakened or scheduled
after the supervised model has learned reliable peaks, and until selection
includes a hard R-peak/QRS guard rather than reconstruction loss alone.

## Artifacts

- Config: `configs/exp_v068_patchgan.yaml`
- Run: `runs/senssmarttech_v068_patchgan_20ep_seed42/`
- Implementation: `src/models/patchgan.py` and the optional PatchGAN path in
  `src/training/trainer.py`
