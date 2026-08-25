# Main Models: 50-Epoch Results

## Protocol

- Dataset: SensSmartTech
- Mapping: 4 PPG -> 4 ECG leads
- Sampling/window: 250 Hz, 8 seconds, 2000 samples
- Split: subject-wise 22/5/5 (`train/val/test`)
- Seed: 42
- Device: NVIDIA GeForce RTX 4060 Ti, `cuda126_env`
- Primary checkpoint: `best.pth` selected on validation loss

The four runs are isolated from the earlier 20-epoch archives. v0.52 selects
the best checkpoint by `ecg_total`; v0.2, v0.61 and v0.64 select by `total`.

## Best Checkpoint Test Results

Lower is better for RMSE, HR error, peak timing error and QRS width error.
Higher is better for PCC and R-peak F1.

| Model | Best epoch | Train RMSE | Train PCC | Test RMSE | Test PCC | Test HR err (bpm) | Test R-peak F1 | Test peak err (ms) | Test QRS err (ms) |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| v0.2 VAE+Flow+GRL/IRM | 14 | 0.7937 | 0.5330 | 0.8586 | 0.3349 | 8.985 | 0.9199 | 27.553 | 18.037 |
| v0.52 Multi-band + frozen cycle | 12 | 0.7746 | 0.5659 | 0.8572 | 0.3078 | 7.734 | 0.9264 | 26.512 | **6.416** |
| v0.61 VAE multi-band latent-128 | 5 | 0.7700 | 0.5771 | 0.8664 | 0.3490 | 8.723 | 0.9226 | **25.657** | 12.472 |
| v0.64 VAE multi-band latent-256 | 2 | 0.8225 | 0.5034 | 0.8326 | 0.3583 | 8.971 | 0.9009 | 26.782 | 18.052 |

## Best Versus Final

The final checkpoint continues optimizing the training windows after the
validation-selected epoch. The resulting Train/Test PCC pairs are:

| Model | Best Train PCC | Best Test PCC | Final Train PCC | Final Test PCC | Final Train RMSE | Final Test RMSE |
|---|---:|---:|---:|---:|---:|---:|
| v0.2 | 0.5330 | 0.3349 | 0.7678 | 0.3135 | 0.5783 | 0.8920 |
| v0.52 | 0.5659 | 0.3078 | 0.8440 | 0.2600 | 0.4913 | 0.9838 |
| v0.61 | 0.5771 | 0.3490 | 0.8614 | 0.2658 | 0.4603 | 0.9742 |
| v0.64 | 0.5034 | 0.3583 | 0.8665 | 0.3018 | 0.4524 | 0.9400 |

This is direct evidence of overfitting under the 50-epoch schedule: final
training PCC rises for every model, while final test PCC falls relative to its
validation-selected checkpoint. The best checkpoint should be used for the
current comparison.

## Artifacts

Checkpoints and histories:

- `runs/senssmarttech_v02_vae_flow_adv_irm_50ep_seed42/`
- `runs/senssmarttech_v052_multiband_frozen_cycle_50ep_seed42/`
- `runs/senssmarttech_v061_vae_multiband_transfer_latent128_50ep_seed42/`
- `runs/senssmarttech_v064_vae_multiband_latent256_transfer_50ep_seed42/`

Each run contains `best.pth`, `final.pth`, `epoch_*.pth`,
`training_history.json`, and `eval_{train,val,test}_{best,final}.json`.

High-resolution Lead-II/A-B comparison figures:

- `results/v0_50ep_figures/v02/`
- `results/v0_50ep_figures/v052/`
- `results/v0_50ep_figures/v061/`
- `results/v0_50ep_figures/v064/`
