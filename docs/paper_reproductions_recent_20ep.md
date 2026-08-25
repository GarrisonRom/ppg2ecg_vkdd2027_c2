# Recent Paper Mechanism Adaptations (20 Epochs)

## Protocol

The three methods below use the same SensSmartTech adaptation as the earlier
CardioGAN/RDDM comparison: `carotid_880nm -> Lead II`, 128 Hz, 4-second (512
sample) windows, 2-second stride, per-recording min-max normalization to
`[-1, 1]`, and subject-wise 80/20 split (26 train and 6 held-out subjects,
seed 42). Every metric is produced by `src.evaluation.metrics.evaluate_all`.

These are mechanism reproductions on a common local protocol, not the papers'
published leaderboard numbers. The original papers use different datasets,
sampling rates, window definitions, and training budgets.

## Test results

Lower is better for RMSE, MAE, HR error, RMSSD error, peak timing error, and
QRS width error. Higher is better for PCC and R-peak F1.

| Method | RMSE | MAE | PCC | HR error (bpm) | RMSSD error (ms) | R-peak F1 | Peak error (ms) | QRS width error (ms) | Params (M) |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| QRS-TransAttn | 0.3849 | 0.3150 | 0.2173 | 15.90 | 91.06 | 0.7185 | 44.64 | 29.62 | 0.650 |
| P2E-WGAN | 0.3682 | 0.2963 | 0.1969 | 12.24 | 73.91 | **0.8611** | **37.54** | 15.83 | 1.508 |
| Li 2024 lightweight | **0.3397** | **0.2784** | 0.1989 | 13.16 | **69.50** | 0.2003 | 79.46 | 94.86 | **0.021** |

## Train/test behavior

| Method | Train RMSE | Test RMSE | Train PCC | Test PCC | Train R-peak F1 | Test R-peak F1 |
|---|---:|---:|---:|---:|---:|---:|
| QRS-TransAttn | 0.2058 | 0.3849 | 0.5423 | 0.2173 | 0.7002 | 0.7185 |
| P2E-WGAN | 0.1911 | 0.3682 | 0.6407 | 0.1969 | 0.8540 | 0.8611 |
| Li 2024 lightweight | 0.3412 | 0.3397 | 0.1775 | 0.1989 | 0.2208 | 0.2003 |

The apparent train/test similarity of the lightweight network is not evidence
of strong generalization: it is underfitting, as shown by its low PCC and
R-peak F1 on both splits. P2E-WGAN is the strongest of this new group for
rhythm and peak timing at the 20-epoch budget. QRS-TransAttn improves the
explicit QRS-focused objective but still needs a longer schedule or a better
peak alignment term to approach the main project's v0.2/v0.61 rhythm results.

## Deviations from the source papers

- QRS-TransAttn: the public code's dataset/training setup is not copied into
  SensSmartTech; the local implementation retains the attention encoder-decoder
  and derives a QRS ROI from the target ECG for the weighted reconstruction.
- P2E-WGAN: the local generator/critic follow the paired conditional WGAN-GP
  mechanism and the large sample reconstruction term. A small QRS-weighted L1
  term is retained so the comparison exposes the peak-sensitive behavior that
  the project measures; this is recorded in the training history. The archived
  20-epoch run uses `ncritic=1` to keep the three-method comparison within the
  same budget; use `--ncritic 3` for the paper-style critic/update ratio.
- Li 2024 lightweight: this is an independent implementation of the reported
  multi-kernel attention/residual design, not a claim of author-code identity.

## Files and rerun

Results:

`paper_repro/runs/senssmarttech_recent_1to1_128hz_seed42/comparison_test.csv`

Rerun:

```powershell
D:\Anaconda\envs\cuda126_env\python.exe paper_repro\reproduce_recent.py --method all --epochs 20
```

Each method directory contains `checkpoint_final.pth`, train/test prediction
arrays, per-split JSON metrics, a training history, and a compact qualitative
train/test plot.
