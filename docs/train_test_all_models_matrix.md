# Unified Train/Test Matrix

> Main protocol: 4 PPG -> 4 ECG, 250 Hz, 8 s, subject-wise 22/5/5. Paper protocol: 1 PPG -> Lead II, 128 Hz, 4 s, subject-wise 26/6. The two protocols are listed together for reference but are not one numerical leaderboard.

## One-Table Summary

| Protocol | Model | Train RMSE / PCC | Test RMSE / PCC | RMSE gap | Test HR err / R-F1 | Test Peak / QRS ms | Test RMSSD err ms | Test SNR / DTW | Test valid |
|---|---|---|---|---|---|---|---|---|---|
| main 4->4 \| 250 Hz \| 8 s \| 22/5/5 | v0.2 | 0.769 / 0.567 | 0.843 / 0.349 | +9.7% | 10.38 / 0.921 | 25.83 / 15.53 | 49.92 | 0.16 / 0.439 | 1.000 |
| main 4->4 \| 250 Hz \| 8 s \| 22/5/5 | v0.52 | 0.772 / 0.573 | 0.855 / 0.314 | +10.8% | 8.44 / 0.892 | 28.22 / 8.85 | 63.79 | -0.04 / 0.437 | 1.000 |
| main 4->4 \| 250 Hz \| 8 s \| 22/5/5 | v0.61 | 0.760 / 0.588 | 0.870 / 0.341 | +14.4% | 9.12 / 0.927 | 25.46 / 11.85 | 52.03 | -0.18 / 0.452 | 1.000 |
| main 4->4 \| 250 Hz \| 8 s \| 22/5/5 | v0.64 | 0.827 / 0.496 | 0.832 / 0.360 | +0.7% | 9.03 / 0.901 | 27.28 / 18.16 | 56.72 | 0.24 / 0.440 | 1.000 |
| paper 1->1 \| 128 Hz \| 4 s \| 26/6 | CardioGAN | 0.551 / 0.013 | 0.547 / 0.038 | -0.8% | 9.74 / 0.838 | 86.69 / 8.81 | 66.91 | -2.25 / 0.432 | 0.999 |
| paper 1->1 \| 128 Hz \| 4 s \| 26/6 | RDDM | 0.518 / 0.022 | 0.554 / 0.035 | +7.1% | 24.73 / 0.774 | 71.39 / 26.51 | 104.00 | -2.21 / 0.462 | 1.000 |
| paper 1->1 \| 128 Hz \| 4 s \| 26/6 | QRS-TransAttn | 0.206 / 0.542 | 0.385 / 0.217 | +87.0% | 15.90 / 0.719 | 44.64 / 29.62 | 91.06 | 0.95 / 0.285 | 1.000 |
| paper 1->1 \| 128 Hz \| 4 s \| 26/6 | P2E-WGAN | 0.191 / 0.641 | 0.368 / 0.197 | +92.7% | 12.24 / 0.861 | 37.54 / 15.83 | 73.91 | 1.20 / 0.259 | 1.000 |
| paper 1->1 \| 128 Hz \| 4 s \| 26/6 | Li 2024 lightweight | 0.341 / 0.178 | 0.340 / 0.199 | -0.5% | 13.16 / 0.200 | 79.46 / 94.86 | 69.50 | 1.91 / 0.247 | 0.984 |

RMSE/PCC: lower/higher is better. HR, Peak, QRS and RMSSD errors: lower is better. QRS amp ratio is best near 1. Valid rate is higher is better.

## Full Long-Form Table

The following rows preserve every recorded metric and place Train/Test beside each other for every model.

| Protocol | Model | Split | Params M | Windows | Subjects | RMSE | MAE | PCC | NRMSE | SNR (dB) | DTW | HR err (bpm) | R-peak F1 | Peak err (ms) | QRS err (ms) | RMSSD err (ms) | QRS amp ratio | Carotid PTT err (ms) | Carotid PTT in range | Valid rate |
|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|
| main 4->4 \| 250 Hz \| 8 s \| 22/5/5 | v0.2 | train | 1.122 | 1428 | 22 | 0.769 | 0.451 | 0.567 | 0.826 | 1.90 | 0.350 | 7.76 | 0.893 | 20.78 | 16.12 | 48.18 | 0.538 | 22.81 | 0.138 | 1.000 |
| main 4->4 \| 250 Hz \| 8 s \| 22/5/5 | v0.2 | test | 1.122 | 318 | 5 | 0.843 | 0.533 | 0.349 | 1.002 | 0.16 | 0.439 | 10.38 | 0.921 | 25.83 | 15.53 | 49.92 | 0.718 | 23.29 | 0.157 | 1.000 |
| main 4->4 \| 250 Hz \| 8 s \| 22/5/5 | v0.52 | train | 1.411 | 1428 | 22 | 0.772 | 0.439 | 0.573 | 0.809 | 2.04 | 0.343 | 9.69 | 0.855 | 21.57 | 10.62 | 67.50 | 0.492 | 29.34 | 0.198 | 1.000 |
| main 4->4 \| 250 Hz \| 8 s \| 22/5/5 | v0.52 | test | 1.411 | 318 | 5 | 0.855 | 0.535 | 0.314 | 1.025 | -0.04 | 0.437 | 8.44 | 0.892 | 28.22 | 8.85 | 63.79 | 0.745 | 27.26 | 0.221 | 1.000 |
| main 4->4 \| 250 Hz \| 8 s \| 22/5/5 | v0.61 | train | 1.510 | 1428 | 22 | 0.760 | 0.442 | 0.588 | 0.812 | 2.05 | 0.341 | 7.03 | 0.901 | 19.90 | 10.94 | 43.13 | 0.580 | 23.48 | 0.139 | 1.000 |
| main 4->4 \| 250 Hz \| 8 s \| 22/5/5 | v0.61 | test | 1.510 | 318 | 5 | 0.870 | 0.549 | 0.341 | 1.049 | -0.18 | 0.452 | 9.12 | 0.927 | 25.46 | 11.85 | 52.03 | 0.857 | 21.71 | 0.163 | 1.000 |
| main 4->4 \| 250 Hz \| 8 s \| 22/5/5 | v0.64 | train | 2.330 | 1428 | 22 | 0.827 | 0.488 | 0.496 | 0.880 | 1.25 | 0.378 | 8.32 | 0.848 | 25.71 | 21.75 | 53.83 | 0.392 | 31.03 | 0.146 | 1.000 |
| main 4->4 \| 250 Hz \| 8 s \| 22/5/5 | v0.64 | test | 2.330 | 318 | 5 | 0.832 | 0.535 | 0.360 | 0.989 | 0.24 | 0.440 | 9.03 | 0.901 | 27.28 | 18.16 | 56.72 | 0.558 | 25.72 | 0.167 | 1.000 |
| paper 1->1 \| 128 Hz \| 4 s \| 26/6 | CardioGAN | train | 1.454 | 3627 | 26 | 0.551 | 0.475 | 0.013 | 2.251 | -1.47 | 0.427 | 10.26 | 0.818 | 83.60 | 9.86 | 65.33 | 0.872 | 72.39 | 0.473 | 1.000 |
| paper 1->1 \| 128 Hz \| 4 s \| 26/6 | CardioGAN | test | 1.454 | 767 | 6 | 0.547 | 0.475 | 0.038 | 2.273 | -2.25 | 0.432 | 9.74 | 0.838 | 86.69 | 8.81 | 66.91 | 0.908 | 71.02 | 0.554 | 0.999 |
| paper 1->1 \| 128 Hz \| 4 s \| 26/6 | RDDM | train | 0.840 | 3627 | 26 | 0.518 | 0.435 | 0.022 | 2.137 | -0.71 | 0.413 | 23.47 | 0.760 | 71.76 | 26.31 | 110.82 | 0.356 | 53.54 | 0.169 | 1.000 |
| paper 1->1 \| 128 Hz \| 4 s \| 26/6 | RDDM | test | 0.840 | 767 | 6 | 0.554 | 0.480 | 0.035 | 2.308 | -2.21 | 0.462 | 24.73 | 0.774 | 71.39 | 26.51 | 104.00 | 0.412 | 46.62 | 0.149 | 1.000 |
| paper 1->1 \| 128 Hz \| 4 s \| 26/6 | QRS-TransAttn | train | 0.650 | 3627 | 26 | 0.206 | 0.123 | 0.542 | 0.841 | 6.86 | 0.087 | 14.63 | 0.700 | 41.15 | 33.78 | 83.61 | 0.375 | 51.88 | 0.245 | 0.999 |
| paper 1->1 \| 128 Hz \| 4 s \| 26/6 | QRS-TransAttn | test | 0.650 | 767 | 6 | 0.385 | 0.315 | 0.217 | 1.574 | 0.95 | 0.285 | 15.90 | 0.719 | 44.64 | 29.62 | 91.06 | 0.428 | 52.81 | 0.252 | 1.000 |
| paper 1->1 \| 128 Hz \| 4 s \| 26/6 | P2E-WGAN | train | 1.508 | 3627 | 26 | 0.191 | 0.129 | 0.641 | 0.784 | 7.53 | 0.091 | 9.57 | 0.854 | 23.93 | 19.00 | 62.27 | 0.521 | 31.78 | 0.187 | 1.000 |
| paper 1->1 \| 128 Hz \| 4 s \| 26/6 | P2E-WGAN | test | 1.508 | 767 | 6 | 0.368 | 0.296 | 0.197 | 1.526 | 1.20 | 0.259 | 12.24 | 0.861 | 37.54 | 15.83 | 73.91 | 0.610 | 37.54 | 0.139 | 1.000 |
| paper 1->1 \| 128 Hz \| 4 s \| 26/6 | Li 2024 lightweight | train | 0.021 | 3627 | 26 | 0.341 | 0.278 | 0.178 | 1.396 | 2.67 | 0.242 | 11.16 | 0.221 | 76.39 | 98.10 | 77.70 | 0.424 | 174.89 | 0.439 | 0.989 |
| paper 1->1 \| 128 Hz \| 4 s \| 26/6 | Li 2024 lightweight | test | 0.021 | 767 | 6 | 0.340 | 0.278 | 0.199 | 1.371 | 1.91 | 0.247 | 13.16 | 0.200 | 79.46 | 94.86 | 69.50 | 0.491 | 186.05 | 0.412 | 0.984 |

## Reading the Matrix

- Train is an in-sample fit reference, not a generalization score.
- A small RMSE gap with near-zero PCC indicates underfitting, not strong invariance.
- Cross-protocol absolute RMSE values must not be ranked directly because sampling, window length, channel count and normalization differ.
- For SCD-oriented interpretation, prioritize Test R-peak F1, Peak error, QRS width error and HR/RMSSD errors over RMSE alone.

Source files: `runs/*/eval_train.json`, `runs/*/eval_test.json`, and `paper_repro/runs/*/metrics_{train,test}.json`.
