# Paper Baseline Reproductions (20 Epochs)

## Scope

This archive records controlled reimplementations of two verified PPG-to-ECG
papers on the project's SensSmartTech protocol. The runs are **paper-mechanism
adaptations**, not exact reproductions of the original papers: the original
experiments use different datasets, channel counts, sampling rates,
normalization, window lengths, hardware, and training budgets.

Common protocol:

- SensSmartTech, 4 PPG channels -> 4 ECG leads (`I`, `II`, `V3`, `V4`);
- 250 Hz, 2000 samples (8 seconds);
- subject-wise 22/5/5 split, seed 42;
- 20 epochs, CUDA 12.6 environment;
- train is in-sample fit, validation selects `best.pth`, test contains five
  unseen subjects.

The test numbers below must therefore be compared with the project's v0.x
rows under the same protocol. They must not be presented as the numerical
results reported by the papers.

## Reproduced methods

### CardioGAN

Reference: Sarkar and Etemad, *CardioGAN: Attentive Generative Adversarial
Network with Cycle-Consistency for ECG Reconstruction from PPG*, AAAI 2021,
DOI [10.1609/aaai.v35i1.16126](https://doi.org/10.1609/aaai.v35i1.16126).

The implementation keeps the paper's main mechanism:

- an attention U-Net for PPG -> ECG;
- a reverse ECG -> PPG attention U-Net;
- time-domain PatchGAN discriminators;
- log-STFT magnitude discriminators;
- least-squares GAN objectives and cycle consistency;
- the paper-inspired weights `alpha_time=3`, `beta_frequency=1`, and
  `cycle_weight=30`.

The original CardioGAN is single-channel and uses a bounded `[-1, 1]`
representation. The current run is 4 -> 4 and leaves the ECG output linear,
because the project's train-statistic z-score ECG has values outside `[-1, 1]`.
The paired batch is shuffled separately in each domain during optimization to
retain the paper's unpaired cycle-training behavior; evaluation remains on the
paired SensSmartTech windows.

Run directory:
`runs/senssmarttech_cardiogan_repro_20ep_seed42/`

### RDDM

Reference: Shome et al., *RDDM: Region-based Diffusion Model for PPG-to-ECG
Translation*, AAAI 2024, DOI
[10.1609/aaai.v38i13.29422](https://doi.org/10.1609/aaai.v38i13.29422), with the
public implementation and checkpoints at
[github.com/DebadityaQU/RDDM](https://github.com/DebadityaQU/RDDM).

The implementation keeps the paper's main mechanism:

- a 1000-step forward noise schedule;
- a target-ECG-derived QRS/ROI mask during training;
- separate ROI-guided and global conditional denoisers;
- ROI and global losses with `lambda_roi=100`, `lambda_global=1`;
- 10-step reverse sampling for the saved predictions.

The original paper is single PPG -> Lead-II ECG. Here the conditional and signal
networks are widened to 4 channels. The paper's `beta_end=0.2` schedule can
underflow in float32 at 1000 steps, so the reverse update clamps cumulative
alpha and sanitizes the reconstructed `x0`. This is a numerical stability
protection, not a change to the model objective.

Run directory:
`runs/senssmarttech_rddm_repro_20ep_seed42/`

## Results on SensSmartTech

All values are subject-level means on the indicated split. Lower is better for
RMSE, MAE, HR error, peak error, QRS width error, and RMSSD error; higher is
better for PCC and R-peak F1.

| Method | Split | RMSE | MAE | PCC | HR error (bpm) | R-peak F1 | Peak error (ms) | QRS width error (ms) | RMSSD error (ms) |
|---|---|---:|---:|---:|---:|---:|---:|---:|---:|
| CardioGAN | train | 1.2341 | 0.8300 | 0.0185 | 10.480 | 0.6271 | 97.808 | 9.563 | 86.802 |
| CardioGAN | val | 1.3902 | 0.9048 | -0.0091 | 2.174 | 0.6896 | 102.493 | 4.882 | 70.074 |
| CardioGAN | test | 1.1534 | 0.7965 | 0.0313 | 7.637 | 0.6194 | 98.302 | 8.156 | 77.784 |
| RDDM | train | 1.3491 | 0.8932 | 0.0567 | 14.696 | 0.8219 | 56.145 | 21.661 | 105.147 |
| RDDM | val | 1.4889 | 0.9534 | 0.0440 | 12.062 | 0.8793 | 55.037 | 14.884 | 113.966 |
| RDDM | test | 1.2920 | 0.8859 | 0.0713 | 16.804 | 0.8202 | 59.960 | 17.027 | 118.608 |

The current v0.x test matrix remains the primary project comparison. In this
20-epoch controlled run, CardioGAN has the lowest test RMSE among these two
paper adaptations, while RDDM has better PCC and substantially better R-peak
F1. Neither matches v0.2/v0.61 on all rhythm and morphology criteria. This is
consistent with the large budget and protocol differences: the original RDDM
paper trains for hundreds of epochs on single-channel public datasets, whereas
these adaptations use 20 epochs and reconstruct four leads simultaneously.

Efficiency recorded in the same JSON files:

| Method | Parameters (M) | Test inference time (s) | Sampling steps |
|---|---:|---:|---:|
| CardioGAN | 1.455 | 0.230 | n/a |
| RDDM | 0.842 | 7.510 | 10 |

The inference time is for the complete held-out split on the local GPU, not a
single 8-second window. RDDM is slower because each sample uses ten denoising
steps.

## Reproduction commands

```powershell
D:\Anaconda\envs\cuda126_env\python.exe scripts\train_paper_baselines.py --method cardiogan --config configs\exp_cardiogan_repro.yaml
D:\Anaconda\envs\cuda126_env\python.exe scripts\train_paper_baselines.py --method rddm --config configs\exp_rddm_repro.yaml
```

Each run stores `config.yaml`, checkpoints at epochs 5/10/15/20,
`best.pth`, `pred_{train,val,test}.npz`, `eval_{train,val,test}.json`, and
`training_history.json`.

## Interpretation boundary

The original published numbers are protocol references only. CardioGAN reports
single-channel waveform metrics on several public datasets, and RDDM reports
single-channel cross-subject results with a much larger training budget. A
fair leaderboard would retrain every method as 1 -> 1 Lead-II on one public
dataset with one subject-wise split, one normalization, and one peak evaluator.
The present runs answer a narrower question: whether the published mechanisms
can be implemented and measured consistently inside the current four-to-four
SensSmartTech experiment.
