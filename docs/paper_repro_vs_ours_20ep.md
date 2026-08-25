# Paper Reproduction Versus Our Models

This note compares the completed 20-epoch paper-protocol adaptation with the
project's recorded v0.x experiments. The same evaluator is used, but the two
groups are **not the same protocol** and must not be treated as one leaderboard.

## Protocol boundary

| Group | Mapping | Sampling/window | Split | Normalization |
|---|---|---|---|---|
| Our v0.x runs | 4 PPG -> 4 ECG | 250 Hz, 8 s, 2000 points | subject-wise 22/5/5 | project `subjectwise-per-lead` preprocessing |
| Paper adaptations | 1 `carotid_880nm` -> 1 Lead II | 128 Hz, 4 s, 512 points | subject-wise 80/20 (26/6) | per-recording min-max to `[-1, 1]` |

The paper adaptation is therefore a controlled implementation comparison, not
an exact reproduction of the original paper datasets or a direct replacement
for the four-channel experiments.

## Test comparison

All rows below use the project's common evaluator. Lower is better for RMSE,
HR error, R-peak timing error and QRS-width error; higher is better for PCC,
R-peak F1.

| Model | Protocol | RMSE | PCC | HR error (bpm) | R-peak F1 | Peak timing error (ms) | QRS width error (ms) |
|---|---|---:|---:|---:|---:|---:|---:|
| v0.2 | our 4->4 | 0.843 | 0.349 | 10.38 | 0.921 | 25.83 | 15.53 |
| v0.52 | our 4->4 | 0.855 | 0.314 | **8.44** | 0.892 | 28.22 | **8.85** |
| v0.61 | our 4->4 | 0.870 | 0.341 | 9.12 | **0.927** | **25.46** | 11.85 |
| v0.64 | our 4->4 | **0.832** | **0.360** | 9.03 | 0.901 | 27.28 | 18.16 |
| CardioGAN | paper 1->1 | **0.547** | 0.038 | 9.74 | 0.838 | 86.69 | 8.81 |
| RDDM | paper 1->1 | 0.554 | 0.035 | 24.73 | 0.774 | 71.39 | 26.51 |
| QRS-TransAttn | paper 1->1 | 0.385 | 0.217 | 15.90 | 0.719 | 44.64 | 29.62 |
| P2E-WGAN | paper 1->1 | 0.368 | 0.197 | 12.24 | **0.861** | **37.54** | 15.83 |
| Li 2024 lightweight | paper 1->1 | **0.340** | 0.199 | 13.16 | 0.200 | 79.46 | 94.86 |

The lower RMSE of the paper rows is primarily a scale/protocol effect: they use
one min-max-normalized lead, 128 Hz and 4-second windows. It is not evidence
that they reconstruct our four-lead task better.

## Interpretation

1. **Waveform shape:** our v0.64 has the best recorded four-channel test RMSE
   and PCC. Among the new paper adaptations, Li's lightweight row has the
   lowest RMSE, but its PCC and rhythm scores show underfitting.
2. **Rhythm:** v0.61 remains the strongest v0.x R-peak reference. P2E-WGAN is
   the strongest new paper adaptation, with R-peak F1 0.861 and 37.54 ms timing
   error. CardioGAN is lower in F1 and has about 87 ms timing error; RDDM has a
   24.7 bpm HR error.
3. **QRS morphology:** CardioGAN's 8.81 ms QRS-width error is comparable to
   v0.52's 8.85 ms. P2E-WGAN reaches 15.83 ms and improves timing, while the
   QRS-TransAttn row is a direct reference for explicit QRS supervision.
4. **Generalization:** CardioGAN train/test RMSE is 0.551/0.547 and RDDM is
   0.518/0.554. The small gaps should not be read as strong invariance: both
   models already have poor train-set PCC (0.013 and 0.022), so a small gap can
   simply mean underfitting. Under its own protocol, v0.2 has substantially
   stronger shape/rhythm metrics on held-out subjects; its RMSE is not directly
   comparable to the paper-adaptation scale.

## Bottom line

For this SensSmartTech study, the paper adaptations are useful baselines and
implementation references, but they do **not** currently beat our main models
on the clinically relevant combination of morphology, R-peak timing and
rhythm. The strongest existing components remain complementary:

- v0.64: best recorded global waveform score;
- v0.61: strongest R-peak F1 and timing among the v0.x models;
- v0.52: strongest QRS-width/HR trade-off;
- CardioGAN: a useful adversarial/cycle baseline whose QRS-width result merits
  an ablation, but whose near-zero PCC needs to be fixed before claiming a gain;
- P2E-WGAN: the strongest new 20-epoch paper adaptation for R-peak F1 and
  timing on this local protocol;
- QRS-TransAttn: a direct QRS-supervision reference for the project's loss and
  decoder experiments;
- RDDM: currently not competitive under the 20-epoch local adaptation;
- Li 2024 lightweight: an efficiency reference, not a rhythm-quality winner at
  this short budget.

Source files:

- `paper_repro/runs/senssmarttech_1to1_128hz_seed42/comparison_test.csv`
- `paper_repro/runs/senssmarttech_recent_1to1_128hz_seed42/comparison_test.csv`
- `docs/experiment_results_full.md`
