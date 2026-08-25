# SensSmartTech Experiment Matrix

## Evaluation protocol

All rows below use the same SensSmartTech subject-wise split (`22/5/5`),
four PPG channels, four ECG leads (`I/II/V3/V4`), `250 Hz`, `2000` samples,
seed `42`, and a 20-epoch budget unless explicitly marked as a diagnostic.
The main table reports the deployable PPG->ECG path on the held-out test split.
Values are subject-level means over the five test subjects.

Train values are reported separately as in-sample fit, validation is used only
for checkpoint selection, and the five-subject test split is the cross-subject
result. Absolute R-peak timing is retained as an auxiliary paired-window
diagnostic; it is not used as the primary ranking criterion for independent ECG
generation or downstream disease recognition.

The matrix was standardized to `best.pth` where available. v0.55-v0.57 were
re-evaluated with their saved `best.pth`; their older archive documents also
contain final-checkpoint numbers, so those values should not be mixed with the
table below.

Metrics:

- `RMSE`, `MAE`: lower is better for global waveform error.
- `PCC`: higher is better for waveform shape correlation.
- `HR err`: absolute heart-rate error in bpm, lower is better.
- `R-peak F1`: higher is better for peak detection and timing.
- `peak err`: matched R-peak timing error in ms, lower is better.
- `QRS width`: absolute QRS width error in ms, lower is better.
- `RMSSD err`: short-window HRV error in ms, lower is better but is less
  statistically reliable for an 8-second window.
- `NRMSE`: RMSE divided by target standard deviation, lower is better and makes
  cross-subject scale differences easier to inspect.
- `SNR`, `DTW`: signal-to-error ratio and a multi-scale time-alignment proxy.
- `RMS/energy/peak ratio`: generated amplitude divided by target amplitude;
  values near `1` indicate amplitude calibration rather than conservative
  smoothing or overshoot.
- `precision`, `recall`, `QRS amplitude ratio`, `RMSSD relative error`, and
  site-specific `PTT` are also recorded in the full report.
- Train/test gaps, A/B activity subgroup results, and inference efficiency are
  reported separately because they are not interchangeable with waveform
  quality.

The expanded comparison (all recorded dimensions, mean +/- subject standard
deviation, train/test gap, activity groups, and efficiency) is generated at
[`docs/experiment_results_full.md`](experiment_results_full.md). It is the
source to use when more than the compact five-to-eight-metric view is needed.

## Channel/split comparability

The main matrix is **4 PPG -> 4 ECG under subject-wise 22/5/5**. It should not
be compared as if it were the usual single-to-single paper protocol. The
existing Lead II rows are output slices from a model that still consumed all
four PPG channels; they are not retrained 1 -> 1 results.

| Comparison group | Input -> output | Split | Status |
|---|---|---|---|
| Main deployable result | 4 -> 4 | subject-wise 22/5/5 | primary ranking |
| In-sample fit | 4 -> 4 | train portion of 22 subjects | fit upper reference; not generalization |
| Lead II slice | 4 -> 1 (selected output) | subject-wise | descriptive; not single-to-single |
| Record-wise/sample-wise audit | 4 -> 4 | overlapping subjects across splits | leakage upper bound; excluded from ranking |
| Literature single-to-single | 1 -> 1 | often beat/cycle, subject-specific or sample-wise | protocol reference only |

The full mapping and the current Lead II slice numbers are documented in
[`evaluation_protocol_and_literature.md`](evaluation_protocol_and_literature.md).

## Method matrix

| ID | Main architecture / intervention | Supervision or constraint | Role in the project |
|---|---|---|---|
| v0.1 | Residual PPG encoder + skip-connected time decoder | MSE only | Deployable reference baseline |
| v0.2 | VAE content/style latent + conditional Flow | GRL subject discriminator, KL, V-REx | Useful disentanglement baseline |
| v0.3 | v0.2 architecture | Target QRS-L1 + derivative + STFT | Negative loss control |
| v0.4 | CardioAlign shared encoder + latent rectified Flow | Paired posterior alignment, InfoNCE, cross-modal reconstruction | Useful PPGFlowECG-inspired axis |
| v0.41 | ECG -> ECG skip autoencoder | Global L1, no PPG/Flow/GRL/IRM | Essential capacity/data diagnostic; not PPG2ECG |
| v0.5 | PPG->ECG plus learned ECG->PPG cycle | Joint direct and cycle L1 | Negative cycle control; collusion observed |
| v0.51 | v0.5 with two-stage training | Direct ECG->PPG pretrain, then freeze reverse branch | Useful training schedule control |
| v0.52-hf4 | Three FFT bands; high weight `4.0` | Normalized band loss | Negative high-band weight ablation |
| v0.52 | Three FFT bands; low/mid/high `0.5/1/2` | Frozen reverse cycle + normalized band loss | **Primary balanced baseline** |
| v0.53 | Time-domain decoder with Symlet-4 SWT loss | Coefficient and time-localized QRS-envelope losses | Useful morphology/wavelet loss candidate |
| v0.54 | Direct Haar coefficient decoder + fixed IDWT | Haar coefficients, QRS envelope, peak interval | Negative architecture control |
| v0.55 | v0.54 + wider PPG encoder | Latent `128->256`, encoder skips widened | Encoder bottleneck diagnostic |
| v0.56 | v0.55 + wider D1/D2 high-frequency decoder | Local residual blocks, high-frequency gain | High-frequency capacity diagnostic |
| v0.57 | v0.56 + zero-started residual high-frequency heads | Additive local residual path | Negative residual control |
| v0.58 | v0.52 + direct QRS/amplitude terms | QRS mask, QRS-L1 `0.50`, peak/RMS `0.75` | Negative over-weighted supervision |
| v0.59 | v0.52 + low-weight QRS/amplitude terms | QRS-L1 `0.10`, peak/RMS `0.20` | Local-detail candidate; global trade-off |
| v0.60 | v0.52 + high-band-only amplitude term | Peak/RMS supervision only on projected `10-40 Hz` branch | Preferred amplitude follow-up; not yet replacement |
| v0.61 | v0.2 VAE latent + v0.52 multi-band decoder | Five-epoch posterior-mean decoder warm-up, then joint GRL/V-REx fine-tuning | Strong rhythm-transfer decoder transplant |
| v0.62 | v0.61 with latent `128->256` | Same losses and schedule; fresh wider VAE encoder | Capacity ablation; lower RMSE but worse QRS physiology |
| v0.63 | v0.61 + light QRS/peak supervision | QRS amplitude `0.05` + differentiable peak interval `0.05` | Negative physiology-loss control |
| v0.64 | v0.62 with overlap-transfer initialization | Copy compatible v0.2 encoder prefixes into widened latent | Capacity/initialization ablation |
| v0.67-control | v0.61 protocol control | Same VAE/GRL/V-REx multi-band path with fixed subject-balanced schedule | Corrected protocol reference |
| v0.67-gated | v0.67-control + full-resolution PPG skip into high branch | Learned high-frequency gate; band loss active | Small QRS/generalization candidate |
| v0.67-residual | v0.67-gated with zero-started residual high-skip gain | `features + tanh(alpha) * residual`, `alpha=0` at initialization | Stability control for gated skip |
| v0.68-PatchGAN | v0.67-residual + conditional 1-D PatchGAN | Hinge loss, separate D optimizer, generator weight `0.02` | Adversarial local-realism ablation |

## Component matrix

`Yes` means the component is active in that experiment. `Target-only` means a
label or mask is used inside the training loss but is not a model input.

| ID | VAE | Flow | GRL / IRM | Frozen reverse cycle | FFT bands | Wavelet loss | Direct wavelet decoder | QRS / amplitude term | Main decoder signal |
|---|---|---|---|---|---|---|---|---|---|
| v0.1 | No | No | No | No | No | No | No | No | Time-domain ECG |
| v0.2 | Yes | Yes | Yes | No | No | No | No | No | Flow-conditioned ECG |
| v0.3 | Yes | Yes | Yes | No | No | No | No | Target-only QRS-L1 | Flow-conditioned ECG |
| v0.4 | Yes | Latent rectified | No | No | No | No | No | No | Aligned latent -> ECG |
| v0.41* | No | No | No | No | No | No | No | No | ECG -> ECG control |
| v0.5 | No | No | No | No | No | No | No | No | Time-domain ECG + cycle |
| v0.51 | No | No | No | Yes | No | No | No | No | Time-domain ECG + cycle |
| v0.52-hf4 | No | No | No | Yes | Yes (`high=4`) | No | No | No | Three-band FFT fusion |
| v0.52 | No | No | No | Yes | Yes (`high=2`) | No | No | No | Three-band FFT fusion |
| v0.53 | No | No | No | Yes | No | Yes (SWT) | No | QRS envelope | Time-domain ECG |
| v0.54 | No | No | No | Yes | No | No | Yes (Haar + IDWT) | QRS envelope + peak interval | IDWT ECG |
| v0.55 | No | No | No | Yes | No | No | Yes | QRS envelope + peak interval | IDWT ECG, wide encoder |
| v0.56 | No | No | No | Yes | No | No | Yes | QRS envelope + peak interval | IDWT ECG, wide high-frequency decoder |
| v0.57 | No | No | No | Yes | No | No | Yes | QRS envelope + peak interval | IDWT ECG + residual high-frequency path |
| v0.58 | No | No | No | Yes | No | No | No | Target-only QRS-L1 + peak/RMS | Three-band FFT fusion |
| v0.59 | No | No | No | Yes | No | No | No | Target-only QRS-L1 + peak/RMS | Three-band FFT fusion |
| v0.60 | No | No | No | Yes | No | No | No | Target-only high-band peak/RMS | Three-band FFT fusion |
| v0.61 | Yes | No | Yes | No | Yes | No | No | No | Three-band FFT fusion |
| v0.62 | Yes | No | Yes | No | Yes | No | No | No | Three-band FFT fusion |
| v0.63 | Yes | No | Yes | No | Yes | No | No | QRS amplitude + peak interval | Three-band FFT fusion |
| v0.64 | Yes | No | Yes | No | Yes | No | No | No | Three-band FFT fusion |
| v0.67-control | Yes | No | Yes | No | Yes | No | No | No | Three-band FFT fusion |
| v0.67-gated | Yes | No | Yes | No | Yes | No | No | No | Three-band FFT + gated high skip |
| v0.67-residual | Yes | No | Yes | No | Yes | No | No | No | Three-band FFT + residual high skip |
| v0.68-PatchGAN | Yes | No | Yes | No | Yes | No | No | No | Three-band FFT + residual high skip + PatchGAN |
| CardioGAN* | No | No | No | No | No | No | No | No | Attention U-Net + time/frequency GAN + cycle |
| RDDM* | No | No | No | No | No | No | No | No | ROI-guided conditional diffusion |

## Test metrics

| ID | ckpt | RMSE | MAE | PCC | HR err (bpm) | R-peak F1 | peak err (ms) | QRS width (ms) | RMSSD err (ms) |
|---|---|---:|---:|---:|---:|---:|---:|---:|---:|
| v0.1 | best | 0.8571 | 0.5515 | 0.2520 | 8.96 | 0.9038 | 33.64 | 33.65 | 61.58 |
| v0.2 | best | 0.8428 | 0.5332 | 0.3486 | 10.38 | **0.9215** | 25.83 | 15.53 | 49.92 |
| v0.3 | best | 0.8254 | 0.5102 | 0.3462 | 8.32 | 0.2449 | 106.36 | 49.03 | 59.03 |
| v0.4 | best | 0.8649 | 0.5477 | 0.2729 | **8.15** | **0.9308** | 29.50 | 22.85 | **49.76** |
| v0.41* | diagnostic | 0.1501 | 0.0860 | 0.9832 | 1.63 | 0.9879 | 1.89 | 3.76 | 12.22 |
| v0.5 | best | 0.8373 | 0.5221 | 0.3022 | 7.89 | 0.1972 | 97.47 | 67.63 | 56.26 |
| v0.51 | best | 0.8454 | 0.5278 | 0.3060 | 14.73 | 0.6018 | 49.11 | 34.03 | 94.70 |
| v0.52-hf4 | best | 0.8485 | 0.5284 | 0.3175 | 9.40 | 0.8555 | 29.23 | 14.64 | 72.81 |
| v0.52 | best | 0.8552 | 0.5345 | 0.3144 | 8.44 | 0.8920 | 28.22 | **8.85** | 63.79 |
| v0.53 | best | 0.8295 | 0.5255 | **0.3497** | 10.34 | 0.8598 | 28.41 | 9.65 | 79.63 |
| v0.54 | best | **0.8196** | **0.5078** | 0.3482 | 9.27 | 0.2199 | 89.22 | 54.75 | 67.04 |
| v0.55 | best | 0.8302 | 0.5169 | 0.3361 | 12.64 | 0.3471 | 71.41 | 55.69 | 68.51 |
| v0.56 | best | 0.8217 | 0.5141 | 0.3403 | 18.76 | 0.5610 | 51.76 | 36.01 | 87.72 |
| v0.57 | best | 0.8277 | 0.5171 | 0.3387 | 19.66 | 0.4024 | 73.49 | 64.79 | 88.28 |
| v0.58 | best | 0.9187 | 0.5710 | 0.2490 | 12.28 | 0.7362 | 39.02 | 20.96 | 107.73 |
| v0.59 | best | 0.9043 | 0.5612 | 0.2850 | 9.45 | 0.8929 | 28.15 | 9.37 | 62.60 |
| v0.60 | best | 0.8856 | 0.5444 | 0.2598 | **8.11** | 0.8926 | 28.80 | 9.27 | 64.98 |
| v0.61 | best_reconstruction | 0.8696 | 0.5489 | 0.3415 | 9.12 | **0.9266** | 25.46 | 11.85 | 52.03 |
| v0.62 | best | **0.8443** | 0.5401 | 0.2805 | 12.27 | 0.8119 | 37.96 | 32.10 | 78.68 |
| v0.63 | best | 0.8569 | 0.5403 | 0.3291 | 9.77 | 0.9188 | 26.07 | 13.76 | 53.80 |
| v0.64 | best | 0.8323 | 0.5352 | 0.3597 | 9.03 | 0.9010 | 27.28 | 18.16 | 56.72 |
| v0.67-control | best | 0.8549 | 0.5408 | 0.3311 | 8.58 | 0.9257 | 24.75 | 13.51 | 55.37 |
| v0.67-gated | best | 0.8417 | 0.5351 | 0.3400 | 9.90 | 0.9245 | 24.59 | 11.36 | 53.51 |
| v0.67-residual | best | 0.8435 | 0.5365 | 0.3322 | 10.04 | 0.9179 | 24.93 | 13.73 | 56.09 |
| v0.68-PatchGAN | best | 0.8340 | 0.5468 | 0.3331 | 11.58 | 0.3234 | 65.69 | 78.31 | 79.30 |
| CardioGAN* | best | 1.1534 | 0.7965 | 0.0313 | 7.64 | 0.6194 | 98.30 | 8.16 | 77.78 |
| RDDM* | best | 1.2920 | 0.8859 | 0.0713 | 16.80 | 0.8202 | 59.96 | 17.03 | 118.61 |

`v0.41*` supplies ECG as the input and therefore is not a valid PPG2ECG
comparison. It is included only to identify whether preprocessing or decoder
capacity is responsible for missing QRS detail.

`CardioGAN*` and `RDDM*` are controlled paper-mechanism adaptations on the
current 4 -> 4 SensSmartTech protocol. They are not the original paper runs;
see [`docs/paper_reproductions_20ep.md`](paper_reproductions_20ep.md) for the
protocol differences, commands, train/validation results, and limitations.

## What is useful

### Keep as the current main line

1. **v0.52** is the best balanced PPG2ECG baseline. It has the best QRS width
   error among deployable models (`8.85 ms`), strong R-peak F1 (`0.8920`), and
   low HR error (`8.44 bpm`) without the severe global-error regression of the
   explicit fused-waveform losses.
2. **v0.60** is the best controlled follow-up for amplitude calibration. Its
   high-band-only term keeps F1 and HR essentially at v0.52 level, but PCC and
   global errors are worse, so it should remain an ablation.
3. **v0.53** is the strongest morphology-loss candidate: it gives the best
   deployable PCC (`0.3497`) and low RMSE (`0.8295`), with QRS width `9.65 ms`.
   Its HR error and F1 do not beat v0.52, so the next wavelet experiment should
   target peak alignment rather than simply increasing wavelet weights.
4. **v0.2/v0.4** should remain the representation-learning axis. v0.2 gives
   a strong F1/QRS compromise and demonstrates GRL/Flow/V-REx; v0.4 has the
   highest deployable F1 (`0.9308`) but weaker global morphology.
5. **v0.61** is the strongest decoder-transplant result: it retains the v0.2
   rhythm advantage while adding the v0.52 multi-band path. It should be the
   next representation/decoder baseline, with v0.52 retained as the balanced
   cycle baseline.
6. **v0.62** is a useful negative capacity result. Doubling latent width lowers
   RMSE but under-amplifies QRS and worsens peak timing, F1, HR, and PCC; latent
   dimension should not be increased without a morphology/peak objective.
7. **v0.63** is a negative light-supervision control. Small fused-output
   QRS-amplitude and differentiable peak-interval terms do not improve the
   held-out peak/QRS metrics under the current 20-epoch VAE/GRL schedule.
8. **v0.64** shows that overlap-transfer initialization rescues much of the
   latent-256 regression: RMSE/PCC/HR improve over v0.62, but R-peak F1 and
   QRS width remain behind v0.61. Keep it as an initialization/capacity
   ablation rather than the main model.
9. **v0.67-gated** is a small structural improvement over its corrected v0.61
   protocol control: test RMSE, PCC, QRS width, and RMSSD all improve, but HR
   error is slightly worse. The residual-gate follow-up is stable but does not
   repeat those gains, so the gated path remains a candidate rather than a
   replacement for v0.52/v0.61.
10. **v0.68-PatchGAN** is a negative adversarial control. Its global RMSE looks
    competitive, but R-peak F1 and QRS width collapse already on the training
    split while the discriminator saturates. This is a local-realism mismatch,
    not a cross-subject generalization gain.

### Keep as diagnostics or ablations

- **v0.41** proves that the cached ECG contains sharp QRS and the skip decoder
  can reconstruct it. The remaining bottleneck is PPG-to-ECG information,
  not a universally broken decoder.
- **v0.51** demonstrates why freezing a directly trained reverse branch is
  preferable to the joint cycle in v0.5, even though its HR transfer is poor.
- **v0.54-v0.57** map the direct wavelet coefficient bottleneck: widening the
  encoder and high-frequency decoder improves peak timing somewhat, but the
  short frozen-cycle stage and PPG information bottleneck still limit QRS
  amplitude.

### Do not use as the next default

- **v0.3, v0.5, v0.52-hf4, v0.57, v0.58** are useful negative controls. They
  show that lower RMSE, a lower cycle loss, or more high-frequency weight does
  not guarantee correct QRS timing and morphology.
- **v0.59** makes local spikes more visible, but its global waveform metrics
  are worse than v0.52 and visual overshoot remains.

## Recommended next experiment block

Use v0.52 as the fixed reference and run a small factorial block:

| Axis | Values | Purpose |
|---|---|---|
| Amplitude target | fused ECG vs high-band only | separate global distortion from QRS calibration |
| QRS term weight | `0.05`, `0.10`, `0.15` | find a non-destructive range |
| Peak alignment | none vs differentiable peak-time term | distinguish amplitude from temporal error |
| Wavelet path | v0.53 loss vs v0.60 high-band term | compare localized coefficients with FFT bands |
| Training schedule | 10 vs 30 frozen-cycle epochs | test whether high-frequency branches simply need more optimization |

Select by a joint report, not one metric: require `R-peak F1`, QRS width, HR
error, and PCC to be reported together. A/B activity remains post-hoc analysis
only and must not be added as a model condition.

## Source files

Each row has a reproducible config and run directory under `configs/` and
`runs/`. The detailed per-version notes are linked from the experiment history
in `README.md`; each run also contains `eval_train.json`, `eval_test.json`,
`training_history.json`, checkpoints, and compact train/test A/B figures.
