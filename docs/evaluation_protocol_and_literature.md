# Evaluation Protocol and Literature Comparison

## Protocol used in this project

The main SensSmartTech result uses a fixed **subject-wise 22/5/5 split**:

- 22 subjects for training;
- 5 subjects for validation and checkpoint selection;
- 5 completely unseen subjects for the final test.

The split is stored in
`data/processed/SensSmartTech/subjectwise_per-lead/split.json` with seed 42.
The input is four PPG channels and the target is four ECG leads (`I/II/V3/V4`),
sampled at 250 Hz in 8-second windows.

The validation set is not a second test set: it is used to select `best.pth`.
The train split is reported as **in-sample fit**, while the test split is the
primary cross-subject result. A separate record-wise split is an intentional
leakage upper bound, not a deployable result.

## Protocol matrix: single-to-single versus multi-channel

The number of input/output channels and the split protocol are two independent
axes. A paper can be single-to-single but still be much easier if it uses
beat-level alignment or lets windows from the same subject appear in train and
test. Conversely, the current four-to-four experiment has more output
morphology to reconstruct and tests five completely unseen subjects.

| Protocol label | Input -> target | Unit | Split | Leakage / difficulty | Can be ranked with the main result? |
|---|---|---|---|---|---|
| Current main | 4 PPG -> 4 ECG (`I/II/V3/V4`) | 8 s window | subject-wise 22/5/5 | no subject overlap; hardest current project protocol | Yes, within this project |
| Current in-sample | 4 PPG -> 4 ECG | 8 s window | train portion of 22 subjects | measures fit, not generalization | No; report as an upper fit reference |
| Current Lead II slice | **4 PPG -> 1 ECG (II)** | 8 s window | subject-wise 22/5/5 | output is reduced, but all four PPG channels remain; **not single-to-single** | No; descriptive only |
| Strict single-to-single target protocol | 1 PPG -> 1 ECG (II) | same window and sampling | subject-wise 22/5/5 | fair single-lead comparison if retrained with one input channel | Not yet measured in this archive |
| Typical single-to-single paper | 1 PPG -> 1 ECG | beat/cycle or short window | varies | fewer channels and often alignment/subject-specific fitting | Protocol reference only |
| Sample-wise / record-wise paper | usually 1 -> 1 | window or beat | random windows; subjects/records may overlap | optimistic leakage; same subject physiology can occur in all splits | Never as cross-subject ranking |
| Multi-to-single | 2-4 PPG -> 1 ECG | window | varies | more input information, less output burden | Compare only with matching protocol |
| Multi-to-multi | 2-4 PPG -> 2-4 ECG | window | varies | output morphology burden is higher | Compare only with matching protocol |

`train` is therefore useful for answering “can this architecture fit the
available mappings?”, while `test` answers “does the mapping transfer to a new
person?”. It is expected that a sample-wise or subject-specific single-to-single
paper reports a higher PCC than the current unseen-subject four-to-four result;
that is a protocol difference, not automatically an architectural advantage.

### Existing Lead II slice (not strict single-to-single)

These values are obtained by evaluating the Lead II output of a model trained
with **four PPG inputs and four ECG outputs**. They are included to make the
comparison transparent, but must not be labeled as a single-channel model.

| Model / protocol | Train RMSE | Train PCC | Subject-wise test RMSE | Subject-wise test PCC |
|---|---:|---:|---:|---:|
| v0.1, 4 -> 4, Lead II slice | 0.869 | 0.441 | 0.816 | 0.176 |
| v0.2, 4 -> 4, Lead II slice | 0.795 | 0.559 | 0.795 | 0.325 |
| v0.52, 4 -> 4, Lead II slice | 0.801 | 0.557 | 0.816 | 0.278 |
| v0.61, 4 -> 4, Lead II slice | 0.780 | 0.584 | 0.835 | 0.312 |
| Record-wise baseline, 4 -> 4, Lead II slice | 0.717 | 0.613 | 0.886 | 0.365 |

The record-wise test PCC is only a leakage upper bound: the same subjects are
allowed in different splits, so it cannot be used to claim cross-subject
generalization. The train PCC being higher than the subject-wise test PCC is
the expected fit-to-transfer gap, even though RMSE can move differently after
normalization.

## What “synchrony” means here

`rpeak_time_err_ms` compares the detected R-peak sample positions of generated
and real ECG in the same 8-second window. It is not an amplitude metric. For an
independent ECG generation or downstream disease classifier, it should not be
the primary ranking metric. The main ranking is instead:

1. R-peak F1/recall and RR/HR error;
2. QRS width and amplitude calibration;
3. PCC/MAE/RMSE for morphology;
4. downstream disease sensitivity/specificity once labels are available.

The absolute R-peak timing and PTT metrics remain recorded as auxiliary
pairwise-reconstruction diagnostics. They are needed when the generated ECG is
used as a time-aligned surrogate of the paired PPG, but they are not evidence
that an ECG is clinically valid by themselves.

## Current v0.2 results

The following values are subject-level mean +/- standard deviation. Train is
in-sample, validation is checkpoint-selection data, and test is unseen-subject
evaluation.

| Split | RMSE | PCC | HR error (bpm) | R-peak F1 | QRS width error (ms) |
|---|---:|---:|---:|---:|---:|
| Train (22 subjects) | 0.769 +/- 0.148 | 0.567 +/- 0.112 | 7.76 +/- 7.40 | 0.893 +/- 0.129 | 16.12 +/- 9.72 |
| Validation (5 subjects) | 1.064 +/- 0.170 | 0.363 +/- 0.093 | 1.74 +/- 1.51 | 0.958 +/- 0.026 | 14.43 +/- 8.45 |
| Test (5 unseen subjects) | 0.843 +/- 0.158 | 0.349 +/- 0.106 | 10.38 +/- 7.90 | 0.922 +/- 0.047 | 15.53 +/- 5.40 |

The validation RMSE is higher than the test RMSE for this small five-subject
sample; that is sampling variation, not evidence of leakage. The important
generalization signal is the train-to-test PCC drop (0.567 to 0.349), which
shows that the representation still has a subject/domain gap. GRL’s final
subject discriminator accuracy is about 0.021, close to the 22-class random
level 0.0455; this supports subject-invariance of the tested content code but
does not prove causality.

## v0.2 versus the project baselines

| Method | Test RMSE | Test PCC | Test HR error | Test R-peak F1 | Test QRS width error |
|---|---:|---:|---:|---:|---:|
| v0.1 deterministic baseline | 0.857 | 0.252 | 8.96 | 0.904 | 33.65 |
| **v0.2 VAE + Flow + GRL + V-REx** | **0.843** | **0.349** | 10.38 | **0.922** | 15.53 |
| v0.52 multi-band decoder | 0.855 | 0.314 | **8.44** | 0.892 | **8.85** |
| v0.61 VAE latent + multi-band decoder | 0.870 | 0.342 | 9.12 | **0.927** | 11.85 |

The result is a genuine trade-off rather than a universal win:

- v0.2 improves global shape correlation, R-peak F1, and QRS width over v0.1;
- v0.52 gives the best HR and QRS-width calibration in the current set;
- v0.61 combines the v0.2 latent path with the multi-band decoder and gives the
  strongest current rhythm-transfer result, while v0.2 remains the cleanest
  disentanglement/representation baseline.

## Record-wise leakage audit

The separate `configs/exp_recordwise_20ep.yaml` run intentionally permits the
same subject to occur in different splits. It is an upper-bound audit only.
Because its metadata contains overlapping subjects, its test score must not be
called out-of-subject generalization.

| Split | Subjects present | RMSE | PCC | HR error (bpm) | R-peak F1 | QRS width error (ms) |
|---|---:|---:|---:|---:|---:|---:|
| Record-wise train | 32 | 0.723 | 0.616 | 5.90 | 0.879 | 15.70 |
| Record-wise validation | 24 | 0.880 | 0.353 | 6.30 | 0.899 | 17.11 |
| Record-wise test | 25 | 0.870 | 0.384 | 9.30 | 0.899 | 17.00 |

The contrast with the strict subject-wise v0.1 test PCC (`0.252`) quantifies
why a window- or record-wise split is unsuitable for the main claim. The
record-wise result is retained in
`runs/senssmarttech_recordwise_baseline_20ep_seed42/` and is excluded from the
main experiment matrix.

## Comparison with published reports

Published numbers below are reported only as protocol references. They are not
an apples-to-apples leaderboard because most papers use one PPG channel to one
ECG lead, beat-level or cycle-aligned windows, different sampling rates and
normalization, and often subject-specific or subject-mixed splits. The protocol
tag is more important than the raw number.

### Literature credibility audit

The audit below separates source credibility from experimental comparability.
“Verified” means that the venue/DOI and the reported protocol were checked
against the paper; it does not mean that every number was independently
reproduced. A weak split or a missing implementation is a reproducibility risk,
not evidence of fabrication.

| Reference | Publication / DOI | Data and mapping | Split and release evidence | Audit classification | Main limitation |
|---|---|---|---|---|---|
| RDDM (Shome et al., 2024) | AAAI 2024; [10.1609/aaai.v38i13.29422](https://doi.org/10.1609/aaai.v38i13.29422) | WESAD, MIMIC-AFib, CAPNO, BIDMC, DALIA; 1 PPG -> Lead-II ECG; 4 s at 128 Hz | 80% subjects train / 20% subjects cross-subject evaluation; [official code and checkpoints](https://github.com/DebadityaQU/RDDM) | **Verified, strongest public reference** | 4 x A100, batch 512, 500 epochs; much larger training budget than this project; RMSE depends on subject-specific z-score and `[-1,1]` scaling |
| CardioGAN (Sarkar & Etemad, 2021) | AAAI 2021; [10.1609/aaai.v35i1.16126](https://doi.org/10.1609/aaai.v35i1.16126) | BIDMC, CAPNO, DALIA, WESAD; 1 PPG -> Lead-II ECG; 4 s at 128 Hz | 80% subjects train / 20% subjects test; the old code link was not available at audit time | **Verified protocol, limited reproducibility** | One lead and one channel; reported aggregate RMSE `0.364`, PRD `8.356`, FD `0.694`, HR MAE `4.77 bpm`; metrics are not on the project's scale |
| CLEP-GAN (Li et al., 2025) | BMC Bioinformatics 2025; [10.1186/s12859-025-06276-0](https://doi.org/10.1186/s12859-025-06276-0) | BIDMC and CapnoBase; 1 PPG -> 1 ECG; 512-point windows | Code on [GitHub](https://github.com/Mathematics-Analytics-Data-Science-Lab/CLEP-GAN) and [Zenodo archive](https://doi.org/10.5281/zenodo.16540236); 15% randomly selected pairs used for test after selecting 34 low-noise pairs | **Verified source, protocol weak for OOD claims** | Pair-wise random split and multiple records per patient can place the same patient in both sets; low-noise selection can inflate results; reported RMSE `0.37` (BIDMC) and `0.33` (CapnoBase) |
| Zhu et al. (2021), “Learning Your Heart Actions From Pulse” | IEEE IoT Journal 2021; [10.1109/JIOT.2021.3097946](https://doi.org/10.1109/JIOT.2021.3097946) | CapnoBase TBME-RR; 1 PPG -> 1 ECG; cycle-level DCT | Each session is one subject; the transform is trained on the first 80% of that session and tested on the remaining 20% | **Verified method, subject-dependent** | PCC `0.954` (SR) / `0.985` (R2R) is not unseen-subject evidence; physical peak alignment, time scaling, and per-cycle normalization make the task substantially easier |
| “PPGFlowECG (2025)” | No unambiguous formal paper/DOI found in the audit | Claimed 1 PPG -> Lead-II ECG | No independently checkable venue, code, or weights located | **Unverified; do not use as a benchmark** | Treat the name and all associated numbers as a hypothesis or secondary note until a citable paper is provided |

The most defensible external numerical references for the current work are
therefore **RDDM first** and **CardioGAN second**. CLEP-GAN is useful for code and
implementation ideas but should be tagged as a random-pair protocol. Zhu et al.
is a valid historical baseline for a subject-dependent cycle task, not a
cross-subject leaderboard entry.

| Reference | Channel mapping | Unit / split tag | Reported result | Correct interpretation |
|---|---|---|---|---|
| RDDM (Shome et al., 2024) | 1 -> 1 | 4 s window, 80/20 subject-wise | RMSE `0.21` WESAD, `0.19` CAPNO, `0.25` DALIA, `0.24` BIDMC; HR MAE `4.49` DALIA and `1.40` WESAD | Best currently verified public reference; normalization and compute are not matched |
| CardioGAN (Sarkar & Etemad, 2021) | 1 -> 1 | 4 s window, 80/20 subject-wise | Aggregate RMSE `0.364`; HR MAE `4.77 bpm` | Reliable protocol reference, but no currently accessible official implementation |
| CLEP-GAN (Li et al., 2025) | 1 -> 1 | 512-point window, random pair test | RMSE `0.37` BIDMC, `0.33` CapnoBase | Protocol is weaker than subject-wise evaluation; do not rank as OOD |
| Zhu et al. (2021), DCT/cycle-level reconstruction | 1 -> 1 | cycle-level; first 80% of each session train, last 20% test | PCC `0.954` (SR), `0.985` (R2R) | Subject-dependent and physically aligned; not a direct four-to-four, unseen-subject comparison |
| Tang et al. (2022), subject-based PPG-to-ECG | 1 -> 1 | subject-based | PCC about 0.818; RMSE about 0.083 mV | Useful single-to-single subject-based reference; units and normalization differ |
| PPG2ECGps (2023), subject-specific model | 1 -> 1 | subject-specific / in-subject | PCC about 0.977; RMSE about 0.037 mV | An in-subject upper reference, not cross-subject evidence |
| HA-CNN-BiLSTM (2024) | 1 -> 1 | beat-level, MIMIC-II; split and overlap must be checked | RMSE about 0.031 in its normalized convention | Beat alignment and scale make raw RMSE non-comparable |
| PPGFlowECG (source not verified) | 1 -> 1 (claimed Lead II) | Claimed 10 s window; protocol not independently checked | Reported HR error about 1.80-3.23 bpm in secondary notes | Exclude from quantitative ranking until a formal citable source is found |
| Peak-oriented diffusion (2026) | 1 -> 1 | subject-level, 4 s | RMSE about 0.220; AF F1 about 0.925 | Downstream disease result is informative, but not a waveform leaderboard match |

### Fair conclusion

The current architecture is **reasonable as a strict cross-subject prototype**,
especially on rhythm structure: v0.2 reaches test R-peak F1 `0.922` and
improves four-to-four PCC from `0.252` to `0.349` over v0.1. It is **not yet
state of the art** on normalized waveform correlation or HR error when compared
with single-lead, aligned, subject-specific papers. The single-lead papers are
useful references for where a retrained 1 -> 1 model might go, but the existing
Lead II slices do not establish that result. The strongest claim we can defend
now is architectural validity and a measurable disentanglement/rhythm benefit
under a harder four-PPG-to-four-ECG, 22/5/5 protocol.

For a publishable head-to-head comparison, re-run all methods on one public
dataset with the same single-Lead-II, subject-wise split, window length,
sampling rate, normalization, and R-peak evaluator. Only then should the
numbers be placed in one leaderboard.

## Reproduction commands

```powershell
D:\Anaconda\envs\cuda126_env\python.exe scripts\evaluate.py --run runs\senssmarttech_vae_flow_adv_irm_20ep_seed42 --split train
D:\Anaconda\envs\cuda126_env\python.exe scripts\evaluate.py --run runs\senssmarttech_vae_flow_adv_irm_20ep_seed42 --split val
D:\Anaconda\envs\cuda126_env\python.exe scripts\evaluate.py --run runs\senssmarttech_vae_flow_adv_irm_20ep_seed42 --split test
```
