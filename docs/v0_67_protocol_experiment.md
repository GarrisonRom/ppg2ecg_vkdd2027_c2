# v0.67 Protocol Experiment

## Protocol

- Dataset: SensSmartTech
- Input/output: 4-channel PPG -> 4-lead ECG
- Sampling: 250 Hz, 2000 samples (8 seconds)
- Subject split: 22 train / 5 validation / 5 test
- Seed: 42
- Activity labels A/B are metadata only and are not model inputs.
- Batch sampler: 4 subjects x 4 windows (16 samples)
- Encoder warm-up: 5 epochs; encoder LR `3e-5`, decoder LR `1e-4`
- Early stopping: patience 5, minimum delta `1e-4`

## Runs

| Run | Main change | Best epoch | Validation loss | Band loss active |
|---|---|---:|---:|---|
| `v067-control` | v0.61 VAE + multi-band backbone, protocol controls | 6 | 1.2452 | yes |
| `v067-gated-highskip` | Same backbone + gated full-resolution PPG skip into high branch | 6 | 1.2349 | yes |
| `v067-residual-highskip` | Gated skip rewritten as zero-started residual | 6 | 1.2350 | yes |

The first exploratory gated run (`runs/senssmarttech_v067_gated_highskip_20ep_seed42`)
was not included in the comparison: the trainer only recognized the exact
`multiband_decoder` name, so its configured band loss was silently disabled for
`gated_multiband_decoder`. The trainer branch was corrected before the formal
rerun.

## Metrics (best checkpoint)

Values are subject-mean metrics from the unified evaluator.

| Split / model | RMSE | PCC | HR error (bpm) | R-peak F1 | Peak error (ms) | QRS width error (ms) | RMSSD error (ms) |
|---|---:|---:|---:|---:|---:|---:|---:|
| Train / control | 0.7751 | 0.5724 | 8.0869 | 0.8961 | 21.0173 | 14.1621 | 45.8730 |
| Val / control | 1.0674 | 0.3493 | 1.4929 | 0.9597 | 23.7382 | 12.6267 | 33.8750 |
| Test / control | 0.8549 | 0.3311 | 8.5843 | 0.9257 | 24.7506 | 13.5142 | 55.3671 |
| Train / gated high-skip | 0.7799 | 0.5591 | 8.1162 | 0.9038 | 20.4671 | 12.6707 | 45.7131 |
| Val / gated high-skip | 1.0587 | 0.3564 | 1.6625 | 0.9662 | 21.8520 | 10.1653 | 30.9679 |
| Test / gated high-skip | 0.8417 | 0.3400 | 9.8992 | 0.9245 | 24.5933 | 11.3606 | 53.5079 |

## Interpretation

The corrected high-skip version improves test RMSE, PCC, QRS width, peak timing,
and RMSSD modestly. Its R-peak F1 is effectively unchanged, while HR error is
slightly worse. The train-test RMSE gap decreases from 10.3% to 7.9%, and the
absolute PCC gap decreases from 0.2413 to 0.2191. This is a small but coherent
generalization benefit, not evidence that the model is clinically usable.

The skip should therefore remain a candidate component, but not be promoted as
the final architecture yet. The next controlled change should preserve the
baseline path with a zero-initialized residual gate or a smaller skip gain,
then re-check HR/R-peak metrics. A signal GAN or larger latent should wait until
that stability check is complete.

## Residual-gate follow-up

The residual form was implemented as
`features + tanh(alpha) * skip_residual`, with `alpha` initialized to zero. The
best checkpoint learned `alpha = 0.0245`, so the high-resolution path remained a
small correction instead of replacing the stable decoder. Relative to the
formal gated run, test RMSE/PCC/QRS width/RMSSD did not improve and test HR error
rose to `10.04 bpm`; it is therefore a stability control, not a new mainline.

## Artifacts

- Control: `runs/senssmarttech_v067_control_v061_protocol_20ep_seed42`
- Formal gated run: `runs/senssmarttech_v067_gated_highskip_bandloss_20ep_seed42`
- Corrected trainer branch: `src/training/trainer.py`
- Formal gated metrics: `eval_train_best.json`, `eval_val_best.json`, `eval_test_best.json`
- Auxiliary epoch-5 test metrics: `eval_test_epoch5.json`
- Residual high-skip metrics: `runs/senssmarttech_v067_residual_highskip_20ep_seed42/eval_{train,val,test}.json`
