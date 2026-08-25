# PPG-to-ECG Paper Reproduction Candidates

This list separates methods that can be reproduced from public code from
methods that are useful references but currently lack a complete public
implementation. It is intentionally conservative: a published number is not
treated as a benchmark result until the data split, normalization, sampling
rate, and peak evaluator are aligned.

## Implemented local adaptations

| Method | Venue and identifier | Public implementation | Why it is useful here | Status |
|---|---|---|---|---|
| QRS-TransAttn | Chiu et al., *Reconstructing QRS Complex From PPG by Transformed Attentional Neural Networks*, IEEE Sensors Journal, 2020, DOI [10.1109/JSEN.2020.3000344](https://doi.org/10.1109/JSEN.2020.3000344) | [james77777778/ppg2ecg-pytorch](https://github.com/james77777778/ppg2ecg-pytorch), MIT | Directly targets QRS reconstruction with temporal/channel attention and an explicit QRS objective. | 20-epoch local adaptation |
| P2E-WGAN | Vo et al., *P2E-WGAN: ECG Waveform Synthesis from PPG with Conditional Wasserstein GANs*, ACM SAC, 2021, DOI [10.1145/3412841.3441979](https://doi.org/10.1145/3412841.3441979) | [khuongav/P2E-WGAN-ecg-ppg-reconstruction](https://github.com/khuongav/P2E-WGAN-ecg-ppg-reconstruction), MIT | Gives an adversarial baseline with conditional critic, U-Net generator, and sample-level waveform supervision. | 20-epoch local adaptation |
| Lightweight PPG2ECG | Li et al., *Inferring Electrocardiography From Optical Sensing Using Lightweight Neural Network*, IEEE TAI, 2024, DOI [10.1109/TAI.2024.3400749](https://doi.org/10.1109/TAI.2024.3400749) | [reproducible-ppg-to-ecg-reconstruction](https://github.com/AnaLovesToCod3/reproducible-ppg-to-ecg-reconstruction), MIT; Zenodo DOI [10.5281/zenodo.21710377](https://doi.org/10.5281/zenodo.21710377) | Recent lightweight temporal/channel attention and multi-kernel design; useful for an efficiency and cross-subject comparison. | Independent implementation, clearly labeled |

The local runs are in `paper_repro/runs/senssmarttech_recent_1to1_128hz_seed42/`.
They are not exact reproductions of the original datasets: all three use the
same SensSmartTech adaptation so that `src.evaluation.metrics.evaluate_all`
can be applied consistently.

## Strong follow-up candidates

| Method | Reference | Reproduction value | Main audit question |
|---|---|---|---|
| HA-CNN-BiLSTM | Ezzat et al., *ECG Signal Reconstruction from PPG Using a Hybrid Attention-Based Deep Learning Network*, EURASIP JASP, 2024, DOI [10.1186/s13634-024-01158-8](https://doi.org/10.1186/s13634-024-01158-8) | Open access, dilated CNN + BiLSTM + attention + scattering-wavelet features; reports RMSE 0.031. | The reported 90/10 split is not clearly subject-wise, so a strict subject holdout is required before trusting the number. |
| Learning Your Heart Actions From Pulse | Zhu et al., IEEE IoT Journal, 2021, DOI [10.1109/JIOT.2021.3097946](https://doi.org/10.1109/JIOT.2021.3097946), preprint DOI [10.1101/815258](https://doi.org/10.1101/815258) | DCT coefficient mapping and a cardiovascular signal model; a useful non-deep frequency-domain baseline. | Whether the reported PCC > 0.92 survives one common subject-wise split and the project's peak evaluator. |
| Cross-Domain Joint Dictionary Learning | IEEE IoT Journal, 2023, DOI [10.1109/JIOT.2022.3231862](https://doi.org/10.1109/JIOT.2022.3231862) | Classical sparse cross-domain representation; tests whether a deep model is actually needed. | Need the authors' data/protocol details; no complete official code was located. |
| Real-Time PPG-to-ECG With On-Device Recalibration | IEEE TIM, 2024, DOI [10.1109/TIM.2024.3450120](https://doi.org/10.1109/TIM.2024.3450120) | Recent device/subject recalibration idea, relevant to deployment and domain shift. | Public implementation and exact data split are currently unavailable. |
| Inception CNN-Transformer | IEEE UR, 2025, DOI [10.1109/UR65550.2025.11078056](https://doi.org/10.1109/UR65550.2025.11078056) | More recent CNN-transformer hybrid; potentially a strong architecture comparison. | No auditable public repository was found at the time of archiving. |
| KANFlow | IEEE IoT Journal, 2026, DOI [10.1109/JIOT.2026.3717960](https://doi.org/10.1109/JIOT.2026.3717960) | Closest to the project's flow-matching direction; shallow KAN flow is a useful future comparison. | No public implementation/checkpoint is currently available. |

## Protocol rule

For every future method, record both the original protocol and the local
adaptation. The local comparison protocol is one PPG channel (`carotid_880nm`)
to Lead II, 128 Hz, 4-second windows, per-recording min-max `[-1, 1]`, and a
subject-wise 80/20 split (seed 42). The main project protocol remains separate:
4 PPG -> 4 ECG, 250 Hz, 8 seconds, subject-wise 22/5/5.
