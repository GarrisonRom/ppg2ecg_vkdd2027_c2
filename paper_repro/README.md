# Paper Protocol Comparison

The single entry point is `reproduce_compare.py`. It prepares a SensSmartTech
adaptation of the common CardioGAN/RDDM signal protocol and evaluates both
methods with the project's shared metrics.

Protocol:

- one PPG channel (`carotid_880nm`) -> one ECG lead (`II`);
- 128 Hz, 4-second windows (512 samples), 2-second stride;
- per-recording min-max normalization to `[-1, 1]`;
- subject-wise 80/20 split, seed 42;
- CardioGAN and RDDM use the same windows and evaluator.

This is not an exact reproduction of the original datasets. It is the closest
paper-protocol adaptation that can be run on the local SensSmartTech data.

Run from the project root:

```powershell
D:\Anaconda\envs\cuda126_env\python.exe paper_repro\reproduce_compare.py --method both --epochs 20
```

Outputs are written under `paper_repro/runs/`:

- `paper_protocol_data.npz` and `protocol.json`;
- one directory per method with final checkpoint, train/test predictions,
  per-split metrics, and training history;
- `comparison.json`, `comparison_train.csv`, `comparison_test.csv`, and
  `comparison_all.csv` for the unified in-sample and out-of-subject comparison.

The interpretation against the project's v0.x runs is recorded in
`docs/paper_repro_vs_ours_20ep.md`; it keeps the 1->1 paper adaptation separate
from the 4->4 main protocol.

The original papers train longer and use their own public datasets. The local
20-epoch run is a protocol sanity check, not their published leaderboard.

## Recent paper mechanisms

`reproduce_recent.py` adds three independently recorded adaptations on the
same cached 1 -> 1 protocol:

```powershell
D:\Anaconda\envs\cuda126_env\python.exe paper_repro\reproduce_recent.py --method all --epochs 20 --ncritic 1
```

- QRS-TransAttn (Chiu et al., IEEE Sensors Journal 2020): temporal/channel
  attention CNN plus target-derived QRS-weighted reconstruction.
- P2E-WGAN (Vo et al., ACM SAC 2021): paired U-Net generator, conditional
  WGAN-GP critic, and the paper's large sample reconstruction term.
- Li et al. lightweight network (IEEE TAI 2024): compact multi-kernel,
  attention, and residual reconstruction implementation based on the public
  reproducibility repository. This row is explicitly marked as an independent
  implementation because no author code was located.

The run directory is `runs/senssmarttech_recent_1to1_128hz_seed42/`. It
contains per-method checkpoints, train/test predictions, JSON metrics,
training histories, compact qualitative plots, and comparison CSV files.
The paper list, source links, protocol boundaries, and follow-up candidates are
recorded in `docs/paper_reproduction_candidates.md` and the measured results
in `docs/paper_reproductions_recent_20ep.md`.

For a paper-style P2E-WGAN update ratio, rerun that method with
`--ncritic 3`; the archived comparison intentionally uses one critic update per
generator update so all three methods receive a comparable short budget.
