# Figure Style Brief

- **Domain / audience:** PPG-to-ECG machine-learning experiment; intended for the researcher and later technical review.
- **Profile:** `technical-neutral`, selected because no publication venue or reference figure was specified.
- **Intake status:** explicit immediate draft; no venue-specific constraints were supplied.
- **Tone:** compact technical diagnostic.
- **Language / medium:** English labels for portability; white background; PDF, 300-DPI PNG, and SVG exports.
- **Evidence rules:** source values come from `training_history.json`; no smoothing or interpolation is applied; the random subject-loss reference is `log(22)` and the random accuracy reference is `1/22`; A/B waveform panels use the existing deterministic group-median-MSE selection.
- **Interpretation limit:** discriminator saturation is reported as a representation diagnostic, not as proof of causal disentanglement or improved clinical utility.

## Lead-II waveform figures

- **Figure goal:** inspect one ECG lead at a time, with Lead II used as the
  rhythm/QRS reference for train-versus-test qualitative comparison.
- **Rendering:** one lead per figure, separate rest (B) and after-activity (A)
  rows, full-window waveform plus a data-derived QRS zoom, 600-DPI PNG and
  matching PDF/SVG.
- **Encoding:** true ECG is blue solid; generated ECG is orange dashed; no
  smoothing, rescaling, or selective removal of difficult samples.
