# Figure Style Brief

- Domain: PPG-to-ECG physiological signal reconstruction.
- Audience: project experiments and future manuscript review.
- Profile: `technical-neutral`, selected because the user requested immediate
  technical comparison figures and did not specify a venue.
- Intake status: explicit immediate draft; no journal, template, language, or
  accessibility constraint was supplied.
- Tone: compact technical report.
- Language: English labels for journal portability; activity definitions are
  shown explicitly as `A = after activity` and `B = before activity/rest`.
- Palette: colorblind-aware blue for measured/true ECG and orange for generated
  ECG, with solid versus dashed line styles as redundant encoding.
- Evidence rules: use the saved `best.pth` checkpoint, preserve the cached
  normalized signals, show one median-error representative window per split and
  activity state, and do not smooth or alter the plotted waveforms.
- Limitation: the representative-window panels are qualitative examples, not
  population summaries; quantitative results remain in `eval_test.json`.
