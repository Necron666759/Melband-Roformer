# Post-inference deep source analysis

A normal `mseparate input.wav` execution now performs a post-inference analysis
of every successfully produced stem unless `--no-deep-analysis` is supplied.

The initial source analysis of the input mix is reused as a baseline. The
post-inference pass loads LAION-CLAP once and evaluates the output files
sequentially, avoiding repeated model initialization for each stem.

## What is recorded

- Relative source-label ranking for each output stem.
- Rank of the expected source family when the stem has a known source mapping.
- Possible off-target source labels for target-specific stems.
- Possible residual-source hints in stems named `other`: a source label must
  rank in the top ten on the original mix and in the top five on the `other`
  output to be reported.
- Existing technical validation warnings from the execution manager.
- Candidate lineage and analysis metadata in
  `mseparate-stems/quarantine.json`.

The JSON report also includes a `deep_output_analysis` object when
`--report` is used.

## Interpretation and safety boundaries

CLAP similarities are relative audio/text similarities, not calibrated
probabilities. A high off-target rank is not proof of leakage. A residual
hint means that a source-like sound may still be present in an `other`
output; it is not proof that the source was missed or that a second model
will improve it.

All analyzed candidates remain in the ledger as `pending`. This pass does
not automatically quarantine, reject, promote or overwrite audio. It does
not subtract stems from the mix. The original input and every model output
remain intact.

A comparison group with more than one candidate is ordered by expected-source
CLAP rank only. That ordering is a semantic alignment hint, not a perceptual
separation-quality ranking. The current normal planner usually schedules one
model per stem, so a single-candidate group explicitly reports that no
model-to-model comparison was possible.

## Disable the additional pass

```bash
mseparate input.wav --no-deep-analysis
```

This disables only post-inference output analysis. The normal input analysis
used for routing remains unchanged.
