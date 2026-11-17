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

This disables post-inference output analysis and therefore also disables the
conditional rescue stage. The normal input analysis used for routing remains
unchanged.


## Explicit primary-vocal model comparison

The normal route still selects one primary vocal model according to manifest
routing preferences. To generate alternative vocal candidates from the same
input mix, run:

```bash
mseparate input.wav --compare-vocal-models
```

This opt-in mode runs the eligible, input-compatible primary vocal models
(MelBand RoFormer Kim, BS-RoFormer Anvuew and MDX23C ZFTurbo) sequentially.
It produces separate candidate files for each model; candidates are not
summed, subtracted or treated as interchangeable without review. The extra
runs increase processing time and asset requirements.

The post-inference analysis groups candidates by stem and orders them by
expected-source CLAP rank. This is only a semantic source-alignment
comparison, not an objective separation-quality score. No winner is
automatically selected. This mode exists to collect side-by-side candidates
for listening and future benchmark-based routing decisions.


## Conditional ZFTurbo 53-stem rescue

After deep analysis, mseparate may run one bounded rescue pass when all of
these conditions hold:

1. A successful MVSep Mega 53 specialist produced an `Other` artifact.
2. Its technical validation status is `valid` (warnings do not trigger rescue).
3. At least one mapped source family ranked in the top five on the original
   mix and in the top three on that exact `Other` artifact.
4. The ZFTurbo 53-stem model passed license, runtime, input-compatibility and
   device/VRAM admission gates, and its pinned assets pass SHA-256 verification.

Only the strongest eligible `Other` parent is selected per invocation. The
rescue requests only the native ZFTurbo labels mapped from the strong hints,
up to eight target stems; it does not request all 53 outputs by default.
The rescue is skipped if any gate fails. Use `--no-rescue` to disable it.

The rescue is an exploratory second analysis, not a mathematical residual
decomposition: MVSep `Other` is not assumed to be exactly mixture-minus-target.
Every ZFTurbo output is recorded as a child candidate of the exact MVSep
`Other` candidate in `quarantine.json`. Candidate files remain independent,
pending and reversible. No output is added to, subtracted from, or used to
overwrite another stem. The original mix, MVSep outputs and rescue outputs
remain intact.

The stricter CLAP ranks are conservative routing gates, not calibrated
probabilities or proof that a source was missed. A rescue result is not
automatically promoted and must be auditioned. The 53-stem model remains
excluded from normal planner routing; this conditional path is a separate,
explicitly auditable rescue stage.
