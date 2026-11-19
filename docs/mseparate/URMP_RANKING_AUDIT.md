# URMP instrument-ranking audit baseline

## Scope and provenance

This document records the existing CPU-only URMP diagnostic baseline from workflow run
`44485638265` (CPU real-audio run #101), repository commit
`6bea24b8e8a14d7276fadcd7f0f865ea92f78ac2`.

The run analyzed four selected URMP mixtures with CLAP and exercised the production
planner. It did not run separation inference and does not establish separation quality.
The report contains metadata and rankings only; no audio is committed.

## Measured baseline

- Ground-truth unique instrument labels: 12.
- Ground-truth labels in CLAP Top-12: 8/12 (66.7%).
- Ground-truth labels in CLAP Top-5: 7/12 (58.3%).
- Mixtures with at least one planner task: 4/4.
- Planner tasks: 4 total, each using the ZFTurbo 53-stem output head.

Known low-ranked ground-truth instruments:

| URMP mixture | Ground-truth instrument | CLAP rank | Score |
|---|---:|---:|---:|
| 11_Maria_ob_vc | Oboe | 14 | 7.547385 |
| 28_Fugue_fl_ob_cl_bn | Bassoon | 13 | 7.983323 |
| 37_Rondeau_fl_vn_va_cl | Flute | 15 | 7.665183 |
| 37_Rondeau_fl_vn_va_cl | Clarinet | 24 | 6.293291 |

## Important planner failure mode: global score-gap suppression

The acoustic-target selector currently applies both a global rank limit and a score-gap
limit measured against the single highest-ranked source. In an orchestral mixture, a
broad family label can dominate the specific instrument prompts:

- Track 28: broad `woodwinds` scores 11.403530; flute is rank 2 at 10.378705 and is
  routed, but oboe is rank 5 at 9.511466 (gap 1.892064) and clarinet is rank 7 at
  9.143563 (gap 2.259967), so both fail the 1.20 dB gap gate.
- Track 37: violin scores 10.094976; oboe is rank 8 at 8.770536 (gap 1.324440), just
  outside the 1.20 dB gate, despite being a ground-truth instrument.

This demonstrates that low rank is not the only issue: a ground-truth instrument can
rank near the top yet be rejected because a broad family prompt has a much higher score.

## False-positive controls

The same baseline also routed non-ground-truth labels:

- Track 12 routes viola although the named ground truth is two violins and cello.
- Track 37 routes broad brass and saxophone although the named ground truth is violin,
  viola, flute and clarinet.

Do not solve missed instruments by globally widening thresholds without testing these
false-positive cases. CLAP scores are relative semantic similarities, not calibrated
probabilities.

## Proposed regression-test matrix

1. **Known-positive recall:** preserve every ground-truth instrument in the 12-label
   URMP subset, including oboe, bassoon, flute and clarinet.
2. **Broad-family competition:** a high `woodwinds` score must not automatically
   suppress a supported specific oboe, clarinet or bassoon candidate.
3. **False-positive restraint:** broad `brass`, saxophone and viola predictions in
   tracks where they are absent must not become automatic routes without independent
   corroboration.
4. **Prompt robustness:** evaluate multiple acoustically distinct text templates per
   instrument (register, articulation, solo/ensemble context) and aggregate normalized
   evidence rather than trusting one wording.
5. **Temporal robustness:** retain evidence for an instrument appearing in only a
   subset of windows; report window coverage and score dispersion instead of only the
   full-track arithmetic mean.
6. **Synthetic/acoustic ambiguity:** compare real-instrument prompts against their
   synthetic counterparts; synthetic timbre alone must neither prove acoustic
   presence nor veto a separately supported real instrument.
7. **Genre independence:** genre predictions must remain contextual and must not be
   mixed into source-instrument ranks.
8. **Deterministic planner tests:** feed fixed prediction fixtures to the planner and
   assert both expected routes and expected non-routes. Keep real CLAP inference as a
   separate scheduled CPU diagnostic, not a unit-test dependency.

## Acceptance criteria for a future ranking change

- Improve ground-truth Top-12 recall above the 8/12 baseline.
- Recover all four specifically missed labels in this four-mixture diagnostic subset.
- Do not increase false-positive automatic acoustic routes in the negative-control
  cases without a documented precision/recall trade-off.
- Preserve synthetic-specialist routing independently of acoustic specialist routing.
- Keep CLAP outputs explicitly provisional; do not reinterpret similarity scores as
  calibrated probabilities.
- Report per-window evidence and per-instrument route reasons so every decision is
  auditable.

No source-analysis prompts, ranking thresholds, model manifests, weights or inference
behavior are changed by this audit document.
