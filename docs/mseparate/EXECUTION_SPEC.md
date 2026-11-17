# mseparate execution and observability specification

Status: implementation contract for the current feature branch.

## 1. Goals and invariants

This specification defines the execution lifecycle and the evidence retained for every run. It does not change the legacy `separate` command.

Hard invariants:

- The original input is immutable.
- A complete, validated plan is built before neural inference.
- Neural inference is globally sequential: one active model residency/inference window at a time, including separate CLI processes sharing the model store.
- Every output is an independent artifact with parent/model/task provenance.
- A child or Rescue output never silently replaces, adds to, subtracts from, or overwrites its parent.
- A failed task does not erase its input or prevent unrelated independent tasks from being attempted.
- Unverified, disabled, or license-blocked manifests fail closed.
- CLAP similarities are relative audio/text similarities, not calibrated probabilities or perceptual quality scores.
- Technical integrity checks must not be presented as perceptual separation-quality metrics. No reference-free SDR/SI-SDR claims.

## 2. Pipeline stages and required evidence

| Order | Stage | Required report evidence |
|---:|---|---|
| 1 | CLI/run initialization | run ID, UTC start, arguments/options, log destination |
| 2 | Input validation and probe | resolved input path, format, duration, sample rate, channels, frames |
| 3 | Signal analysis | measurements, notes, clipping/silence indicators |
| 4 | Semantic/source analysis | provider/model, source and genre predictions, relative scores, window count, device/fallback |
| 5 | Registry and compatibility | registry path/version, eligible and rejected models, reasons |
| 6 | Planning | deterministic task list, task IDs, input stem IDs, dependencies, target stems, routing score and rationale |
| 7 | VRAM/resource preflight | requested/effective device, available/total VRAM where available, admission profile and reasons |
| 8 | Asset verification | model ID, asset role, path, cache/download status, bytes, SHA-256/size verification outcome |
| 9 | Primary inference | per-task start/end, backend/model, device, input, requested/actual outputs, status, elapsed time, exception |
| 10 | Output validation | duration/rate/channels, finite samples, clipping/silence, cross-stem duplication warnings |
| 11 | Deep output analysis | candidate ID, parent ID, stem, target rank, top source rankings, off-target hints, review flags |
| 12 | Residual/Other gate | parent identity, source hints, mix rank, Other rank, thresholds, selected/rejected targets and reason |
| 13 | Rescue preflight and inference | model/asset verification, VRAM decision, parent artifact, target stems, per-task status and duration |
| 14 | Rescue output analysis | independent child candidates, relative CLAP rankings, review flags, no auto-acceptance |
| 15 | Finalization | summary counts, report path, log path, exit status |

A stage must explicitly report `completed`, `skipped`, `blocked`, `failed`, or `not_started` when applicable. Missing information is represented as null/unknown, not inferred.

## 3. Routing contract

Routing is a deterministic planning heuristic, not a model-quality contest.

1. Reject hard-incompatible models before ranking (format, sample rate, channels, license, runtime and resource requirements).
2. Consider only enabled and verified registry entries.
3. Prefer a manifest-declared specialist for an overlapping target.
4. Apply an explicit operator preference only where it does not violate compatibility or policy.
5. Use manifest planner priority and source salience as routing preferences.
6. Record the exact rationale and all eligible/rejected candidates.
7. Do not infer source presence from genre labels alone.
8. Keep HT-Demucs 6-stem ONNX out of automatic routing; explicit use remains a separate command.
9. Execute planned tasks serially, preserving every successful artifact.

## 4. Other and Rescue contract

`Other` is a model-produced output category. It is not assumed to be a mathematically exact mixture residual.

The current conservative gate is:

- parent output must be technically valid;
- source must rank at or above the configured relevance cutoff in the mix (`mix_rank <= 5`);
- source must rank within the first three results for the parent's `Other` (`other_rank <= 3`);
- at most one parent output is processed per Rescue invocation;
- no more than eight target stems are sent to Rescue;
- selected broad target stems and the source hints that caused selection are both retained in the report.

The target rank measured after Rescue is a separate observation from the parent hint rank used to admit the target. For example, a selected broad `drums` target may rank poorly after analysis; it remains `pending_review` and must not be auto-accepted.

Rescue outputs are independent candidates. No mixture subtraction, summing, destructive replacement, or automatic promotion is allowed.

## 5. Candidate lifecycle

Allowed states:

- `created`: output artifact was created.
- `technical_valid`: file integrity/alignment checks passed.
- `pending_review`: technically valid, but perceptual correctness is not established.
- `accepted`: explicit review or a future validated acceptance gate approves it.
- `rejected`: explicit review or a validated gate rejects it.
- `blocked` / `failed`: execution could not safely produce a candidate.

Every candidate retains candidate ID, task ID, model ID, parent candidate ID, input path, output path, target stem, technical findings, analysis provenance, decision, and review flags. Existing quarantine/ledger records remain append-safe and are never silently rewritten to imply acceptance.

## 6. Logging and report contract

Two complementary artifacts are supported:

- Human-readable terminal log: exact text written to stdout and stderr, mirrored while still displaying in the terminal.
- Machine-readable JSON report: structured analysis, plan, assets, task results, candidates, Rescue decision, and run timeline.

The terminal log is append-mode UTF-8 and includes progress output emitted through Python stdout/stderr. The report is atomic JSON. A report is not a substitute for the raw log: the report is structured and intentionally does not embed every progress-bar refresh or backend diagnostic line.

The run timeline uses UTC ISO-8601 timestamps and monotonic elapsed seconds. Every event contains `stage`, `status`, and optional structured details. Stage duration is measured with a monotonic clock. Wall-clock timestamps are for correlation only.

The top-level report schema is version 2. Version 2 retains the existing analysis/plan/execution fields and adds `observability`, `log_file`, and `run_options`; consumers should ignore unknown fields and branch on `schema_version` when strict validation is required.

No secrets, credentials, authorization headers, or environment dumps may be logged. Paths and command options may contain private filenames; users control where logs are stored and shared.

## 7. CLI contract

- `--report [PATH]`: write the structured JSON report; without PATH uses `<output-dir>/mseparate-report.json`.
- `--log-file [PATH]`: mirror stdout and stderr to a UTF-8 text log; without PATH uses `<output-dir>/mseparate.log`.
- Relative paths are resolved under `--output-dir`.
- Both options may be used together.
- Log capture does not require `--report`.
- Log capture must not alter terminal visibility, model selection, execution order, or inference behavior.

Example:

```bash
mseparate input.wav --execute --report --log-file
```

Explicit paths:

```bash
mseparate input.wav --execute \
  --report reports/session-01.json \
  --log-file logs/session-01.log
```

## 8. Current implementation boundary

The current report schema contains the complete plan, asset outcomes, task results, Deep CLAP candidate summaries, Rescue gate and Rescue execution. The new run timeline and terminal log mirror add run-level observability.

Per-task load/inference/unload timing, full backend-native diagnostic capture, hardware telemetry sampling during chunks, and detailed per-stage progress callbacks remain follow-up work. Do not claim these are present until they are emitted and covered by tests.
