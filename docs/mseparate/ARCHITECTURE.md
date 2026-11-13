# mseparate architecture

Status: foundation slice implemented on feature/mseparate-foundation. Neural inference, source prediction and asset downloads are not enabled yet.

## 1. Compatibility boundary

- `separate` remains the existing MelBand RoFormer workflow. Its arguments, defaults, exit codes, output layout, model selection, and GUI integration must not change as part of the initial `mseparate` work.
- `mseparate` is a separate CLI entry point and orchestration path.
- Existing upstream inference remains behind an adapter. New model families must not become imports required by the legacy command.
- The AppImage command shell must add an explicit `mseparate` allow-listed command. It must not gain arbitrary shell execution.

## 2. High-level pipeline

1. Validate input and output paths; probe audio metadata.
2. Decode/read audio and run non-neural signal analysis.
3. Run optional source/semantic analysis providers. Every prediction carries confidence, provenance, and model/provider identity.
4. Resolve model registry entries and verify local checkpoint/configuration assets.
5. Build and validate a complete execution plan before neural inference starts.
6. Execute the plan through one global sequential inference manager.
7. Validate every produced stem and compare refinements against the preserved parent stem.
8. Persist outputs, an execution report, and diagnostic events.

Analysis and asset checks may overlap where safe. Neural inference for a single job is globally sequential: no two model inference calls and no two large model residency windows may overlap.

## 3. Proposed package boundaries

The names below describe responsibilities; exact filenames may be adjusted to fit the current package layout after implementation inspection.

- `mseparate.cli`: independent parser and command entry point.
- `mseparate.audio`: decoding, metadata, channel/sample-rate normalization, signal features.
- `mseparate.analysis`: source likelihoods, structure, BPM/style hints, uncertainty-aware analysis.
- `mseparate.registry`: model manifests, schema validation, registry discovery and filtering.
- `mseparate.assets`: download, resume, SHA-256 verification, atomic installation, local cache.
- `mseparate.planner`: candidate scoring, dependency graph, depth/branch limits, VRAM and availability constraints.
- `mseparate.backends.base`: backend protocol and common result types.
- `mseparate.backends.melband`, `msst`, `pymss`: optional adapters, loaded lazily and isolated where dependency conflicts require it.
- `mseparate.execution`: global sequential model lifecycle and plan executor.\n- `mseparate.quality`: bounded-memory integrity, silence, clipping and alignment checks; no reference-free SDR claims.
- `mseparate.quality`: stem integrity, leakage/artifact heuristics, refinement acceptance.
- `mseparate.reporting`: human-readable terminal events and machine-readable report.
- `mseparate.storage`: job workspace, intermediate stems and final output persistence.

The legacy package must not import optional backend packages at module import time.

## 4. Model manifest

Each model entry should describe at least:

- stable `model_id`, display name, architecture and upstream/source URL;
- backend adapter and backend-specific configuration;
- checkpoint and config asset URLs, expected SHA-256 and expected byte size when known;
- license identifier, license URL, redistribution/use notes and verification status;
- native output stems, derived/residual outputs, meaningful target stems, dummy/placeholder stems;
- supported refinement targets and input/output constraints;
- sample rate, channel policy, segment/overlap requirements and inference parameters;
- measured VRAM and runtime observations, with hardware/software provenance;
- availability and compatibility status, including reasons for rejection.

Unknown values remain unknown. Derived/residual stems (for example mixture-minus-vocals instrumental output) must be explicitly distinguished from native model outputs. A model is not eligible for automatic use until its source, license, required files, and output semantics have been reviewed. Dummy outputs must never be advertised as independent source separation.

## 5. Asset store

Default root: `~/.local/share/mseparate/`

Suggested layout:

- `models/<model_id>/`: verified checkpoint and config files;
- `registry/`: versioned local registry metadata;
- `cache/`: disposable analysis/cache data;
- `jobs/<job_id>/`: intermediate stems and reports, subject to retention policy.

Downloads use a temporary `.part` file, bounded timeouts/retries, optional HTTP Range resume, checksum verification, and atomic rename. A valid local asset is reused without downloading again. A corrupt asset is quarantined or marked invalid; it is not silently trusted. Network failure for one model marks that model unavailable for the current plan and must not abort unrelated model checks. Models are never automatically deleted.

## 6. Execution plan and sequential invariant

The planner creates a finite DAG of source-separation and refinement tasks. The DAG is a logical dependency graph, not a concurrency schedule.

Before execution, validate:
- every task's input stem exists or is produced by a predecessor;
- every selected model and all required assets are available and verified;
- model/backend compatibility and resource estimates are acceptable;
- no cycles exist;
- maximum depth, task count and refinement count are respected;
- every branch has a preservation/fallback path.

The executor runs one task at a time. For each task it loads one model, runs inference, persists results, releases references to model/tensor/audio buffers, requests backend cleanup, checks resource state, and only then loads the next model. `torch.cuda.empty_cache()` is best-effort cache release and is not treated as a substitute for dropping live tensor/model references.

A failed task does not erase its input. If no compatible alternative exists, preserve the parent stem and continue independent planned tasks. A refinement replaces a parent only when validation supports accepting it; otherwise retain the original.

## 7. Backend contract

A backend adapter provides capability discovery, compatibility checks, asset requirements, model load, inference, result normalization, unload/cleanup, and diagnostics. The orchestration layer must not depend on MSST/PyMSS/MelBand-specific classes.

Where backend dependency constraints conflict with the main PyTorch environment, use a separately managed environment or subprocess protocol. Do not upgrade/downgrade the existing application environment merely to accommodate a candidate backend.

## 8. Analysis and quality semantics

Analysis outputs are estimates, not ground truth. Each detected source includes a calibrated/declared confidence where available, evidence, and an explicit unknown/uncertain state.

Without reference stems, quality checks can detect invalid duration/alignment, silence, clipping, channel/sample-rate mismatch, extreme spectral anomalies and possible leakage/artifacts. They must not report SDR/SI-SDR as objective measured quality without a reference target.

## 9. Observability

Every stage emits structured events with timestamp, job ID, stage, status, elapsed time, model ID, input/output stem IDs, resource observations and error/fallback reason as applicable. The terminal renderer presents detailed progress; the report writer persists the same event stream in a machine-readable format.

## 10. Non-goals for the first implementation slice

- No claim that all listed architectures are supported until an adapter and a verified manifest entry exist.
- No automatic download of unreviewed or unlicensed weights.
- No parallel neural inference.
- No destructive overwrite of the only copy of an input stem.
- No changes to legacy `separate` behavior.
