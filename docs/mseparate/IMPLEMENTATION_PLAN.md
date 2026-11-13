# mseparate implementation plan

Status: proposed sequence. Each phase must leave the legacy `separate` path working.

## Phase 0 — baseline and safety

- Record the current main-branch commit and inspect the complete package/build/test layout.
- Establish a feature branch and a clean rollback point.
- Capture baseline CLI help/version/self-test behavior and existing test results.
- Add regression tests for legacy parser and command dispatch before changing integration points.

Exit criteria: a reproducible baseline and a documented list of compatibility surfaces.

## Phase 1 — core contracts (no neural inference)

- Add independent `mseparate` CLI routing without changing `separate`.
- Define typed models for audio metadata, analysis findings, source/stem identities, model manifests, execution tasks/plans, backend results, quality findings and events.
- Add registry schema validation and a small test-only fixture registry.
- Add persistent storage path resolution and report serialization.
- Add analysis-only and plan-only modes.

Exit criteria: CLI can inspect an input, validate fixture manifests and print a deterministic plan without importing optional neural backends.

## Phase 2 — asset manager

- Implement local asset inventory and checksum verification.
- [x] Implement safe downloads to `.part`, bounded retries/timeouts, resume where supported, and atomic finalization.
- [x] Isolate failures per asset/model; continue checking the rest of the registry.
- [x] Add CPU-only tests for offline mode, partial downloads, checksum mismatch, Range handling and reuse of verified assets.
- [x] Cover HTTP 403/404/500 responses and partial-file Range resume in the local-HTTP test suite; CI executes these CPU-only tests.

Exit criteria: asset manager is deterministic and never re-downloads a valid checkpoint unnecessarily.

## Phase 3 — backend adapters

- [x] Define the backend adapter contract and common stem/result types.\n- [x] Implement a strictly sequential execution manager with unload-before-next-load ordering, output validation and per-task failure isolation.\n- [x] Add an isolated MelBand adapter that invokes the repository's existing `melband-roformer` wrapper CLI in a child process; no Torch import is added to the coordinator.\n- [ ] Verify the adapter against the exact pinned upstream package and a real checkpoint on the target Debian/GPU environment before enabling any production manifest.
- Investigate MSST and PyMSS compatibility in isolated environments before enabling either.
- Add capability/compatibility reports and model-specific manifests only after checking source, license, checkpoint/config pairing and output semantics.
- Record real VRAM/runtime measurements on supported hardware; do not invent estimates.

Exit criteria: one backend can run a controlled inference task through the common interface; optional backends fail closed and do not affect the legacy environment.

## Phase 4 — planner and sequential executor

- Implement candidate scoring using source likelihoods, stem targets, model capabilities, availability, resource constraints and prior validation results.
- Generate a bounded acyclic execution plan.
- Implement the global sequential model lifecycle manager.
- Add explicit preservation/fallback rules and refinement acceptance.
- Add invariant tests proving no overlapping inference calls or simultaneous model residency.

Exit criteria: all neural tasks in a job execute strictly one at a time and every failure preserves usable input audio.

## Phase 5 — analysis and quality

- Add deterministic audio measurements first.
- Add optional genre/structure/BPM/source-analysis providers with confidence and provenance.
- Add stem integrity and artifact/leakage heuristics.
- Use reference-based metrics only when reference stems exist.

Exit criteria: reports distinguish measurements, model predictions, heuristics and unknowns.

## Phase 6 — integration and packaging

- Add the `mseparate` console entry point.
- Add explicit AppImage shell allow-list routing.
- Verify Debian packaging, AppImage runtime paths, GUI coexistence and optional dependency isolation.
- Keep new heavyweight dependencies out of the legacy base environment unless strictly necessary.

Exit criteria: installed package exposes both commands; legacy GUI and `separate` remain unchanged.

## Phase 7 — validation and release

- Run unit, integration, packaging and regression tests.
- Test offline and partial-network-failure paths.
- Test representative tracks and reference datasets where legally available.
- Compare outputs with baselines and retain logs/reports.
- Document unsupported models and hardware limits honestly.

Exit criteria: tests and hardware actually exercised are recorded; no unsupported quality claims are made.

## Current status

- Repository access and write permissions confirmed.
- Source-level audit of the CLI, model download wrapper, packaging rules, AppImage shell and current tests completed to the extent available through repository files.
- Feature branch `feature/mseparate-foundation` created from `main`.
- Phase 1 foundation code added: independent `mseparate` entry point, audio metadata/signal measurements, typed contracts, validated local registry, persistent data layout, atomic JSON reports and plan validation.
- Debian launcher/manual page and AppImage restricted-shell routing added. Legacy CLI implementation and dependency pins were not changed.
- CPU-only tests were authored but have not been executed in a local runtime. No build, AppImage, GPU or real-audio inference test has been claimed.
- Phase 2 asset-manager implementation is now present: SHA-256-gated downloads, resumable `.part` files, atomic installation, local reuse, corrupt-file preservation, HTTP(S)-only URLs, per-asset failure isolation and offline verification.
- A lightweight GitHub Actions workflow runs the two `mseparate` CPU-only test modules without downloading PyTorch or model weights. CI passed for the asset-manager implementation and HTTP 404 coverage; the expanded 403/500 cases are queued in the latest workflow run. The initial broader test attempt failed because it included unrelated legacy tests without their heavyweight dependencies.
- Model manifests now distinguish `native_stems` from `derived_stems`; the MelBand adapter declares vocals as native and instrumental as a residual output. The executor rejects tasks requesting stems that the manifest does not declare as native or derived.\n- `app/mseparate/quality.py` adds bounded-memory output checks for decodability, finite samples, silence, clipping, duration, sample rate and channel alignment. It intentionally does not report SDR/SI-SDR without reference stems.\n- Backend foundation added in `app/mseparate/execution.py`; CPU-only fake-backend tests cover maximum single-model residency, unload/resource-check ordering, output validation, independent-task continuation and dependency skipping.\n- A first MelBand adapter is present in `app/mseparate/melband.py`. It calls the existing wrapper with explicit checkpoint/config paths, so inference runs in a child process and Torch/model references are released when that process exits. Its unit tests mock the process and do not claim real inference.\n- The workflow now includes the execution test module. The adapter contract is in place; concrete MelBand wiring remains gated on confirming the API in the pinned `melband-roformer-infer==0.1.5` release (upstream main currently advertises 0.1.6, so its newer session facade must not be assumed available).\n- Next: verify the pinned upstream release API and implement its adapter without changing the legacy wrapper or Torch pins.
