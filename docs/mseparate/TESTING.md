# mseparate testing strategy

Status: required test matrix; no runtime tests are implied by this document.

## Legacy regression

- Existing `separate --help`, `--version`, `--info`, `--self-test` and model-list/download dispatch.
- Existing GUI invocation path.
- AppImage shell rejects unknown/arbitrary commands and continues to route legacy commands.
- Debian build/install entry points and existing console scripts.

## Audio and analysis

- WAV/FLAC/MP3 decode success and invalid/corrupt inputs.
- Mono/stereo/multichannel metadata and conversion.
- Silence, clipped audio, short clips, long tracks and non-finite samples.
- Stable feature extraction and deterministic report serialization.
- Confidence/provenance and unknown-source handling.

## Registry and assets

- Valid/invalid manifest schema.
- Missing files, incorrect size, incorrect SHA-256 and truncated checkpoints.
- Download success, interrupted transfer, HTTP Range resume, unsupported Range, timeout and retry exhaustion.
- 404/403/5xx for one model while other model checks continue.
- Offline/no-download behavior.
- Verified local checkpoint reuse without network access.
- No automatic deletion of model assets.

## Planning and execution

- Acyclic plan validation, missing dependencies, depth/task limits and unavailable-model fallback.
- Global sequential inference invariant: maximum active inference count equals one.
- No next model load before previous backend unload/cleanup completes.
- Failure preserves original input stem.
- Refinement is accepted only when validation criteria pass.
- VRAM guard and CPU fallback behavior.
- Backend import/load/inference/unload exceptions.

## Quality and outputs

- Stem duration, channel count, sample rate and alignment.
- Empty/silent output, clipping and invalid samples.
- Heuristic artifact/leakage flags are not mislabeled as reference metrics.
- SDR/SI-SDR only with aligned reference stems.
- Intermediate and final output persistence and report integrity.

## Integration

- `mseparate` command entry point.
- AppImage explicit allow-list.
- Debian package installation and runtime data paths.
- Optional backend dependency isolation.
- No changes to legacy `separate` output behavior.

## Real-audio evaluation

Use a documented collection spanning different arrangements and source densities, not only genre labels. Keep source references where licensing permits. Record model versions, checkpoint hashes, hardware/software stack, elapsed time, VRAM observations and evaluation method. Synthetic test tones validate plumbing only; they do not establish separation quality.


## Asset manager test coverage

The CPU-only asset tests use a local HTTP server and cover verified download/reuse, Range resume, servers that ignore Range, checksum mismatch, preservation of corrupt files, offline checks, and failure isolation between assets. They do not download production checkpoints or validate model quality.
