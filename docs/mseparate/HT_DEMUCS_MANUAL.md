# Explicit HT-Demucs 6-stem execution

HT-Demucs 6-stem (ONNX) is excluded from the normal `mseparate` automatic
planner and cannot be selected through `--prefer-model`. The model is
available only through the dedicated terminal command:

```bash
mseparate-demucs /path/to/input.wav
```

The default device is strict CUDA. The command does not silently fall back to
CPU if CUDA is unavailable. CPU execution must be explicitly requested:

```bash
mseparate-demucs /path/to/input.wav --device cpu
```

Other options:

```bash
mseparate-demucs input.wav --output-dir ./demucs-stems
mseparate-demucs input.wav --models-dir ~/.cache/mseparate-models
mseparate-demucs input.wav --no-download
```

The command creates six outputs: drums, bass, other, vocals, guitar and piano.
Output files are written below the selected output directory in the
`htdemucs-6stem-manual` task folder. The normal MSeparate planner does not
schedule this model, even when a source analysis suggests guitar or piano.

## Why it is opt-in

On the user's RTX 3060 Ti test, after warm-up a 7.8-second chunk took about
363.7 seconds (RTF approximately 63.35). For a 275.6-second input, the measured
steady-state estimate was about 17,455 seconds (roughly 4 hours 50 minutes),
excluding decode, overlap-add and file writing. This is a measured estimate
from that test environment, not a universal runtime guarantee.

## Candidate quarantine

The quarantine ledger stores candidate path, model, stem, parent lineage,
analysis metadata, current state and append-only decision history. Its states
are `pending`, `accepted`, `quarantined`, `rejected` and `superseded`.
A quarantined candidate is held for review; this is not proof of a bad
separation. State transitions are reversible for re-evaluation. The ledger
does not move, delete or overwrite audio files.

Automatic rescue scheduling and perceptual promotion are not enabled by this
foundation slice. They require comparative candidate evaluation; technical
integrity checks alone cannot reliably rank musical separation quality
without reference stems.
