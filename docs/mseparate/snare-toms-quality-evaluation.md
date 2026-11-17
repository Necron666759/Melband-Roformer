# Snare and Toms quality evaluation

## Purpose

The Snare/Toms checkpoint smoke proves that the pinned ZFTurbo 53-stem checkpoint loads and returns finite output arrays. It does **not** measure separation quality. This benchmark measures quality against aligned reference stems and compares the estimates with the unprocessed mixture.

## Dataset

Use the **StemGMD Eval Session** as the first reproducible benchmark subset:

- Download: https://zenodo.org/records/7882857
- File: `StemGMD_eval_session.zip` (approximately 5 GB)
- License: Creative Commons Attribution 4.0 International (CC BY 4.0).
- It contains a manageable evaluation subset of 400 drum mixtures. StemGMD supplies isolated stems for Snare, High Tom, Mid-Low Tom, Floor Tom, Kick, Open/Closed Hi-Hat, Crash and Ride.
- The audio is synthesized from expressive MIDI performances using ten acoustic drum-kit sample libraries. It is useful for controlled acoustic-kit evaluation, but it is **not a substitute for live-microphone recordings**.

Retain the dataset's attribution and paper citation when publishing results. Do not commit dataset audio or generated mixtures/references to this repository.

## Prepare a CSV manifest

Prepare a CSV with these exact columns:

```csv
track_id,mixture,snare,toms
kit01_groove001,mixes/kit01_groove001.wav,stems/kit01_groove001_snare.wav,stems/kit01_groove001_high_tom.wav;stems/kit01_groove001_mid_low_tom.wav;stems/kit01_groove001_floor_tom.wav
```

- `mixture` is the stereo full-kit drum mix supplied by the evaluation subset, or a locally created sum of all nine isolated components if the archive provides only components.
- `snare` points to the isolated Snare reference.
- `toms` may contain one or more semicolon-separated isolated tom references; the runner sums them.
- Paths are relative to the CSV file or absolute.
- Keep the original 44.1 kHz stereo sample alignment. Do not normalize stems independently, crop, pad, or shift them.
- The example row is illustrative only; replace it with paths that exist in the downloaded archive.

## Run

Install the same application runtime used by the project, including `pymss==2.3.5`, `numpy`, and `soundfile`. From the repository root:

```bash
PYTHONPATH=app python scripts/evaluate_mseparate_drum_heads.py \
  --manifest /path/to/stemgmd_eval.csv \
  --output-dir /path/to/snare-toms-results \
  --device cuda \
  --low-vram-profile
```

The 10-second low-VRAM profile is opt-in and checks the project's existing minimum of 7 GiB total and 6 GiB free VRAM. If those requirements are not met, the script stops; it never silently falls back to CPU. Omit `--low-vram-profile` only when the standard model VRAM admission requirements are satisfied. `--max-tracks 4` can be used for a short pilot before a full run.

## Metrics and interpretation

For each track and target, the runner reports:

- **SI-SDR (dB):** scale-invariant signal-to-distortion ratio of the model estimate against the reference.
- **Mixture baseline SI-SDR (dB):** the same metric when the unprocessed mixture is treated as the estimate.
- **SI-SDRi (dB):** improvement over that mixture baseline. Positive values indicate that the model estimate is better than passing the mixture through unchanged.

The JSON and CSV reports include per-track metrics and mean/median/min/max summaries for Snare and Toms separately.

The runner enforces exact sample rate, stereo channel count and frame alignment. It does not silently resample, crop, pad, or time-align audio. The 53-stem outputs are not assumed to be additive, so no artificial residual or mixture-consistency score is calculated.

## Follow-up live-recording validation

StemGMD uses sample-library-rendered acoustic kits. Before making a quality claim about live acoustic recordings, run a separate evaluation on a legally accessible multitrack/live-drum corpus with isolated or close-mic references, and conduct blind listening on a small set of real songs. Keep synthetic-kit and live-recording results separate in the report.

## Dataset reference

A. I. Mezza, R. Giampiccolo, A. Bernardini, and A. Sarti, "Toward Deep Drum Source Separation," *Pattern Recognition Letters*, vol. 183, pp. 86–91, 2024, DOI: 10.1016/j.patrec.2024.04.026.

Dataset: https://doi.org/10.5281/zenodo.7882857
