# CPU real-audio routing evaluation

This experiment exercises the production MSeparate analysis and planner on a
small acoustic URMP reference subset and the 20-track BabySlakh debug subset,
then runs bounded CPU inference on up to four BabySlakh excerpts.

## Dataset and scope

- Dataset: [BabySlakh, Zenodo record 4603844](https://zenodo.org/records/4603844).
- The archive contains the first 20 Slakh2100 tracks, rendered from MIDI with
  sample-based virtual instruments. It is a set of musical arrangements, not a
  collection of live studio recordings.
- The workflow downloads the archive at runtime and verifies its published MD5.
  Audio, extracted stems, model checkpoints and inference outputs are not
  committed to this repository or uploaded as workflow artifacts.
- Slakh attribution: Ethan Manilow, Gordon Wichern, Prem Seetharaman and
  Jonathan Le Roux, “Cutting Music Source Separation Some Slakh,” WASPAA 2019.

## Acoustic reference subset: URMP

The workflow also downloads a fixed four-piece subset of URMP directly to the
ephemeral GitHub Actions runner:

- `11_Maria_ob_vc`: oboe + cello
- `12_Spring_vn_vn_vc`: two violins + cello
- `28_Fugue_fl_ob_cl_bn`: flute + oboe + clarinet + bassoon
- `37_Rondeau_fl_vn_va_cl`: flute + violin + viola + clarinet

Only four mixtures and their 13 isolated instrument recordings are fetched.
The download script verifies the SHA-256 of all four mixtures. It does not
download videos or the full URMP archive. The runner inspects WAV sample rate,
channel count, frame count, duration and subtype, then runs the production
analysis and CPU planner against the four mixtures using the known instrument
lists as reference labels.

The URMP files remain in the runner's temporary directory. They are not
committed to Git, Git LFS, or workflow artifacts. Only metadata and routing
reports are uploaded. The workflow uses the dataset for evaluation; users
must continue to observe the dataset provider's terms and attribution
requirements.

This stage does not yet run specialist separation inference or calculate
SDR/SI-SDR. The isolated AuSep recordings provide a path toward reference-based
evaluation, but temporal alignment, sample-rate consistency, mixture/source
level relationships and stem-to-output mapping must be validated before
publishing those metrics.

## Stages

### A — source and style analysis

- Run bounded-memory signal analysis on each mixture.
- Run LAION-CLAP source/style analysis in one CPU model session across the
  collection.
- Preserve ranked source and genre similarities. These are relative semantic
  similarities, not calibrated probabilities.

### B — intelligent routing

- Pass the real analysis reports to the production build_plan() function.
- Apply the production CPU eligibility filter unchanged.
- Record style profile, evidence, selected models, requested output stems,
  routing rationale and CPU-excluded models.
- HT-Demucs remains excluded from normal routing.
- ZFTurbo 53-stem remains GPU/VRAM-gated; no CPU fallback is introduced.

### C — bounded real-model inference

- Select up to four examples across electronic, orchestral, mixed and ambiguous
  evidence classes where the dataset actually provides them.
- Crop a deterministic 12-second center excerpt from each selected mixture.
- Invoke the production mseparate --execute --device cpu --allow-resample
  command, including its post-inference integrity checks and CLAP candidate
  analysis. Conditional rescue is evaluated normally; GPU-gated rescue is
  expected to remain unavailable on a CPU runner.
- Keep every generated output local to the ephemeral runner. Upload only JSON
  reports and concise console logs; never upload audio.

## Interpretation

This is an integration/routing experiment, not a proof of perceptual
superiority. BabySlakh is small and MIDI-rendered, has no reliable vocal/live
recording coverage, and may not contain examples of every desired style family.
The report records actual coverage rather than pretending all four categories
were present.

The first iteration does not calculate SDR/SI-SDR: reference-source grouping
and alignment between Slakh's individual instrument tracks and MSeparate's
native stem labels still need a separately validated mapping. No CLAP score,
routing score, RMS check or integrity warning is presented as a separation
quality metric.

A successful run demonstrates that the production analysis, planner, CPU
specialist inference, output integrity checks and candidate ledger operate on
actual musical mixtures. It does not validate CUDA performance or establish
that the chosen routing sounds better than a baseline.

## Running locally

Install the same CPU dependencies as the workflow, then run:

~~~bash
python scripts/evaluate_mseparate_babyslakh.py \
  --work-dir ./mseparate-evaluation \
  --max-tracks 20 \
  --max-cases 4 \
  --excerpt-seconds 12 \
  --inference-timeout 600
~~~

The machine needs enough disk space for the dataset, Hugging Face CLAP cache,
selected checkpoints and temporary outputs. A CPU-only run can take hours.
