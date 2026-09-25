# melband-roformer (Debian package)

Native `.deb` for Debian 13 (Trixie), amd64, that installs a local CLI/GUI
wrapper around the upstream **[melband-roformer-infer](https://github.com/openmirlab/melband-roformer-infer)**
package (MIT license, PyPI, Python API `mel_band_roformer`) to separate vocals
from instrumental/music using **Mel-Band RoFormer**, with NVIDIA CUDA
acceleration on GPUs such as the RTX 3060 Ti, and CPU fallback.

This wrapper does **not** reimplement Mel-Band RoFormer. It is a thin,
auditable launcher + packaging layer around the real upstream project, its
real CLI flags (`--input_folder`, `--store_dir`, `--config_path`,
`--model_path`, `--model`, `--models_dir`), and its real model registry
(`melband-roformer-download`).

See `README.debian.md` for packaging-specific details (venv layout, build
process, why the package is large, what is/isn't bundled).

---

## 1. System Requirements

**OS / architecture**
- Debian 13 (Trixie), `amd64` only. The `.deb` is built and tested against
  this exact release/arch combination; other distros/architectures are
  not supported out of the box.
- Python >= 3.11 (built and tested against Trixie's system Python 3.13).

**Required (Depends)**
- `ffmpeg`

**Recommended, GUI only (Recommends)**
- `gir1.2-gtk-4.0`, `python3-gi`, `libgl1`, `libglib2.0-0` — needed only
  for `melband-roformer-gui`; the CLI works without them.

**Build-time only (Build-Depends, for rebuilding the package)**
- `debhelper-compat (= 13)`, `python3-venv`, `python3-pip`,
  `python3-pytest`, `ca-certificates`

**Python environment (bundled in the package's own venv — no system pip
required)**
- `torch==2.11.0` (CUDA 12.6 / `cu126` build)
- `torchaudio==2.11.0`
- `melband-roformer-infer==0.1.5`
- `soundfile>=0.12,<0.13`

**GPU (optional, for acceleration)**
- NVIDIA GPU, Ampere (`sm_86`) or newer recommended (target hardware:
  RTX 3060 Ti).
- NVIDIA driver **>= 560.28.03** — hard floor for the `cu126` PyTorch
  build; below this, CUDA will not initialize.
- The package does **not** install or manage the NVIDIA driver or CUDA
  toolkit — install these yourself first.
- No GPU/driver present → automatic CPU fallback (slower, but works).
- Indicative VRAM usage on an RTX 3060 Ti: ~7.4 GiB free out of 8 GiB
  (see `--info` output in §7).

**Disk**
- The `.deb` itself is several GB (bundled venv + CUDA PyTorch runtime).
- Model weights are downloaded separately on first use: ~913 MB
  (`MelBandRoformer.ckpt`, from Hugging Face), with explicit user
  confirmation and sha256 verification.
- Cache location: `~/.cache/melband-roformer-infer/<model-slug>/`.

**Network**
- Not required for `apt install` (everything needed is already in the
  package).
- Required once, to download the model weights (Hugging Face), with
  explicit user consent.
- Inference itself is fully local — no audio is ever sent anywhere.

---

## 2. Install

```bash
sudo apt install ./melband-roformer_0.1.0_amd64.deb
```

No network access is required for `apt install` itself — the Python venv
(including PyTorch) is baked into the package at *build* time. The
**model weights** (~913 MB) are downloaded separately, on demand, at first
use (see §4), because Debian packages should not ship near-gigabyte binary
blobs that most users may not even need immediately, and because you should
explicitly consent to that download.

## 3. Remove

```bash
sudo apt remove melband-roformer
```

This removes the application and its bundled venv. It does **not** delete
downloaded model weights in `~/.cache/melband-roformer-infer/` (per-user
cache, outside package management) — remove that directory yourself if
wanted.

## 4. Basic usage

```bash
melband-roformer input.wav
melband-roformer input.wav --output-dir ./output
melband-roformer input.wav --device cuda
melband-roformer input.wav --device cpu
melband-roformer input.wav --model melband-roformer-kim-vocals
melband-roformer ./input/*.wav --output-dir ./output   # batch
```

Output: `<name>_vocals.wav` and `<name>_instrumental.wav` next to (or in
`--output-dir`), same sample rate as the input where the model allows it.

## 5. Model download

```bash
melband-roformer --list-models
melband-roformer --download-model                     # fetches the default: melband-roformer-kim-vocals
melband-roformer --download-model --model <slug>
```

On first real separation run, if the model isn't cached yet, you'll be
prompted:

```
Model not found: melband-roformer-kim-vocals
Download MelBand RoFormer Kim (~913 MB) from Hugging Face
(huggingface.co/KimberleyJSN/melbandroformer)? [y/N]
```

Nothing is downloaded silently. The checksum is verified against upstream's
recorded sha256 (see `mel_band_roformer/data/checksums.json`) before the
file is used; a mismatch deletes the file and refuses to run inference with
it.

Cache location: `~/.cache/melband-roformer-infer/<model-slug>/` (upstream's
own default — this wrapper does not relocate it, so it composes with any
other tool using the same upstream package). Override with
`MELBAND_ROFORMER_MODELS_PATH` or `--models-dir`.

## 6. CUDA / GPU

```bash
melband-roformer --info
```

```
Application version: 0.1.0
Python:               3.12.7 (system, via venv)
PyTorch:               2.11.0+cu126
CUDA runtime (torch):  12.6
CUDA available:        yes
GPU:                   NVIDIA GeForce RTX 3060 Ti
GPU memory:             8192 MiB total, 7431 MiB free
Selected device:       cuda:0
Model:                 melband-roformer-kim-vocals (not cached)
```

The wrapper never crashes with a raw CUDA traceback if the GPU is
unavailable — it prints a clear message and falls back to CPU:

```
CUDA is not available. Falling back to CPU.
Run `melband-roformer --info` for details.
```

`--device cuda` forces GPU and *fails loudly* (not silently to CPU) if CUDA
isn't actually available, so you know your GPU setup is broken rather than
silently getting a slow CPU run.

## 7. Diagnostics

```bash
melband-roformer --self-test
```

```
[OK]   Python
[OK]   PyTorch import
[OK]   CUDA available
[OK]   GPU: NVIDIA GeForce RTX 3060 Ti
[OK]   Model present: melband-roformer-kim-vocals
[OK]   Model loads
[OK]   Inference on 2s test tone

Mel-Band RoFormer is ready.
```

Any `[OK]` becomes `[FAIL] <reason>` with an actionable message
instead of a stack trace when something is missing (no GPU, no model, wrong
driver, etc).

## 8. Model weights location

`~/.cache/melband-roformer-infer/<model-slug>/` (upstream default, see §4).

## 9. Output location

Current directory by default, or `--output-dir <path>` / `--store_dir`
(upstream's underlying flag; `--output-dir` is accepted as the friendlier
alias documented in this wrapper's `--help`).

## 10. Known limitations

- **This is a vocal/instrumental *music* separation model, not a
  speech-recognition or dedicated dialogue-isolation model.** Extracting a
  short spoken monologue from the intro of an industrial-music track is
  *not* the model's primary trained use case (its training data is
  music vocals, i.e. singing, not spoken word). It often works
  reasonably on spoken intros because spoken voice and sung vocals share a
  lot of spectral structure, but quality is **not guaranteed** and depends
  heavily on the specific mix (how loud/dry the voice is relative to the
  music bed, whether there's heavy processing/distortion typical of
  industrial music, etc). Always listen to the result; do not assume a
  clean separation. See `tests/` for the explicit music+spoken-voice test
  case and its caveats.
- CPU inference is slow for anything beyond short clips; expect real GPU
  need for full tracks.
- Only 37 of the upstream registry's 89 models currently have live,
  checksummed download URLs as of the 2026-07-12 upstream audit; the rest
  (`--list-models`) may 404. `melband-roformer-kim-vocals` (the default) is
  confirmed live.
- GTK4 GUI (`melband-roformer-gui`) is a thin optional layer; it has not
  been visually tested in this environment (no display server available
  here — see README.debian.md "What was actually verified").
