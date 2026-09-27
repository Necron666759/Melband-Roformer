# melband-roformer (Debian package)

`amd64`, built from one source tree into two installable forms: a native
`.deb` (built and released against Debian 13 / Trixie, but compatible
with recent Debian/Ubuntu derivatives — see §1a for exactly what
"compatible" means and the one case where it isn't guaranteed) and a
self-contained AppImage with its own restricted, sandboxed terminal
(§1b). Both wrap the upstream
**[melband-roformer-infer](https://github.com/openmirlab/melband-roformer-infer)**
package (MIT license, PyPI, Python API `mel_band_roformer`) to separate vocals
from instrumental/music using **Mel-Band RoFormer**, with NVIDIA CUDA
acceleration on GPUs such as the RTX 3060 Ti, and CPU fallback. The
`.deb` additionally ships an optional GTK4 GUI (`melband-roformer-gui`);
the AppImage does not, offering its own bundled terminal UI instead (§4).

This wrapper does **not** reimplement Mel-Band RoFormer. It is a thin,
auditable launcher + packaging layer around the real upstream project, its
real CLI flags (`--input_folder`, `--store_dir`, `--config_path`,
`--model_path`, `--model`, `--models_dir`), and its real model registry
(`melband-roformer-download`).

See `README.debian.md` for packaging-specific details (venv layout, build
process, why the package is large, what is/isn't bundled).

---

## 0. System Requirements

This section describes the `.deb` (its `debian/control` is the source of
truth for the table/lists below). The AppImage shares the same PyTorch/
CUDA/driver/model requirements — it bundles the identical pinned
versions — with three differences: it needs no `apt install` step (§1b),
it has no GTK4 GUI dependency because it has no GUI (it bundles its own
restricted terminal instead, §4), and — unlike the `.deb` — it cannot
declare `ffmpeg` as a package dependency, so `ffmpeg` must already be on
your `$PATH` for it to work, exactly as with the `.deb`.

**Quick summary**

| | |
|---|---|
| OS | Debian 13 (Trixie) |
| Architecture | `amd64` / x86-64 |
| Python | >= 3.11; the pre-built Debian 13 package's bundled venv runs on Python 3.13 (Trixie's own system `python3`) |
| Disk space | Several GB required to install (bundled venv, including the CUDA PyTorch runtime, is the majority of it); the pre-built, compressed `.deb` itself is ~2.2 GB — see §1/§2. No installed/unpacked-size figure is tracked separately in this repo's docs. |
| Model | ~913 MB additional, for `melband-roformer-kim-vocals` — downloaded separately on first use, not bundled in the package |
| FFmpeg | Required (`Depends`) |
| NVIDIA GPU | Required for CUDA hardware acceleration; CPU fallback is also supported |
| NVIDIA driver | >= 580.65.06 — hard floor for the bundled `cu130` (CUDA 13.0) PyTorch build. This is a new major-branch floor, not a point bump: CUDA 13.x raised the minimum required driver from the CUDA 12.x family's `>=525` up to `>=580`. |
| System CUDA Toolkit | Not required |
| CUDA runtime | Already included in the package, bundled alongside PyTorch (`cu130`) |
| RAM | Exact minimum not specified in this repo's docs |
| VRAM | Exact minimum not specified; a configuration with an RTX 3060 Ti (8 GB) is confirmed working |
| GUI (optional) | Additionally requires `gir1.2-gtk-4.0`, `python3-gi`, `libgl1`, `libglib2.0-0` (`Recommends`, GUI-only — the CLI works without them) |

**OS / architecture**
- `amd64` only.
- Built and tested against Debian 13 (Trixie), Python >= 3.11 (Trixie
  ships 3.13). The build tooling itself (`scripts/build-deb.sh`,
  `debian/rules`) is distro-agnostic `dpkg-buildpackage`/`debhelper` — it
  was also test-built and smoke-tested (wrapper package install + import +
  the non-GPU/non-model test tier) on **Ubuntu 24.04 LTS** with no changes
  needed, and its `Build-Depends`/`Depends`/`Recommends` all resolve
  cleanly there too. Expected to work the same way on any current
  Debian/Ubuntu release, LTS or not (e.g. Ubuntu 25.10, 26.04) — the
  distro/version only has to satisfy the two points below.
- **Two things actually gate compatibility, not the distro name:**
  1. **Python >= 3.11** available as `python3` (Ubuntu 22.04 ships 3.10 —
     too old; 23.10+/24.04+ are fine).
  2. **glibc version, direction matters.** glibc is backward- but not
     forward-compatible: a binary built against an *older* glibc runs
     fine on a *newer* one, never the reverse. The official `.deb` is
     built on Debian 13 (glibc 2.41) and embeds a copy of that machine's
     Python interpreter and compiled extensions (torch, numpy, scipy) via
     `--copies` in `debian/rules`. That means it should run without issue
     on anything with glibc >= 2.41 (Ubuntu 25.10 and 26.04+ are already
     ahead, at 2.42) — including releases that don't exist yet, since
     glibc only ever adds symbols going forward. It is **not guaranteed
     on Ubuntu 24.04 LTS**, whose glibc (2.39) is older than the build
     machine's. **Rebuilding locally with `scripts/build-deb.sh` sidesteps
     this entirely**, since the venv then embeds the target system's own
     interpreter and libraries instead of Debian 13's — this is the
     recommended path on any system other than Debian 13 itself.
- Package **names can still drift between releases** independently of
  glibc — e.g. Ubuntu >= 24.04 renamed `libglib2.0-0` to
  `libglib2.0-0t64` as part of its 64-bit `time_t` transition, so that
  exact `Recommends` entry in `debian/control` won't resolve by that name
  there (harmless for the CLI; only affects the optional GTK4 GUI).
  Rebuilding locally also avoids this, since `apt`/`debhelper` resolve
  whatever the target system currently calls its packages.

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
- `torch==2.13.0` (CUDA 13.0 / `cu130` build)
- `torchaudio==2.11.0` (upstream's final, maintenance-mode release —
  documented as compatible with later torch minors as-is)
- `melband-roformer-infer==0.1.5`
- `soundfile>=0.12,<0.13`

**GPU (optional, for acceleration)**
- NVIDIA GPU, Ampere (`sm_86`) or newer recommended (target hardware:
  RTX 3060 Ti).
- NVIDIA driver **>= 580.65.06** — hard floor for the `cu130` PyTorch
  build; below this, CUDA will not initialize. (This is a new major-branch
  floor, not a point bump: CUDA 13.x raised the minimum required driver
  from the 12.x family's `>=525` up to `>=580`.)
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

## 1. Build from source

Two build outputs come from the same source tree: a native `.deb`
package (§1a) and a self-contained AppImage (§1b). Both bundle their own
CUDA PyTorch runtime; neither needs system Python packages beyond what's
listed in §0, and the AppImage additionally needs no `apt install` step
at all.

### 1a. `.deb` package

```bash
git clone https://github.com/Necron666759/Melband-Roformer
cd Melband-Roformer
./scripts/build-deb.sh
```

Building locally like this is generally the better way to get the
package, for two reasons:

- **It sidesteps the glibc/package-naming caveats in §0** entirely — the
  venv baked into the `.deb` ends up built with *your own* system's Python
  interpreter, compiled extensions, and package names, rather than a copy
  of the Debian 13 build machine's.
- **The pre-built `.deb` is ~2.2 GB** (bundled CUDA PyTorch runtime — see
  `README.debian.md` for why), which doesn't fit as a single GitHub
  Release asset without splitting into archive volumes. Building locally
  avoids downloading a multi-gigabyte split archive at all; only PyPI/
  PyTorch-index network access is needed, and only at build time.

`scripts/build-deb.sh` installs its own build dependencies via `apt`
(pass `--no-deps` to skip that if you already have them — see the script's
own header comment for the full list and reasoning), builds a throwaway
venv, runs the non-GPU/non-model test tier against it, then invokes
`dpkg-buildpackage -us -uc -b`. The resulting `.deb`, `.changes`, and
`.buildinfo` are collected into `./out/`:

```bash
sudo apt install ./out/melband-roformer_*.deb
```

No root is required for the build itself (only for the one-time `apt-get`
install of build tools, and even that is skipped automatically if they're
already present). Expect the build to take a while and to download
several GB (PyTorch's CUDA wheel is 2–3.5 GB by itself) — this only
happens once, not on every `apt install`.

If you'd rather not build locally, a pre-built `.deb` may be attached to
this repo's GitHub Releases (split into `.7z` volumes, see the release
notes for reassembly instructions) — see §2.

### 1b. AppImage

```bash
./scripts/build-appimage.sh
```

Output: `./out/melband-roformer-<version>-x86_64.AppImage` (e.g.
`melband-roformer-0.1.14-x86_64.AppImage`). Unlike the `.deb`, this is
**not** a general-purpose CLI you install system-wide: the AppImage's
only entry point is a restricted, single-purpose terminal — an
interactive REPL that understands exactly the `separate` /
`list-models` / `download-model` / `info` / `self-test` / `version`
commands (see §4) — and nothing else; no general shell access, no
`cd`/`ls`/other filesystem browsing. Useful for a portable, sandboxed
way to run separations with no `apt install` step at all.

Building it downloads the same multi-gigabyte CUDA PyTorch wheels as the
`.deb` build, plus a small portable CPython interpreter (pinned, from
`astral-sh/python-build-standalone` — not the build host's own
`python3`), so the resulting image runs on any glibc >= 2.17 system
(Debian 8+/Ubuntu 14.04+ and effectively everything current) regardless
of what machine built it. No root or `apt-get` step is required for this
build at all — everything the script needs is fetched as plain archives,
not installed system-wide.

Run it like any AppImage:

```bash
chmod +x ./out/melband-roformer-*-x86_64.AppImage
./out/melband-roformer-*-x86_64.AppImage
# or, in a container/CI without FUSE:
./out/melband-roformer-*-x86_64.AppImage --appimage-extract-and-run
```

See `README.appimage.md` for the full design/security rationale (why a
restricted shell, how it's enforced, the bundled HUD terminal).

## 2. Install

This section covers the `.deb`. The AppImage (§1b) has no separate
install step — `chmod +x` and run it directly.

If you built locally (§1a), install straight from `./out/`. If you're
using a pre-built `.deb` (e.g. reassembled from a split GitHub Release
archive):

```bash
sudo apt install ./melband-roformer_0.1.14_amd64.deb
```

No network access is required for `apt install` itself — the Python venv
(including PyTorch) is baked into the package at *build* time. The
**model weights** (~913 MB) are downloaded separately, on demand, at first
use (see §5), because Debian packages should not ship near-gigabyte binary
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

The AppImage has nothing to "remove" in the `apt` sense — it's a single
file; deleting it is enough. Same model-weight cache note applies: it's
shared with the `.deb` (see §5), so delete
`~/.cache/melband-roformer-infer/` yourself if you no longer want the
downloaded weights either.

## 4. Basic usage

### `.deb`: direct CLI

```bash
melband-roformer input.wav
melband-roformer input.wav --output-dir ./output
melband-roformer input.wav --device cuda
melband-roformer input.wav --device cpu
melband-roformer input.wav --model melband-roformer-kim-vocals
melband-roformer ./input/*.wav --output-dir ./output   # batch, glob expanded internally
```

### AppImage: interactive terminal

The AppImage is interactive-only — it does **not** take melband-roformer
flags directly on its own command line (`./melband-roformer.AppImage
input.wav` will *not* separate that file). Launch it, then type commands
at its prompt:

```bash
./melband-roformer-0.1.14-x86_64.AppImage
```

- **Launched by double-clicking** (no controlling terminal): opens the
  AppImage's own bundled "HUD" terminal window running this same prompt.
- **Launched from a shell you already have open**: runs the prompt
  directly in that terminal, no extra window.

```
[melband-roformer]> separate input.wav
[melband-roformer]> separate input.wav --output-dir ./output --device cuda
[melband-roformer]> separate ./input/*.wav --output-dir ./output
[melband-roformer]> help
[melband-roformer]> exit
```

Every flag above (`--output-dir`, `--device`, `--model`, ...) works the
same way after `separate` as it does on the `.deb`'s command line — the
REPL just dispatches each whitelisted subcommand (`separate`,
`list-models`, `download-model`, `info`, `self-test`, `version`, plus
`clear`/`help`/`exit`/`quit`) to the identical underlying code. There is
no general shell behind this prompt: no `cd`/`ls`, no `;`/`&&`/`|`/`` ` ``
command chaining, no way to run anything outside that whitelist (see
`README.appimage.md` for the full security rationale).

Output (both): `<name>_vocals.wav` and `<name>_instrumental.wav` next to
(or in `--output-dir`), same sample rate as the input where the model
allows it.

## 5. Model download

```bash
melband-roformer --list-models
melband-roformer --download-model                     # fetches the default: melband-roformer-kim-vocals
melband-roformer --download-model --model <slug>
```

Same thing from the AppImage's prompt (see §4):

```
[melband-roformer]> list-models
[melband-roformer]> download-model
[melband-roformer]> download-model --model <slug>
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

From the AppImage prompt, this is just `info` (see §4).

```
Application version: 0.1.0
Python:               3.12.7 (system, via venv)
PyTorch:               2.13.0+cu130
CUDA runtime (torch):  13.0
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

### GPU attention backend

On a GPU run you'll see exactly one of these two lines, once per
separation:

```
A100 GPU detected, using flash attention if input tensor is on cuda
```
```
Non-A100 GPU detected, using math or mem efficient attention if input tensor is on cuda
```

This is upstream (`mel_band_roformer`) picking which PyTorch SDPA backend
to use, based on your GPU's compute capability: flash attention on A100
(compute capability 8.0), math/memory-efficient attention on everything
else (Ampere `sm_86` cards like the RTX 3060 Ti included). **Neither line
is an error or a degraded mode** — both backends are correct, numerically
equivalent implementations; only the A100 line means "flash attention was
picked" specifically because that's the one GPU generation flash attention
is unconditionally available for in this code path.

A second, separate message you will *not* see, because this package
suppresses it deliberately: a `UserWarning` from `torch.stft()` about "A
window was not provided". Upstream triggers this once per model
construction with a shape-probe call against dummy random noise (used
only to count frequency bins from the configured FFT size) — every other
`torch.stft()`/`istft()` call that actually processes your audio already
passes an explicit window, so the spectral-leakage concern the warning
describes never applies to real separation output. See
`app/sitecustomize.py` (`_silence_benign_stft_window_warning` /
`_patch_attend_gpu_banner_readme_pointer`) for exactly what is filtered
and why, if you want to double check this yourself.

## 7. Diagnostics

```bash
melband-roformer --self-test
```

From the AppImage prompt, this is just `self-test` (see §4).

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

`~/.cache/melband-roformer-infer/<model-slug>/` (upstream default, see
§5). Same location and same cache for both the `.deb` and the AppImage.

## 9. Output location

Current directory by default, or `--output-dir <path>` / `--store_dir`
(upstream's underlying flag; `--output-dir` is accepted as the friendlier
alias documented in this wrapper's `--help`). Same for the AppImage's
`separate --output-dir <path>` (§4).

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
- GTK4 GUI (`melband-roformer-gui`) ships with the `.deb` only; it is a
  thin optional layer and has not been visually tested in this
  environment (no display server available here — see README.debian.md
  "What was actually verified"). The AppImage does not include it — it
  has its own bundled restricted terminal instead (§1b/§4).
- The AppImage's restricted terminal is interactive-only: it does not
  accept melband-roformer flags on the AppImage's own command line, and
  has no scripted/one-shot invocation mode — every run drops into the
  `[melband-roformer]>` prompt (see §4).

<p align="center">
  <img src="ScreenShots/my_cat.png" width="50%">
</p>

Donate $5 to buy food for a cat:
<br>
USDT(TRC20): TWEmMHfc5DbQuDru8oaXNoXxTNkqYJbsYv<br>
BTC(BEP20): 0x147d19ae0e1b50ca6c87d32b2f716068e6ba5b17<br>
SOL(SOL): j3kkX7VuKfchcnH8Y9bUCbqjjA4ssLpspvsw4YZi8ia<br>
