# Packaging notes (melband-roformer for Debian 13 / Trixie, amd64)

## Why the package is large, and why that's the correct tradeoff here

GPU (CUDA) PyTorch wheels are 2-3.5 GB by themselves (they bundle cuBLAS,
cuDNN, NCCL runtime .so files so they work against nothing but the user's
NVIDIA *driver*). Combined with the rest of the venv (numpy, scipy,
librosa, soundfile, ml_collections, pyyaml, einops, rotary-embedding-torch,
the `melband-roformer-infer` package itself, etc.) the bundled venv is the
majority of the package's size.

Two designs were considered:

1. **Download PyTorch/deps during `postinst`.** Rejected: Debian maintainer
   scripts must not require network access during package installation/
   configuration (and pulling `pip install` inside a root `postinst` is
   exactly the kind of "curl|bash-shaped" risk this task explicitly forbids
   — arbitrary code from the network, running as root, outside apt's trust
   model). It would also silently break `apt install` on air-gapped or
   offline machines.
2. **Bake the venv into the package payload at *build* time, model weights
   excluded and fetched on demand by the *user* at runtime with explicit
   consent and checksum verification.** Chosen. Build machines are expected
   to have network (same as any Debian package with vendored/downloaded
   build-time dependencies); the resulting `.deb` installs offline
   with **zero network access** and **zero code execution beyond dpkg's own
   unpacking of a payload built by us at a known commit/hash** — which is
   the actual security property you want. Model weights are excluded from
   the payload because unlike PyTorch (a fixed, auditable build-time
   dependency) they are a large, swappable, user-chosen asset that
   shouldn't silently balloon the base package and that upstream explicitly
   supports fetching independently with its own checksum registry.

This is why "keep the package small" and "no CUDA weights baked into a
`Depends:`-only Debian-repo package" *cannot both hold* for a GPU-accelerated
deep learning tool that must install offline. If instead you want a small
package and are fine with `postinst` requiring network (explicitly against
Debian policy but sometimes done pragmatically by third-party `.deb`
repositories, e.g. many proprietary vendors), that is a different, less
safe design — happy to produce that variant instead if you actually want
it, but it wasn't what "no curl|bash at install" pointed to.

## Layout inside the package

```
/usr/bin/melband-roformer               -> thin dispatcher, execs the venv's python
/usr/bin/melband-roformer-gui           -> thin dispatcher for the GTK4 GUI (optional)
/usr/lib/melband-roformer/venv/         -> self-contained venv (python3, pip-installed deps, incl. torch+cuda)
/usr/lib/melband-roformer/app/          -> our wrapper package (melband_roformer_wrapper/*.py), pip-installed
                                            into the same venv in editable-less (built wheel) form
/usr/share/applications/melband-roformer.desktop
/usr/share/icons/hicolor/scalable/apps/melband-roformer.svg
/usr/share/doc/melband-roformer/{README.md,README.debian.md,changelog.Debian.gz,copyright}
/usr/share/man/man1/melband-roformer.1.gz
```

The venv is created with `--system-site-packages`, for one specific reason:
**PyGObject (the `gi` module GTK4 bindings need) is a system package
(`python3-gi`) built against the host's GObject-Introspection/GTK
libraries, not something reasonably pip-installable/vendorable** without
adding heavy build-time dependencies (`libgirepository`-dev, `libcairo2`-
dev, a C compiler) purely for an *optional* GUI. `--system-site-packages`
lets the bundled venv see `python3-gi` if the user has it installed
(`Recommends`, see below), without affecting the pinned CUDA
torch/numpy/scipy versions -- packages actually pip-installed *inside* the
venv always shadow same-named packages from system site-packages on
`sys.path`, so this does not reintroduce the "system Python pollution"
problem; it only adds *read* visibility into system packages, one-way.

The venv is **not** relocatable-sensitive: it's built with `--copies` (not
symlinks to a build-machine python) and its shebangs are rewritten to the
final `/usr/lib/melband-roformer/venv` path before packaging, so it works
regardless of what Python the host's `/usr/bin/python3` is.

`/usr/bin/melband-roformer` does **not** put the venv's `bin/` on `PATH`
system-wide; it simply execs
`/usr/lib/melband-roformer/venv/bin/python -m melband_roformer_wrapper.cli "$@"`.
This is the standard "vendored venv behind a thin launcher" pattern used by
tools like `mypy`, `black`, etc. when installed via pipx-style isolation —
it cannot collide with the system Python or any other Python application,
and `apt remove` cleanly deletes the whole `/usr/lib/melband-roformer/` tree.

## Silencing upstream's `torch.backends.cuda.sdp_kernel()` FutureWarning

Upstream `melband-roformer-infer`'s attention code still opens its backend
selection with the now-deprecated
`torch.backends.cuda.sdp_kernel(enable_flash=..., enable_math=...,
enable_mem_efficient=...)` context manager, which prints on every
separation run:

```
FutureWarning: `torch.backends.cuda.sdp_kernel()` is deprecated. In the
future, this context manager will be removed. Please see
`torch.nn.attention.sdpa_kernel()` for the new context manager, with
updated signature.
```

The actual separation work runs as a **subprocess** (`cli.py` execs the
`melband-roformer-infer` console-script directly; see "Layout inside the
package" above), so a monkeypatch placed inside
`melband_roformer_wrapper/__init__.py` would never reach it -- that
subprocess is a fresh interpreter importing upstream's own code, not ours.
`app/sitecustomize.py` is installed as a **top-level** module of the venv
(`[tool.setuptools] py-modules` in `app/pyproject.toml`, not nested inside
`melband_roformer_wrapper/`) specifically so Python's `site` machinery
auto-imports it at the start of *every* interpreter invocation in this
venv -- our CLI, our GUI, and upstream's own console-scripts alike -- and
replaces `torch.backends.cuda.sdp_kernel` with a same-signature shim built
on the new, non-deprecated `torch.nn.attention.sdpa_kernel`. Upstream's
call site keeps working completely unmodified; it just no longer touches
the deprecated code path, so the warning does not fire at all. See
`app/sitecustomize.py`'s own docstring for the full reasoning.

## Runtime dependencies actually needed from Debian

| Package                       | Why |
|--------------------------------|-----|
| `python3 (>= 3.11)`            | to build/run the venv (Trixie ships 3.13; venv's own interpreter is what actually runs, but `python3-venv`-created venvs still need a system python3 present at build time) |
| `ffmpeg`                        | upstream `melband-roformer-infer` / its underlying loader use it (via `soundfile`/`librosa`/`pydub`-style backends and for input formats beyond WAV/FLAC, e.g. MP3/OGG) for decoding. Not vendored — it's a normal Debian package with its own large dependency tree (codecs), exactly the kind of thing that *should* come from the distro, not be bundled. |
| `libgl1`, `libglib2.0-0`        | runtime shared libs GTK4/PyGObject need if the GUI is used (already present on any desktop Trixie install; only pulled as `Recommends`, not hard `Depends`, since the CLI works without a desktop) |
| `gir1.2-gtk-4.0`, `python3-gi`  | **only** for `melband-roformer-gui` (separate `Recommends`, not `Depends` — CLI must work headless/on servers) |

NOT taken from Debian (bundled in the venv instead), and why:

| Component | Why not from Debian repos |
|---|---|
| PyTorch (CUDA build) | Debian's `python3-torch` in the archive is typically CPU-only or a much older/mismatched CUDA build; pinning the exact CUDA-compatible wheel upstream tested against (see below) requires PyPI, not apt. |
| `melband-roformer-infer`, `ml_collections`, `rotary-embedding-torch`, `einops` | not packaged in Debian at all. |
| `numpy`/`scipy`/`librosa`/`soundfile` Python bindings | pinned to versions compatible with the above rather than Trixie's system versions, to avoid ABI mismatches with the bundled torch build; the system's own `python3-numpy` etc. are left untouched. |

`nvidia-smi` and the NVIDIA kernel module/driver are explicitly **assumed
pre-installed by the user** (as required) — the package only *calls*
`nvidia-smi` if present and checks `torch.cuda.is_available()`; it never
attempts to install, load, or manage any NVIDIA driver or kernel component.

## PyTorch / CUDA compatibility used for build pinning

- Target GPU: RTX 3060 Ti = Ampere, SM 8.6 (`sm_86`). This has been a
  supported PyTorch CUDA target since PyTorch 1.9 / CUDA 11.1 and remains
  fully supported by every current PyTorch CUDA 11.8/12.6/12.8 wheel
  (Ampere is not deprecated in any current PyTorch release channel).
- The build script pins `torch==2.11.0` / `torchaudio==2.11.0` with the
  `cu126` wheel index (`https://download.pytorch.org/whl/cu126`), which
  requires NVIDIA driver **>= 560.28.03** (CUDA 12.6 GA minimum driver).
  Bumped from `torch/torchaudio==2.6.0`+`cu124` in 0.1.6 on request; both
  packages ship cp313 wheels on `cu126` for the 2.11.0 release, so this
  carries no repeat of the earlier cp313-coverage gaps (see below). This
  is a real, checkable constraint at `--self-test`/`--info` time: the
  wrapper compares `nvidia-smi`'s reported driver/CUDA version against
  this floor and warns (not crashes) if the host driver is older.
  (History: switched from `cu121` in 0.1.2, since PyTorch's `cu121`
  channel never published a `torchaudio` wheel for CPython 3.13 -- it has
  `torch` wheels for cp313, just not the matching `torchaudio` ones,
  which is what Debian 13 (Trixie) ships as `python3`. That alone wasn't
  enough, though: `cu124` turned out to have the same gap at
  `torchaudio==2.5.1` -- `torchaudio` didn't publish *any* cp313 wheel
  for the whole 2.5.x series, on any index. `torchaudio`'s first cp313
  wheels landed in the 2.6.0 release, so both pins were bumped to 2.6.0
  together (0.1.4) to get real cp313 coverage on `cu124`, and later to
  2.11.0/`cu126` together (0.1.6).)
- No CUDA *toolkit* or driver `.deb` is installed or bundled; only the
  PyPI PyTorch wheel's bundled runtime `.so` files (cuBLAS/cuDNN/NCCL),
  which is exactly what upstream's own `pip install melband-roformer-infer`
  users get — we're not changing that story, just packaging it.

## What was actually verified in preparing this deliverable, vs. assumed

This project was prepared in a sandboxed environment with **no outbound
network access from the build/runtime container, no NVIDIA GPU, and no
Debian 13 host to run `dpkg-buildpackage`/`apt install` against.** See the
final "Что реально проверено, а что только предполагается" section in the
chat response for the explicit, itemized list — do not treat this package
as pre-verified on real hardware. `scripts/build-deb.sh` and
`scripts/test-deb.sh` are provided precisely so *you* can run the real
verification on your Trixie + RTX 3060 Ti machine, and they print clear
pass/fail output rather than assuming success.
