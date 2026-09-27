# Changelog

All notable changes to `melband-roformer` are documented here. This file is
a human-readable summary; the authoritative, fully-detailed record (exact
rationale, file-by-file changes, verification notes) is `debian/changelog`.

Versions below are Debian package versions (`debian/changelog`), which are
independent of the wrapper's internal Python `__version__` (`app/pyproject.toml`,
currently `0.1.0` and not tied 1:1 to the package version).

## [0.1.14] - 2026-09-27

### Changed
- **Rewrote the terminal progress display** for the wrapped
  `melband-roformer-infer` subprocess (`app/melband_roformer_wrapper/progress.py`,
  new). Previously two uncoordinated signals reached the terminal: a
  `tqdm` bar keyed on completed *tracks* (with a single input file it just
  jumps from 0% to 100%, never animating) and a separate plain-text
  "Estimated time remaining" line with no bar. These are replaced by a
  single bar sized by total input audio duration (via `soundfile`) rather
  than track count, so it now fills smoothly in real time and reflects
  proportional progress across multi-file batches. The finished line is
  also shorter: `Tracks: 100%|<bar>| 1/1 [24.20s/track]`.
  - Implemented by running the subprocess behind a pty and filtering its
    output line-by-line; every other line (device/CUDA banner, "Processing
    track i/N: ...", warnings, tracebacks) still passes through unchanged.
  - Applies everywhere the CLI is invoked: the `.deb`, the AppImage
    terminal UI, and the GTK4 GUI (which shells out to the same CLI).

### Dependencies
- Added `tqdm>=4.64` as a direct dependency (`app/requirements.txt`,
  `app/pyproject.toml`) — `progress.py` now imports it directly, instead
  of relying on it only being present transitively via
  `melband-roformer-infer`.

## [0.1.13] - 2026-09-27

### Fixed
- **AppImage HUD terminal (`appimage/melband-roformer-hud-term.py`):
  Ctrl+Shift+C / Ctrl+Shift+V (copy/paste) did nothing, and any keypress
  cleared the current text selection instead, under non-Latin keyboard
  layouts (e.g. Cyrillic ЙЦУКЕН).**
  - Root cause: the copy/paste chord was matched against the X11 keysym
    (which depends on the *active layout*, not the physical key), so under
    a Cyrillic layout the physical C/V keys never matched `"c"`/`"v"` and
    fell through to the "clear selection" branch.
  - Fix: matching now happens primarily on the physical hardware keycode
    (layout-independent), with a keysym fallback covering both Latin and
    the common Cyrillic keysym names.
- **Same file: a second, layout-independent bug** where pressing Ctrl or
  Shift *on its own*, before the chord's letter key, already cleared the
  selection (X11 delivers each modifier as its own keypress, and
  `event.state` reflects the state *before* that key). Bare modifier
  keypresses (Ctrl/Shift/Alt/Meta/Super/Hyper/Level-shift/Caps/Num/Scroll
  lock, etc.) are now treated as no-ops instead of "any other keystroke"
  events, so the selection survives until the actual chord letter arrives.
  - Both fixes were verified with standalone simulations of the relevant
    (keysym, keycode) sequences, since this build environment has no X
    display to drive the real widget through.

## [0.1.12] - 2026-09-27

### Changed
- **Bumped the bundled CUDA PyTorch build from `cu126`/`torch==2.11.0` to
  `cu130`/`torch==2.13.0`** (`app/requirements.txt`, `app/pyproject.toml`,
  `debian/rules`, `scripts/build-appimage.sh`). `cu130` is now PyTorch's
  own default stable channel; `cu126` is deprecated upstream and slated
  for removal in PyTorch 2.15.
  - `torchaudio` is deliberately **left at `2.11.0`**: upstream put it
    into maintenance mode at that release and documents it as compatible
    with later `torch` minors without a rebuild.
- **Raised the minimum required NVIDIA driver from `560.28.03` to
  `580.65.06`** (`app/melband_roformer_wrapper/gpuinfo.py`) to match the
  `cu130` bump. This is a new major-branch floor (CUDA 13.x requires
  `>=580`, vs. the CUDA 12.x family's `>=525`), not a point release —
  older drivers that worked with the previous build will need updating.
- Updated `README.md` / `README.debian.md`: dependency table, `--info`
  sample output, and CUDA-compatibility section now reflect `cu130` /
  `torch==2.13.0` / driver `>=580.65.06`, with the `torchaudio`
  maintenance-mode reasoning documented inline for future bumps.
- No CLI/behavioral changes beyond the above — `melband-roformer-infer`
  only requires `torch>=2.0`, so it is unaffected by the `torch` bump.

---

## Upgrading from 0.1.11

If you're moving straight from `0.1.11`, the two changes that actually
affect you day-to-day are:

1. **Driver requirement went up**: you now need NVIDIA driver
   `>= 580.65.06` (was `>= 560.28.03`). Check with `melband-roformer --info`
   or `--self-test` after upgrading — GPU runs will fail loudly rather than
   silently falling back to CPU if your driver is too old.
2. **Progress output looks different**: a single, smoothly-animating,
   duration-based progress bar instead of the old per-track `tqdm` bar
   plus separate ETA line.

Everything else in this release is a packaging/build fix (CUDA wheel
channel, AppImage HUD terminal keyboard handling) with no change to
separation output quality or CLI flags.

See `debian/changelog` for the complete, unabridged history back to the
initial `0.1.0` release.
