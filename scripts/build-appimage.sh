#!/usr/bin/env bash
# Build melband-roformer.AppImage: a self-contained AppImage whose only
# entry point is a restricted terminal (appimage/melband-roformer-shell.py)
# that runs exclusively melband-roformer commands. See appimage/AppRun and
# README.appimage.md for the design and its security reasoning.
#
# Usage:
#   ./scripts/build-appimage.sh
#
# Output: ./out/melband-roformer-<version>-x86_64.AppImage
#
# Portability: the venv's base interpreter is a prebuilt, relocatable
# CPython from astral-sh/python-build-standalone (glibc floor 2.17,
# Debian 8+/Ubuntu 14.04+), fetched at build time -- NOT the build host's
# own python3. That means, unlike a naive `python3 -m venv --copies`
# approach, this script produces the same glibc-compatibility result
# whether you run it on Debian 13, Ubuntu 24.04, or any other current
# release; you do not need to build on your oldest target. See
# README.appimage.md for how this was verified (`objdump -T | grep
# GLIBC` against the portable interpreter and the venv copied from it).
set -euo pipefail

HERE="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$HERE"

VERSION="$(sed -n 's/^melband-roformer (\([^)]*\)).*/\1/p' debian/changelog | head -1)"
OUT_DIR="$HERE/out"
BUILD_DIR="$HERE/.appimage-build"
APPDIR="$BUILD_DIR/AppDir"
TORCH_INDEX="https://download.pytorch.org/whl/cu130"
PYPI_INDEX="https://pypi.org/simple"

TOOLS_DIR="$BUILD_DIR/tools"
APPIMAGETOOL="$TOOLS_DIR/appimagetool"

# ---------------------------------------------------------------------------
# Progress/animation helpers.
#
# Why this exists: curl's `-s` and pip's `-q` flags (used below) silence
# *their own* download progress meters -- which is exactly the information
# that was missing (what's downloading right now, at what speed). Fix: stop
# silencing them, so curl's native transfer meter and pip's native
# per-package "Downloading ... |####| N.N MB N.N MB/s" bars show for real,
# rather than reinventing a worse version of what they already have.
#
# What's added on top, for the parts that have no native progress output at
# all (e.g. copying the portable interpreter into place, which silently
# creates hundreds of files): a small spinner with an elapsed-time counter,
# plus an overall "[n/5]" step counter so it's always clear which of the
# five build stages is currently running. Both degrade to plain, static
# lines (no \r animation) when stdout isn't a terminal (redirected to a
# log file, CI), so piped output stays clean and grep-able there too.
# ---------------------------------------------------------------------------
TOTAL_STEPS=5
CURRENT_STEP=0
BUILD_START=$SECONDS

_IS_TTY=0
[ -t 1 ] && _IS_TTY=1

step() {
    CURRENT_STEP=$((CURRENT_STEP + 1))
    echo
    echo "==> [$CURRENT_STEP/$TOTAL_STEPS] $*"
}

substep() {
    echo "    -> $*"
}

SPINNER_PID=""

# start_spinner LABEL: shows an animated "<frame> LABEL (mm:ss)" line that
# updates in place, for a command that produces no output of its own.
# Always pair with stop_spinner once that command finishes (success or
# failure alike -- the EXIT trap below covers the "script aborts mid-step"
# case too, so a failed build never leaves a stuck spinner or a hidden
# cursor behind).
start_spinner() {
    local label="$1"
    if [ "$_IS_TTY" -ne 1 ]; then
        echo "    -> $label ..."
        return
    fi
    (
        frames='⠋⠙⠹⠸⠼⠴⠦⠧⠇⠏'
        i=0
        t0=$SECONDS
        while :; do
            el=$((SECONDS - t0))
            printf "\r    %s %s (%02d:%02d)  " \
                "${frames:i%${#frames}:1}" "$label" $((el / 60)) $((el % 60))
            i=$((i + 1))
            sleep 0.1
        done
    ) &
    SPINNER_PID=$!
    disown "$SPINNER_PID" 2>/dev/null || true
}

stop_spinner() {
    if [ -n "$SPINNER_PID" ]; then
        kill "$SPINNER_PID" 2>/dev/null || true
        wait "$SPINNER_PID" 2>/dev/null || true
        SPINNER_PID=""
        [ "$_IS_TTY" -eq 1 ] && printf "\r\033[K"
    fi
}
trap stop_spinner EXIT

echo "==> melband-roformer $VERSION -> AppImage"

rm -rf "$APPDIR"
mkdir -p "$APPDIR" "$OUT_DIR" "$TOOLS_DIR"

# ---------------------------------------------------------------------------
# 1. Build tools (appimagetool) -- downloaded once, cached in
#    .appimage-build/tools/. It ships as an AppImage itself; run it with
#    --appimage-extract-and-run so this also works in containers/CI
#    without FUSE.
# ---------------------------------------------------------------------------
step "Build tools (appimagetool)"
fetch_tool() {
    local url="$1" dest="$2"
    if [ ! -x "$dest" ]; then
        substep "fetching $(basename "$dest")"
        # No -s here on purpose: that flag silences curl's own transfer
        # meter, which is exactly what shows the current download speed and
        # percentage/ETA live -- removing it, not reimplementing it, is the
        # fix.
        curl -L -o "$dest" "$url"
        chmod +x "$dest"
    else
        substep "$(basename "$dest") already cached, skipping download"
    fi
}
fetch_tool "https://github.com/AppImage/AppImageKit/releases/download/continuous/appimagetool-x86_64.AppImage" "$APPIMAGETOOL"

# ---------------------------------------------------------------------------
# 2. Portable CPython interpreter (astral-sh/python-build-standalone), NOT
#    the build machine's own python3.
#
#    Why: `python3 -m venv --copies` duplicates whichever interpreter runs
#    it. If that's the *build host's* system python3, the resulting binary
#    is linked against the host's glibc -- meaning the AppImage inherits
#    that machine's glibc floor and, per glibc's backward-but-not-forward
#    compatibility guarantee, is only guaranteed to run on machines with an
#    equal-or-newer glibc than the one it was built on (see the portability
#    caveat further down / README.appimage.md).
#
#    python-build-standalone publishes prebuilt, relocatable CPython
#    binaries that only ever reference glibc symbols up to version 2.17
#    (released 2014; verified in this project's own testing with `objdump
#    -T | grep GLIBC`), regardless of what machine builds with them. Using
#    one of *those* as the venv's base interpreter instead of the build
#    host's own python3 makes the resulting AppImage's glibc floor a fixed,
#    low constant (2.17 -- Debian 8+/Ubuntu 14.04+) independent of whether
#    you build on Debian 13, Ubuntu 24.04, or anything newer. This is what
#    actually removes the "build on your oldest target" requirement, not
#    just documents around it.
# ---------------------------------------------------------------------------
PBS_RELEASE="20260825"
PBS_ASSET="cpython-3.13.15+${PBS_RELEASE}-x86_64-unknown-linux-gnu-install_only_stripped.tar.gz"
PBS_URL="https://github.com/astral-sh/python-build-standalone/releases/download/${PBS_RELEASE}/${PBS_ASSET/+/%2B}"
PBS_ARCHIVE="$TOOLS_DIR/$PBS_ASSET"
PBS_DIR="$BUILD_DIR/portable-python"

step "Portable CPython interpreter (python-build-standalone $PBS_RELEASE)"
# PBS_DIR is a fixed path across script runs/versions, so an "is the binary
# there" check alone would silently keep serving a *previous* PBS_ASSET's
# interpreter after this script is edited to point at a different CPython
# version/release (e.g. bumping 3.12.14 -> 3.13.15 here) -- caching would
# then ship the stale interpreter with no error or warning. Stamp the asset
# name actually extracted into $PBS_DIR and compare against it, so a version
# change always forces a fresh fetch+extract, and only a byte-for-byte
# rerun with the same PBS_ASSET is treated as cache-hit.
PBS_STAMP="$PBS_DIR/.pbs_asset"
if [ -x "$PBS_DIR/python/bin/python3" ] && [ "$(cat "$PBS_STAMP" 2>/dev/null)" = "$PBS_ASSET" ]; then
    substep "portable CPython already cached ($PBS_ASSET), skipping download"
else
    substep "fetching $PBS_ASSET"
    curl -L -o "$PBS_ARCHIVE" "$PBS_URL"  # see fetch_tool() above: no -s, so the real download speed/progress shows
    rm -rf "$PBS_DIR"
    mkdir -p "$PBS_DIR"
    start_spinner "extracting $PBS_ASSET"
    tar -xzf "$PBS_ARCHIVE" -C "$PBS_DIR"
    stop_spinner
    echo "$PBS_ASSET" > "$PBS_STAMP"
fi
BASE_PYTHON="$PBS_DIR/python/bin/python3"

VENV="$APPDIR/usr/lib/melband-roformer/venv"

# ---------------------------------------------------------------------------
# 3. AppDir skeleton: our own files only. No general-purpose shell or
#    terminal emulator is ever added here -- melband-roformer-shell.py is
#    the sole entry point, and melband-roformer-hud-term.py (tkinter +
#    pty + pyte, see that file's own docstring) is this AppImage's whole
#    windowed-terminal UI. Both run with a hardcoded argv (see AppRun);
#    neither is ever given a user-supplied command line.
# ---------------------------------------------------------------------------
step "Assembling AppDir"

install -d "$APPDIR/usr/bin"
install -m 0755 appimage/melband-roformer-shell.py "$APPDIR/usr/bin/melband-roformer-shell.py"
install -m 0755 appimage/melband-roformer-hud-term.py "$APPDIR/usr/bin/melband-roformer-hud-term.py"
install -m 0755 appimage/AppRun "$APPDIR/AppRun"
install -m 0644 appimage/melband-roformer.desktop "$APPDIR/melband-roformer.desktop"
install -m 0644 appimage/melband-roformer.png "$APPDIR/melband-roformer.png"

# HUD terminal font (see melband-roformer-hud-term.py's own docstring):
# the bundled Tk has no Xft/fontconfig linkage, so it can only ever
# rasterize the X server's non-antialiased core bitmap fonts. The HUD
# terminal sidesteps that entirely by rendering glyphs itself with
# Pillow/FreeType from this bundled TrueType font, rather than asking Tk
# to draw text. Installed under usr/share (not usr/bin) so it is found
# the same way regardless of whether AppRun sets $APPDIR or the script is
# run straight out of this checkout (melband-roformer-hud-term.py also
# checks next to itself, appimage/fonts/, for the latter case).
install -d "$APPDIR/usr/share/melband-roformer/fonts"
install -m 0644 appimage/fonts/DejaVuSansMono.ttf \
    "$APPDIR/usr/share/melband-roformer/fonts/DejaVuSansMono.ttf"
install -m 0644 appimage/fonts/DejaVuSansMono-Bold.ttf \
    "$APPDIR/usr/share/melband-roformer/fonts/DejaVuSansMono-Bold.ttf"
install -m 0644 appimage/fonts/LICENSE_DEJAVU \
    "$APPDIR/usr/share/melband-roformer/fonts/LICENSE_DEJAVU"

# ---------------------------------------------------------------------------
# 4. Self-contained interpreter + venv, built from the portable interpreter
#    fetched above (no --system-site-packages: unlike the .deb, the
#    AppImage should never depend on anything from the host's Python, that
#    is the entire point of the format).
#
#    IMPORTANT, and DIFFERENT from an earlier version of this script: this
#    does NOT use `python3 -m venv --copies`. A venv only copies the
#    interpreter binary itself; the standard library it depends on (Lib/,
#    lib-dynload/) stays referenced from the *base* interpreter's location
#    via pyvenv.cfg's "home" key -- which, here, is this build's own
#    ephemeral .appimage-build/portable-python/ directory. That directory
#    does not exist on an end user's machine, so a `venv --copies` build
#    of this AppImage would fail at runtime with `ModuleNotFoundError:
#    encodings` (or similar) as soon as the interpreter tried to start up,
#    on any machine other than the one it was built on. This was verified
#    directly: moving/removing the base interpreter out from under such a
#    venv reproduces exactly that failure.
#
#    python-build-standalone's interpreters are, true to their name,
#    actually relocatable: they locate their own standard library relative
#    to their own binary's path, not via any baked-in absolute path or
#    pyvenv.cfg. So instead of a venv, this step copies the *entire*
#    fetched interpreter tree (bin/, lib/, include/, share/ -- which is
#    also how tkinter's own Tcl/Tk script libraries travel along
#    automatically, with no separate copying step needed for them) into
#    place under $APPDIR, and installs packages into that copy directly.
#    This was also verified directly: copying the tree to an arbitrary new
#    path and running `import os`/`import tkinter` out of the copy, with
#    the original extraction directory removed, both succeed unmodified.
# ---------------------------------------------------------------------------
install -d "$(dirname "$VENV")"

step "Building self-contained interpreter + venv"
substep "base interpreter: $BASE_PYTHON"
start_spinner "copying portable interpreter into place"
rm -rf "$VENV"
cp -a "$PBS_DIR/python/." "$VENV/"
stop_spinner

# No -q on any of the pip calls below, on purpose: -q also silences pip's
# own per-package progress ("Downloading torch-2.13.0+cu130-....whl
# (830.5 MB) |████████████| 830.5 MB 42.1 MB/s eta 0:00:00"), which is
# exactly what shows what's currently being fetched and at what speed --
# torch/cuda alone is multiple GB, so this is the single longest, most
# opaque part of the whole build if left quiet.
substep "upgrading pip/wheel"
"$VENV/bin/python3" -m pip install --upgrade pip wheel
substep "installing pinned requirements (torch, torchaudio, melband-roformer-infer, ...) -- this is the big one"
"$VENV/bin/python3" -m pip install \
    --index-url "$TORCH_INDEX" \
    --extra-index-url "$PYPI_INDEX" \
    -r app/requirements.txt
substep "installing the wrapper package itself"
"$VENV/bin/python3" -m pip install --no-deps ./app

# pyte (pure-Python VT100/xterm terminal-state emulation) and Pillow
# (antialiased glyph rendering, replacing the bundled Tk's own Xft-less,
# bitmap-only font engine) back melband-roformer-hud-term.py's rendering
# -- both are real runtime dependencies of this AppImage's terminal UI,
# not a build-time or test-only tool, so (unlike pytest just below) they
# are installed here, before the pre/post-pytest `pip freeze` snapshots,
# so the diff-based pytest cleanup further down never mistakes them for
# something to remove.
substep "installing pyte (terminal-state emulation for the bundled HUD terminal)"
"$VENV/bin/python3" -m pip install "pyte==0.8.2"
substep "installing Pillow (antialiased glyph rendering for the bundled HUD terminal)"
"$VENV/bin/python3" -m pip install "pillow>=9"

# pytest is deliberately NOT in app/requirements.txt (it must not ship
# inside the AppImage) and, unlike debian/rules' venv, this one has no
# --system-site-packages to borrow a host-installed python3-pytest from --
# see the "Why the venv is built differently than the .deb's" section in
# README.appimage.md. So it needs installing here, for the test run only,
# then removing again before packaging. Diffing `pip freeze` before/after
# (rather than a hardcoded `pip uninstall -y pytest iniconfig pluggy ...`)
# removes exactly whatever pytest actually pulled in this run, no more and
# no less, so this stays correct even if pytest's own dependency set
# changes in a future version.
substep "installing pytest (build-time only, removed again before packaging)"
PRE_TEST_PKGS="$("$VENV/bin/python3" -m pip freeze)"
"$VENV/bin/python3" -m pip install pytest

substep "running build-time test tier (non-GPU/non-model)"
"$VENV/bin/python3" -m pytest -v "$HERE/tests" -k "not RUN_MODEL and not RUN_GPU"

substep "removing pytest and its build-time-only dependencies (not shipped)"
POST_TEST_PKGS="$("$VENV/bin/python3" -m pip freeze)"
TEST_ONLY_PKGS="$(comm -13 <(sort <<<"$PRE_TEST_PKGS") <(sort <<<"$POST_TEST_PKGS") | cut -d= -f1)"
if [ -n "$TEST_ONLY_PKGS" ]; then
    # shellcheck disable=SC2086
    "$VENV/bin/python3" -m pip uninstall -y $TEST_ONLY_PKGS
fi

# Unlike debian/rules, there is no fixed final install path to rewrite
# shebangs to -- an AppImage's mount point is a random path chosen at
# *run* time. Point every venv script's shebang at `env python3` instead,
# and make AppRun put this venv's bin/ first on PATH; that resolves
# correctly regardless of where the AppImage ends up mounted/extracted.
# (pip's own pip3/pip3.12 scripts are unaffected by -- and don't need --
# this: they already re-locate themselves via `dirname "$(realpath "$0")")`
# at their own top, which is why their shebang is `#!/bin/sh` rather than
# a python3 path in the first place.)
substep "relocating shebangs (env python3, not a baked-in path)"
grep -rIl '^#!.*python3' "$VENV/bin" | while read -r f; do
    sed -i "1s|^#!.*|#!/usr/bin/env python3|" "$f"
done

# ---------------------------------------------------------------------------
# 5. Package.
# ---------------------------------------------------------------------------
step "Packaging the AppImage (appimagetool)"
substep "appimagetool has its own live output below"
OUT_FILE="$OUT_DIR/melband-roformer-${VERSION}-x86_64.AppImage"
ARCH=x86_64 "$APPIMAGETOOL" --appimage-extract-and-run "$APPDIR" "$OUT_FILE"

BUILD_ELAPSED=$((SECONDS - BUILD_START))
echo
echo "==> done in $((BUILD_ELAPSED / 60))m $((BUILD_ELAPSED % 60))s: $OUT_FILE"
du -h "$OUT_FILE"
