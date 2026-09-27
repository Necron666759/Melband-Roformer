#!/bin/bash
# Build melband-roformer_*.deb from a clean Debian 13 (Trixie) amd64
# machine. Requires network access (PyPI + download.pytorch.org) -- see
# README.debian.md for why that's a build-time-only requirement, not an
# install-time one. Does NOT require root, except for the one apt-get
# invocation that installs build dependencies (clearly separated below so
# you can skip it if you already have them).
#
# Usage:
#   ./scripts/build-deb.sh            # installs build-deps via sudo, then builds
#   ./scripts/build-deb.sh --no-deps  # skip apt-get, assume deps already present
#
# If invoked as root (e.g. `sudo ./scripts/build-deb.sh`), only the
# apt-get step runs as root; the actual build (venv, pip, dpkg-buildpackage)
# is re-exec'd as $SUDO_USER, matching debian/rules' "Rules-Requires-Root:
# no" -- running the build itself as root would leave the source tree,
# .venv-build/, and the resulting .deb/.buildinfo/.changes root-owned, and
# can also poison ~/.cache/pip with root-owned entries for the real user.
#
# If invoked as a genuine root login (`su -`, direct root shell -- no
# $SUDO_USER set, unlike `sudo`), the script instead drops to whoever
# already owns this checkout, on the same reasoning. Only if that user
# can't be determined (project dir owned by root/unknown, or `sudo` isn't
# installed) does the build actually run as root, with a warning.
set -euo pipefail

SCRIPT_PATH="$(readlink -f "$0")"
cd "$(dirname "$SCRIPT_PATH")/.."
PROJECT_ROOT="$(pwd)"

# Fix ownership of a directory (recursively) if any file under it is not
# owned by $1, but only bother chown-ing if such a file actually exists
# (avoids a no-op recursive chown, and its log line, on every normal run).
# Used for every place build state can accumulate: the project directory
# itself, the target user's pip/other caches, and pytest's own tmp root
# -- all of which can end up root-owned by an *earlier* build that ran as
# plain root (before this drop-privileges logic existed, or via a direct
# `sudo dpkg-buildpackage`/`sudo pytest`/etc. that bypassed this script),
# and then break (permission-denied, sometimes fixture-level pytest
# errors, sometimes a silent "pip cache disabled" warning) once the build
# correctly drops to a non-root user.
_fix_stale_root_ownership() {
    local target_user="$1" dir="$2"
    [ -e "$dir" ] || return 0
    if find "$dir" \! -user "$target_user" -print -quit | grep -q .; then
        echo "==> Found root-owned files under $dir (left over from an" \
             "earlier root-run build) -- fixing ownership..."
        chown -R "$target_user:$(id -gn "$target_user")" "$dir"
    fi
}

# All the places a stale root-owned leftover can break a later
# non-root-run build: the project checkout itself (out/, .venv-build/,
# dpkg-buildpackage's debian/melband-roformer/, debian/.debhelper/,
# debian/files, etc.), the target user's whole ~/.cache (pip's wheel/http
# cache is the one we've actually hit, but this covers any other cache
# tooling ever writes there too), and pytest's own tmp root
# (/tmp/pytest-of-<user>, created by debian/rules' override_dh_auto_test).
_fix_stale_root_ownership_everywhere() {
    local target_user="$1" target_home="$2"
    _fix_stale_root_ownership "$target_user" "$PROJECT_ROOT"
    [ -n "$target_home" ] && _fix_stale_root_ownership "$target_user" "$target_home/.cache"
    _fix_stale_root_ownership "$target_user" "${TMPDIR:-/tmp}/pytest-of-$target_user"
}

BUILD_DEPS=(
    build-essential
    debhelper
    python3
    python3-venv
    python3-pip
    python3-pytest
    ca-certificates
    devscripts
    dpkg-dev
)

# True only if every package in BUILD_DEPS is already installed. Lets the
# caller skip apt-get update entirely on a machine that already has them --
# apt-get update refreshes *every* configured source (this project's own
# InRelease fetch list routinely includes unrelated third-party repos such
# as browser/remote-desktop/GPU-vendor ones that ship none of these
# packages), so it's a real, avoidable network round trip on every build
# when nothing is actually missing, not just log noise.
_all_build_deps_installed() {
    local pkg
    for pkg in "${BUILD_DEPS[@]}"; do
        dpkg-query -W -f='${Status}' "$pkg" 2>/dev/null | grep -q "^install ok installed$" || return 1
    done
    return 0
}

if [ "$(id -u)" -eq 0 ]; then
    if [ "${1:-}" != "--no-deps" ]; then
        if _all_build_deps_installed; then
            echo "==> All build dependencies already installed; skipping apt-get" \
                 "update (avoids an unnecessary refresh of every configured" \
                 "apt source, including unrelated third-party repos)."
        else
            echo "==> Installing build dependencies (running as root already)..."
            apt-get update
            apt-get install -y "${BUILD_DEPS[@]}"
        fi
    else
        echo "==> Skipping apt-get (--no-deps given); assuming build-deps are present."
    fi

    if [ -n "${SUDO_USER:-}" ] && [ "$SUDO_USER" != "root" ]; then
        sudo_user_home="$(getent passwd "$SUDO_USER" | cut -d: -f6)"
        _fix_stale_root_ownership_everywhere "$SUDO_USER" "$sudo_user_home"

        echo "==> Dependencies installed. Dropping root and re-running the build as $SUDO_USER..."
        exec sudo -u "$SUDO_USER" -H "$SCRIPT_PATH" --no-deps
    fi

    # No usable $SUDO_USER -- e.g. a genuine root login or `su -`, not
    # `sudo ./build-deb.sh` (plain `su`/direct root login never sets
    # $SUDO_USER; only `sudo` itself does, `sudo -i` included). Fall back
    # to a heuristic: whoever already owns this checkout is overwhelmingly
    # likely to be the intended non-root build user (they cloned/own it),
    # so drop to them too, for the same reason as the $SUDO_USER case
    # above -- avoid leaving root-owned build artifacts that break a
    # later `sudo`-based build. Only if `sudo` is actually installed (a
    # bare-root box may not have it) and the owner is a real, non-root
    # account; otherwise there is no safe target to drop to and we fall
    # through to building as root, as before.
    project_owner="$(stat -c '%U' "$PROJECT_ROOT" 2>/dev/null || true)"
    if command -v sudo >/dev/null 2>&1 \
       && [ -n "$project_owner" ] && [ "$project_owner" != "root" ] \
       && [ "$project_owner" != "UNKNOWN" ] \
       && id -u "$project_owner" >/dev/null 2>&1; then
        echo "==> No sudo invocation detected (direct root login), but"
        echo "    $PROJECT_ROOT is owned by '$project_owner'. Dropping to"
        echo "    that user for the actual build instead of building as"
        echo "    root (avoids root-owned build artifacts -- see the"
        echo "    comment at the top of this script)."
        target_home="$(getent passwd "$project_owner" | cut -d: -f6)"
        _fix_stale_root_ownership_everywhere "$project_owner" "$target_home"
        exec sudo -u "$project_owner" -H "$SCRIPT_PATH" --no-deps
    fi

    echo "WARNING: running as root with no non-root user to drop to (logged in" >&2
    echo "as root directly, rather than via 'sudo ./scripts/build-deb.sh', and" >&2
    echo "the project directory's owner is root or unresolvable)." >&2
    echo "Continuing the build as root -- this works, but debian/rules does" >&2
    echo "not require it, and the resulting files/venv/.deb will end up" >&2
    echo "root-owned." >&2
else
    if [ "${1:-}" != "--no-deps" ]; then
        if _all_build_deps_installed; then
            echo "==> All build dependencies already installed; skipping apt-get" \
                 "update (avoids an unnecessary refresh of every configured" \
                 "apt source, including unrelated third-party repos)."
        else
            echo "==> Installing build dependencies (requires sudo)..."
            sudo apt-get update
            sudo apt-get install -y "${BUILD_DEPS[@]}"
        fi
    else
        echo "==> Skipping apt-get (--no-deps given); assuming build-deps are present."
    fi
fi

echo "==> Checking required build tools..."
for tool in dpkg-buildpackage python3; do
    command -v "$tool" >/dev/null || { echo "Missing required tool: $tool" >&2; exit 1; }
done

echo "==> Checking network reachability of package indexes (build-time only)..."
if ! python3 -c "import urllib.request; urllib.request.urlopen('https://pypi.org/simple/', timeout=10)" 2>/dev/null; then
    echo "WARNING: could not reach pypi.org. The build will fail at the" >&2
    echo "'pip install' step inside debian/rules if this machine has no" >&2
    echo "network access. This is expected/required only at BUILD time." >&2
fi

echo "==> Building with dpkg-buildpackage -us -uc (no root needed for the build itself)..."
dpkg-buildpackage -us -uc -b

# dpkg-buildpackage always drops its output one directory *above* the
# source tree (../melband-roformer_*.deb etc.) -- that's fine for a single
# checkout, but it means the built package ends up sitting next to
# whatever else happens to be in the parent directory, with nothing
# marking it as "the output of this project". Collect it into ./out/
# instead, right next to the sources, so there's exactly one place to
# look after a build.
OUT_DIR="$PROJECT_ROOT/out"
mkdir -p "$OUT_DIR"
shopt -s nullglob
artifacts=(../melband-roformer_*.deb ../melband-roformer_*.changes ../melband-roformer_*.buildinfo)
shopt -u nullglob

if [ "${#artifacts[@]}" -eq 0 ]; then
    echo "No build artifacts found in the parent directory -- check the build log above." >&2
    exit 1
fi

mv -f "${artifacts[@]}" "$OUT_DIR/"

echo
echo "==> Build finished. Output collected in: $OUT_DIR"
ls -la "$OUT_DIR"

echo
echo "Install with:"
echo "  sudo apt install $(ls "$OUT_DIR"/melband-roformer_*.deb | head -n1)"
