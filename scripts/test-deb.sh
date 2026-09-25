#!/bin/bash
# Installs the most recently built melband-roformer_*.deb and runs smoke
# tests against it. Intended to be run on the actual target machine
# (Debian 13 amd64, ideally with the RTX 3060 Ti + NVIDIA driver present,
# though it degrades gracefully to CPU-only checks if not).
#
# This script requires root for the apt install/remove steps; the actual
# smoke tests run as the invoking (non-root) user, since normal usage of
# melband-roformer must not require root (per the task's own requirement).
set -euo pipefail

cd "$(dirname "$0")/.."

# scripts/build-deb.sh collects its output into ./out/ (next to the
# sources); fall back to the parent directory too, in case the .deb was
# built directly with dpkg-buildpackage instead of via that script.
DEB="$(ls -t out/melband-roformer_*.deb ../melband-roformer_*.deb 2>/dev/null | head -n1 || true)"
if [ -z "$DEB" ]; then
    echo "No melband-roformer_*.deb found in ./out/ or ../ -- run scripts/build-deb.sh first." >&2
    exit 1
fi
echo "==> Using package: $DEB"

echo "==> Installing (sudo)..."
sudo apt-get install -y "$DEB"

fail=0

echo "==> [1/6] melband-roformer --version"
melband-roformer --version || fail=1

echo "==> [2/6] melband-roformer --info"
melband-roformer --info || fail=1

echo "==> [3/6] melband-roformer --list-models"
melband-roformer --list-models | head -n 5 || fail=1

echo "==> [4/6] melband-roformer --self-test"
melband-roformer --self-test || echo "(non-zero exit is expected if the model hasn't been downloaded yet; see output above)"

echo "==> [5/6] man melband-roformer"
man -P cat melband-roformer >/dev/null || fail=1

echo "==> [6/6] Desktop/icon files present"
[ -f /usr/share/applications/melband-roformer.desktop ] || { echo "Missing .desktop file"; fail=1; }
[ -f /usr/share/icons/hicolor/scalable/apps/melband-roformer.svg ] || { echo "Missing icon"; fail=1; }

echo
if [ "$fail" -eq 0 ]; then
    echo "Smoke tests PASSED."
else
    echo "Smoke tests FAILED -- see output above." >&2
fi

read -r -p "Remove the package now? [y/N] " ans
if [ "${ans,,}" = "y" ]; then
    sudo apt-get remove -y melband-roformer
fi

exit "$fail"
