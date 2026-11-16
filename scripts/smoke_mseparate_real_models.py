#!/usr/bin/env python3
"""Download, hash-check, strict-load and run real MVSep Mega 53 specialist weights.

This is an opt-in integration smoke test, not part of the fast CPU unit suite.
The runtime's PyTorch load_state_dict() uses strict=True by default.
"""
from __future__ import annotations

import argparse
import hashlib
import importlib.util
import os
import shutil
import subprocess
import sys
import tarfile
import tempfile
from dataclasses import replace
from pathlib import Path
from urllib.request import Request, urlopen

# Make the source checkout importable when this file is launched directly
# from scripts/ (without requiring PYTHONPATH=app).
APP_DIR = Path(__file__).resolve().parents[1] / "app"
if str(APP_DIR) not in sys.path:
    sys.path.insert(0, str(APP_DIR))


def _missing_runtime_modules() -> list[str]:
    required = ("numpy", "soundfile", "torch", "pymss", "tqdm", "soxr")
    return [name for name in required if importlib.util.find_spec(name) is None]


def _bootstrap_runtime() -> None:
    """Re-exec with a cached, isolated CPython + the application's pinned MSS runtime.

    The smoke test is intentionally self-contained: a fresh Debian/Ubuntu
    checkout needs only Python 3 and network access. The portable interpreter
    is the same python-build-standalone release used by build-appimage.sh.
    """
    requested_device = "cpu"
    for index, argument in enumerate(sys.argv[1:], start=1):
        if argument.startswith("--device="):
            requested_device = argument.split("=", 1)[1]
            break
        if argument == "--device" and index + 1 < len(sys.argv):
            requested_device = sys.argv[index + 1]
            break

    missing = _missing_runtime_modules()
    cpu_only_torch = False
    if "torch" not in missing and requested_device == "cuda":
        try:
            import torch as _torch
            cpu_only_torch = _torch.version.cuda is None
        except Exception:
            cpu_only_torch = True

    if not missing and not cpu_only_torch:
        return
    if os.environ.get("MSEPARATE_SMOKE_BOOTSTRAPPED") == "1":
        raise RuntimeError(
            "The isolated smoke-test Python is still missing runtime modules: "
            + ", ".join(_missing_runtime_modules())
        )

    cache_home = Path(os.environ.get("XDG_CACHE_HOME", Path.home() / ".cache"))
    cache_root = cache_home / "mseparate-smoke"
    python_root = cache_root / "python"
    python = python_root / "bin" / "python3"
    if not python.is_file():
        cache_root.mkdir(parents=True, exist_ok=True)
        asset = "cpython-3.13.15+20260825-x86_64-unknown-linux-gnu-install_only_stripped.tar.gz"
        archive = Path(__file__).resolve().parents[1] / ".appimage-build" / "tools" / asset
        if not archive.is_file():
            archive = cache_root / asset
            if not archive.is_file():
                url = (
                    "https://github.com/astral-sh/python-build-standalone/releases/"
                    "download/20260825/cpython-3.13.15%2B20260825-x86_64-unknown-linux-gnu-install_only_stripped.tar.gz"
                )
                print(f"Downloading portable CPython 3.13.15: {url}", flush=True)
                request = Request(url, headers={"User-Agent": "mseparate-real-model-smoke/1.0"})
                with urlopen(request, timeout=60) as response, archive.open("wb") as output:
                    while True:
                        block = response.read(1024 * 1024)
                        if not block:
                            break
                        output.write(block)

        expected_archive_sha256 = "8af9a8214c71b2dd698005e39fab87aad02a994330508857da4e6d1ba7e6ddb6"
        archive_hash = hashlib.sha256()
        with archive.open("rb") as archive_stream:
            for block in iter(lambda: archive_stream.read(1024 * 1024), b""):
                archive_hash.update(block)
        actual_archive_sha256 = archive_hash.hexdigest()
        if actual_archive_sha256 != expected_archive_sha256:
            raise RuntimeError(
                "Portable CPython archive SHA-256 mismatch: "
                f"expected {expected_archive_sha256}, got {actual_archive_sha256}"
            )

        print(f"Preparing isolated Python runtime in {python_root}", flush=True)
        extract_root = cache_root / "extract"
        shutil.rmtree(extract_root, ignore_errors=True)
        extract_root.mkdir(parents=True, exist_ok=True)
        try:
            with tarfile.open(archive, "r:gz") as bundle:
                bundle.extractall(extract_root, filter="data")
            extracted_python = extract_root / "python"
            if not (extracted_python / "bin" / "python3").is_file():
                raise RuntimeError("Portable CPython archive has an unexpected layout")
            shutil.rmtree(python_root, ignore_errors=True)
            shutil.move(str(extracted_python), str(python_root))
        finally:
            shutil.rmtree(extract_root, ignore_errors=True)

    if not python.is_file():
        raise RuntimeError(f"Could not prepare portable Python at {python}")

    print(
        "Installing the pinned smoke-test runtime (PyTorch 2.13.0 CUDA 13.0, "
        "torchaudio 2.11.0, pymss 2.3.5). First run downloads several GB; "
        "subsequent runs reuse the cache.",
        flush=True,
    )
    subprocess.run(
        [
            str(python), "-m", "pip", "install",
            "--disable-pip-version-check",
            "--index-url", "https://download.pytorch.org/whl/cu130",
            "--extra-index-url", "https://pypi.org/simple",
            "torch==2.13.0", "torchaudio==2.11.0", "pymss==2.3.5",
            "numpy", "soundfile", "tqdm", "soxr",
        ],
        check=True,
    )
    env = os.environ.copy()
    env["MSEPARATE_SMOKE_BOOTSTRAPPED"] = "1"
    current_pythonpath = env.get("PYTHONPATH", "")
    env["PYTHONPATH"] = str(APP_DIR) + (os.pathsep + current_pythonpath if current_pythonpath else "")
    os.execve(str(python), [str(python), str(Path(__file__).resolve()), *sys.argv[1:]], env)


_bootstrap_runtime()

import numpy as np
import soundfile as sf

from mseparate.assets import ensure_model_assets
from mseparate.bs_roformer import BSRoformerAdapter
from mseparate.registry import bundled_registry_path, load_registry


MODEL_IDS = (
    "mvsep-mega53-synth-specialist-candidate",
    "mvsep-mega53-keys-specialist-candidate",
    "mvsep-mega53-bass-specialist-candidate",
    "mvsep-mega53-kick-specialist-candidate",
    "mvsep-mega53-drums-specialist-candidate",
    "mvsep-mega53-organ-specialist-candidate",
    "mvsep-mega53-digital-piano-specialist-candidate",
    "mvsep-mega53-percussion-specialist-candidate",
    "mvsep-mega53-hh-specialist-candidate",
    "zfturbo-bs-roformer-53stem-candidate",
)


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def make_fixture(path: Path, seconds: float = 20.0, sample_rate: int = 44100) -> None:
    frames = int(seconds * sample_rate)
    time = np.arange(frames, dtype=np.float32) / sample_rate
    left = (
        0.16 * np.sin(2 * np.pi * 220.0 * time)
        + 0.11 * np.sin(2 * np.pi * 440.0 * time)
        + 0.07 * np.sin(2 * np.pi * 880.0 * time)
    )
    right = (
        0.15 * np.sin(2 * np.pi * 221.0 * time)
        + 0.10 * np.sin(2 * np.pi * 441.0 * time)
        + 0.06 * np.sin(2 * np.pi * 881.0 * time)
    )
    sf.write(path, np.column_stack((left, right)), sample_rate, subtype="FLOAT")


def download_and_pin_assets(manifest, model_dir: Path):
    """Bootstrap hashes for an unpinned candidate, then verify its local bytes."""
    model_dir.mkdir(parents=True, exist_ok=True)
    pinned_assets = []
    for asset in manifest.assets:
        path = model_dir / asset.relative_path
        path.parent.mkdir(parents=True, exist_ok=True)
        request = Request(asset.url, headers={"User-Agent": "mseparate-real-model-smoke/1.0"})
        with urlopen(request, timeout=120) as response, path.open("wb") as output:
            while True:
                block = response.read(1024 * 1024)
                if not block:
                    break
                output.write(block)
        digest = sha256(path)
        size = path.stat().st_size
        if asset.sha256 and digest != asset.sha256:
            raise RuntimeError(f"{asset.role} digest mismatch: expected {asset.sha256}, got {digest}")
        if asset.size_bytes and size != asset.size_bytes:
            raise RuntimeError(f"{asset.role} size mismatch: expected {asset.size_bytes}, got {size}")
        print(f"DISCOVERED {manifest.model_id} {asset.role}: {size} bytes sha256={digest}", flush=True)
        pinned_assets.append(replace(asset, sha256=digest, size_bytes=size))
    return replace(manifest, assets=tuple(pinned_assets))


def run_model(manifest, models_root: Path, fixture: Path, output_root: Path, device: str,
              discover_checksums: bool = False, low_vram_profile: bool = False) -> None:
    is_mega53 = manifest.model_id == "zfturbo-bs-roformer-53stem-candidate"
    if low_vram_profile and not is_mega53:
        raise ValueError("--low-vram-profile applies only to the ZFTurbo 53-stem model")
    if low_vram_profile and device != "cuda":
        raise ValueError("--low-vram-profile must be validated on CUDA, not CPU")
    if low_vram_profile:
        manifest = replace(
            manifest,
            inference_params={
                "batch_size": 1, "chunk_size": 441000,
                "overlap_size": 220500, "normalize": False,
            },
        )
        print("Using experimental low-VRAM profile: 10-second chunks.", flush=True)
    print(f"\\n=== {manifest.model_id} ({device}) ===", flush=True)
    model_dir = models_root / manifest.model_id
    if discover_checksums:
        manifest = download_and_pin_assets(manifest, model_dir)
    else:
        results = ensure_model_assets(manifest, models_root, timeout=120.0, retries=3)
        failures = [item for item in results if item.status not in {"cached", "downloaded", "recovered"}]
        if failures:
            raise RuntimeError("; ".join(
                f"{item.role}: {item.status}: {item.message or 'asset verification failed'}"
                for item in failures
            ))

    for asset in manifest.assets:
        path = model_dir / asset.relative_path
        actual = sha256(path)
        if actual != asset.sha256:
            raise RuntimeError(
                f"{asset.role} digest mismatch after download: expected {asset.sha256}, got {actual}"
            )
        print(f"{asset.role}: {path.stat().st_size} bytes sha256={actual}", flush=True)

    adapter = BSRoformerAdapter()
    compatible, reason = adapter.compatibility(manifest)
    if not compatible:
        raise RuntimeError(f"adapter rejected manifest: {reason}")

    model_output = output_root / manifest.model_id
    try:
        if device == "cuda":
            import torch
            torch.cuda.reset_peak_memory_stats(0)
            free_bytes, _ = torch.cuda.mem_get_info(0)
            total_bytes = torch.cuda.get_device_properties(0).total_memory
            print(
                "CUDA before load: free={:.2f} GiB / total={:.2f} GiB".format(
                    free_bytes / (1024 ** 3), total_bytes / (1024 ** 3)
                ),
                flush=True,
            )
        adapter.load(manifest, model_dir, device)
        smoke_stems = (
            ("bass", "keys", "synth", "vocal")
            if is_mega53
            else manifest.target_stems
        )
        outputs = adapter.infer(fixture, model_output, smoke_stems)
        expected_stems = set(smoke_stems)
        actual_stems = {item.stem for item in outputs}
        if actual_stems != expected_stems:
            raise RuntimeError(
                f"unexpected output stems: expected {sorted(expected_stems)}, got {sorted(actual_stems)}"
            )
        non_silent_outputs = []
        for item in outputs:
            samples, sample_rate = sf.read(item.path, dtype="float32", always_2d=True)
            if sample_rate != manifest.sample_rate or samples.shape != (20 * sample_rate, 2):
                raise RuntimeError(
                    f"invalid {item.stem} output shape/rate: shape={samples.shape}, rate={sample_rate}"
                )
            rms = float(np.sqrt(np.mean(samples ** 2)))
            if not np.isfinite(samples).all():
                raise RuntimeError(f"{item.stem} output contains NaN/Inf")
            if rms > 1e-7:
                non_silent_outputs.append(item.stem)
            elif not is_mega53:
                raise RuntimeError(f"{item.stem} output is silent")
            print(
                f"PASS: {item.stem}, {samples.shape}, {sample_rate} Hz, RMS={rms:.6f}",
                flush=True,
            )
        if is_mega53 and not non_silent_outputs:
            raise RuntimeError("All selected 53-stem outputs were silent")
        if device == "cuda":
            import torch
            torch.cuda.synchronize()
            print(
                "CUDA peak memory: allocated={:.2f} GiB, reserved={:.2f} GiB".format(
                    torch.cuda.max_memory_allocated(0) / (1024 ** 3),
                    torch.cuda.max_memory_reserved(0) / (1024 ** 3),
                ),
                flush=True,
            )
    finally:
        adapter.unload()


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--device", choices=("cpu", "cuda"), default="cpu")
    parser.add_argument("--models-dir", type=Path, default=None, help="Persistent model cache directory; defaults to a temporary directory")
    parser.add_argument("--models", nargs="*", default=list(MODEL_IDS), choices=MODEL_IDS)
    parser.add_argument("--discover-checksums", action="store_true", help="Download unpinned assets, calculate SHA-256/size, then run real inference")
    parser.add_argument("--low-vram-profile", action="store_true", help="Test the experimental 10-second chunk profile on CUDA; this does not enable it for automatic routing")
    args = parser.parse_args()

    if args.device == "cuda":
        import torch
        if not torch.cuda.is_available():
            raise SystemExit(
                "CUDA is unavailable in the smoke-test runtime. This test uses the "
                "application's PyTorch 2.13.0+cu130 build and requires an NVIDIA "
                "driver >= 580.65.06. Check nvidia-smi and update the driver if needed. "
                f"torch={torch.__version__}, torch CUDA build={torch.version.cuda!r}"
            )
        props = torch.cuda.get_device_properties(0)
        free_bytes, total_bytes = torch.cuda.mem_get_info(0)
        print(
            f"CUDA device: {props.name}; VRAM total={total_bytes / (1024 ** 3):.2f} GiB, "
            f"free={free_bytes / (1024 ** 3):.2f} GiB; "
            f"PyTorch={torch.__version__}, CUDA runtime={torch.version.cuda}",
            flush=True,
        )

    registry = load_registry(bundled_registry_path())
    with tempfile.TemporaryDirectory(prefix="mseparate-real-model-smoke-") as temp:
        root = Path(temp)
        fixture = root / "fixture.wav"
        models_root = args.models_dir.expanduser() if args.models_dir else root / "models"
        models_root.mkdir(parents=True, exist_ok=True)
        make_fixture(fixture)
        for model_id in args.models:
            run_model(registry[model_id], models_root, fixture, root / "outputs", args.device, args.discover_checksums, args.low_vram_profile)
    print("\nAll requested real-model smoke tests passed.", flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
