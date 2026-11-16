#!/usr/bin/env python3
"""Download, hash-check, strict-load and run real MVSep Mega 53 specialist weights.

This is an opt-in integration smoke test, not part of the fast CPU unit suite.
The runtime's PyTorch load_state_dict() uses strict=True by default.
"""
from __future__ import annotations

import argparse
import hashlib
import tempfile
from dataclasses import replace
from pathlib import Path
from urllib.request import Request, urlopen

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
    parser.add_argument("--models", nargs="*", default=list(MODEL_IDS), choices=MODEL_IDS)
    parser.add_argument("--discover-checksums", action="store_true", help="Download unpinned assets, calculate SHA-256/size, then run real inference")
    parser.add_argument("--low-vram-profile", action="store_true", help="Test the experimental 10-second chunk profile on CUDA; this does not enable it for automatic routing")
    args = parser.parse_args()

    registry = load_registry(bundled_registry_path())
    with tempfile.TemporaryDirectory(prefix="mseparate-real-model-smoke-") as temp:
        root = Path(temp)
        fixture = root / "fixture.wav"
        make_fixture(fixture)
        for model_id in args.models:
            run_model(registry[model_id], root / "models", fixture, root / "outputs", args.device, args.discover_checksums, args.low_vram_profile)
    print("\nAll requested real-model smoke tests passed.", flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
