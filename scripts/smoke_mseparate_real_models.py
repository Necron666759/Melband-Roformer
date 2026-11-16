#!/usr/bin/env python3
"""Download, hash-check, strict-load and run the real MVSep Synth/Keys weights.

This is an opt-in integration smoke test, not part of the fast CPU unit suite.
The runtime's PyTorch load_state_dict() uses strict=True by default.
"""
from __future__ import annotations

import argparse
import hashlib
import tempfile
from pathlib import Path

import numpy as np
import soundfile as sf

from mseparate.assets import ensure_model_assets
from mseparate.bs_roformer import BSRoformerAdapter
from mseparate.registry import bundled_registry_path, load_registry


MODEL_IDS = (
    "mvsep-mega53-synth-specialist-candidate",
    "mvsep-mega53-keys-specialist-candidate",
)


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def make_fixture(path: Path, seconds: float = 8.0, sample_rate: int = 44100) -> None:
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


def run_model(manifest, models_root: Path, fixture: Path, output_root: Path, device: str) -> None:
    print(f"\n=== {manifest.model_id} ({device}) ===", flush=True)
    results = ensure_model_assets(manifest, models_root, timeout=60.0, retries=3)
    failures = [item for item in results if item.status not in {"cached", "downloaded", "recovered"}]
    if failures:
        raise RuntimeError("; ".join(
            f"{item.role}: {item.status}: {item.message or 'asset verification failed'}"
            for item in failures
        ))

    model_dir = models_root / manifest.model_id
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
        # MSSeparator constructs the real BS-RoFormer from the pinned YAML and
        # checkpoint. PyTorch load_state_dict defaults to strict=True.
        adapter.load(manifest, model_dir, device)
        outputs = adapter.infer(fixture, model_output, manifest.target_stems)
        if len(outputs) != 1 or outputs[0].stem != manifest.target_stems[0]:
            raise RuntimeError(f"unexpected output contract: {outputs!r}")
        samples, sample_rate = sf.read(outputs[0].path, dtype="float32", always_2d=True)
        if sample_rate != manifest.sample_rate or samples.shape != (8 * sample_rate, 2):
            raise RuntimeError(
                f"invalid output shape/rate: shape={samples.shape}, rate={sample_rate}"
            )
        if not np.isfinite(samples).all() or float(np.sqrt(np.mean(samples ** 2))) <= 1e-7:
            raise RuntimeError("output is silent or contains NaN/Inf")
        print(
            f"PASS: {outputs[0].stem}, {samples.shape}, {sample_rate} Hz, "
            f"RMS={float(np.sqrt(np.mean(samples ** 2))):.6f}",
            flush=True,
        )
    finally:
        adapter.unload()


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--device", choices=("cpu", "cuda"), default="cpu")
    parser.add_argument("--models", nargs="*", default=list(MODEL_IDS), choices=MODEL_IDS)
    args = parser.parse_args()

    registry = load_registry(bundled_registry_path())
    with tempfile.TemporaryDirectory(prefix="mseparate-real-model-smoke-") as temp:
        root = Path(temp)
        fixture = root / "fixture.wav"
        make_fixture(fixture)
        for model_id in args.models:
            run_model(registry[model_id], root / "models", fixture, root / "outputs", args.device)
    print("\nAll requested real-model smoke tests passed.", flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
