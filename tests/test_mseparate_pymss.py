"""CPU-only contract tests for the optional pymss model-family adapters."""
from __future__ import annotations

import hashlib
import re
import sys
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import soundfile as sf

from mseparate.bs_roformer import BSRoformerAdapter
from mseparate.mdx23c import MDX23CAdapter
from mseparate.types import ModelAsset, ModelManifest


def _manifest(tmp_path: Path, backend: str) -> ModelManifest:
    assets = []
    for role in ("checkpoint", "config"):
        path = tmp_path / f"{role}.bin"
        payload = f"fixture-{role}".encode()
        path.write_bytes(payload)
        assets.append(ModelAsset(
            role=role,
            url=f"https://example.invalid/{role}",
            relative_path=path.name,
            sha256=hashlib.sha256(payload).hexdigest(),
            size_bytes=len(payload),
            required=True,
        ))
    return ModelManifest(
        model_id=f"fixture-{backend}",
        name="Fixture vocal model",
        architecture="fixture",
        backend=backend,
        source_url="https://example.invalid/model",
        license="MIT",
        license_url=None,
        license_status="verified",
        assets=tuple(assets),
        native_stems=("vocals",),
        derived_stems=("instrumental",),
        target_stems=("vocals", "instrumental"),
        sample_rate=44100,
        channels="stereo",
        input_formats=("wav",),
        category="vocals",
        status="available",
    )


def test_bs_roformer_adapter_loads_verified_assets_and_writes_vocals(tmp_path, monkeypatch, capsys):
    state = {}

    class FakeSeparator:
        def __init__(self, **kwargs):
            state["kwargs"] = kwargs
            state["closed"] = False

        def separate(self, mix, *, pbar, stems):
            state["mix_shape"] = mix.shape
            state["stems"] = tuple(stems)
            state["pbar"] = pbar
            callback = state["kwargs"]["progress_callback"]
            callback(0, 48, "Processing audio")
            callback(24, 48, "Processing audio")
            sys.stderr.write("\nRuntime warning: fixture warning\n")
            return {stem: np.full((mix.shape[1], mix.shape[0]), 0.125, dtype=np.float32)
                    for stem in stems}

        def close(self):
            state["closed"] = True

    monkeypatch.setitem(sys.modules, "pymss", SimpleNamespace(MSSeparator=FakeSeparator))
    manifest = _manifest(tmp_path, "bs-roformer-pymss")
    source = tmp_path / "input.wav"
    sf.write(source, np.full((4096, 2), 0.05, dtype=np.float32), 44100, subtype="FLOAT")

    adapter = BSRoformerAdapter()
    assert adapter.compatibility(manifest) == (True, "")
    adapter.load(manifest, tmp_path, "cpu")
    outputs = adapter.infer(source, tmp_path / "out", ("vocals", "instrumental"))

    captured = capsys.readouterr()
    assert captured.out == ""
    assert "Processing audio:" in captured.err
    assert "50%" in captured.err
    assert re.search(r"Processing audio:\\s+0%", captured.err) is None
    assert "Runtime warning: fixture warning" in captured.err
    assert state["pbar"] is False
    assert state["kwargs"]["model_type"] == "bs_roformer"
    assert state["kwargs"]["device"] == "cpu"
    assert state["stems"] == ("vocals",)
    assert state["mix_shape"] == (2, 4096)
    assert len(outputs) == 2
    samples, rate = sf.read(tmp_path / "out" / "vocals.wav", dtype="float32", always_2d=True)
    residual, residual_rate = sf.read(
        tmp_path / "out" / "instrumental.wav", dtype="float32", always_2d=True
    )
    assert rate == residual_rate == 44100
    assert samples.shape == residual.shape == (4096, 2)
    assert np.allclose(samples, 0.125)
    assert np.allclose(residual, -0.075, atol=1e-6)
    adapter.unload()
    assert state["closed"] is True


def test_mdx23c_adapter_uses_mdx23c_runtime_type(tmp_path, monkeypatch):
    state = {}

    class FakeSeparator:
        def __init__(self, **kwargs):
            state["kwargs"] = kwargs

        def separate(self, mix, *, pbar, stems):
            return {stem: np.zeros((mix.shape[1], mix.shape[0]), dtype=np.float32)
                    for stem in stems}

        def close(self):
            state["closed"] = True

    monkeypatch.setitem(sys.modules, "pymss", SimpleNamespace(MSSeparator=FakeSeparator))
    manifest = _manifest(tmp_path, "mdx23c-pymss")
    source = tmp_path / "input.wav"
    sf.write(source, np.zeros((2048, 2), dtype=np.float32), 44100)

    adapter = MDX23CAdapter()
    adapter.load(manifest, tmp_path, "cpu")
    outputs = adapter.infer(source, tmp_path / "mdx-out", ("vocals", "instrumental"))

    assert state["kwargs"]["model_type"] == "mdx23c"
    assert len(outputs) == 2
    adapter.unload()
    assert state["closed"] is True


def test_pymss_adapter_rejects_manifest_for_another_backend(tmp_path):
    manifest = _manifest(tmp_path, "mdx23c-pymss")
    compatible, reason = BSRoformerAdapter().compatibility(manifest)
    assert compatible is False
    assert "bs-roformer-pymss" in reason


def test_pymss_adapter_rejects_corrupt_or_unpinned_asset(tmp_path, monkeypatch):
    manifest = _manifest(tmp_path, "bs-roformer-pymss")
    checkpoint = tmp_path / "checkpoint.bin"
    checkpoint.write_bytes(b"corrupt")
    monkeypatch.setitem(
        sys.modules, "pymss", SimpleNamespace(MSSeparator=lambda **kwargs: None)
    )
    adapter = BSRoformerAdapter()
    try:
        adapter.load(manifest, tmp_path, "cpu")
    except RuntimeError as exc:
        assert "failed SHA-256/size verification" in str(exc)
    else:
        raise AssertionError("corrupt checkpoint must not reach pymss")

