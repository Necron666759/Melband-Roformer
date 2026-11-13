"""Tests for the isolated MelBand wrapper adapter; no Torch import or GPU."""
from __future__ import annotations

from pathlib import Path

import pytest

from mseparate.melband import MelBandCLIAdapter
from mseparate.types import ModelAsset, ModelManifest


def _manifest():
    return ModelManifest(
        model_id="melband-fixture", name="MelBand fixture",
        architecture="MelBand RoFormer", backend="melband-roformer",
        source_url="https://example.invalid/project", license="MIT",
        license_url=None, license_status="verified",
        assets=(
            ModelAsset("checkpoint", "https://example.invalid/model", "model.ckpt", "a" * 64),
            ModelAsset("config", "https://example.invalid/config", "config.yaml", "b" * 64),
        ),
        native_stems=("vocals", "instrumental"), target_stems=("vocals", "instrumental"),
        status="available",
    )


def test_melband_adapter_checks_manifest_and_assets(tmp_path, monkeypatch):
    manifest = _manifest()
    adapter = MelBandCLIAdapter()
    assert adapter.compatibility(manifest) == (True, "")

    monkeypatch.setattr("mseparate.melband.shutil.which", lambda _: "/venv/bin/melband-roformer")
    model_dir = tmp_path / manifest.model_id
    model_dir.mkdir()
    (model_dir / "model.ckpt").write_bytes(b"checkpoint")
    (model_dir / "config.yaml").write_text("training: {}")
    adapter.load(manifest, model_dir, "cpu")
    assert adapter.info().name == "melband-roformer-cli"
    assert adapter.info().details["isolation"] == "subprocess"


def test_melband_adapter_invokes_wrapper_and_maps_declared_outputs(tmp_path, monkeypatch):
    manifest = _manifest()
    model_dir = tmp_path / manifest.model_id
    model_dir.mkdir()
    (model_dir / "model.ckpt").write_bytes(b"checkpoint")
    (model_dir / "config.yaml").write_text("training: {}")
    source = tmp_path / "mix.wav"
    source.write_bytes(b"input")
    output_dir = tmp_path / "out"

    monkeypatch.setattr("mseparate.melband.shutil.which", lambda _: "/venv/bin/melband-roformer")
    calls = []

    def fake_run(args, **kwargs):
        calls.append((args, kwargs))
        output_dir.mkdir(parents=True, exist_ok=True)
        (output_dir / "mix_vocals.wav").write_bytes(b"vocals")
        (output_dir / "mix_instrumental.wav").write_bytes(b"instrumental")

        class Result:
            returncode = 0
        return Result()

    monkeypatch.setattr("mseparate.melband.subprocess.run", fake_run)
    adapter = MelBandCLIAdapter(timeout=900)
    adapter.load(manifest, model_dir, "cpu")
    artifacts = adapter.infer(source, output_dir, ("vocals", "instrumental"))

    assert [item.stem for item in artifacts] == ["vocals", "instrumental"]
    args, kwargs = calls[0]
    assert args[0] == "/venv/bin/melband-roformer"
    assert "--config-path" in args and "--model-path" in args
    assert "--device" in args and args[args.index("--device") + 1] == "cpu"
    assert kwargs["timeout"] == 900
    adapter.unload()
    assert adapter.info().details["model_id"] == ""


def test_melband_adapter_reports_child_process_failure(tmp_path, monkeypatch):
    manifest = _manifest()
    model_dir = tmp_path / manifest.model_id
    model_dir.mkdir()
    (model_dir / "model.ckpt").write_bytes(b"checkpoint")
    (model_dir / "config.yaml").write_text("training: {}")
    source = tmp_path / "mix.wav"
    source.write_bytes(b"input")
    monkeypatch.setattr("mseparate.melband.shutil.which", lambda _: "/venv/bin/melband-roformer")

    class Failed:
        returncode = 3

    monkeypatch.setattr("mseparate.melband.subprocess.run", lambda *a, **k: Failed())
    adapter = MelBandCLIAdapter()
    adapter.load(manifest, model_dir, "auto")
    with pytest.raises(RuntimeError, match="status 3"):
        adapter.infer(source, tmp_path / "out", ("vocals",))


def test_melband_adapter_rejects_missing_required_asset_role():
    manifest = _manifest()
    invalid = ModelManifest(
        **{**manifest.__dict__, "assets": (manifest.assets[0],)}
    )
    compatible, reason = MelBandCLIAdapter().compatibility(invalid)
    assert not compatible
    assert "config" in reason
