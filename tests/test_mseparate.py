"""CPU-only tests for the mseparate foundation; no model weights or GPU required."""
from __future__ import annotations

import json
import wave
from pathlib import Path

import pytest

from mseparate.audio import analyze_audio
from mseparate.cli import main
from mseparate.planner import build_plan, validate_plan
from mseparate.registry import RegistryError, eligible_models, load_registry, parse_manifest
from mseparate.storage import ensure_layout, write_json_atomic


def _write_wav(path: Path, seconds: float = 0.25, sample_rate: int = 8000) -> Path:
    import math
    import struct

    count = int(seconds * sample_rate)
    samples = [int(12000 * math.sin(2 * math.pi * 440 * i / sample_rate)) for i in range(count)]
    with wave.open(str(path), "wb") as wav:
        wav.setnchannels(1)
        wav.setsampwidth(2)
        wav.setframerate(sample_rate)
        wav.writeframes(b"".join(struct.pack("<h", value) for value in samples))
    return path


def test_audio_analysis_reports_basic_measurements(tmp_path):
    source = _write_wav(tmp_path / "tone.wav")
    result = analyze_audio(source)
    assert result.audio.sample_rate == 8000
    assert result.audio.channels == 1
    assert result.audio.duration_seconds == pytest.approx(0.25)
    assert result.audio.peak is not None and result.audio.peak > 0
    assert result.audio.rms is not None and result.audio.rms > 0
    assert result.predictions == []
    assert 0.0 <= result.measurements["low_band_energy_ratio"] <= 1.0
    assert 0.0 <= result.measurements["spectral_flatness"] <= 1.0
    assert result.measurements["spectral_centroid_hz"] > 0
    assert result.measurements["transient_activity"] >= 0


def test_audio_analysis_rejects_missing_file(tmp_path):
    with pytest.raises(FileNotFoundError):
        analyze_audio(tmp_path / "missing.wav")


def test_registry_is_empty_when_not_created(tmp_path):
    assert load_registry(tmp_path / "missing.json") == {}


def test_registry_rejects_path_traversal():
    with pytest.raises(RegistryError):
        parse_manifest({
            "model_id":"bad","name":"Bad","architecture":"x","backend":"x",
            "source_url":"https://example.invalid","assets":[{
                "role":"checkpoint","url":"https://example.invalid/a","relative_path":"../escape"
            }]
        })


def test_registry_only_exposes_verified_enabled_entries():
    raw = {
        "model_id":"candidate-a","name":"Candidate","architecture":"test",
        "backend":"test","source_url":"https://example.invalid",
        "license":"MIT","license_url":"https://example.invalid/license",
        "license_status":"verified","status":"available","assets":[{
            "role":"checkpoint","url":"https://example.invalid/a","relative_path":"a.ckpt",
            "sha256":"a" * 64
        }],
    }
    manifest = parse_manifest(raw)
    assert eligible_models({manifest.model_id: manifest}) == [manifest]


def test_registry_rejects_bad_checksum():
    with pytest.raises(RegistryError):
        parse_manifest({
            "model_id":"bad","name":"Bad","architecture":"x","backend":"x",
            "source_url":"https://example.invalid","assets":[{
                "role":"checkpoint","url":"https://example.invalid/a",
                "relative_path":"a.ckpt","sha256":"not-a-sha"
            }]
        })


def test_plan_is_explicitly_empty_and_valid():
    plan = build_plan()
    validate_plan(plan)
    assert plan.tasks == ()


def test_storage_layout_and_atomic_json(tmp_path):
    paths = ensure_layout(tmp_path / "state")
    assert paths["models"].is_dir()
    target = write_json_atomic(paths["root"] / "report.json", {"ok": True})
    assert json.loads(target.read_text()) == {"ok": True}


def test_cli_analysis_and_report(tmp_path, monkeypatch, capsys):
    source = _write_wav(tmp_path / "input.wav")
    monkeypatch.setenv("MSEPARATE_HOME", str(tmp_path / "state"))
    rc = main([str(source), "--analysis-only", "--report", str(tmp_path / "report.json")])
    assert rc == 0
    assert "Audio analysis" in capsys.readouterr().out
    payload = json.loads((tmp_path / "report.json").read_text())
    assert payload["inference_performed"] is False
    assert payload["plan"] is None


def test_cli_no_input_returns_usage(capsys):
    assert main([]) == 2
    assert "usage:" in capsys.readouterr().err


def test_cli_accepts_no_download_flag(tmp_path, monkeypatch):
    source = _write_wav(tmp_path / "offline.wav")
    monkeypatch.setenv("MSEPARATE_HOME", str(tmp_path / "state"))
    assert main([str(source), "--analysis-only", "--no-download"]) == 0


def test_cli_honors_models_dir_override(tmp_path, monkeypatch):
    source = _write_wav(tmp_path / "custom.wav")
    monkeypatch.setenv("MSEPARATE_HOME", str(tmp_path / "state"))
    custom_models = tmp_path / "custom-models"
    assert main([str(source), "--analysis-only", "--models-dir", str(custom_models)]) == 0
    assert custom_models.is_dir()



def test_cli_prepare_models_offline_reports_missing_asset(tmp_path, monkeypatch, capsys):
    source = _write_wav(tmp_path / "prepare.wav")
    state = tmp_path / "state"
    registry_dir = state / "registry"
    registry_dir.mkdir(parents=True)
    registry = [{
        "model_id": "fixture-model",
        "name": "Fixture Model",
        "architecture": "test",
        "backend": "test",
        "source_url": "https://example.invalid/project",
        "license": "MIT",
        "license_url": "https://example.invalid/license",
        "license_status": "verified",
        "status": "available",
        "assets": [{
            "role": "checkpoint",
            "url": "https://example.invalid/checkpoint",
            "relative_path": "model.ckpt",
            "sha256": "a" * 64,
            "size_bytes": 10
        }]
    }]
    (registry_dir / "models.json").write_text(json.dumps(registry), encoding="utf-8")
    monkeypatch.setenv("MSEPARATE_HOME", str(state))

    rc = main([str(source), "--prepare-models", "--no-download",
               "--report", str(tmp_path / "asset-report.json")])
    assert rc == 0
    output = capsys.readouterr().out
    assert "fixture-model / checkpoint: missing" in output
    payload = json.loads((tmp_path / "asset-report.json").read_text())
    assert payload["asset_results"][0]["status"] == "missing"
    assert payload["inference_performed"] is False
