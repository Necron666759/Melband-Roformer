"""CPU-only tests for the mseparate foundation; no Torch, weights or GPU required."""
from __future__ import annotations

import json
import wave
from pathlib import Path

import pytest

from mseparate.audio import analyze_audio
from mseparate.cli import main
from mseparate.planner import build_plan, validate_plan
from mseparate.registry import (
    RegistryError, bundled_registry_path, eligible_models, load_registry,
    model_audio_compatibility, parse_manifest,
)
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


def test_registry_rejects_duplicate_asset_roles():
    raw = {
        "model_id":"duplicate","name":"Duplicate","architecture":"x","backend":"x",
        "source_url":"https://example.invalid",
        "assets":[
            {"role":"checkpoint","url":"https://example.invalid/a","relative_path":"a","sha256":"a"*64},
            {"role":"checkpoint","url":"https://example.invalid/b","relative_path":"b","sha256":"b"*64},
        ],
    }
    with pytest.raises(RegistryError, match="Duplicate asset role"):
        parse_manifest(raw)


def test_bundled_registry_contains_pinned_kim_vocals_model():
    registry = load_registry(bundled_registry_path())
    model = registry["melband-roformer-kim-vocals"]
    assert model in eligible_models(registry)
    assert model.upstream_version == "melband-roformer-infer==0.1.5"
    assert {asset.role for asset in model.assets} == {"checkpoint", "config"}
    assert all(asset.sha256 and asset.size_bytes for asset in model.assets)
    assert model.native_stems == ("vocals",)
    assert model.derived_stems == ("instrumental",)


def test_new_pymss_models_are_pinned_and_license_reviewed():
    registry = load_registry(bundled_registry_path())
    eligible = {model.model_id for model in eligible_models(registry)}

    bs = registry["bs-roformer-anvuew-vocals"]
    mdx = registry["mdx23c-zfturbo-vocals"]
    assert bs.model_id in eligible and mdx.model_id in eligible
    assert bs.license == "GPL-3.0"
    assert mdx.license == "CC-BY-NC-SA-4.0"
    assert bs.backend == "bs-roformer-pymss"
    assert mdx.backend == "mdx23c-pymss"
    assert bs.native_stems == ("vocals",)
    assert mdx.native_stems == ("vocals",)
    assert all(asset.sha256 and asset.size_bytes for asset in (*bs.assets, *mdx.assets))
    assert bs.planner_priority > mdx.planner_priority


def test_model_audio_compatibility_enforces_sample_rate_and_channels():
    from mseparate.types import AudioMetadata
    model = load_registry(bundled_registry_path())["melband-roformer-kim-vocals"]
    good = AudioMetadata("mix.wav", 1.0, 44100, 2, 44100, "PCM_16")
    wrong_rate = AudioMetadata("mix.wav", 1.0, 48000, 2, 48000, "PCM_16")
    too_many_channels = AudioMetadata("mix.wav", 1.0, 44100, 6, 44100, "PCM_16")
    wrong_container = AudioMetadata("mix.flac", 1.0, 44100, 2, 44100, "PCM_16")
    assert model_audio_compatibility(model, good) == (True, "")
    assert not model_audio_compatibility(model, wrong_rate)[0]
    assert "resampling is not supported" in model_audio_compatibility(model, wrong_rate)[1]
    assert not model_audio_compatibility(model, too_many_channels)[0]
    assert not model_audio_compatibility(model, wrong_container)[0]
    assert "requires wav input" in model_audio_compatibility(model, wrong_container)[1]


def test_plan_is_explicitly_empty_and_valid():
    plan = build_plan()
    validate_plan(plan)
    assert plan.tasks == ()


def test_incompatible_audio_does_not_schedule_model():
    from mseparate.types import AnalysisReport, AudioMetadata
    registry = load_registry(bundled_registry_path())
    audio = AnalysisReport(
        schema_version=2,
        audio=AudioMetadata("mix.wav", 1.0, 48000, 2, 48000, "PCM_16"),
    )
    plan = build_plan(analysis=audio, registry=registry)
    assert {task.model_id for task in plan.tasks} == {
        "bs-roformer-anvuew-vocals",
        "demucs-onnx-htdemucs-6s",
    }
    by_model = {task.model_id: task for task in plan.tasks}
    assert by_model["bs-roformer-anvuew-vocals"].target_stems == ("instrumental", "vocals")
    assert set(by_model["demucs-onnx-htdemucs-6s"].target_stems) == {
        "drums", "bass", "other", "guitar", "piano",
    }
    assert any("melband-roformer-kim-vocals: requires 44100 Hz" in note for note in plan.notes)


def test_planner_can_opt_into_melband_sample_rate_conversion():
    from mseparate.types import AnalysisReport, AudioMetadata

    registry = load_registry(bundled_registry_path())
    audio = AnalysisReport(
        schema_version=2,
        audio=AudioMetadata("mix.wav", 1.0, 48000, 2, 48000, "PCM_16"),
    )

    default_plan = build_plan(analysis=audio, registry=registry)
    assert "melband-roformer-kim-vocals" not in {
        task.model_id for task in default_plan.tasks
    }

    converted_plan = build_plan(
        analysis=audio, registry=registry, allow_resample=True,
    )
    by_model = {task.model_id: task for task in converted_plan.tasks}
    assert "melband-roformer-kim-vocals" in by_model
    assert any("Sample-rate conversion is enabled" in note for note in converted_plan.notes)


def test_cli_plan_only_can_enable_sample_rate_conversion(tmp_path, monkeypatch):
    source = _write_wav(tmp_path / "input-48k.wav", sample_rate=48000)
    report = tmp_path / "plan.json"
    monkeypatch.setenv("MSEPARATE_HOME", str(tmp_path / "state"))

    rc = main([
        str(source), "--plan-only", "--allow-resample",
        "--report", str(report),
    ])

    assert rc == 0
    payload = json.loads(report.read_text())
    assert payload["allow_resample"] is True
    assert "melband-roformer-kim-vocals" in {
        task["model_id"] for task in payload["plan"]["tasks"]
    }


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


def test_cli_list_models_does_not_require_audio(tmp_path, monkeypatch, capsys):
    monkeypatch.setenv("MSEPARATE_HOME", str(tmp_path / "state"))
    assert main(["--list-models"]) == 0
    output = capsys.readouterr().out
    assert "melband-roformer-kim-vocals [eligible]" in output
    assert "bs-roformer-anvuew-vocals [eligible]" in output
    assert "mdx23c-zfturbo-vocals [eligible]" in output
    assert "Planner priority: 50" in output
    assert "Registry source:" in output


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


def test_other_stem_uses_strongest_matching_clap_source():
    from mseparate.planner import _stem_score

    score, reason = _stem_score("other", {}, [
        {"source": "synthesizer", "score": 0.12},
        {"source": "strings", "score": 0.31},
        {"source": "brass", "score": 0.24},
        {"source": "woodwinds", "score": 0.18},
        {"source": "sound_effects", "score": 0.20},
    ])
    assert score == 1.0
    assert "strings" in reason


def test_feature_guided_planner_selects_models_from_declared_stems():
    from mseparate.types import AnalysisReport, AudioMetadata
    from mseparate.planner import build_plan

    def manifest(model_id, stems):
        return parse_manifest({
            "model_id": model_id, "name": model_id, "architecture": "test",
            "backend": "test", "source_url": "https://example.invalid",
            "license": "MIT", "license_status": "verified", "status": "available",
            "assets": [{"role": "checkpoint", "url": "https://example.invalid/a",
                        "relative_path": model_id + ".ckpt", "sha256": "a" * 64}],
            "native_stems": stems,
        })

    audio = AnalysisReport(
        schema_version=2,
        audio=AudioMetadata("mix.wav", 1.0, 44100, 2, 44100, "PCM_16"),
        measurements={"low_band_energy_ratio": 0.8, "mid_band_energy_ratio": 0.1,
                      "high_band_energy_ratio": 0.1, "transient_activity": 0.05,
                      "spectral_flatness": 0.2},
    )
    bass = manifest("bass-model", ["bass"])
    vocal = manifest("vocal-model", ["vocals"])
    plan = build_plan(analysis=audio, registry={bass.model_id: bass, vocal.model_id: vocal})
    validate_plan(plan)
    assert len(plan.tasks) == 2
    by_model = {task.model_id: task for task in plan.tasks}
    assert by_model["bass-model"].target_stems == ("bass",)
    assert by_model["bass-model"].rationale
    assert 0 <= by_model["bass-model"].confidence <= 1


def test_planner_prefers_vocal_specialist_over_general_demucs_route():
    registry = load_registry(bundled_registry_path())
    plan = build_plan(registry=registry)
    validate_plan(plan)

    by_model = {task.model_id: task for task in plan.tasks}
    assert set(by_model) == {
        "melband-roformer-kim-vocals",
        "demucs-onnx-htdemucs-6s",
    }
    assert set(by_model["melband-roformer-kim-vocals"].target_stems) == {
        "vocals", "instrumental",
    }
    assert set(by_model["demucs-onnx-htdemucs-6s"].target_stems) == {
        "drums", "bass", "other", "guitar", "piano",
    }
    assert "manifest-declared specialist" in by_model[
        "melband-roformer-kim-vocals"
    ].rationale


def test_pymss_models_allow_manager_sample_rate_conversion():
    from mseparate.types import AudioMetadata

    registry = load_registry(bundled_registry_path())
    audio = AudioMetadata("mix.wav", 1.0, 48000, 2, 48000, "PCM_16")
    for model_id in ("bs-roformer-anvuew-vocals", "mdx23c-zfturbo-vocals"):
        assert model_audio_compatibility(registry[model_id], audio) == (True, "")


def test_planner_explicit_model_preference_overrides_default_priority():
    registry = load_registry(bundled_registry_path())
    plan = build_plan(registry=registry, preferred_model_id="mdx23c-zfturbo-vocals")
    by_model = {task.model_id: task for task in plan.tasks}

    assert "mdx23c-zfturbo-vocals" in by_model
    assert by_model["mdx23c-zfturbo-vocals"].target_stems == ("instrumental", "vocals")
    assert "melband-roformer-kim-vocals" not in by_model
    assert "explicitly preferred" in by_model["mdx23c-zfturbo-vocals"].rationale


def test_planner_rejects_unknown_preferred_model():
    registry = load_registry(bundled_registry_path())
    with pytest.raises(ValueError, match="not eligible or compatible"):
        build_plan(registry=registry, preferred_model_id="unknown-model")


def test_planner_uses_semantic_source_rankings_when_available():
    from mseparate.planner import _stem_score

    predictions = [
        {"source": "vocals", "score": 0.31},
        {"source": "synthesizer", "score": 0.42},
        {"source": "drums", "score": 0.27},
    ]
    score, reason = _stem_score("vocals", {}, predictions)
    assert score == 0.5
    assert "CLAP relative similarity" in reason
    assert "not a calibrated" in reason


def test_cli_execute_runs_plan_only_after_assets_are_verified(tmp_path, monkeypatch, capsys):
    from mseparate.assets import AssetResult
    from mseparate.execution import ExecutionSummary, TaskResult

    source = _write_wav(tmp_path / "input.wav", sample_rate=44100)
    state = {"manager_args": None, "execute_args": None, "assets": []}

    def prepare(model, models_root, **kwargs):
        state["assets"].append(model.model_id)
        return [AssetResult("checkpoint", "cached", str(models_root / model.model_id / "model.ckpt")),
                AssetResult("config", "cached", str(models_root / model.model_id / "config.yaml"))]

    class FakeManager:
        def __init__(self, factories, **kwargs):
            state["manager_args"] = (factories, kwargs)

        def execute(self, plan, manifests, inputs):
            state["execute_args"] = (plan, manifests, inputs)
            return ExecutionSummary((TaskResult("separate-01", "melband-roformer-kim-vocals", "success"),))

    monkeypatch.setattr("mseparate.cli.ensure_model_assets", prepare)
    monkeypatch.setattr("mseparate.source_analysis.analyze_sources", lambda *a, **k: [])
    monkeypatch.setattr("mseparate.cli.ExecutionManager", FakeManager)
    monkeypatch.setenv("MSEPARATE_HOME", str(tmp_path / "state"))
    output = tmp_path / "stems"
    report = tmp_path / "execution.json"

    rc = main([str(source), "--execute", "--device", "cpu",
               "--output-dir", str(output), "--report", str(report)])

    assert rc == 0
    assert state["assets"] == ["melband-roformer-kim-vocals"]
    factories, manager_kwargs = state["manager_args"]
    assert "melband-roformer" in factories
    assert "demucs-onnx" in factories
    assert manager_kwargs["device"] == "cpu"
    assert manager_kwargs["output_root"] == output / "mseparate-stems"
    plan, manifests, inputs = state["execute_args"]
    assert len(plan.tasks) == 2
    by_model = {task.model_id: task for task in plan.tasks}
    assert set(by_model["demucs-onnx-htdemucs-6s"].target_stems) == {
        "drums", "bass", "other", "guitar", "piano",
    }
    assert set(by_model["melband-roformer-kim-vocals"].target_stems) == {
        "instrumental", "vocals",
    }
    assert "melband-roformer-kim-vocals" in manifests
    assert "demucs-onnx-htdemucs-6s" in manifests
    assert inputs["mix"] == source.resolve()
    payload = json.loads(report.read_text())
    assert payload["inference_attempted"] is True
    assert payload["inference_performed"] is True
    assert payload["execution"]["results"][0]["status"] == "success"
    assert "Execution results" in capsys.readouterr().out



def test_cli_executes_by_default_and_retries_cuda_failure_on_cpu(tmp_path, monkeypatch, capsys):
    from mseparate.assets import AssetResult
    from mseparate.execution import ExecutionSummary, TaskResult

    source = _write_wav(tmp_path / "default.wav", sample_rate=44100)
    calls = []

    monkeypatch.setattr(
        "mseparate.cli.ensure_model_assets",
        lambda model, root, **kwargs: [
            AssetResult("checkpoint", "cached", str(root / model.model_id / "model.ckpt")),
            AssetResult("config", "cached", str(root / model.model_id / "config.yaml")),
        ],
    )
    monkeypatch.setattr("mseparate.source_analysis.analyze_sources", lambda *a, **k: [])
    monkeypatch.setattr("mseparate.cli._resolve_auto_device", lambda: ("cuda", None))
    monkeypatch.setenv("MSEPARATE_HOME", str(tmp_path / "state"))

    class FakeManager:
        def __init__(self, factories, **kwargs):
            self.device = kwargs["device"]
            calls.append(self.device)

        def execute(self, plan, manifests, inputs):
            if self.device == "cuda":
                return ExecutionSummary((
                    TaskResult("separate-01", "melband-roformer-kim-vocals",
                               "failed", message="CUDA initialization failed"),
                ))
            return ExecutionSummary((
                TaskResult("separate-01", "melband-roformer-kim-vocals", "success"),
            ))

    monkeypatch.setattr("mseparate.cli.ExecutionManager", FakeManager)
    report = tmp_path / "default-report.json"
    rc = main([str(source), "--report", str(report)])

    assert rc == 0
    assert calls == ["cuda", "cpu"]
    output = capsys.readouterr()
    assert "retrying the complete plan on CPU" in output.err
    assert "Execution results" in output.out
    payload = json.loads(report.read_text())
    assert payload["inference_performed"] is True
    assert payload["requested_device"] == "auto"
    assert payload["effective_device"] == "cpu"
    assert "CUDA initialization failed" in payload["device_fallback_reason"]


def test_cli_execute_is_blocked_when_assets_are_not_verified(tmp_path, monkeypatch, capsys):
    from mseparate.assets import AssetResult

    source = _write_wav(tmp_path / "input.wav", sample_rate=44100)
    called = {"manager": False}

    monkeypatch.setattr(
        "mseparate.cli.ensure_model_assets",
        lambda *a, **k: [AssetResult("checkpoint", "missing", message="offline")],
    )
    monkeypatch.setattr("mseparate.source_analysis.analyze_sources", lambda *a, **k: [])

    class ForbiddenManager:
        def __init__(self, *args, **kwargs):
            called["manager"] = True
            raise AssertionError("execution must not start with missing assets")

    monkeypatch.setattr("mseparate.cli.ExecutionManager", ForbiddenManager)
    monkeypatch.setenv("MSEPARATE_HOME", str(tmp_path / "state"))
    rc = main([str(source), "--execute", "--no-download"])
    output = capsys.readouterr()
    assert rc == 1
    assert not called["manager"]
    assert "Execution blocked" in output.err
    assert "Inference was not started" in output.out


def test_cli_rejects_execute_with_analysis_only():
    with pytest.raises(SystemExit) as exc:
        main(["input.wav", "--execute", "--analysis-only"])
    assert exc.value.code == 2


def test_demucs_strict_cuda_accepts_cpu_provider_registered_after_cuda(tmp_path):
    from mseparate.demucs_onnx import _prewarm_strict_cuda

    class FakeSession:
        def __init__(self):
            self.fallback_disabled = False

        def get_providers(self):
            return ["CUDAExecutionProvider", "CPUExecutionProvider"]

        def disable_fallback(self):
            self.fallback_disabled = True

    class FakePool:
        def __init__(self):
            self.session = FakeSession()
            self.requested = None

        def get(self, model_path, providers):
            self.requested = (model_path, providers)
            return self.session

    class FakeRuntime:
        def __init__(self):
            self.pool = FakePool()
            self.prewarm_args = None

        def prewarm(self, **kwargs):
            self.prewarm_args = kwargs
            return {"htdemucs_6s": tmp_path / "htdemucs_6s.onnx"}

        def session_pool(self):
            return self.pool

    runtime = FakeRuntime()
    paths = _prewarm_strict_cuda(runtime, tmp_path, "fp16weights")

    assert paths["htdemucs_6s"] == tmp_path / "htdemucs_6s.onnx"
    assert runtime.pool.session.fallback_disabled is True
    assert runtime.pool.requested == (
        tmp_path / "htdemucs_6s.onnx", ["CUDAExecutionProvider"]
    )


@pytest.mark.parametrize(
    "providers",
    [
        ["CPUExecutionProvider", "CUDAExecutionProvider"],
        ["CPUExecutionProvider"],
        [],
    ],
)
def test_demucs_strict_cuda_rejects_cuda_missing_or_not_first(tmp_path, providers):
    from mseparate.demucs_onnx import _prewarm_strict_cuda

    class FakeSession:
        def get_providers(self):
            return providers

        def disable_fallback(self):
            raise AssertionError("fallback must not be disabled for an invalid session")

    class FakePool:
        def get(self, model_path, requested):
            return FakeSession()

    class FakeRuntime:
        def prewarm(self, **kwargs):
            return {"htdemucs_6s": tmp_path / "htdemucs_6s.onnx"}

        def session_pool(self):
            return FakePool()

    with pytest.raises(RuntimeError, match="CUDAExecutionProvider first"):
        _prewarm_strict_cuda(FakeRuntime(), tmp_path, "fp16weights")


def test_demucs_strict_cuda_requires_disable_fallback_support(tmp_path):
    from mseparate.demucs_onnx import _prewarm_strict_cuda

    class FakeSession:
        def get_providers(self):
            return ["CUDAExecutionProvider", "CPUExecutionProvider"]

    class FakePool:
        def get(self, model_path, requested):
            return FakeSession()

    class FakeRuntime:
        def prewarm(self, **kwargs):
            return {"htdemucs_6s": tmp_path / "htdemucs_6s.onnx"}

        def session_pool(self):
            return FakePool()

    with pytest.raises(RuntimeError, match="disable_fallback"):
        _prewarm_strict_cuda(FakeRuntime(), tmp_path, "fp16weights")
