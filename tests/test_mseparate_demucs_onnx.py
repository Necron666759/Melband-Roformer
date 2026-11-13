"""CPU-only contract tests for the optional six-stem ONNX backend."""
from __future__ import annotations

import sys
import types
from pathlib import Path

import pytest

from mseparate.demucs_onnx import DemucsOnnxAdapter
from mseparate.registry import bundled_registry_path, eligible_models, load_registry


def _manifest():
    return load_registry(bundled_registry_path())["demucs-onnx-htdemucs-6s"]


def test_six_stem_manifest_is_eligible_without_pinned_local_assets():
    model = _manifest()
    assert model in eligible_models({model.model_id: model})
    assert model.assets == ()
    assert model.native_stems == ("drums", "bass", "other", "vocals", "guitar", "piano")


def test_adapter_requires_all_six_declared_native_stems():
    from dataclasses import replace
    adapter = DemucsOnnxAdapter()
    model = _manifest()
    assert adapter.compatibility(model) == (True, "")
    ok, reason = adapter.compatibility(replace(model, native_stems=("vocals",)))
    assert not ok
    assert "missing native stems" in reason


def test_adapter_forces_cuda_and_writes_all_requested_outputs(tmp_path, monkeypatch):
    state = {"preload": None, "call": None, "cleared": False}

    class Pool:
        def clear(self):
            state["cleared"] = True

    fake_ort = types.ModuleType("onnxruntime")
    fake_ort.get_available_providers = lambda: ["CUDAExecutionProvider", "CPUExecutionProvider"]
    fake_ort.preload_dlls = lambda **kwargs: state.update(preload=kwargs)
    fake_runtime = types.ModuleType("demucs_onnx")
    fake_runtime.session_pool = lambda: Pool()

    def separate(input_path, output_dir, **kwargs):
        state["call"] = kwargs
        for stem in kwargs["stems"]:
            Path(output_dir, f"{stem}.wav").write_bytes(b"wav")

    fake_runtime.separate = separate
    monkeypatch.setitem(sys.modules, "onnxruntime", fake_ort)
    monkeypatch.setitem(sys.modules, "demucs_onnx", fake_runtime)

    adapter = DemucsOnnxAdapter()
    adapter.load(_manifest(), tmp_path / "models", "cuda")
    source = tmp_path / "mix.wav"
    source.write_bytes(b"input")
    artifacts = adapter.infer(source, tmp_path / "out", ("drums", "guitar", "piano"))

    assert state["preload"] == {"directory": ""}
    assert state["call"]["model"] == "htdemucs_6s"
    assert state["call"]["providers"] == "CUDAExecutionProvider"
    assert state["call"]["precision"] == "fp16weights"
    assert [item.stem for item in artifacts] == ["drums", "guitar", "piano"]
    adapter.unload()
    assert state["cleared"]


def test_cuda_mode_fails_instead_of_silently_using_cpu(tmp_path, monkeypatch):
    fake_ort = types.ModuleType("onnxruntime")
    fake_ort.get_available_providers = lambda: ["CPUExecutionProvider"]
    fake_ort.preload_dlls = lambda **kwargs: None
    fake_runtime = types.ModuleType("demucs_onnx")
    monkeypatch.setitem(sys.modules, "onnxruntime", fake_ort)
    monkeypatch.setitem(sys.modules, "demucs_onnx", fake_runtime)

    with pytest.raises(RuntimeError, match="CUDAExecutionProvider is unavailable"):
        DemucsOnnxAdapter().load(_manifest(), tmp_path / "models", "cuda")


def test_no_download_uses_huggingface_offline_mode(tmp_path, monkeypatch):
    import os
    fake_ort = types.ModuleType("onnxruntime")
    fake_ort.get_available_providers = lambda: ["CUDAExecutionProvider", "CPUExecutionProvider"]
    fake_ort.preload_dlls = lambda **kwargs: None
    fake_runtime = types.ModuleType("demucs_onnx")
    fake_runtime.session_pool = lambda: types.SimpleNamespace(clear=lambda: None)
    state = {}

    def separate(*args, **kwargs):
        state["offline"] = os.environ.get("HF_HUB_OFFLINE")
        for stem in kwargs["stems"]:
            Path(args[1], f"{stem}.wav").write_bytes(b"wav")

    fake_runtime.separate = separate
    monkeypatch.setitem(sys.modules, "onnxruntime", fake_ort)
    monkeypatch.setitem(sys.modules, "demucs_onnx", fake_runtime)
    monkeypatch.delenv("HF_HUB_OFFLINE", raising=False)

    adapter = DemucsOnnxAdapter(no_download=True)
    adapter.load(_manifest(), tmp_path / "models", "cuda")
    source = tmp_path / "mix.wav"
    source.write_bytes(b"input")
    adapter.infer(source, tmp_path / "out", ("vocals",))
    assert state["offline"] == "1"
    assert "HF_HUB_OFFLINE" not in os.environ
    adapter.unload()

def test_planner_schedules_all_six_demucs_stems():
    from mseparate.planner import build_plan
    from mseparate.types import AnalysisReport, AudioMetadata

    model = _manifest()
    analysis = AnalysisReport(
        schema_version=2,
        audio=AudioMetadata("mix.wav", 12.0, 48000, 2, 576000, "PCM_16"),
    )
    plan = build_plan(analysis=analysis, registry={model.model_id: model})
    assert len(plan.tasks) == 1
    assert plan.tasks[0].model_id == model.model_id
    assert set(plan.tasks[0].target_stems) == set(DemucsOnnxAdapter.STEMS)
    assert not any("demucs-onnx-htdemucs-6s: requires 44100 Hz" in note for note in plan.notes)
