"""CPU-only contract tests for the optional six-stem ONNX backend."""
from __future__ import annotations

import io
import os
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
    state = {
        "preload": None, "call": None, "cleared": False,
        "prewarm": None, "session_providers": None, "fallback_disabled": False,
    }

    class Session:
        def get_providers(self):
            return ["CUDAExecutionProvider"]

        def disable_fallback(self):
            state["fallback_disabled"] = True

    class Pool:
        def get(self, model_path, providers):
            state["session_providers"] = providers
            return Session()

        def clear(self):
            state["cleared"] = True

    fake_ort = types.ModuleType("onnxruntime")
    fake_ort.get_available_providers = lambda: ["CUDAExecutionProvider", "CPUExecutionProvider"]
    fake_ort.preload_dlls = lambda **kwargs: state.update(preload=kwargs)
    fake_runtime = types.ModuleType("demucs_onnx")
    fake_runtime.session_pool = lambda: Pool()
    def prewarm(**kwargs):
        state["prewarm"] = kwargs
        return {"htdemucs_6s": Path("htdemucs_6s.onnx")}
    fake_runtime.prewarm = prewarm

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
    assert state["prewarm"]["providers"] == "CUDAExecutionProvider"
    assert state["session_providers"] == ["CUDAExecutionProvider"]
    assert state["fallback_disabled"]
    assert state["call"]["precision"] == "fp16weights"
    assert [item.stem for item in artifacts] == ["drums", "guitar", "piano"]
    adapter.unload()
    assert state["cleared"]


def test_strict_cuda_accepts_cpu_provider_after_cuda_and_disables_fallback():
    from mseparate.demucs_onnx import _prewarm_strict_cuda

    state = {"fallback_disabled": False}

    class Session:
        def get_providers(self):
            return ["CUDAExecutionProvider", "CPUExecutionProvider"]

        def disable_fallback(self):
            state["fallback_disabled"] = True

    class Pool:
        def get(self, model_path, providers):
            assert providers == ["CUDAExecutionProvider"]
            return Session()

    class Runtime:
        @staticmethod
        def prewarm(**kwargs):
            assert kwargs["providers"] == "CUDAExecutionProvider"
            return {"htdemucs_6s": Path("htdemucs_6s.onnx")}

        @staticmethod
        def session_pool():
            return Pool()

    paths = _prewarm_strict_cuda(Runtime, Path("cache"), "fp16weights")
    assert paths["htdemucs_6s"] == Path("htdemucs_6s.onnx")
    assert state["fallback_disabled"]


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
    state = {"fallback_disabled": False}

    class Session:
        def get_providers(self):
            return ["CUDAExecutionProvider"]

        def disable_fallback(self):
            state["fallback_disabled"] = True

    class Pool:
        def get(self, model_path, providers):
            return Session()

        def clear(self):
            pass

    fake_runtime.session_pool = lambda: Pool()
    fake_runtime.prewarm = lambda **kwargs: {"htdemucs_6s": Path("htdemucs_6s.onnx")}

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
    assert state["fallback_disabled"]
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

 
def _run_real_demucs_onnx_integration(tmp_path, device):
    import numpy as np
    import soundfile as sf

    sample_rate = 44100
    duration = 10
    samples = np.arange(sample_rate * duration, dtype=np.float32) / sample_rate
    audio = np.stack((
        0.15 * np.sin(2 * np.pi * 220 * samples),
        0.12 * np.sin(2 * np.pi * 330 * samples),
    ), axis=1)
    source = tmp_path / "integration-input.wav"
    sf.write(source, audio, sample_rate, subtype="PCM_16")

    adapter = DemucsOnnxAdapter(precision="fp16weights")
    adapter.load(_manifest(), tmp_path / "model-cache", device)
    try:
        artifacts = adapter.infer(
            source, tmp_path / "stems", DemucsOnnxAdapter.STEMS,
        )
        assert {item.stem for item in artifacts} == set(DemucsOnnxAdapter.STEMS)
        for artifact in artifacts:
            info = sf.info(artifact.path)
            assert info.samplerate == sample_rate
            assert info.channels == 2
            assert info.frames == len(samples)
            assert info.frames > 0
    finally:
        adapter.unload()


@pytest.mark.skipif(
    os.environ.get("MSEPARATE_RUN_DEMUCS_ONNX_INTEGRATION") != "1",
    reason="set MSEPARATE_RUN_DEMUCS_ONNX_INTEGRATION=1 to download and run the real ONNX model",
)
def test_RUN_MODEL_demucs_onnx_cpu_integration(tmp_path):
    _run_real_demucs_onnx_integration(tmp_path, "cpu")


@pytest.mark.skipif(
    os.environ.get("MSEPARATE_RUN_DEMUCS_ONNX_GPU") != "1",
    reason="set MSEPARATE_RUN_DEMUCS_ONNX_GPU=1 on a CUDA host to run real GPU inference",
)
def test_RUN_GPU_demucs_onnx_cuda_integration(tmp_path):
    import onnxruntime as ort

    if "CUDAExecutionProvider" not in ort.get_available_providers():
        pytest.skip("ONNX Runtime CUDAExecutionProvider is not available")
    _run_real_demucs_onnx_integration(tmp_path, "cuda")



def test_chunk_progress_stream_uses_native_tqdm_bar_and_redraws_chunk_counter():
    import re
    from mseparate.demucs_onnx import _ChunkProgressStream

    target = io.StringIO()
    progress = _ChunkProgressStream(target)
    progress.write("  input:  12,151,885 samples\\n")
    progress.write("    chunk 1/48: 0.8s elapsed\\n")
    progress.write("    chunk 24/48: 12.4s elapsed\\n")
    progress.write("    chunk 48/48: 25.0s elapsed\\n")
    progress.finish()

    raw = target.getvalue()
    output = re.sub(r"\\x1b\\[[0-9;]*[A-Za-z]", "", raw)
    assert "  input:  12,151,885 samples\\n" in output
    assert "Processing audio:" in output
    assert "2%|" in output
    assert "50%|" in output
    assert "100%|" in output
    assert "Обрабатывается фрагмент: 1 / 48" in output
    assert "Обрабатывается фрагмент: 24 / 48" in output
    assert "Обработано фрагментов: 48 / 48" in output
    assert "first chunk + warm-up" not in output
    assert "last chunk:" not in output
    assert "\\x1b[1A" in raw  # later updates redraw the two-line display in place
    assert raw.endswith("\\n")


def test_chunk_progress_stream_flushes_after_each_progress_event():
    from mseparate.demucs_onnx import _ChunkProgressStream

    class FlushTrackingStream(io.StringIO):
        def __init__(self):
            super().__init__()
            self.flush_count = 0

        def flush(self):
            self.flush_count += 1
            super().flush()

    target = FlushTrackingStream()
    progress = _ChunkProgressStream(target)
    progress.write("    chunk 3/10: 2.1s elapsed\\n")

    assert target.flush_count == 1
    assert "30%|" in target.getvalue()
    assert "Обрабатывается фрагмент: 3 / 10" in target.getvalue()
