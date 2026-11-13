"""CPU-only tests for sequential backend execution and failure isolation."""
from __future__ import annotations

from dataclasses import replace
from pathlib import Path

import numpy as np
import soundfile as sf

from mseparate.execution import BackendInfo, ExecutionManager, StemArtifact
from mseparate.types import ExecutionPlan, ExecutionTask, ModelAsset, ModelManifest


class FakeBackend:
    def __init__(self, state, fail=False):
        self.state = state
        self.fail = fail

    def compatibility(self, manifest):
        return True, ""

    def load(self, manifest, model_dir, device):
        self.state["active"] += 1
        self.state["max_active"] = max(self.state["max_active"], self.state["active"])
        self.state["events"].append(("load", manifest.model_id))
        if self.state["active"] > 1:
            raise AssertionError("overlapping model residency")

    def infer(self, input_audio, output_dir, target_stems):
        self.state["events"].append(("infer", input_audio.name))
        if self.fail:
            raise RuntimeError("simulated model failure")
        outputs = []
        for stem in target_stems:
            path = output_dir / f"{stem}.wav"
            samples = np.sin(np.linspace(0, 100, 48000, dtype=np.float32)) * 0.2
            sf.write(path, np.column_stack((samples, samples)), 48000)
            outputs.append(StemArtifact(stem, path))
        return outputs

    def unload(self):
        self.state["events"].append(("unload",))
        self.state["active"] -= 1

    def info(self):
        return BackendInfo("fake")


def _manifest(model_id):
    return ModelManifest(
        model_id=model_id, name=model_id, architecture="fixture",
        backend="fake", source_url="https://example.invalid",
        license="MIT", license_url=None, license_status="verified",
        assets=(ModelAsset("checkpoint", "https://example.invalid/a",
                           "model.ckpt", "a" * 64),),
        native_stems=("vocals",), target_stems=("vocals",),
        status="available",
    )


def test_execution_is_strictly_sequential_and_unloads_before_next(tmp_path):
    source = tmp_path / "input.wav"
    sf.write(source, np.full((48000, 2), 0.1, dtype=np.float32), 48000)
    state = {"active": 0, "max_active": 0, "events": []}

    manager = ExecutionManager(
        {"fake": lambda: FakeBackend(state)},
        models_root=tmp_path / "models",
        output_root=tmp_path / "outputs",
        resource_check=lambda: state["events"].append(("resource-check", state["active"])),
    )
    plan = ExecutionPlan(1, (
        ExecutionTask("task-1", "model-a", "mix", ("vocals",)),
        ExecutionTask("task-2", "model-b", "mix", ("vocals",)),
    ))
    summary = manager.execute(
        plan, {"model-a": _manifest("model-a"), "model-b": _manifest("model-b")},
        {"mix": source},
    )

    assert summary.succeeded == 2
    assert summary.failed == summary.skipped == 0
    assert state["max_active"] == 1
    assert state["active"] == 0
    assert state["events"] == [
        ("load", "model-a"), ("infer", "input.wav"), ("unload",),
        ("resource-check", 0),
        ("load", "model-b"), ("infer", "input.wav"), ("unload",),
        ("resource-check", 0),
    ]


def test_failed_task_does_not_abort_independent_task_but_skips_dependents(tmp_path):
    source = tmp_path / "input.wav"
    sf.write(source, np.full((48000, 2), 0.1, dtype=np.float32), 48000)
    state = {"active": 0, "max_active": 0, "events": []}

    def factory():
        # model-a deliberately fails; later independent model-b works.
        return FakeBackend(state, fail=state["events"].count(("load", "model-a")) == 0)

    manager = ExecutionManager(
        {"fake": factory}, models_root=tmp_path / "models",
        output_root=tmp_path / "outputs",
    )
    plan = ExecutionPlan(1, (
        ExecutionTask("task-a", "model-a", "mix", ("vocals",)),
        ExecutionTask("task-child", "model-b", "mix", ("vocals",), depends_on=("task-a",)),
        ExecutionTask("task-independent", "model-b", "mix", ("vocals",)),
    ))
    summary = manager.execute(
        plan, {"model-a": _manifest("model-a"), "model-b": _manifest("model-b")},
        {"mix": source},
    )

    assert [item.status for item in summary.results] == ["failed", "skipped", "success"]
    assert summary.failed == summary.skipped == summary.succeeded == 1
    assert state["active"] == 0


def test_execution_rejects_missing_requested_stem(tmp_path):
    source = tmp_path / "input.wav"
    sf.write(source, np.zeros((4410, 2), dtype=np.float32), 44100)

    class IncompleteBackend(FakeBackend):
        def infer(self, input_audio, output_dir, target_stems):
            return []

    state = {"active": 0, "max_active": 0, "events": []}
    manager = ExecutionManager(
        {"fake": lambda: IncompleteBackend(state)},
        models_root=tmp_path / "models", output_root=tmp_path / "outputs",
    )
    plan = ExecutionPlan(1, (
        ExecutionTask("task", "model-a", "mix", ("vocals", "drums")),
    ))
    manifest = replace(_manifest("model-a"), derived_stems=("drums",))
    summary = manager.execute(plan, {"model-a": manifest}, {"mix": source})
    assert summary.failed == 1
    assert "did not produce requested stems" in summary.results[0].message
    assert state["active"] == 0


def test_backend_is_unloaded_even_when_load_raises_part_way(tmp_path):
    source = tmp_path / "input.wav"
    source.write_bytes(b"input")

    class PartialLoadBackend(FakeBackend):
        def load(self, manifest, model_dir, device):
            self.state["active"] += 1
            self.state["events"].append(("partial-load", manifest.model_id))
            raise RuntimeError("failed during load")

    state = {"active": 0, "max_active": 0, "events": []}
    manager = ExecutionManager(
        {"fake": lambda: PartialLoadBackend(state)},
        models_root=tmp_path / "models", output_root=tmp_path / "outputs",
    )
    plan = ExecutionPlan(1, (
        ExecutionTask("task", "model-a", "mix", ("vocals",)),
    ))
    summary = manager.execute(plan, {"model-a": _manifest("model-a")}, {"mix": source})

    assert summary.failed == 1
    assert state["active"] == 0
    assert state["events"] == [("partial-load", "model-a"), ("unload",)]



def test_executor_attaches_quality_report_for_each_output(tmp_path):
    source = tmp_path / "input.wav"
    sf.write(source, np.full((48000, 2), 0.1, dtype=np.float32), 48000)
    state = {"active": 0, "max_active": 0, "events": []}
    manager = ExecutionManager(
        {"fake": lambda: FakeBackend(state)},
        models_root=tmp_path / "models", output_root=tmp_path / "outputs",
    )
    plan = ExecutionPlan(1, (
        ExecutionTask("task", "model-a", "mix", ("vocals",)),
    ))

    summary = manager.execute(plan, {"model-a": _manifest("model-a")}, {"mix": source})
    result = summary.results[0]

    assert result.status == "success"
    assert len(result.quality_reports) == 1
    assert result.quality_reports[0].status == "valid"
    assert result.quality_reports[0].sample_rate == 48000
    assert result.quality_reports[0].channels == 2


def test_executor_fails_structurally_invalid_output_but_keeps_report(tmp_path):
    source = tmp_path / "input.wav"
    sf.write(source, np.full((48000, 2), 0.1, dtype=np.float32), 48000)

    class MismatchedBackend(FakeBackend):
        def infer(self, input_audio, output_dir, target_stems):
            path = output_dir / "vocals.wav"
            sf.write(path, np.full((24000, 1), 0.1, dtype=np.float32), 44100)
            return [StemArtifact("vocals", path)]

    state = {"active": 0, "max_active": 0, "events": []}
    manager = ExecutionManager(
        {"fake": lambda: MismatchedBackend(state)},
        models_root=tmp_path / "models", output_root=tmp_path / "outputs",
    )
    plan = ExecutionPlan(1, (
        ExecutionTask("task", "model-a", "mix", ("vocals",)),
    ))

    summary = manager.execute(plan, {"model-a": _manifest("model-a")}, {"mix": source})
    result = summary.results[0]

    assert result.status == "failed"
    assert result.outputs == ()
    assert len(result.quality_reports) == 1
    assert result.quality_reports[0].status == "invalid"
    assert "Output quality validation failed" in result.message


def test_executor_keeps_warning_outputs_available(tmp_path):
    source = tmp_path / "input.wav"
    sf.write(source, np.full((48000, 2), 0.1, dtype=np.float32), 48000)

    class SilentBackend(FakeBackend):
        def infer(self, input_audio, output_dir, target_stems):
            outputs = []
            for stem in target_stems:
                path = output_dir / f"{stem}.wav"
                sf.write(path, np.zeros((48000, 2), dtype=np.float32), 48000)
                outputs.append(StemArtifact(stem, path))
            return outputs

    state = {"active": 0, "max_active": 0, "events": []}
    manager = ExecutionManager(
        {"fake": lambda: SilentBackend(state)},
        models_root=tmp_path / "models", output_root=tmp_path / "outputs",
    )
    plan = ExecutionPlan(1, (
        ExecutionTask("task", "model-a", "mix", ("vocals",)),
    ))

    result = manager.execute(
        plan, {"model-a": _manifest("model-a")}, {"mix": source}
    ).results[0]

    assert result.status == "success"
    assert len(result.outputs) == 1
    assert result.quality_reports[0].status == "warning"
    assert "output is effectively silent" in result.quality_reports[0].issues



def test_executor_propagates_cross_stem_duplicate_warning(tmp_path):
    source = tmp_path / "input.wav"
    sf.write(source, np.full((48000, 2), 0.1, dtype=np.float32), 48000)

    class DuplicateBackend(FakeBackend):
        def infer(self, input_audio, output_dir, target_stems):
            outputs = []
            signal = np.sin(np.linspace(0, 100, 48000, dtype=np.float32)) * 0.2
            stereo = np.column_stack((signal, signal))
            for stem in target_stems:
                path = output_dir / f"{stem}.wav"
                sf.write(path, stereo, 48000)
                outputs.append(StemArtifact(stem, path))
            return outputs

    state = {"active": 0, "max_active": 0, "events": []}
    manager = ExecutionManager(
        {"fake": lambda: DuplicateBackend(state)},
        models_root=tmp_path / "models", output_root=tmp_path / "outputs",
    )
    manifest = replace(
        _manifest("model-a"),
        native_stems=("vocals", "drums"),
        target_stems=("vocals", "drums"),
    )
    plan = ExecutionPlan(1, (
        ExecutionTask("task", "model-a", "mix", ("vocals", "drums")),
    ))

    result = manager.execute(
        plan, {"model-a": manifest}, {"mix": source}
    ).results[0]

    assert result.status == "success"
    assert len(result.outputs) == 2
    assert all(report.status == "warning" for report in result.quality_reports)
    assert all(
        any("possible duplicated content" in issue for issue in report.issues)
        for report in result.quality_reports
    )



def test_execution_resamples_96k_input_for_model_and_restores_output_rate(tmp_path):
    source = tmp_path / "input-96k.wav"
    signal = np.sin(np.linspace(0, 500, 96000, dtype=np.float32)) * 0.2
    sf.write(source, np.column_stack((signal, signal)), 96000)

    class RateAwareBackend(FakeBackend):
        def infer(self, input_audio, output_dir, target_stems):
            rate = sf.info(str(input_audio)).samplerate
            self.state["model_input_rate"] = rate
            outputs = []
            for stem in target_stems:
                path = output_dir / f"{stem}.wav"
                samples = np.sin(np.linspace(0, 100, rate, dtype=np.float32)) * 0.1
                sf.write(path, np.column_stack((samples, samples)), rate)
                outputs.append(StemArtifact(stem, path))
            return outputs

    state = {"active": 0, "max_active": 0, "events": []}
    manifest = replace(_manifest("model-44k"), sample_rate=44100)
    manager = ExecutionManager(
        {"fake": lambda: RateAwareBackend(state)},
        models_root=tmp_path / "models", output_root=tmp_path / "outputs",
    )
    plan = ExecutionPlan(1, (
        ExecutionTask("task", "model-44k", "mix", ("vocals",)),
    ))

    result = manager.execute(plan, {"model-44k": manifest}, {"mix": source}).results[0]
    assert result.status == "success", result.message
    assert state["model_input_rate"] == 44100
    assert sf.info(str(result.outputs[0].path)).samplerate == 96000
    assert result.quality_reports[0].sample_rate == 96000
