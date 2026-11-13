"""CPU-only tests for sequential backend execution and failure isolation."""
from __future__ import annotations

from pathlib import Path

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
            path.write_bytes(b"fake wav payload")
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
    source.write_bytes(b"input")
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
    source.write_bytes(b"input")
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
    source.write_bytes(b"input")

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
    summary = manager.execute(plan, {"model-a": _manifest("model-a")}, {"mix": source})
    assert summary.failed == 1
    assert "did not produce requested stems" in summary.results[0].message
    assert state["active"] == 0
