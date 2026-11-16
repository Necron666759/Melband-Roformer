"""Tests for the deliberately opt-in HT-Demucs command."""
from __future__ import annotations

from pathlib import Path

from mseparate.demucs_cli import build_parser, main
from mseparate.execution import ExecutionSummary, TaskResult
from mseparate.registry import bundled_registry_path, load_registry


def test_manual_demucs_command_defaults_to_strict_cuda():
    args = build_parser().parse_args(["track.wav"])
    assert args.device == "cuda"
    assert args.no_download is False
    assert args.output_dir == "./mseparate-demucs-stems"

    cpu_args = build_parser().parse_args(["track.wav", "--device", "cpu"])
    assert cpu_args.device == "cpu"


def test_manual_demucs_command_schedules_all_six_stems_only_after_explicit_call(
    tmp_path, monkeypatch,
):
    source = tmp_path / "track.wav"
    source.write_bytes(b"test input")
    state = {}

    monkeypatch.setattr(
        "mseparate.demucs_cli.ensure_layout",
        lambda: {"models": tmp_path / "models"},
    )

    class FakeManager:
        def __init__(self, factories, **kwargs):
            state["factories"] = factories
            state["kwargs"] = kwargs

        def execute(self, plan, manifests, inputs):
            state["plan"] = plan
            state["manifests"] = manifests
            state["inputs"] = inputs
            return ExecutionSummary((
                TaskResult("htdemucs-6stem-manual", "demucs-onnx-htdemucs-6s", "success"),
            ))

    monkeypatch.setattr("mseparate.demucs_cli.ExecutionManager", FakeManager)
    rc = main([str(source), "--device", "cuda", "--output-dir", str(tmp_path / "out")])

    assert rc == 0
    assert set(state["factories"]) == {"demucs-onnx"}
    assert state["kwargs"]["device"] == "cuda"
    assert state["kwargs"]["output_root"] == tmp_path / "out"
    task = state["plan"].tasks[0]
    assert task.model_id == "demucs-onnx-htdemucs-6s"
    assert set(task.target_stems) == {
        "drums", "bass", "other", "vocals", "guitar", "piano",
    }
    assert state["inputs"]["mix"] == source.resolve()
