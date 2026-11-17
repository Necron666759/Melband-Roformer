"""Tests for the bounded HT-Demucs ONNX diagnostic."""
from __future__ import annotations

import json

from mseparate.cli import _print_onnx_profile_summary, build_parser


def test_diagnose_parser_accepts_chunk_limit_and_device():
    args = build_parser().parse_args([
        "input.wav", "--diagnose", "3", "--device", "cuda",
    ])
    assert args.diagnose == 3
    assert args.device == "cuda"


def test_profile_summary_reports_provider_placement_and_slowest_cpu_nodes(tmp_path, capsys):
    profile = tmp_path / "ort-profile.json"
    profile.write_text(json.dumps([
        {
            "cat": "Node",
            "name": "CUDA_Conv_0_kernel_time",
            "dur": 1_500_000,
            "args": {"provider": "CUDAExecutionProvider", "op_name": "Conv"},
        },
        {
            "cat": "Node",
            "name": "CPU_Transpose_1_kernel_time",
            "dur": 250_000,
            "args": {"provider": "CPUExecutionProvider", "op_name": "Transpose"},
        },
        {
            "cat": "Node",
            "name": "CPU_Reshape_2_kernel_time",
            "dur": 100_000,
            "args": {"provider": "CPUExecutionProvider", "op_name": "Reshape"},
        },
        {"cat": "Kernel", "name": "cuda_kernel", "dur": 99_000_000, "args": {}},
    ]), encoding="utf-8")

    _print_onnx_profile_summary(profile)
    output = capsys.readouterr().out

    assert "CUDAExecutionProvider: 1 nodes" in output
    assert "CPUExecutionProvider: 2 nodes" in output
    assert "Slowest CPU-assigned nodes" in output
    assert "Transpose: CPU_Transpose_1_kernel_time" in output
    assert "cumulative node time" in output
    assert "not wall-clock time" in output


def test_profile_summary_handles_profile_without_provider_events(tmp_path, capsys):
    profile = tmp_path / "ort-profile.json"
    profile.write_text(json.dumps([
        {"cat": "Session", "name": "model_loading", "dur": 1, "args": {}},
    ]), encoding="utf-8")

    _print_onnx_profile_summary(profile)
    output = capsys.readouterr().out

    assert "No provider-tagged node events were recorded" in output
