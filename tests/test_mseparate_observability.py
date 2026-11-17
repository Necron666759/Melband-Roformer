"""CPU-only tests for mseparate run observability."""
from __future__ import annotations

import io

import numpy as np
import soundfile as sf

from mseparate.observability import RunJournal, TeeTextIO, resolve_log_path


def test_run_journal_records_stage_duration_and_order():
    journal = RunJournal()
    journal.begin("model_inference", model_id="fixture")
    journal.end("model_inference", outputs=2)
    snapshot = journal.snapshot()

    assert snapshot["started_at"]
    assert snapshot["elapsed_seconds"] >= 0
    assert [event["status"] for event in snapshot["events"] if event["stage"] == "model_inference"] == [
        "started", "completed",
    ]
    completed = snapshot["events"][-1]
    assert completed["details"]["duration_seconds"] >= 0
    assert completed["details"]["outputs"] == 2


def test_tee_text_io_mirrors_stdout_and_stderr():
    terminal = io.StringIO()
    log = io.StringIO()
    stream = TeeTextIO(terminal, log)

    assert stream.write("first line\n") == 11
    stream.flush()

    assert terminal.getvalue() == "first line\n"
    assert log.getvalue() == "first line\n"


def test_resolve_log_path_uses_output_directory(tmp_path):
    path = resolve_log_path("logs/run.log", tmp_path)
    assert path == tmp_path / "logs" / "run.log"
    assert path.parent.is_dir()


def test_resolve_log_path_preserves_absolute_path(tmp_path):
    target = tmp_path / "absolute.log"
    assert resolve_log_path(str(target), "/ignored") == target



def test_cli_log_file_option_accepts_default_and_explicit_path():
    from mseparate.cli import build_parser

    parser = build_parser()
    default = parser.parse_args(["track.wav", "--log-file"])
    explicit = parser.parse_args(["track.wav", "--log-file", "logs/custom.log"])

    assert default.log_file == "mseparate.log"
    assert explicit.log_file == "logs/custom.log"



def test_cli_mirrors_terminal_output_to_log_file(tmp_path, monkeypatch, capsys):
    from mseparate.cli import main

    source = tmp_path / "input.wav"
    samples = np.sin(np.linspace(0, 100, 44100, dtype=np.float32)) * 0.1
    sf.write(source, np.column_stack((samples, samples)), 44100)
    log_path = tmp_path / "logs" / "run.log"
    monkeypatch.setenv("MSEPARATE_HOME", str(tmp_path / "state"))

    assert main([
        str(source), "--analysis-only", "--log-file", str(log_path),
    ]) == 0

    terminal = capsys.readouterr().out
    log_text = log_path.read_text(encoding="utf-8")
    assert "Audio analysis" in terminal
    assert "Audio analysis" in log_text
    assert "Execution plan" not in log_text
