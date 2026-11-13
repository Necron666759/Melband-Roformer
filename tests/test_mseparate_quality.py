"""CPU-only tests for technical output validation."""
from __future__ import annotations

from dataclasses import replace

import numpy as np
import soundfile as sf

from mseparate.quality import decide_refinement, inspect_stem, inspect_stem_set


def test_inspect_stem_accepts_aligned_audio(tmp_path):
    source = tmp_path / "mix.wav"
    stem = tmp_path / "vocals.wav"
    samples = np.sin(np.linspace(0, 100, 48000, dtype=np.float32)) * 0.2
    stereo = np.column_stack((samples, samples))
    sf.write(source, stereo, 48000)
    sf.write(stem, stereo, 48000)

    report = inspect_stem(stem, reference=source, block_frames=4096)
    assert report.status == "valid"
    assert report.sample_rate == 48000
    assert report.channels == 2
    assert report.peak > 0
    assert report.rms > 0
    assert report.issues == ()


def test_inspect_stem_flags_silence_without_claiming_separation_score(tmp_path):
    silent = tmp_path / "silent.wav"
    sf.write(silent, np.zeros((24000, 2), dtype=np.float32), 48000)

    report = inspect_stem(silent)
    assert report.status == "warning"
    assert "output is effectively silent" in report.issues
    assert not hasattr(report, "sdr")
    assert not hasattr(report, "si_sdr")


def test_inspect_stem_rejects_duration_and_format_mismatch(tmp_path):
    reference = tmp_path / "mix.wav"
    candidate = tmp_path / "stem.wav"
    sf.write(reference, np.ones((48000, 2), dtype=np.float32) * 0.1, 48000)
    sf.write(candidate, np.ones((32000, 1), dtype=np.float32) * 0.1, 44100)

    report = inspect_stem(candidate, reference=reference)
    assert report.status == "invalid"
    assert any("duration mismatch" in issue for issue in report.issues)
    assert any("sample-rate mismatch" in issue for issue in report.issues)
    assert any("channel-count mismatch" in issue for issue in report.issues)


def test_inspect_stem_reports_unreadable_file(tmp_path):
    broken = tmp_path / "broken.wav"
    broken.write_bytes(b"not audio")
    report = inspect_stem(broken)
    assert report.status == "invalid"
    assert report.issues[0].startswith("cannot inspect audio:")



def test_inspect_stem_set_flags_near_duplicate_stems(tmp_path):
    first = tmp_path / "vocals.wav"
    second = tmp_path / "other.wav"
    t = np.arange(48000, dtype=np.float32) / 48000
    signal = (0.2 * np.sin(2 * np.pi * 440 * t)).astype(np.float32)
    stereo = np.column_stack((signal, signal))
    sf.write(first, stereo, 48000)
    sf.write(second, stereo.copy(), 48000)

    reports = (inspect_stem(first), inspect_stem(second))
    checked = inspect_stem_set(
        (("vocals", first), ("other", second)), reports
    )

    assert all(report.status == "warning" for report in checked)
    assert all(
        any("possible duplicated content" in issue for issue in report.issues)
        for report in checked
    )
    assert all("not proof of source leakage" in report.issues[-1] for report in checked)


def test_inspect_stem_set_does_not_flag_distinct_signals(tmp_path):
    first = tmp_path / "vocals.wav"
    second = tmp_path / "drums.wav"
    t = np.arange(48000, dtype=np.float32) / 48000
    left = np.sin(2 * np.pi * 220 * t).astype(np.float32) * 0.2
    right = np.sin(2 * np.pi * 1700 * t).astype(np.float32) * 0.2
    sf.write(first, np.column_stack((left, left)), 48000)
    sf.write(second, np.column_stack((right, right)), 48000)

    reports = (inspect_stem(first), inspect_stem(second))
    checked = inspect_stem_set((("vocals", first), ("drums", second)), reports)

    assert all(report.status == "valid" for report in checked)
    assert all(not report.issues for report in checked)


def test_refinement_gate_only_accepts_clearing_all_parent_warnings(tmp_path):
    parent_path = tmp_path / "parent.wav"
    candidate_path = tmp_path / "candidate.wav"
    sf.write(parent_path, np.zeros((48000, 2), dtype=np.float32), 48000)
    signal = np.sin(np.linspace(0, 100, 48000, dtype=np.float32)) * 0.1
    sf.write(candidate_path, np.column_stack((signal, signal)), 48000)

    parent = inspect_stem(parent_path)
    candidate = inspect_stem(candidate_path)
    decision = decide_refinement(parent, candidate)

    assert decision.accepted is True
    assert "clears all warnings" in decision.reason

    no_improvement = decide_refinement(candidate, candidate)
    assert no_improvement.accepted is False

    invalid = replace(candidate, status="invalid")
    rejected = decide_refinement(parent, invalid)
    assert rejected.accepted is False
    assert "preserve parent" in rejected.reason
