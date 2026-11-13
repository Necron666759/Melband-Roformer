"""CPU-only tests for technical output validation."""
from __future__ import annotations

import numpy as np
import soundfile as sf

from mseparate.quality import inspect_stem


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
