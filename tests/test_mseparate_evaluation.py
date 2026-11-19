from __future__ import annotations

import numpy as np
import pytest

from mseparate.evaluation import si_sdr_db


def test_si_sdr_is_scale_invariant_for_clean_estimate():
    reference = np.linspace(-1.0, 1.0, 2048, dtype=np.float32)
    estimate = 0.37 * reference
    assert si_sdr_db(reference, estimate) > 100.0


def test_si_sdr_penalizes_added_interference():
    reference = np.sin(np.linspace(0.0, 40.0, 4096, dtype=np.float32))
    interference = np.cos(np.linspace(0.0, 40.0, 4096, dtype=np.float32))
    clean = si_sdr_db(reference, reference)
    noisy = si_sdr_db(reference, reference + 0.5 * interference)
    assert clean == float("inf")
    assert noisy < 20.0


def test_si_sdr_rejects_shape_mismatch_and_silent_reference():
    with pytest.raises(ValueError, match="shapes differ"):
        si_sdr_db(np.ones(8), np.ones(9))
    with pytest.raises(ValueError, match="silent"):
        si_sdr_db(np.zeros(8), np.ones(8))


def test_si_sdr_rejects_non_finite_audio():
    with pytest.raises(ValueError, match="NaN or infinity"):
        si_sdr_db(np.ones(8), np.array([1, 1, 1, 1, 1, 1, np.nan, 1]))


def test_level_diagnostics_flags_large_target_estimate_mismatch():
    from pathlib import Path
    import sys
    sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "scripts"))
    from evaluate_mseparate_drum_heads import _level_diagnostics

    reference = np.ones(100, dtype=np.float32) * 0.001
    estimate = np.ones(100, dtype=np.float32) * 0.02
    result = _level_diagnostics(reference, estimate)
    assert result["estimate_to_reference_rms_ratio"] == pytest.approx(20.0)
    assert result["estimate_reference_level_mismatch_db"] == pytest.approx(26.0206, abs=1e-3)
    assert result["level_mismatch_warning"] is True


def test_level_diagnostics_does_not_flag_near_matched_levels():
    from pathlib import Path
    import sys
    sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "scripts"))
    from evaluate_mseparate_drum_heads import _level_diagnostics

    reference = np.ones(100, dtype=np.float32)
    estimate = np.ones(100, dtype=np.float32) * 1.5
    assert _level_diagnostics(reference, estimate)["level_mismatch_warning"] is False



def test_tom_component_diagnostics_reports_each_reference_and_mixture_levels():
    from pathlib import Path
    import sys
    sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "scripts"))
    from evaluate_mseparate_drum_heads import _tom_component_diagnostics

    hi = np.full((64, 2), 0.1, dtype=np.float32)
    mid = np.full((64, 2), 0.2, dtype=np.float32)
    low = np.full((64, 2), 0.3, dtype=np.float32)
    aggregate = hi + mid + low
    estimate = np.full((64, 2), 0.3, dtype=np.float32)
    mixture = np.full((64, 2), 0.9, dtype=np.float32)

    result = _tom_component_diagnostics(
        {"hi_tom": hi, "mid_tom": mid, "low_tom": low},
        aggregate, estimate, mixture,
    )

    assert [item["name"] for item in result["tom_component_levels"]] == [
        "hi_tom", "mid_tom", "low_tom"
    ]
    assert [item["rms"] for item in result["tom_component_levels"]] == pytest.approx(
        [0.1, 0.2, 0.3]
    )
    assert result["tom_component_rms_sum"] == pytest.approx(0.6)
    assert result["tom_component_rms_sum_to_aggregate_db"] == pytest.approx(0.0, abs=1e-5)
    assert result["tom_reference_to_mixture_db"] == pytest.approx(
        20 * np.log10(0.6 / 0.9)
    )
    assert result["tom_estimate_to_mixture_db"] == pytest.approx(
        20 * np.log10(0.3 / 0.9)
    )


def test_tom_component_diagnostics_is_empty_for_non_tom_target():
    from pathlib import Path
    import sys
    sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "scripts"))
    from evaluate_mseparate_drum_heads import _tom_component_diagnostics

    audio = np.ones((16, 2), dtype=np.float32)
    result = _tom_component_diagnostics({}, audio, audio, audio)
    assert result == {
        "tom_component_levels": [],
        "tom_component_rms_sum": None,
        "tom_component_rms_sum_to_aggregate_db": None,
        "tom_reference_to_mixture_db": None,
        "tom_estimate_to_mixture_db": None,
    }

def test_tom_reference_salience_bands_are_diagnostic_only():
    from pathlib import Path
    import sys
    sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "scripts"))
    from evaluate_mseparate_drum_heads import _tom_reference_salience_band

    assert _tom_reference_salience_band(-19.9) == "prominent_ge_-20_db"
    assert _tom_reference_salience_band(-20.0) == "prominent_ge_-20_db"
    assert _tom_reference_salience_band(-25.0) == "low_-30_to_-20_db"
    assert _tom_reference_salience_band(-30.0) == "low_-30_to_-20_db"
    assert _tom_reference_salience_band(-30.1) == "very_low_lt_-30_db"
    assert _tom_reference_salience_band(None) == "not_applicable"


def test_tom_component_diagnostics_reports_salience_without_changing_audio():
    from pathlib import Path
    import sys
    sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "scripts"))
    from evaluate_mseparate_drum_heads import _tom_component_diagnostics

    toms = np.full((32, 2), 0.01, dtype=np.float32)
    estimate = toms * 2.0
    mixture = np.full((32, 2), 0.1, dtype=np.float32)
    result = _tom_component_diagnostics(
        {"hi_tom": toms, "mid_tom": np.zeros_like(toms), "low_tom": np.zeros_like(toms)},
        toms, estimate, mixture,
    )
    assert result["tom_reference_salience_band"] == "low_-30_to_-20_db"
    assert result["tom_reference_to_mixture_db"] == pytest.approx(-20.0)
    assert result["tom_component_levels"][0]["rms"] == pytest.approx(0.01)
