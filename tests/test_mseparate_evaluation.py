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
