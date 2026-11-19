"""Small, dependency-light metrics for source-separation evaluation."""
from __future__ import annotations

import math

import numpy as np


def si_sdr_db(reference: np.ndarray, estimate: np.ndarray) -> float:
    """Compute scale-invariant SDR in dB for aligned, equal-shape audio arrays.

    The calculation follows the projection definition and deliberately does not
    time-align, resample, or mean-center signals: dataset preparation must
    preserve sample alignment and the model-facing sample rate.
    """
    reference = np.asarray(reference, dtype=np.float64)
    estimate = np.asarray(estimate, dtype=np.float64)
    if reference.shape != estimate.shape:
        raise ValueError(
            f"Reference and estimate shapes differ: {reference.shape} != {estimate.shape}"
        )
    if reference.size == 0:
        raise ValueError("Cannot score empty audio")
    if not np.isfinite(reference).all() or not np.isfinite(estimate).all():
        raise ValueError("Audio contains NaN or infinity")

    ref_energy = float(np.dot(reference.reshape(-1), reference.reshape(-1)))
    if ref_energy <= 1e-20:
        raise ValueError("Reference stem is silent; SI-SDR is undefined")

    ref = reference.reshape(-1)
    est = estimate.reshape(-1)
    scale = float(np.dot(est, ref) / ref_energy)
    target = scale * ref
    residual = est - target
    target_energy = float(np.dot(target, target))
    residual_energy = float(np.dot(residual, residual))
    if target_energy <= 1e-20:
        return float("-inf")
    if residual_energy <= 1e-20:
        return float("inf")
    return 10.0 * math.log10(target_energy / residual_energy)
