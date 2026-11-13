"""Bounded-memory technical and spectral audio analysis."""
from __future__ import annotations

import math
from pathlib import Path

from .types import AudioMetadata, AnalysisReport


def analyze_audio(path: str | Path) -> AnalysisReport:
    """Extract conservative signal features without identifying sound sources.

    Audio is streamed in blocks; spectral statistics are computed on short,
    non-overlapping frames. No neural model is loaded and no audio is uploaded.
    """
    import numpy as np
    import soundfile as sf

    source = Path(path).expanduser()
    if not source.is_file():
        raise FileNotFoundError(f"Input audio file does not exist: {source}")
    info = sf.info(str(source))
    if info.frames <= 0 or info.samplerate <= 0 or info.channels <= 0:
        raise ValueError(f"Audio file has invalid stream metadata: {source}")

    frame_size = 1024
    peak_value = sum_squares = 0.0
    sample_count = clipped_count = silent_count = 0
    frame_rms: list[float] = []
    centroids: list[float] = []
    rolloffs: list[float] = []
    flatnesses: list[float] = []
    band_totals = np.zeros(3, dtype=np.float64)
    spectral_total = 0.0
    transient_sum = 0.0
    transient_count = 0
    previous_rms: float | None = None
    stereo_cross = stereo_left = stereo_right = stereo_side = 0.0
    stereo_samples = 0
    frequencies = np.fft.rfftfreq(frame_size, 1.0 / info.samplerate)
    band_masks = (frequencies < 150, (frequencies >= 150) & (frequencies < 2000), frequencies >= 2000)

    with sf.SoundFile(str(source), mode="r") as audio:
        while True:
            block = audio.read(frames=262144, dtype="float32", always_2d=True)
            if block.size == 0:
                break
            finite = np.nan_to_num(block, copy=False, nan=0.0, posinf=0.0, neginf=0.0)
            absolute = np.abs(finite)
            peak_value = max(peak_value, float(absolute.max(initial=0.0)))
            sum_squares += float(np.square(finite, dtype=np.float64).sum())
            sample_count += int(finite.size)
            clipped_count += int(np.count_nonzero(absolute >= 0.999))
            silent_count += int(np.count_nonzero(absolute < 1e-4))
            if info.channels >= 2:
                left = finite[:, 0].astype(np.float64, copy=False)
                right = finite[:, 1].astype(np.float64, copy=False)
                stereo_cross += float(np.dot(left, right))
                stereo_left += float(np.dot(left, left))
                stereo_right += float(np.dot(right, right))
                side = (left - right) * 0.5
                stereo_side += float(np.dot(side, side))
                stereo_samples += len(left)

            mono = finite.mean(axis=1, dtype=np.float64)
            usable = (len(mono) // frame_size) * frame_size
            if usable == 0:
                continue
            frames = mono[:usable].reshape(-1, frame_size)
            rms_values = np.sqrt(np.mean(frames * frames, axis=1))
            frame_rms.extend(float(x) for x in rms_values)
            silent_count += 0
            for value in rms_values:
                current = float(value)
                if previous_rms is not None:
                    transient_sum += max(0.0, current - previous_rms) / max(current, previous_rms, 1e-9)
                    transient_count += 1
                previous_rms = current
            windowed = frames * np.hanning(frame_size)
            spectrum = np.abs(np.fft.rfft(windowed, axis=1))
            power = spectrum * spectrum
            totals = power.sum(axis=1)
            valid = totals > 1e-18
            if not np.any(valid):
                continue
            spectral_total += float(totals[valid].sum())
            weighted = (power * frequencies[None, :]).sum(axis=1)
            centroids.extend((weighted[valid] / totals[valid]).tolist())
            cumulative = np.cumsum(power, axis=1)
            thresholds = totals * 0.85
            roll_indices = np.argmax(cumulative >= thresholds[:, None], axis=1)
            rolloffs.extend(frequencies[roll_indices[valid]].tolist())
            positive = np.maximum(power, 1e-18)
            geometric = np.exp(np.mean(np.log(positive), axis=1))
            arithmetic = np.mean(positive, axis=1)
            flatnesses.extend((geometric[valid] / np.maximum(arithmetic[valid], 1e-18)).tolist())
            for index, mask in enumerate(band_masks):
                band_totals[index] += float(power[:, mask].sum())

    peak = rms = clipping_fraction = None
    if sample_count:
        peak = peak_value
        rms = math.sqrt(sum_squares / sample_count)
        clipping_fraction = clipped_count / sample_count
    if not frame_rms:
        raise ValueError("Audio contains no analyzable frames")

    frame_array = np.asarray(frame_rms, dtype=np.float64)
    active = frame_array[frame_array > 1e-7]
    dynamic_range_db = None
    if active.size:
        dynamic_range_db = float(20 * np.log10(max(np.percentile(active, 95), 1e-9) /
                                                max(np.percentile(active, 10), 1e-9)))
    band_sum = float(band_totals.sum())
    band_ratios = band_totals / band_sum if band_sum > 0 else np.zeros(3)
    stereo_correlation = None
    side_energy_ratio = None
    if stereo_samples and stereo_left > 0 and stereo_right > 0:
        stereo_correlation = stereo_cross / math.sqrt(stereo_left * stereo_right)
        stereo_correlation = max(-1.0, min(1.0, stereo_correlation))
        side_energy_ratio = stereo_side / max((stereo_left + stereo_right) * 0.5, 1e-18)

    measurements = {
        "peak_dbfs": 20 * math.log10(max(peak, 1e-12)) if peak is not None else None,
        "rms_dbfs": 20 * math.log10(max(rms, 1e-12)) if rms is not None else None,
        "clipping_fraction": clipping_fraction,
        "silence_fraction": silent_count / sample_count if sample_count else None,
        "crest_factor_db": 20 * math.log10(max(peak / max(rms, 1e-12), 1e-12)) if peak is not None and rms is not None else None,
        "dynamic_range_db_p10_p95": dynamic_range_db,
        "spectral_centroid_hz": float(np.mean(centroids)) if centroids else 0.0,
        "spectral_rolloff_85_hz": float(np.mean(rolloffs)) if rolloffs else 0.0,
        "spectral_flatness": float(np.mean(flatnesses)) if flatnesses else 0.0,
        "low_band_energy_ratio": float(band_ratios[0]),
        "mid_band_energy_ratio": float(band_ratios[1]),
        "high_band_energy_ratio": float(band_ratios[2]),
        "transient_activity": transient_sum / transient_count if transient_count else 0.0,
        "stereo_correlation": stereo_correlation,
        "stereo_side_energy_ratio": side_energy_ratio,
    }
    metadata = AudioMetadata(
        path=str(source.resolve()), duration_seconds=info.frames / float(info.samplerate),
        sample_rate=int(info.samplerate), channels=int(info.channels), frames=int(info.frames),
        subtype=getattr(info, "subtype", None), peak=peak, rms=rms,
        clipping_fraction=clipping_fraction,
    )
    notes = [
        "Spectral and dynamic measurements describe the signal; they do not identify instruments or prove that a source is present.",
        "Transient activity is a frame-to-frame level-change heuristic, not a drum detector.",
    ]
    if clipping_fraction and clipping_fraction > 0:
        notes.append("Samples at or above 0.999 full scale were detected; this is a clipping-risk indicator, not proof of audible distortion.")
    if info.channels > 2:
        notes.append("Stereo correlation and side-energy metrics use the first two channels only.")
    return AnalysisReport(schema_version=2, audio=metadata, measurements=measurements,
                          predictions=[], notes=notes)
