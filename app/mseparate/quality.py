"""Technical validation for separated audio files.

These checks detect malformed or misaligned artifacts. They are not a
perceptual separation score and never claim SDR/SI-SDR without references.
"""
from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path

import numpy as np
import soundfile as sf


@dataclass(frozen=True)
class StemQualityReport:
    path: str
    status: str
    duration_seconds: float | None = None
    sample_rate: int | None = None
    channels: int | None = None
    peak: float | None = None
    rms: float | None = None
    clipping_fraction: float | None = None
    silent_frame_fraction: float | None = None
    issues: tuple[str, ...] = field(default_factory=tuple)


def inspect_stem(
    path: str | Path,
    *,
    reference: str | Path | None = None,
    block_frames: int = 262144,
) -> StemQualityReport:
    """Inspect an output in bounded-memory blocks and optionally compare alignment."""
    audio_path = Path(path)
    issues: list[str] = []
    try:
        info = sf.info(str(audio_path))
        if info.frames <= 0 or info.samplerate <= 0 or info.channels <= 0:
            raise ValueError("Audio has invalid frame count, sample rate or channel count")
        peak = 0.0
        square_sum = 0.0
        sample_count = 0
        clipped = 0
        silent_frames = 0
        frame_count = 0
        with sf.SoundFile(str(audio_path), "r") as stream:
            while True:
                block = stream.read(block_frames, dtype="float32", always_2d=True)
                if not len(block):
                    break
                if not np.isfinite(block).all():
                    issues.append("audio contains NaN or infinite samples")
                    break
                absolute = np.abs(block)
                peak = max(peak, float(absolute.max(initial=0.0)))
                square_sum += float(np.square(block, dtype=np.float64).sum())
                sample_count += int(block.size)
                clipped += int(np.count_nonzero(absolute >= 0.999))
                silent_frames += int(np.count_nonzero(np.max(absolute, axis=1) <= 1e-6))
                frame_count += int(block.shape[0])
        rms = float(np.sqrt(square_sum / sample_count)) if sample_count else 0.0
        clipping_fraction = clipped / sample_count if sample_count else 0.0
        silent_fraction = silent_frames / frame_count if frame_count else 1.0

        if peak <= 1e-8 or silent_fraction >= 0.999:
            issues.append("output is effectively silent")
        if clipping_fraction > 0.001:
            issues.append("output has substantial clipped samples")

        if reference is not None:
            ref = sf.info(str(reference))
            duration = info.frames / info.samplerate
            ref_duration = ref.frames / ref.samplerate
            tolerance = max(0.05, ref_duration * 0.001)
            if abs(duration - ref_duration) > tolerance:
                issues.append(
                    f"duration mismatch ({duration:.3f}s vs {ref_duration:.3f}s)"
                )
            if info.samplerate != ref.samplerate:
                issues.append(
                    f"sample-rate mismatch ({info.samplerate} Hz vs {ref.samplerate} Hz)"
                )
            if info.channels != ref.channels:
                issues.append(
                    f"channel-count mismatch ({info.channels} vs {ref.channels})"
                )

        hard_failure = any(
            item.startswith("audio contains")
            or item.startswith("duration mismatch")
            or item.startswith("sample-rate mismatch")
            or item.startswith("channel-count mismatch")
            for item in issues
        )
        status = "invalid" if hard_failure else ("warning" if issues else "valid")
        return StemQualityReport(
            path=str(audio_path),
            status=status,
            duration_seconds=info.frames / info.samplerate,
            sample_rate=info.samplerate,
            channels=info.channels,
            peak=peak,
            rms=rms,
            clipping_fraction=clipping_fraction,
            silent_frame_fraction=silent_fraction,
            issues=tuple(issues),
        )
    except (OSError, RuntimeError, ValueError) as exc:
        return StemQualityReport(
            path=str(audio_path),
            status="invalid",
            issues=(f"cannot inspect audio: {type(exc).__name__}: {exc}",),
        )
