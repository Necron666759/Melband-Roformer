"""Technical validation for separated audio files.

These checks detect malformed or misaligned artifacts. They are not a
perceptual separation score and never claim SDR/SI-SDR without references.
"""
from __future__ import annotations

from dataclasses import dataclass, field, replace
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
    allow_sample_rate_mismatch: bool = False,
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
            if info.samplerate != ref.samplerate and not allow_sample_rate_mismatch:
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



def inspect_stem_set(
    artifacts: tuple[tuple[str, str | Path], ...] | list[tuple[str, str | Path]],
    reports: tuple[StemQualityReport, ...],
    *,
    block_frames: int = 65536,
    correlation_threshold: float = 0.9995,
    level_ratio_tolerance: float = 0.10,
) -> tuple[StemQualityReport, ...]:
    """Flag near-duplicate output stems as a conservative leakage/duplication hint.

    This is deliberately not a general leakage detector. It only catches pairs
    that are almost sample-identical (or polarity-inverted) and have comparable
    RMS levels. All comparisons are streamed with bounded memory.
    """
    report_by_path = {str(Path(item.path)): item for item in reports}
    additions: dict[str, list[str]] = {key: [] for key in report_by_path}
    for index, (left_name, left_path) in enumerate(artifacts):
        left_path = Path(left_path)
        left_report = report_by_path.get(str(left_path))
        if left_report is None or left_report.status == "invalid" or left_report.rms is None:
            continue
        for right_name, right_path in artifacts[index + 1:]:
            right_path = Path(right_path)
            right_report = report_by_path.get(str(right_path))
            if right_report is None or right_report.status == "invalid" or right_report.rms is None:
                continue
            if (
                left_report.sample_rate != right_report.sample_rate
                or left_report.channels != right_report.channels
                or left_report.duration_seconds is None
                or right_report.duration_seconds is None
                or abs(left_report.duration_seconds - right_report.duration_seconds) > 1e-6
            ):
                continue
            if min(left_report.rms, right_report.rms) <= 1e-8:
                continue
            level_ratio = min(left_report.rms, right_report.rms) / max(
                left_report.rms, right_report.rms
            )
            if level_ratio < 1.0 - level_ratio_tolerance:
                continue

            sum_x = sum_y = sum_x2 = sum_y2 = sum_xy = 0.0
            count = 0
            try:
                with sf.SoundFile(str(left_path), "r") as left, sf.SoundFile(str(right_path), "r") as right:
                    while True:
                        x = left.read(block_frames, dtype="float64", always_2d=True)
                        y = right.read(block_frames, dtype="float64", always_2d=True)
                        if not len(x) and not len(y):
                            break
                        if x.shape != y.shape or not len(x):
                            count = 0
                            break
                        sum_x += float(x.sum())
                        sum_y += float(y.sum())
                        sum_x2 += float(np.square(x).sum())
                        sum_y2 += float(np.square(y).sum())
                        sum_xy += float(np.multiply(x, y).sum())
                        count += int(x.size)
            except (OSError, RuntimeError, ValueError):
                continue
            if count <= 1:
                continue
            covariance = sum_xy - (sum_x * sum_y / count)
            variance_x = sum_x2 - (sum_x * sum_x / count)
            variance_y = sum_y2 - (sum_y * sum_y / count)
            denominator = float(np.sqrt(max(variance_x, 0.0) * max(variance_y, 0.0)))
            if denominator <= 1e-12:
                continue
            correlation = abs(covariance / denominator)
            if correlation >= correlation_threshold:
                message = (
                    f"possible duplicated content between '{left_name}' and "
                    f"'{right_name}' (absolute correlation {correlation:.5f}); "
                    "heuristic only, not proof of source leakage"
                )
                additions[str(left_path)].append(message)
                additions[str(right_path)].append(message)

    updated: list[StemQualityReport] = []
    for report in reports:
        extra = additions.get(str(Path(report.path)), [])
        if not extra:
            updated.append(report)
            continue
        updated.append(replace(
            report,
            status="warning" if report.status == "valid" else report.status,
            issues=(*report.issues, *extra),
        ))
    return tuple(updated)


@dataclass(frozen=True)
class RefinementDecision:
    accepted: bool
    reason: str


def decide_refinement(
    parent: StemQualityReport,
    candidate: StemQualityReport,
) -> RefinementDecision:
    """Accept only a candidate that clears every parent warning without new ones.

    This intentionally conservative gate evaluates technical warnings only;
    it cannot establish perceptual separation quality or detect all leakage.
    """
    if parent.status == "invalid":
        return RefinementDecision(False, "Parent is structurally invalid; preserve it for diagnosis")
    if candidate.status == "invalid":
        return RefinementDecision(False, "Candidate failed structural validation; preserve parent")
    if candidate.status != "valid":
        return RefinementDecision(False, "Candidate still has quality warnings; preserve parent")
    if parent.status == "valid":
        return RefinementDecision(False, "No measurable technical warning was resolved")
    return RefinementDecision(
        True,
        "Candidate is structurally valid and clears all warnings present on the parent",
    )
