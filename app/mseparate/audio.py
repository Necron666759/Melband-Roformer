"""Conservative audio probing and signal measurements."""
from __future__ import annotations

from pathlib import Path

from .types import AudioMetadata, AnalysisReport


def analyze_audio(path: str | Path) -> AnalysisReport:
    """Read basic technical metadata and signal measurements.

    soundfile is already a dependency of the parent application. No neural
    model is loaded and no audio is uploaded or modified by this function.
    """
    import numpy as np
    import soundfile as sf

    source = Path(path).expanduser()
    if not source.is_file():
        raise FileNotFoundError(f"Input audio file does not exist: {source}")

    info = sf.info(str(source))
    if info.frames <= 0 or info.samplerate <= 0 or info.channels <= 0:
        raise ValueError(f"Audio file has invalid stream metadata: {source}")

    peak = rms = clipping_fraction = None
    # Read in bounded blocks to avoid holding a long track in memory.
    peak_value = 0.0
    sum_squares = 0.0
    sample_count = 0
    clipped_count = 0
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

    if sample_count:
        peak = peak_value
        rms = (sum_squares / sample_count) ** 0.5
        clipping_fraction = clipped_count / sample_count

    metadata = AudioMetadata(
        path=str(source.resolve()),
        duration_seconds=info.frames / float(info.samplerate),
        sample_rate=int(info.samplerate),
        channels=int(info.channels),
        frames=int(info.frames),
        subtype=getattr(info, "subtype", None),
        peak=peak,
        rms=rms,
        clipping_fraction=clipping_fraction,
    )
    notes = [
        "This initial analysis reports signal measurements only.",
        "Source identities, genre, BPM and musical structure are not inferred in this phase.",
    ]
    if clipping_fraction is not None and clipping_fraction > 0:
        notes.append("Samples at or above 0.999 full scale were detected; this is a clipping-risk indicator, not proof of audible distortion.")
    return AnalysisReport(
        schema_version=1,
        audio=metadata,
        measurements={
            "peak_dbfs": (20.0 * __import__("math").log10(max(peak, 1e-12))) if peak is not None else None,
            "rms_dbfs": (20.0 * __import__("math").log10(max(rms, 1e-12))) if rms is not None else None,
            "clipping_fraction": clipping_fraction,
        },
        predictions=[],
        notes=notes,
    )
