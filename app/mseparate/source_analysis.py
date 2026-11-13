"""Open-vocabulary musical-source analysis using LAION-CLAP.

Scores are relative audio/text similarities, not calibrated probabilities. The
analyzer runs short overlapping windows and releases the model before stem
separation so consumer GPUs do not need both networks resident simultaneously.
"""
from __future__ import annotations

import gc
from pathlib import Path

import numpy as np
import soundfile as sf
import soxr

MODEL_ID = "laion/clap-htsat-unfused"
SOURCES = {
    "vocals": "human singing voice and vocal harmonies",
    "drums": "acoustic or electronic drum kit and percussion",
    "bass": "bass guitar and synthesized bass line",
    "guitar": "electric or acoustic guitar",
    "piano": "piano, electric piano and keyboard",
    "synthesizer": "electronic synthesizer leads, pads and arpeggios",
    "strings": "orchestral strings, violin and cello",
    "brass": "brass instruments and horns",
    "woodwinds": "woodwind instruments, flute and saxophone",
    "sound_effects": "electronic sound effects, noise textures and risers",
}

def analyze_sources(path: str | Path, *, device: str = "auto",
                    cache_dir: str | Path | None = None,
                    no_download: bool = False) -> list[dict]:
    """Rank candidate sources across 10-second windows with CLAP."""
    try:
        import torch
        from transformers import ClapModel, ClapProcessor
    except ImportError as exc:
        raise RuntimeError(
            "Source recognition requires transformers and torch. "
            "Rebuild the application with the optional source-analysis dependencies."
        ) from exc

    source = Path(path)
    info = sf.info(str(source))
    if info.channels < 1:
        raise ValueError("Audio stream has no channels")
    target_device = "cuda" if device in {"auto", "cuda"} and torch.cuda.is_available() else "cpu"
    if device == "cuda" and target_device != "cuda":
        raise RuntimeError("CUDA was requested for source analysis but PyTorch CUDA is unavailable")

    kwargs = {"cache_dir": str(cache_dir)} if cache_dir else {}
    if no_download:
        kwargs["local_files_only"] = True
    processor = ClapProcessor.from_pretrained(MODEL_ID, **kwargs)
    model = ClapModel.from_pretrained(MODEL_ID, **kwargs).to(target_device).eval()

    # CLAP's audio encoder expects 48 kHz. Read in bounded chunks to avoid
    # holding a long 96 kHz master in memory.
    window_seconds, hop_seconds, sample_rate = 10.0, 7.5, 48000
    window = int(window_seconds * sample_rate)
    similarities = np.zeros(len(SOURCES), dtype=np.float64)
    count = 0
    labels = [f"The recording contains {description}." for description in SOURCES.values()]
    with torch.inference_mode():
        with sf.SoundFile(str(source), "r") as audio:
            # Seeking lets us keep memory bounded for long recordings.
            start = 0
            while start < info.frames:
                audio.seek(start)
                frames = audio.read(frames=int(window_seconds * info.samplerate),
                                    dtype="float32", always_2d=True)
                if len(frames) == 0:
                    break
                mono = frames.mean(axis=1)
                resampled = soxr.resample(mono, info.samplerate, sample_rate).astype(np.float32)
                if len(resampled) < window:
                    resampled = np.pad(resampled, (0, window - len(resampled)))
                else:
                    resampled = resampled[:window]
                inputs = processor(text=labels, audio=resampled,
                                   sampling_rate=sample_rate, return_tensors="pt", padding=True)
                inputs = {key: value.to(target_device) for key, value in inputs.items()}
                outputs = model(**inputs)
                scores = outputs.logits_per_audio.squeeze(0)
                similarities += scores.detach().float().cpu().numpy()
                count += 1
                start += max(1, int(hop_seconds * info.samplerate))

    result = []
    if count:
        mean_scores = similarities / count
        for name, score in zip(SOURCES, mean_scores):
            result.append({"source": name, "score": round(float(score), 6),
                           "windows": count, "model": MODEL_ID})
    del model, processor, outputs, inputs, frames, mono, resampled, scores
    if target_device == "cuda":
        torch.cuda.empty_cache()
    gc.collect()
    return sorted(result, key=lambda item: item["score"], reverse=True)
