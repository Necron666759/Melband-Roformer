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
# Source-family prompts and genre prompts are scored separately. Genre scores
# describe broad musical context; they must never be treated as proof that a
# particular source stem exists.
SOURCES = {
    "vocals": "human singing voice, lead vocals and vocal harmonies",
    "drums": "acoustic drum kit, electronic drum kit and percussion",
    "electronic_drums": "programmed electronic drums, drum machine and synthesized percussion",
    "breakbeats": "breakbeat drums, chopped breaks and sampled drum loops",
    "bass": "bass guitar and electric bass line",
    "synth_bass": "synthesizer bass, sub bass and electronic bass line",
    "guitar": "electric guitar and acoustic guitar",
    "piano": "acoustic piano, electric piano and keyboard",
    "synthesizer": "electronic synthesizer sounds and synthesizer instruments",
    "synth_lead": "monophonic synthesizer lead melody and electronic lead synth",
    "synth_pad": "sustained synthesizer pad, atmospheric synth chords and evolving textures",
    "synth_arp": "sequenced synthesizer arpeggio and repeating electronic synth pattern",
    "synth_keys": "electronic keyboard, synthesizer keys and digital keyboard",
    "synth_strings": "synthesized string ensemble and orchestral synthesizer",
    "synth_brass": "synthesized brass stabs and electronic horn sounds",
    "synth_vocals": "vocoder, talkbox and synthetic vocal effect",
    "synth_fx": "electronic risers, sweeps, impacts, noise bursts and sound design effects",
    "strings": "orchestral strings, violin, viola and cello",
    "brass": "acoustic brass instruments and horns",
    "woodwinds": "woodwind instruments, flute and saxophone",
    "sound_effects": "electronic sound effects, noise textures and transitions",
    "samples": "sampled musical loops, chopped samples and audio chops",
}

GENRES = {
    "techno": "techno electronic dance music with a repetitive four on the floor kick",
    "house": "house music with a four on the floor groove, bass and dance synthesizers",
    "trance": "trance electronic dance music with melodic arpeggios and long synthesizer builds",
    "drum_and_bass": "drum and bass electronic music with fast breakbeats and deep bass",
    "dubstep": "dubstep electronic music with heavy wobble bass and half time drums",
    "breakbeat": "breakbeat electronic dance music with syncopated chopped drum loops",
    "ambient_electronic": "ambient electronic music with atmospheric pads and evolving textures",
    "synthwave": "synthwave music with retro analog synthesizers and electronic drums",
    "industrial_electronic": "industrial electronic music with distorted synths and mechanical percussion",
    "electro": "electro electronic music with synthetic bass lines and drum machine rhythms",
}

CLAP_CANDIDATES = tuple(
    (kind, name, description)
    for kind, collection in (("source", SOURCES), ("genre", GENRES))
    for name, description in collection.items()
)

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
    similarities = np.zeros(len(CLAP_CANDIDATES), dtype=np.float64)
    count = 0
    labels = [f"The recording contains {description}." for _, _, description in CLAP_CANDIDATES]
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
        for (kind, name, _), score in zip(CLAP_CANDIDATES, mean_scores):
            result.append({"kind": kind, "source": name, "score": round(float(score), 6),
                           "windows": count, "model": MODEL_ID})
    # Function-local tensors and arrays are released on return. Avoid an
    # unconditional del of per-window locals: an empty valid stream may not
    # have created those names, and that must not mask the analysis result.
    del model, processor
    if target_device == "cuda":
        torch.cuda.empty_cache()
    gc.collect()
    return sorted(result, key=lambda item: item["score"], reverse=True)
