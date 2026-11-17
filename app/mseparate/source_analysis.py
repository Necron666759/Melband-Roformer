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
    "vocals": "human singing voice, lead vocals and vocal harmonies",
    "drums": "acoustic drum kit, electronic drum kit and percussion",
    "electronic_drums": "programmed electronic drums, drum machine and synthesized percussion",
    "kick": "electronic kick drum, four on the floor kick and sub kick",
    "snare": "snare drum, clap snare and electronic snare hits",
    "hi_hat": "closed hi-hat, open hi-hat and fast electronic hi-hat patterns",
    "breakbeats": "breakbeat drums, chopped breaks and sampled drum loops",
    "bass": "bass guitar and electric bass line",
    "synth_bass": "synthesizer bass, sub bass and electronic bass line",
    "guitar": "real acoustic or electric guitar performance, not a synthesizer",
    "percussion": "acoustic hand percussion and non-kit percussion instruments",
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
    "strings": "acoustic orchestral strings, violin, viola and cello section",
    "brass": "acoustic brass instruments and horns",
    "woodwinds": "acoustic woodwind instruments and orchestral woodwinds",
    "violin": "solo acoustic violin playing a bowed melody, not a synthesizer",
    "viola": "acoustic viola playing a bowed string part, not a synthesizer",
    "cello": "acoustic cello playing a bowed low string part, not a synthesizer",
    "double_bass": "acoustic double bass or orchestral contrabass, not synthesized bass",
    "harp": "acoustic concert harp playing plucked strings, not a synthesizer",
    "flute": "acoustic concert flute playing a wind instrument melody, not a synthesizer",
    "clarinet": "acoustic clarinet playing a woodwind part, not a synthesizer",
    "oboe": "acoustic oboe playing an orchestral woodwind part, not a synthesizer",
    "bassoon": "acoustic bassoon playing a low woodwind part, not a synthesizer",
    "saxophone": "acoustic saxophone playing a jazz or orchestral wind part, not a synthesizer",
    "trombone": "acoustic trombone playing a brass section part, not a synthesizer",
    "trumpet": "acoustic trumpet playing a brass melody or fanfare, not a synthesizer",
    "tuba": "acoustic tuba playing a low brass part, not a synthesizer",
    "french_horn": "acoustic French horn playing an orchestral brass part, not a synthesizer",
    "acoustic_guitar": "real acoustic guitar strings being plucked or strummed",
    "electric_guitar": "real electric guitar performance with amplified guitar tone",
    "banjo": "acoustic banjo playing plucked strings",
    "mandolin": "acoustic mandolin playing tremolo or plucked strings",
    "accordion": "acoustic accordion or concertina performance",
    "harpsichord": "acoustic harpsichord playing plucked keyboard strings",
    "acoustic_piano": "acoustic grand or upright piano performance with hammers and strings",
    "digital_piano": "digital piano or electric keyboard performance",
    "organ": "organ keyboard performance with sustained pipe or tonewheel notes",
    "marimba": "acoustic marimba playing wooden pitched percussion",
    "glockenspiel": "acoustic glockenspiel or orchestral bells",
    "timpani": "acoustic orchestral timpani drums",
    "congas": "acoustic conga drums played by hand",
    "tambourine": "acoustic tambourine percussion",
    "sitar": "acoustic sitar playing plucked strings",
    "ukulele": "acoustic ukulele performance",
    "synth_flute": "synthesized flute patch or electronic flute-like lead",
    "synth_woodwinds": "synthesized woodwind or flute-like orchestral patch",
    "synth_guitar": "synthesized guitar-like electronic timbre, not a real guitar",
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
    "electro_industrial": "electro-industrial music with distorted electronic synths, sequenced bass and mechanical drum machines",
    "classical": "classical music performed by acoustic orchestral instruments",
    "orchestral": "orchestral music with acoustic strings brass and woodwinds",
    "chamber_music": "classical chamber music performed by a small acoustic ensemble",
    "baroque": "baroque classical music with strings harpsichord and acoustic instruments",
    "folk": "acoustic folk music with traditional live instruments",
    "jazz": "jazz music with acoustic instruments and improvisation",
    "blues": "blues music with guitar bass and live rhythm section",
    "country": "country music with acoustic guitar fiddle and live band",
    "rock": "rock music with electric guitars bass and live drums",
    "metal": "heavy metal music with distorted guitars bass and drums",
    "pop": "pop music with vocals melody and contemporary production",
    "hip_hop": "hip hop music with programmed beats bass and samples",
    "funk": "funk music with syncopated bass guitar and rhythm section",
    "soul": "soul music with expressive vocals and live band instruments",
    "reggae": "reggae music with offbeat guitar or keyboard and bass groove",
}
GENRE_FAMILIES = {
    "techno": "electronic", "house": "electronic", "trance": "electronic",
    "drum_and_bass": "electronic", "dubstep": "electronic", "breakbeat": "electronic",
    "ambient_electronic": "electronic", "synthwave": "electronic",
    "industrial_electronic": "electronic", "electro": "electronic",
    "electro_industrial": "electronic",
    "classical": "orchestral", "orchestral": "orchestral",
    "chamber_music": "orchestral", "baroque": "orchestral",
    "folk": "acoustic_band", "jazz": "acoustic_band", "blues": "acoustic_band",
    "country": "acoustic_band", "rock": "band", "metal": "band",
    "pop": "popular", "hip_hop": "popular", "funk": "popular",
    "soul": "popular", "reggae": "popular",
}


def classify_style(predictions: list[dict] | None) -> dict:
    """Return a transparent, provisional style label from CLAP genre rankings."""
    genres = sorted(
        (item for item in (predictions or []) if item.get("kind") == "genre"),
        key=lambda item: float(item.get("score", 0.0)),
        reverse=True,
    )
    if not genres:
        return {
            "label": "unknown", "family": "unknown", "status": "unclassified",
            "score": None, "margin": None, "alternatives": [],
        }
    best = genres[0]
    runner_up = genres[1] if len(genres) > 1 else None
    margin = (
        float(best.get("score", 0.0)) - float(runner_up.get("score", 0.0))
        if runner_up else None
    )
    ambiguous = margin is not None and margin < 0.12
    label = str(best.get("source", "unknown"))
    return {
        "label": label,
        "family": "mixed_or_uncertain" if ambiguous else GENRE_FAMILIES.get(label, "unknown"),
        "status": "ambiguous" if ambiguous else "provisional",
        "score": round(float(best.get("score", 0.0)), 6),
        "margin": round(margin, 6) if margin is not None else None,
        "alternatives": [
            {"label": str(item.get("source", "")), "score": float(item.get("score", 0.0))}
            for item in genres[:3]
        ],
        "interpretation": "CLAP relative genre similarity; not a calibrated genre probability",
    }


CLAP_CANDIDATES = tuple(
    (kind, name, description)
    for kind, collection in (("source", SOURCES), ("genre", GENRES))
    for name, description in collection.items()
)


def analyze_sources_many(
    paths: list[str | Path] | tuple[str | Path, ...],
    *,
    device: str = "auto",
    cache_dir: str | Path | None = None,
    no_download: bool = False,
) -> dict[str, list[dict]]:
    """Analyze several files with one CLAP model load, sequentially and bounded-memory."""
    unique_paths = list(dict.fromkeys(Path(path).expanduser() for path in paths))
    if not unique_paths:
        return {}
    try:
        import torch
        from transformers import ClapModel, ClapProcessor
    except ImportError as exc:
        raise RuntimeError(
            "Source recognition requires transformers and torch. "
            "Rebuild the application with the optional source-analysis dependencies."
        ) from exc

    target_device = "cuda" if device in {"auto", "cuda"} and torch.cuda.is_available() else "cpu"
    if device == "cuda" and target_device != "cuda":
        raise RuntimeError("CUDA was requested for source analysis but PyTorch CUDA is unavailable")
    kwargs = {"cache_dir": str(cache_dir)} if cache_dir else {}
    if no_download:
        kwargs["local_files_only"] = True
    processor = ClapProcessor.from_pretrained(MODEL_ID, **kwargs)
    model = ClapModel.from_pretrained(MODEL_ID, **kwargs).to(target_device).eval()
    labels = [f"The recording contains {description}." for _, _, description in CLAP_CANDIDATES]
    results: dict[str, list[dict]] = {}
    sample_rate, window_seconds, hop_seconds = 48000, 10.0, 7.5
    window = int(window_seconds * sample_rate)
    try:
        with torch.inference_mode():
            for source in unique_paths:
                info = sf.info(str(source))
                if info.channels < 1 or info.frames <= 0:
                    raise ValueError(f"Audio stream is empty or invalid: {source}")
                similarities = np.zeros(len(CLAP_CANDIDATES), dtype=np.float64)
                count = 0
                with sf.SoundFile(str(source), "r") as audio:
                    start = 0
                    while start < info.frames:
                        audio.seek(start)
                        frames = audio.read(
                            frames=int(window_seconds * info.samplerate),
                            dtype="float32", always_2d=True,
                        )
                        if len(frames) == 0:
                            break
                        mono = frames.mean(axis=1)
                        resampled = soxr.resample(mono, info.samplerate, sample_rate).astype(np.float32)
                        if len(resampled) < window:
                            resampled = np.pad(resampled, (0, window - len(resampled)))
                        else:
                            resampled = resampled[:window]
                        inputs = processor(
                            text=labels, audio=resampled, sampling_rate=sample_rate,
                            return_tensors="pt", padding=True,
                        )
                        inputs = {key: value.to(target_device) for key, value in inputs.items()}
                        outputs = model(**inputs)
                        similarities += outputs.logits_per_audio.squeeze(0).detach().float().cpu().numpy()
                        count += 1
                        start += max(1, int(hop_seconds * info.samplerate))
                predictions = []
                if count:
                    for (kind, name, _), score in zip(CLAP_CANDIDATES, similarities / count):
                        predictions.append({
                            "kind": kind, "source": name, "score": round(float(score), 6),
                            "windows": count, "model": MODEL_ID,
                        })
                results[str(source)] = sorted(predictions, key=lambda item: item["score"], reverse=True)
    finally:
        del model, processor
        if target_device == "cuda":
            torch.cuda.empty_cache()
        gc.collect()
    return results


def analyze_sources(
    path: str | Path,
    *,
    device: str = "auto",
    cache_dir: str | Path | None = None,
    no_download: bool = False,
) -> list[dict]:
    """Rank candidate sources across 10-second windows with CLAP."""
    source = Path(path).expanduser()
    return analyze_sources_many(
        [source], device=device, cache_dir=cache_dir, no_download=no_download,
    ).get(str(source), [])
