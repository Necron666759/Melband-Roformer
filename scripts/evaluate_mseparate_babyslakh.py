#!/usr/bin/env python3
"""CPU evaluation of MSeparate routing on the public BabySlakh music dataset.

Stage A: analyze real musical arrangements with LAION-CLAP.
Stage B: build and record the source-aware CPU execution plan for every mix.
Stage C: execute bounded excerpts from representative mixes and collect
technical output checks, post-inference source hints and rescue decisions.

BabySlakh is rendered MIDI music, not live studio recordings. This is a
reproducible routing/integration evaluation, not a claim of real-world
perceptual superiority. Audio and model outputs are never uploaded as artifacts.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import os
import shutil
import subprocess
import sys
import time
import zipfile
from pathlib import Path
from urllib.request import Request, urlopen

import numpy as np
import soundfile as sf

ROOT = Path(__file__).resolve().parents[1]
APP = ROOT / "app"
if str(APP) not in sys.path:
    sys.path.insert(0, str(APP))

DATA_URL = "https://zenodo.org/records/4603844/files/babyslakh_16k.zip?download=1"
DATA_MD5 = "ea1797fc57689a0e33c759c17a2292f5"
MIX_NAMES = {"mix.wav", "mix.flac", "mixture.wav", "mixture.flac"}
SYNTH = {
    "synthesizer", "synth_lead", "synth_pad", "synth_arp", "synth_keys",
    "synth_strings", "synth_brass", "synth_bass", "synth_fx",
    "synth_flute", "synth_woodwinds", "synth_guitar",
}
ACOUSTIC = {
    "strings", "violin", "viola", "cello", "double_bass", "harp",
    "brass", "woodwinds", "flute", "clarinet", "oboe", "bassoon",
    "saxophone", "trombone", "trumpet", "tuba", "french_horn",
    "acoustic_guitar", "guitar", "acoustic_piano", "harpsichord",
    "banjo", "mandolin", "accordion", "sitar", "ukulele",
}


def emit(message: str) -> None:
    print(f"[mseparate-eval] {message}", flush=True)


def download_dataset(cache: Path) -> Path:
    cache.mkdir(parents=True, exist_ok=True)
    archive = cache / "babyslakh_16k.zip"
    if archive.exists():
        digest = hashlib.md5()
        with archive.open("rb") as cached:
            for block in iter(lambda: cached.read(1024 * 1024), b""):
                digest.update(block)
        if digest.hexdigest() == DATA_MD5:
            emit(f"Using verified cached BabySlakh archive ({archive.stat().st_size} bytes)")
            return archive
        archive.unlink()
    temporary = archive.with_suffix(".zip.part")
    request = Request(DATA_URL, headers={"User-Agent": "MSeparate-CPU-Evaluation/1.0"})
    digest = hashlib.md5()
    total = 0
    emit("Downloading BabySlakh from Zenodo (about 883 MB)")
    with urlopen(request, timeout=60) as response, temporary.open("wb") as output:
        while True:
            block = response.read(1024 * 1024)
            if not block:
                break
            output.write(block)
            digest.update(block)
            total += len(block)
            if total % (64 * 1024 * 1024) < len(block):
                emit(f"Downloaded {total / (1024 * 1024):.0f} MiB")
    if digest.hexdigest() != DATA_MD5:
        temporary.unlink(missing_ok=True)
        raise RuntimeError(
            f"BabySlakh MD5 mismatch: expected {DATA_MD5}, got {digest.hexdigest()}"
        )
    temporary.replace(archive)
    emit(f"Dataset verified ({total} bytes, MD5 {digest.hexdigest()})")
    return archive


def safe_extract(archive: Path, destination: Path) -> None:
    marker = destination / ".babyslakh-extracted"
    if marker.exists():
        return
    destination.mkdir(parents=True, exist_ok=True)
    with zipfile.ZipFile(archive) as bundle:
        root = destination.resolve()
        for item in bundle.infolist():
            target = (destination / item.filename).resolve()
            if target != root and root not in target.parents:
                raise RuntimeError(f"Unsafe path in dataset archive: {item.filename}")
        bundle.extractall(destination)
    marker.write_text("verified archive extracted\n", encoding="utf-8")


def find_mixes(root: Path, limit: int) -> list[Path]:
    mixes = [
        path for path in root.rglob("*")
        if path.is_file() and path.name.lower() in MIX_NAMES
    ]
    mixes.sort(key=lambda item: str(item).lower())
    if not mixes:
        # Some dataset mirrors use a mixture suffix rather than mix.wav.
        mixes = sorted(
            (path for path in root.rglob("*") if path.is_file()
             and path.stem.lower() in {"mixture", "mix"}),
            key=lambda item: str(item).lower(),
        )
    return mixes[:limit]


def source_sets(predictions: list[dict]) -> tuple[set[str], set[str]]:
    ranked = [
        item for item in predictions
        if item.get("kind", "source") == "source"
    ]
    # Use top-12 labels to avoid treating every weak CLAP match as evidence.
    ranked.sort(key=lambda item: float(item.get("score", 0.0)), reverse=True)
    top = {str(item.get("source")) for item in ranked[:12]}
    return top & SYNTH, top & ACOUSTIC


def classify_case(plan: dict, predictions: list[dict]) -> tuple[str, dict]:
    style = plan.get("style_profile") or {}
    family = style.get("family", "unknown")
    synth, acoustic = source_sets(predictions)
    if synth and acoustic:
        category = "mixed"
    elif family == "electronic" or synth:
        category = "electronic"
    elif family == "orchestral" or acoustic:
        category = "orchestral"
    elif style.get("status") == "ambiguous" or family in {"unknown", "mixed_or_uncertain"}:
        category = "ambiguous"
    else:
        category = "other"
    return category, {
        "style_family": family,
        "style_label": style.get("label"),
        "style_status": style.get("status"),
        "synth_evidence_top12": sorted(synth),
        "acoustic_evidence_top12": sorted(acoustic),
    }


def choose_cases(rows: list[dict], maximum: int) -> list[dict]:
    chosen: list[dict] = []
    used: set[str] = set()
    for category in ("electronic", "orchestral", "mixed", "ambiguous"):
        candidates = [
            row for row in rows
            if row["category"] == category and row["track"] not in used
            and row["plan"].get("tasks")
        ]
        if candidates:
            # Prefer a non-empty, source-specific plan and stronger style margin.
            candidates.sort(
                key=lambda row: (
                    len(row["plan"].get("tasks", [])),
                    float(row["plan"].get("style_profile", {}).get("margin") or 0.0),
                ),
                reverse=True,
            )
            chosen.append(candidates[0])
            used.add(candidates[0]["track"])
        if len(chosen) >= maximum:
            return chosen
    remaining = [row for row in rows if row["track"] not in used and row["plan"].get("tasks")]
    remaining.sort(key=lambda row: len(row["plan"].get("tasks", [])), reverse=True)
    chosen.extend(remaining[: max(0, maximum - len(chosen))])
    return chosen


def run_cli(
    input_path: Path, output: Path, models_dir: Path, timeout: int,
) -> tuple[int, dict | None, str]:
    output.mkdir(parents=True, exist_ok=True)
    report_path = output / "mseparate-report.json"
    command = [
        sys.executable, "-m", "mseparate", str(input_path),
        "--execute", "--device", "cpu", "--allow-resample",
        "--output-dir", str(output), "--models-dir", str(models_dir),
        "--inference-timeout", str(timeout),
        "--report", str(report_path), "--log-file",
    ]
    emit(f"Stage C: {input_path.name} (CPU, per-task timeout {timeout}s)")
    started = time.monotonic()
    process = subprocess.run(
        command, cwd=ROOT, text=True, stdout=subprocess.PIPE,
        stderr=subprocess.STDOUT, timeout=timeout * 6 + 600,
        env=os.environ.copy(),
    )
    elapsed = time.monotonic() - started
    log = process.stdout or ""
    (output / "console.log").write_text(log, encoding="utf-8")
    payload = None
    if report_path.is_file():
        try:
            payload = json.loads(report_path.read_text(encoding="utf-8"))
        except json.JSONDecodeError:
            pass
    emit(f"Stage C finished: exit={process.returncode}, wall={elapsed:.1f}s")
    return process.returncode, payload, log[-12000:]


def crop_center(source: Path, destination: Path, seconds: float) -> None:
    info = sf.info(str(source))
    length = min(info.frames, max(1, int(seconds * info.samplerate)))
    start = max(0, (info.frames - length) // 2)
    with sf.SoundFile(str(source), "r") as stream:
        stream.seek(start)
        audio = stream.read(length, dtype="float32", always_2d=True)
    if not np.isfinite(audio).all() or not len(audio):
        raise RuntimeError(f"Invalid excerpt from {source}")
    sf.write(destination, audio, info.samplerate, subtype="PCM_16")


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--work-dir", type=Path, default=Path("evaluation-work"))
    parser.add_argument("--max-tracks", type=int, default=20)
    parser.add_argument("--max-cases", type=int, default=4)
    parser.add_argument("--excerpt-seconds", type=float, default=12.0)
    parser.add_argument("--inference-timeout", type=int, default=1800)
    args = parser.parse_args()

    work = args.work_dir.resolve()
    dataset_cache = work / "dataset-cache"
    dataset_root = work / "dataset"
    models_dir = work / "models"
    output_root = work / "outputs"
    report_root = work / "reports"
    for folder in (models_dir, output_root, report_root):
        folder.mkdir(parents=True, exist_ok=True)

    from mseparate.audio import analyze_audio
    from mseparate.planner import build_plan
    from mseparate.registry import bundled_registry_path, load_registry
    from mseparate.source_analysis import analyze_sources_many
    from mseparate.types import to_jsonable
    from mseparate.vram import filter_registry_for_device

    started = time.monotonic()
    archive = download_dataset(dataset_cache)
    safe_extract(archive, dataset_root)
    mixes = find_mixes(dataset_root, args.max_tracks)
    if not mixes:
        raise RuntimeError(
            "No mix.wav/mix.flac files found in BabySlakh archive; "
            "inspect archive layout before proceeding."
        )
    emit(f"Found {len(mixes)} musical mixtures")

    # Stage A: deterministic signal analysis and one batched CLAP model session.
    audio_reports = {}
    for index, mix in enumerate(mixes, 1):
        audio_reports[str(mix)] = analyze_audio(mix)
        emit(f"Stage A signal analysis {index}/{len(mixes)}: {mix.parent.name}")
    predictions = analyze_sources_many(
        mixes, device="cpu", cache_dir=models_dir / "source-analysis",
        no_download=False,
    )
    emit("Stage A CLAP source/style analysis completed")

    # Stage B: use the production planner and CPU eligibility policy unchanged.
    registry = load_registry(bundled_registry_path())
    cpu_registry, cpu_notes = filter_registry_for_device(registry, "cpu")
    rows: list[dict] = []
    for mix in mixes:
        report = audio_reports[str(mix)]
        found = predictions.get(str(mix), [])
        analysis = type(report)(
            schema_version=report.schema_version,
            audio=report.audio,
            measurements=report.measurements,
            predictions=found,
            notes=report.notes,
        )
        plan = build_plan(
            max_depth=3, analysis=analysis, registry=cpu_registry,
            allow_resample=True,
        )
        plan_json = to_jsonable(plan)
        category, evidence = classify_case(plan_json, found)
        row = {
            "track": mix.parent.name,
            "input": str(mix.relative_to(dataset_root)),
            "duration_seconds": report.audio.duration_seconds,
            "sample_rate": report.audio.sample_rate,
            "category": category,
            "classification_evidence": evidence,
            "top_source_predictions": [
                item for item in found if item.get("kind", "source") == "source"
            ][:12],
            "top_genre_predictions": [
                item for item in found if item.get("kind") == "genre"
            ][:5],
            "plan": plan_json,
        }
        rows.append(row)
        emit(
            f"Stage B {mix.parent.name}: {category}; "
            f"{len(plan.tasks)} task(s): "
            + (", ".join(task.model_id for task in plan.tasks) or "none")
        )

    cases = choose_cases(rows, args.max_cases)
    emit(
        "Selected Stage C cases: "
        + (", ".join(f'{row["track"]}/{row["category"]}' for row in cases) or "none")
    )
    stage_c = []
    for index, row in enumerate(cases, 1):
        source = dataset_root / row["input"]
        excerpt_dir = work / "excerpts"
        excerpt_dir.mkdir(parents=True, exist_ok=True)
        excerpt = excerpt_dir / f"{row['track']}-center-{args.excerpt_seconds:g}s.wav"
        crop_center(source, excerpt, args.excerpt_seconds)
        case_output = output_root / row["track"]
        try:
            code, payload, tail = run_cli(
                excerpt, case_output, models_dir, args.inference_timeout,
            )
            execution = (payload or {}).get("execution") or {}
            rescue = (payload or {}).get("conditional_rescue") or {}
            deep = (payload or {}).get("deep_output_analysis") or {}
            stage_c.append({
                "track": row["track"],
                "category": row["category"],
                "exit_code": code,
                "execution_succeeded": execution.get("succeeded", 0),
                "execution_failed": execution.get("failed", 0),
                "execution_skipped": execution.get("skipped", 0),
                "task_results": execution.get("results", []),
                "deep_analysis_status": deep.get("status", "completed" if deep else "missing"),
                "candidate_count": deep.get("candidate_count", 0),
                "residual_source_hints": deep.get("residual_source_hints", []),
                "conditional_rescue": rescue,
                "effective_device": (payload or {}).get("effective_device"),
                "vram_preflight_notes": (payload or {}).get("vram_preflight_notes", []),
                "log_tail": tail,
            })
            emit(
                f"Stage C {index}/{len(cases)}: "
                f"{execution.get('succeeded', 0)} succeeded, "
                f"{execution.get('failed', 0)} failed, rescue={rescue.get('status', 'unknown')}"
            )
        except subprocess.TimeoutExpired as exc:
            stage_c.append({
                "track": row["track"], "category": row["category"],
                "status": "timeout", "timeout_seconds": args.inference_timeout * 6 + 600,
                "log_tail": str(exc.stdout or "")[-12000:],
            })
            emit(f"Stage C {row['track']}: timed out; remaining cases will continue")
        except Exception as exc:
            stage_c.append({
                "track": row["track"], "category": row["category"],
                "status": "error", "error": f"{type(exc).__name__}: {exc}",
            })
            emit(f"Stage C {row['track']}: {type(exc).__name__}: {exc}")

    result = {
        "schema_version": 1,
        "repository_commit": os.environ.get("GITHUB_SHA"),
        "dataset": {
            "name": "BabySlakh",
            "source": "https://zenodo.org/records/4603844",
            "archive_md5": DATA_MD5,
            "tracks_analyzed": len(rows),
            "note": (
                "Rendered MIDI compositions with sample-based virtual instruments; "
                "not live studio recordings. Dataset terms and attribution apply."
            ),
        },
        "runtime": {
            "device": "cpu",
            "torch_threads": os.environ.get("OMP_NUM_THREADS"),
            "elapsed_seconds": round(time.monotonic() - started, 3),
            "models_dir": str(models_dir),
        },
        "stages": {
            "A_analysis": {
                "status": "completed",
                "tracks": len(rows),
                "clap_model": "laion/clap-htsat-unfused",
                "note": "CLAP similarities are relative semantic hints, not probabilities.",
            },
            "B_routing": {
                "status": "completed",
                "tracks": rows,
                "cpu_registry_model_ids": sorted(cpu_registry),
                "cpu_excluded_model_notes": list(cpu_notes),
                "coverage": {
                    category: sum(1 for row in rows if row["category"] == category)
                    for category in ("electronic", "orchestral", "mixed", "ambiguous", "other")
                },
            },
            "C_inference_and_output_review": {
                "status": "completed" if stage_c else "not_run",
                "excerpt_seconds": args.excerpt_seconds,
                "selected_cases": stage_c,
                "note": (
                    "Technical checks and CLAP output hints do not prove perceptual quality. "
                    "ZFTurbo 53-stem is GPU/VRAM-gated and is intentionally not CPU-fallback eligible."
                ),
            },
        },
        "limitations": [
            "BabySlakh is rendered MIDI music; live acoustic recordings and vocals are not represented reliably.",
            "The first 20 tracks are a small debug subset, not a statistically representative benchmark.",
            "No SDR/SI-SDR is reported because reference-source grouping and alignment are not yet implemented in this harness.",
            "A CPU run validates the actual planner and CPU-capable specialist inference, not RTX 3060 Ti CUDA performance.",
            "HT-Demucs remains excluded from normal routing and was not invoked.",
        ],
    }
    report_path = report_root / "evaluation-report.json"
    report_path.write_text(json.dumps(result, indent=2, ensure_ascii=False), encoding="utf-8")
    summary = {
        "tracks_analyzed": len(rows),
        "routing_coverage": result["stages"]["B_routing"]["coverage"],
        "selected_cases": len(stage_c),
        "inference_successes": sum(int(item.get("execution_succeeded", 0)) for item in stage_c),
        "inference_failures": sum(int(item.get("execution_failed", 0)) for item in stage_c),
        "report": str(report_path),
    }
    (report_root / "summary.json").write_text(
        json.dumps(summary, indent=2, ensure_ascii=False), encoding="utf-8",
    )
    emit(json.dumps(summary, ensure_ascii=False))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
