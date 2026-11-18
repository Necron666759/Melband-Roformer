#!/usr/bin/env python3
"""CPU evaluation of MSeparate routing on the public BabySlakh music dataset.

Stage A: analyze rendered MIDI arrangements and isolated drum stems with LAION-CLAP.
Stage B: build and record the source-aware CPU execution plan for every mix.
Stage C: execute bounded excerpts from representative mixes and collect
technical output checks, post-inference source hints and rescue decisions.

BabySlakh is rendered MIDI music, not live studio recordings. This is a
reproducible routing/integration evaluation, not a claim of real-world
perceptual superiority. Audio and model outputs are never uploaded as artifacts.
"""
from __future__ import annotations

import argparse
import json
import os
import subprocess
import sys
import time
from pathlib import Path

import numpy as np
import soundfile as sf

ROOT = Path(__file__).resolve().parents[1]
APP = ROOT / "app"
if str(APP) not in sys.path:
    sys.path.insert(0, str(APP))

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


def execution_counts(payload: dict | None) -> tuple[int, int, list[dict], int]:
    """Read serialized ExecutionSummary results, not non-serialized properties."""
    payload = payload or {}
    execution = payload.get("execution") or {}
    results = execution.get("results") or []
    succeeded = sum(item.get("status") == "success" for item in results)
    failed = sum(item.get("status") == "failed" for item in results)
    plan = payload.get("plan") or {}
    excerpt_task_count = len(plan.get("tasks") or [])
    return succeeded, failed, results, excerpt_task_count

def stage_c_progress_message(
    index: int, total: int, succeeded: int, failed: int, rescue: dict,
) -> str:
    """Format Stage C progress from the already parsed serialized result counts."""
    rescue_status = (rescue or {}).get("status", "unknown")
    return (
        f"Stage C {index}/{total}: {succeeded} succeeded, "
        f"{failed} failed, rescue={rescue_status}"
    )


def run_cli(
    input_path: Path, output: Path, models_dir: Path, timeout: int,
) -> tuple[int, dict | None, str]:
    output.mkdir(parents=True, exist_ok=True)
    report_path = output / "mseparate-report.json"
    command = [
        sys.executable, "-m", "mseparate", str(input_path),
        "--execute", "--device", "cpu", "--allow-resample", "--allow-mono-to-stereo",
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
    parser.add_argument("--max-tracks", type=int, default=8)
    parser.add_argument("--max-cases", type=int, default=4)
    parser.add_argument("--excerpt-seconds", type=float, default=12.0)
    parser.add_argument("--inference-timeout", type=int, default=1800)
    args = parser.parse_args()

    work = args.work_dir.resolve()
    dataset_root = work / "dataset"
    models_dir = work / "models"
    output_root = work / "outputs"
    report_root = work / "reports"
    for folder in (dataset_root, models_dir, output_root, report_root):
        folder.mkdir(parents=True, exist_ok=True)

    from mseparate.audio import analyze_audio
    from mseparate.planner import build_plan
    from mseparate.registry import bundled_registry_path, load_registry
    from mseparate.source_analysis import analyze_sources_many
    from mseparate.types import to_jsonable
    from mseparate.vram import filter_registry_for_device

    started = time.monotonic()
    subset_manifest = report_root / "babyslakh-subset-manifest.json"
    if not subset_manifest.is_file():
        downloader = ROOT / "scripts" / "download_mseparate_babyslakh_subset.py"
        subprocess.run(
            [
                sys.executable, str(downloader),
                "--output-dir", str(dataset_root),
                "--manifest", str(subset_manifest),
                "--max-tracks", str(args.max_tracks),
            ],
            cwd=ROOT, check=True, env=os.environ.copy(),
        )
    else:
        emit("Reusing the pre-fetched selective BabySlakh subset")
    subset = json.loads(subset_manifest.read_text(encoding="utf-8"))
    subset_rows = subset.get("tracks", [])
    if not subset_rows:
        raise RuntimeError("Selective BabySlakh downloader returned no tracks")
    mixes = [dataset_root / row["mix"] for row in subset_rows]
    missing = [str(path) for path in mixes if not path.is_file()]
    if missing:
        raise FileNotFoundError("Missing selectively downloaded mixtures: " + ", ".join(missing))
    metadata_by_mix = {str(dataset_root / row["mix"]): row for row in subset_rows}
    emit(
        f"Found {len(mixes)} BabySlakh mixtures with isolated drum ground truth; "
        f"downloaded {subset.get('extracted_files')} selected files "
        f"({subset.get('extracted_bytes', 0) / (1024 * 1024):.1f} MiB)"
    )

    # Stage A: deterministic signal analysis and one batched CLAP model session.
    audio_reports = {}
    for index, mix in enumerate(mixes, 1):
        audio_reports[str(mix)] = analyze_audio(mix)
        emit(f"Stage A signal analysis {index}/{len(mixes)}: {mix.parent.name}")
    drum_stem_paths = sorted({
        dataset_root / stem_path
        for row in subset_rows for stem_path in row["drum_stems"]
    })
    analysis_paths = mixes + drum_stem_paths
    predictions = analyze_sources_many(
        analysis_paths, device="cpu", cache_dir=models_dir / "source-analysis",
        no_download=False,
    )
    emit(
        f"Stage A CLAP analysis completed for {len(mixes)} mixtures and "
        f"{len(drum_stem_paths)} isolated drum stems"
    )

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
            allow_resample=True, allow_mono_to_stereo=True,
        )
        plan_json = to_jsonable(plan)
        category, evidence = classify_case(plan_json, found)
        subset_row = metadata_by_mix[str(mix)]
        row = {
            "track": subset_row["track"],
            "input": str(mix.relative_to(dataset_root)),
            "ground_truth": subset_row["ground_truth"],
            "drum_stem_paths": subset_row["drum_stems"],
            "isolated_drum_stem_analysis": [
                {
                    "stem": stem_path,
                    "top_source_predictions": sorted(
                        [
                            item for item in predictions.get(
                                str(dataset_root / stem_path), []
                            )
                            if item.get("kind", "source") == "source"
                        ],
                        key=lambda item: float(item.get("score", 0.0)),
                        reverse=True,
                    )[:12],
                }
                for stem_path in subset_row["drum_stems"]
            ],
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
            succeeded, failed, task_results, excerpt_task_count = execution_counts(payload)
            rescue = (payload or {}).get("conditional_rescue") or {}
            deep = (payload or {}).get("deep_output_analysis") or {}
            full_track_task_count = len(row["plan"].get("tasks") or [])
            stage_c.append({
                "track": row["track"],
                "category": row["category"],
                "exit_code": code,
                "status": (
                    "completed" if succeeded or failed
                    else "no_tasks" if excerpt_task_count == 0
                    else "no_execution_result"
                ),
                "full_track_planner_tasks": full_track_task_count,
                "excerpt_planner_tasks": excerpt_task_count,
                "routing_drift": full_track_task_count > 0 and excerpt_task_count == 0,
                "execution_succeeded": succeeded,
                "execution_failed": failed,
                "execution_skipped": sum(item.get("status") == "skipped" for item in task_results),
                "task_results": task_results,
                "deep_analysis_status": deep.get("status", "completed" if deep else "missing"),
                "candidate_count": deep.get("candidate_count", 0),
                "residual_source_hints": deep.get("residual_source_hints", []),
                "conditional_rescue": rescue,
                "effective_device": (payload or {}).get("effective_device"),
                "vram_preflight_notes": (payload or {}).get("vram_preflight_notes", []),
                "log_tail": tail,
            })
            emit(stage_c_progress_message(
                index, len(cases), succeeded, failed, rescue,
            ))
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
            "license": "CC BY 4.0",
            "tracks_analyzed": len(rows),
            "download_method": subset.get("download_method"),
            "full_archive_downloaded": subset.get("full_archive_downloaded"),
            "extracted_files": subset.get("extracted_files"),
            "extracted_bytes": subset.get("extracted_bytes"),
            "note": (
                "Rendered MIDI compositions with sample-based virtual instruments; "
                "not live studio recordings. Ground truth identifies combined drum "
                "sources from metadata and isolated drum stems, not Snare/Toms components."
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
            "The selected 4-8 tracks are a compact routing diagnostic, not a statistically representative benchmark.",
            "BabySlakh ground truth identifies combined rendered drum sources; it does not distinguish acoustic versus electronic kits or provide isolated Snare/Toms references.",
            "No SDR/SI-SDR is reported because this harness does not align and score the selected combined drum stems against Snare/Toms outputs.",
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
        "drum_ground_truth_tracks": sum(
            1 for row in rows if row.get("ground_truth", {}).get("drums_present")
        ),
        "isolated_drum_stems_analyzed": len(drum_stem_paths),
        "inference_successes": sum(int(item.get("execution_succeeded", 0)) for item in stage_c),
        "inference_failures": sum(int(item.get("execution_failed", 0)) for item in stage_c),
        "cases_without_excerpt_tasks": sum(item.get("status") == "no_tasks" for item in stage_c),
        "routing_drift_cases": sum(bool(item.get("routing_drift")) for item in stage_c),
        "report": str(report_path),
    }
    (report_root / "summary.json").write_text(
        json.dumps(summary, indent=2, ensure_ascii=False), encoding="utf-8",
    )
    emit(json.dumps(summary, ensure_ascii=False))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
