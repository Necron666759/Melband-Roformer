#!/usr/bin/env python3
"""Run production MSeparate analysis/planning against the selected URMP mixtures.

Audio is read from an ephemeral runner directory. The report contains metadata,
ground-truth instrument labels, analysis hints and planner decisions only.
No audio or model weights are uploaded.
"""
from __future__ import annotations

import argparse
import json
import os
import sys
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
APP = ROOT / "app"
if str(APP) not in sys.path:
    sys.path.insert(0, str(APP))

MIXES = {
    "11_Maria_ob_vc": ["oboe", "cello"],
    "12_Spring_vn_vn_vc": ["violin", "violin", "cello"],
    "28_Fugue_fl_ob_cl_bn": ["flute", "oboe", "clarinet", "bassoon"],
    "37_Rondeau_fl_vn_va_cl": ["flute", "violin", "viola", "clarinet"],
}


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--audio-dir", type=Path, required=True)
    parser.add_argument("--report", type=Path, required=True)
    args = parser.parse_args()

    import soundfile as sf
    from mseparate.audio import analyze_audio
    from mseparate.planner import build_plan
    from mseparate.registry import bundled_registry_path, load_registry
    from mseparate.source_analysis import analyze_sources_many
    from mseparate.types import to_jsonable
    from mseparate.vram import filter_registry_for_device

    audio_root = args.audio_dir.resolve()
    mix_paths = [audio_root / name / f"AuMix_{name}.wav" for name in MIXES]
    missing = [str(path) for path in mix_paths if not path.is_file()]
    if missing:
        raise FileNotFoundError("Missing URMP mixtures: " + ", ".join(missing))

    started = time.monotonic()
    audio_reports = {}
    file_metadata = {}
    for path in mix_paths:
        info = sf.info(str(path))
        file_metadata[str(path)] = {
            "sample_rate": info.samplerate,
            "channels": info.channels,
            "frames": info.frames,
            "duration_seconds": round(info.duration, 6),
            "subtype": info.subtype,
        }
        audio_reports[str(path)] = analyze_audio(path)

    models_dir = Path(os.environ.get("RUNNER_TEMP", "/tmp")) / "mseparate-urmp-models"
    predictions = analyze_sources_many(
        mix_paths, device="cpu", cache_dir=models_dir / "source-analysis",
        no_download=False,
    )

    registry = load_registry(bundled_registry_path())
    cpu_registry, cpu_notes = filter_registry_for_device(registry, "cpu")
    rows = []
    for path in mix_paths:
        report = audio_reports[str(path)]
        found = predictions.get(str(path), [])
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
        rows.append({
            "track": path.parent.name,
            "ground_truth_instruments": MIXES[path.parent.name],
            "audio": file_metadata[str(path)],
            "top_source_predictions": sorted(
                [item for item in found if item.get("kind", "source") == "source"],
                key=lambda item: float(item.get("score", 0.0)),
                reverse=True,
            )[:12],
            "top_genre_predictions": sorted(
                [item for item in found if item.get("kind") == "genre"],
                key=lambda item: float(item.get("score", 0.0)),
                reverse=True,
            )[:5],
            "plan": to_jsonable(plan),
            "planner_task_count": len(plan.tasks),
            "planner_tasks": [
                {"model_id": task.model_id, "requested_stems": list(task.requested_stems)}
                for task in plan.tasks
            ],
        })

    args.report.parent.mkdir(parents=True, exist_ok=True)
    payload = {
        "schema_version": 1,
        "dataset": "URMP",
        "dataset_note": (
            "Four manually selected acoustic chamber pieces. Ground-truth instrument "
            "labels are taken from the dataset piece names and isolated AuSep tracks."
        ),
        "runtime": {
            "device": "cpu",
            "elapsed_seconds": round(time.monotonic() - started, 3),
            "repository_commit": os.environ.get("GITHUB_SHA"),
        },
        "analysis": {
            "clap_model": "laion/clap-htsat-unfused",
            "note": "CLAP scores are relative semantic similarities, not probabilities or ground truth.",
        },
        "cpu_registry_model_ids": sorted(cpu_registry),
        "cpu_excluded_model_notes": list(cpu_notes),
        "tracks": rows,
        "summary": {
            "mixtures": len(rows),
            "stereo_mixtures": sum(row["audio"]["channels"] >= 2 for row in rows),
            "mono_mixtures": sum(row["audio"]["channels"] == 1 for row in rows),
            "mixtures_with_planner_tasks": sum(row["planner_task_count"] > 0 for row in rows),
            "total_planner_tasks": sum(row["planner_task_count"] for row in rows),
        },
        "limitations": [
            "This stage evaluates source analysis and routing only; it does not run separation inference.",
            "CLAP similarities do not establish instrument presence or separation quality.",
            "URMP references may require temporal alignment and level handling before valid SDR/SI-SDR calculation.",
            "CPU eligibility does not establish CUDA compatibility or RTX 3060 Ti VRAM behavior.",
        ],
    }
    args.report.write_text(json.dumps(payload, indent=2, ensure_ascii=False), encoding="utf-8")
    print(json.dumps(payload["summary"], indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
