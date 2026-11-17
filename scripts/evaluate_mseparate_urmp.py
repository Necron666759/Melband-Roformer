#!/usr/bin/env python3
"""Run production MSeparate analysis/planning against the selected URMP mixtures.

Audio is read from an ephemeral runner directory. The report contains metadata,
ground-truth instrument labels, analysis hints and planner decisions only.
No audio or model weights are uploaded.
"""
from __future__ import annotations

import argparse
import json
from dataclasses import replace
import os
import sys
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
APP = ROOT / "app"
if str(APP) not in sys.path:
    sys.path.insert(0, str(APP))

GIB = 1024 ** 3


def _rtx3060ti_low_vram_boundary() -> tuple[int, int]:
    """Synthetic admission-boundary probe; not a live CI or user GPU reading."""
    return 6 * GIB, int(7.68 * GIB)


def _ground_truth_ranking(predictions: list[dict], instruments: list[str]) -> dict:
    """Report where known instrument labels rank in CLAP source similarities."""
    ranked = sorted(
        (item for item in predictions if item.get("kind", "source") == "source"),
        key=lambda item: float(item.get("score", 0.0)),
        reverse=True,
    )
    positions = {str(item.get("source")): (rank, float(item.get("score", 0.0)))
                 for rank, item in enumerate(ranked, start=1)}
    unique = list(dict.fromkeys(instruments))
    rows = []
    for instrument in unique:
        rank_score = positions.get(instrument)
        rows.append({
            "instrument": instrument,
            "rank": rank_score[0] if rank_score else None,
            "score": round(rank_score[1], 6) if rank_score else None,
            "in_top_5": bool(rank_score and rank_score[0] <= 5),
            "in_top_12": bool(rank_score and rank_score[0] <= 12),
            "mixture_count": instruments.count(instrument),
        })
    found = sum(row["in_top_12"] for row in rows)
    return {
        "unique_ground_truth_instruments": len(unique),
        "matched_in_top_5": sum(row["in_top_5"] for row in rows),
        "matched_in_top_12": found,
        "top_5_recall": round(sum(row["in_top_5"] for row in rows) / len(unique), 4) if unique else None,
        "top_12_recall": round(found / len(unique), 4) if unique else None,
        "instruments": rows,
        "duplicate_instrument_note": (
            "Repeated instruments share one mixture-level label rank; CLAP cannot count "
            "how many separate performers or instances are present."
        ),
    }


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
    # This deterministic boundary scenario asks whether the already-verified
    # RTX 3060 Ti low-VRAM profile would admit the model if >=6 GiB were free.
    # It is not a live GPU probe and does not execute CUDA inference in CI.
    cuda_registry, cuda_notes = filter_registry_for_device(
        registry, "cuda", memory_probe=_rtx3060ti_low_vram_boundary,
    )
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
        cuda_mono_plan = build_plan(
            max_depth=3, analysis=analysis, registry=cuda_registry,
            allow_resample=True,
        )
        # Metadata-only preview: tests which acoustic heads would be planned
        # after a future, explicit mono-to-dual-mono adapter. No audio is
        # altered here and this preview is not an executable inference plan.
        stereo_preview_analysis = replace(
            analysis, audio=replace(analysis.audio, channels=2),
        )
        cuda_stereo_preview_plan = build_plan(
            max_depth=3, analysis=stereo_preview_analysis, registry=cuda_registry,
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
            "ground_truth_ranking": _ground_truth_ranking(found, MIXES[path.parent.name]),
            "top_genre_predictions": sorted(
                [item for item in found if item.get("kind") == "genre"],
                key=lambda item: float(item.get("score", 0.0)),
                reverse=True,
            )[:5],
            "plan": to_jsonable(plan),
            "planner_task_count": len(plan.tasks),
            "planner_tasks": [to_jsonable(task) for task in plan.tasks],
            "cuda_low_vram_mono_plan": to_jsonable(cuda_mono_plan),
            "cuda_low_vram_mono_task_count": len(cuda_mono_plan.tasks),
            "cuda_low_vram_stereo_adapter_preview": to_jsonable(cuda_stereo_preview_plan),
            "cuda_low_vram_stereo_adapter_preview_task_count": len(cuda_stereo_preview_plan.tasks),
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
        "cuda_low_vram_scenario": {
            "name": "RTX 3060 Ti verified low-VRAM admission boundary",
            "assumed_total_vram_gib": 7.68,
            "assumed_free_vram_gib": 6.0,
            "registry_model_ids": sorted(cuda_registry),
            "preflight_notes": list(cuda_notes),
            "warning": "Synthetic threshold scenario only; not a live GPU probe or a CUDA inference test.",
        },
        "tracks": rows,
        "summary": {
            "mixtures": len(rows),
            "stereo_mixtures": sum(row["audio"]["channels"] >= 2 for row in rows),
            "mono_mixtures": sum(row["audio"]["channels"] == 1 for row in rows),
            "mixtures_with_planner_tasks": sum(row["planner_task_count"] > 0 for row in rows),
            "total_planner_tasks": sum(row["planner_task_count"] for row in rows),
            "mixtures_with_cuda_low_vram_mono_tasks": sum(row["cuda_low_vram_mono_task_count"] > 0 for row in rows),
            "mixtures_with_cuda_low_vram_stereo_preview_tasks": sum(row["cuda_low_vram_stereo_adapter_preview_task_count"] > 0 for row in rows),
            "total_cuda_low_vram_stereo_preview_tasks": sum(row["cuda_low_vram_stereo_adapter_preview_task_count"] for row in rows),
            "ground_truth_instrument_labels": sum(row["ground_truth_ranking"]["unique_ground_truth_instruments"] for row in rows),
            "ground_truth_labels_in_clap_top12": sum(row["ground_truth_ranking"]["matched_in_top_12"] for row in rows),
        },
        "limitations": [
            "This stage evaluates source analysis and routing only; it does not run separation inference.",
            "CLAP similarities do not establish instrument presence or separation quality.",
            "URMP references may require temporal alignment and level handling before valid SDR/SI-SDR calculation.",
            "The CUDA low-VRAM scenario uses a deterministic synthetic 7.68 GiB total / 6.0 GiB free admission-boundary probe; it is not a live hardware reading.",
            "The stereo-adapter preview changes channel metadata only. It does not duplicate audio, run inference, or validate the quality of mono-to-stereo adaptation.",
            "Ground-truth CLAP ranks are retrieval diagnostics, not calibrated instrument-presence probabilities; repeated performers cannot be counted from mixture-level similarities.",
        ],
    }
    args.report.write_text(json.dumps(payload, indent=2, ensure_ascii=False), encoding="utf-8")
    print(json.dumps(payload["summary"], indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
