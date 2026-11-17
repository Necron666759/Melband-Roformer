#!/usr/bin/env python3
"""Evaluate ZFTurbo 53-stem Snare/Toms on aligned, ground-truth drum mixtures.

CSV columns: track_id,mixture,snare,toms
Paths are relative to the CSV file unless absolute. The toms cell may contain
multiple paths separated by semicolons; those reference stems are summed.
"""
from __future__ import annotations

import argparse
import csv
import json
import math
import sys
from dataclasses import replace
from pathlib import Path

APP_DIR = Path(__file__).resolve().parents[1] / "app"
if str(APP_DIR) not in sys.path:
    sys.path.insert(0, str(APP_DIR))

import numpy as np
import soundfile as sf

from mseparate.assets import ensure_model_assets
from mseparate.bs_roformer import BSRoformerAdapter
from mseparate.evaluation import si_sdr_db
from mseparate.registry import bundled_registry_path, load_registry

MODEL_ID = "zfturbo-bs-roformer-53stem-candidate"
TARGETS = ("snare", "toms")


def _resolve(base: Path, value: str) -> list[Path]:
    paths = [part.strip() for part in value.split(";") if part.strip()]
    if not paths:
        raise ValueError("A reference path cell is empty")
    return [(Path(part).expanduser() if Path(part).is_absolute()
             else (base / part).resolve()) for part in paths]


def _read_audio(paths: list[Path], *, label: str, sample_rate: int) -> np.ndarray:
    combined = None
    for path in paths:
        if not path.is_file():
            raise FileNotFoundError(f"{label} audio does not exist: {path}")
        samples, rate = sf.read(str(path), dtype="float32", always_2d=True)
        if rate != sample_rate:
            raise ValueError(f"{label} sample rate mismatch in {path}: {rate} != {sample_rate}")
        if samples.shape[1] != 2:
            raise ValueError(f"{label} must be stereo; got {samples.shape[1]} channels in {path}")
        if not np.isfinite(samples).all():
            raise ValueError(f"{label} contains NaN/Inf: {path}")
        if combined is None:
            combined = samples.astype(np.float64)
        else:
            if samples.shape != combined.shape:
                raise ValueError(
                    f"{label} stem lengths/channels differ: {path} has {samples.shape}, "
                    f"expected {combined.shape}"
                )
            combined += samples
    assert combined is not None
    return combined.astype(np.float32)


def _load_rows(manifest_path: Path, limit: int | None) -> list[dict[str, str]]:
    with manifest_path.open("r", encoding="utf-8-sig", newline="") as stream:
        reader = csv.DictReader(stream)
        required = {"track_id", "mixture", "snare", "toms"}
        missing = required - set(reader.fieldnames or ())
        if missing:
            raise ValueError("CSV is missing required columns: " + ", ".join(sorted(missing)))
        rows = []
        for row in reader:
            track_id = row.get("track_id", "").strip()
            if not track_id or track_id in {".", ".."} or "/" in track_id or "\\" in track_id:
                raise ValueError("track_id must be a non-empty filename-safe identifier")
            if any(not row.get(key, "").strip() for key in ("mixture", "snare", "toms")):
                raise ValueError(f"Track {row['track_id']} has an empty required path")
            rows.append(row)
            if limit and len(rows) >= limit:
                break
    if not rows:
        raise ValueError("Evaluation manifest contains no tracks")
    return rows


def _summary(values: list[float]) -> dict:
    finite = [value for value in values if math.isfinite(value)]
    infinite_count = sum(math.isinf(value) for value in values)
    if not finite:
        return {"count": 0, "infinite_count": infinite_count,
                "mean_db": None, "median_db": None, "min_db": None, "max_db": None}
    return {
        "count": len(finite),
        "infinite_count": infinite_count,
        "mean_db": round(float(np.mean(finite)), 4),
        "median_db": round(float(np.median(finite)), 4),
        "min_db": round(float(np.min(finite)), 4),
        "max_db": round(float(np.max(finite)), 4),
    }


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--manifest", type=Path, required=True, help="CSV with mixture and reference stem paths")
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--models-dir", type=Path, default=Path.home() / ".cache/mseparate/models")
    parser.add_argument("--device", choices=("cuda", "cpu"), default="cuda")
    parser.add_argument("--low-vram-profile", action="store_true",
                        help="Use the separately validated 10-second chunk profile; CUDA only")
    parser.add_argument("--max-tracks", type=int, default=None)
    args = parser.parse_args()

    if args.max_tracks is not None and args.max_tracks < 1:
        parser.error("--max-tracks must be >= 1")
    if args.low_vram_profile and args.device != "cuda":
        parser.error("--low-vram-profile requires --device cuda")

    manifest_path = args.manifest.expanduser().resolve()
    output_dir = args.output_dir.expanduser().resolve()
    models_dir = args.models_dir.expanduser().resolve()
    rows = _load_rows(manifest_path, args.max_tracks)

    registry = load_registry(bundled_registry_path())
    model = registry[MODEL_ID]
    if args.low_vram_profile:
        import torch
        if not torch.cuda.is_available():
            raise RuntimeError("CUDA was requested but is unavailable; no CPU fallback is performed")
        free_bytes, total_bytes = torch.cuda.mem_get_info(0)
        total_gib, free_gib = total_bytes / 1024**3, free_bytes / 1024**3
        if total_gib < 7.0 or free_gib < 6.0:
            raise RuntimeError(
                f"Low-VRAM profile requires >=7 GiB total and >=6 GiB free; "
                f"detected {total_gib:.2f} GiB total / {free_gib:.2f} GiB free"
            )
        model = replace(model, inference_params=model.low_vram_profile["inference_params"])
    elif args.device == "cuda":
        import torch
        if not torch.cuda.is_available():
            raise RuntimeError("CUDA was requested but is unavailable; no CPU fallback is performed")
        if not args.low_vram_profile:
            free_bytes, total_bytes = torch.cuda.mem_get_info(0)
            total_gib, free_gib = total_bytes / 1024**3, free_bytes / 1024**3
            if total_gib < (model.min_vram_gb or 0) or free_gib < (model.min_free_vram_gb or 0):
                raise RuntimeError(
                    f"Standard 53-stem profile requires >= {model.min_vram_gb:g} GiB total "
                    f"and >= {model.min_free_vram_gb:g} GiB free; detected "
                    f"{total_gib:.2f} GiB total / {free_gib:.2f} GiB free. "
                    "Use --low-vram-profile only if its 7/6 GiB admission check passes."
                )

    model_dir = models_dir / MODEL_ID
    results = ensure_model_assets(model, models_dir, timeout=120.0, retries=3)
    failures = [item for item in results if item.status not in {"cached", "downloaded", "recovered"}]
    if failures:
        raise RuntimeError("; ".join(
            f"{item.role}: {item.status}: {item.message or 'asset verification failed'}"
            for item in failures
        ))

    adapter = BSRoformerAdapter()
    compatible, reason = adapter.compatibility(model)
    if not compatible:
        raise RuntimeError(f"Model manifest rejected: {reason}")

    metric_rows: list[dict] = []
    try:
        adapter.load(model, model_dir, args.device)
        for index, row in enumerate(rows, start=1):
            track_id = row["track_id"].strip()
            base = manifest_path.parent
            mixture_path = _resolve(base, row["mixture"])[0]
            mixture = _read_audio([mixture_path], label=f"{track_id} mixture", sample_rate=model.sample_rate)
            refs = {
                target: _read_audio(_resolve(base, row[target]), label=f"{track_id} {target}",
                                    sample_rate=model.sample_rate)
                for target in TARGETS
            }
            for target, reference in refs.items():
                if reference.shape != mixture.shape:
                    raise ValueError(
                        f"{track_id} {target} shape {reference.shape} does not match mixture {mixture.shape}; "
                        "do not silently crop, pad, or time-align benchmark audio"
                    )

            track_output = output_dir / track_id
            artifacts = adapter.infer(mixture_path, track_output, TARGETS)
            estimates = {item.stem: item.path for item in artifacts}
            for target in TARGETS:
                estimate = _read_audio([estimates[target]], label=f"{track_id} {target} estimate",
                                       sample_rate=model.sample_rate)
                if estimate.shape != refs[target].shape:
                    raise ValueError(f"{track_id} {target} estimate shape differs from reference")
                score = si_sdr_db(refs[target], estimate)
                baseline = si_sdr_db(refs[target], mixture)
                metric_rows.append({
                    "track_id": track_id,
                    "target": target,
                    "si_sdr_db": score,
                    "mixture_baseline_si_sdr_db": baseline,
                    "si_sdri_db": score - baseline,
                    "reference_rms": float(np.sqrt(np.mean(refs[target] ** 2))),
                    "estimate_rms": float(np.sqrt(np.mean(estimate ** 2))),
                })
            print(f"[{index}/{len(rows)}] {track_id}: "
                  + ", ".join(
                      f"{target} SI-SDRi={metric_rows[-2 + offset]['si_sdri_db']:.2f} dB"
                      for offset, target in enumerate(TARGETS)
                  ), flush=True)
    finally:
        adapter.unload()

    summaries = {
        target: {
            "si_sdr": _summary([r["si_sdr_db"] for r in metric_rows if r["target"] == target]),
            "si_sdri": _summary([r["si_sdri_db"] for r in metric_rows if r["target"] == target]),
        }
        for target in TARGETS
    }
    output_dir.mkdir(parents=True, exist_ok=True)
    report = {
        "schema_version": 1,
        "model_id": MODEL_ID,
        "device": args.device,
        "low_vram_profile": args.low_vram_profile,
        "sample_rate": model.sample_rate,
        "tracks_evaluated": len(rows),
        "metric": "SI-SDR and SI-SDR improvement over the unprocessed mixture",
        "alignment_policy": "strict: identical sample rate, stereo channels and frame count; no implicit resampling/cropping/padding",
        "summary": summaries,
        "tracks": metric_rows,
    }
    def json_safe(value):
        if isinstance(value, float) and not math.isfinite(value):
            return "inf" if value > 0 else "-inf"
        if isinstance(value, dict):
            return {key: json_safe(item) for key, item in value.items()}
        if isinstance(value, list):
            return [json_safe(item) for item in value]
        return value

    report_path = output_dir / "snare_toms_evaluation.json"
    report_path.write_text(json.dumps(json_safe(report), indent=2, allow_nan=False), encoding="utf-8")
    csv_path = output_dir / "snare_toms_evaluation.csv"
    with csv_path.open("w", encoding="utf-8", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=list(metric_rows[0]))
        writer.writeheader()
        writer.writerows(metric_rows)
    print(json.dumps({"summary": summaries, "report": str(report_path), "csv": str(csv_path)}, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
