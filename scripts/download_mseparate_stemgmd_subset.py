#!/usr/bin/env python3
"""Download a compact, aligned StemGMD Snare/Toms evaluation subset.

Only selected WAV members are fetched through HTTP Range requests. The full
5 GB archive is never downloaded. StemGMD stores the nine isolated kit pieces,
not a separate mixture file, so each mixture is reconstructed by summing the
nine aligned reference stems without normalization.
"""
from __future__ import annotations

import argparse
import csv
import json
import re
import shutil
from collections import defaultdict
from pathlib import Path, PurePosixPath
from typing import Any

ARCHIVE_URL = (
    "https://zenodo.org/records/7882857/files/"
    "StemGMD_eval_session.zip?download=1"
)
INSTRUMENTS = (
    "kick", "snare", "hi_tom", "mid_tom", "low_tom",
    "hihat_closed", "hihat_open", "crash_left", "ride",
)
TOM_INSTRUMENTS = ("hi_tom", "mid_tom", "low_tom")
SAMPLE_RATE = 44100
CHANNELS = 2


def _track_id(drummer: str, kit: str, groove: str) -> str:
    raw = f"{drummer}__{kit}__{PurePosixPath(groove).stem}"
    return re.sub(r"[^A-Za-z0-9_.-]+", "_", raw).strip("._-")


def index_archive_members(infos: list[Any]) -> list[dict[str, Any]]:
    """Index complete nine-stem groups from the observed Eval Session layout."""
    grouped: dict[tuple[str, str, str], dict[str, Any]] = {}
    for info in infos:
        path = PurePosixPath(info.filename)
        if path.suffix.lower() != ".wav" or len(path.parts) != 7:
            continue
        if path.parts[:2] != ("StemGMD", "audio") or path.parts[3] != "eval_session":
            continue
        drummer, kit, instrument, groove = (
            path.parts[2], path.parts[4], path.parts[5], path.parts[6]
        )
        if instrument not in INSTRUMENTS:
            continue
        key = (drummer, kit, groove)
        row = grouped.setdefault(key, {
            "drummer": drummer,
            "kit": kit,
            "groove": groove,
            "members": {},
        })
        if instrument in row["members"]:
            raise ValueError(f"Duplicate {instrument} member in StemGMD group {key}")
        row["members"][instrument] = info.filename

    complete = [
        row for row in grouped.values()
        if set(row["members"]) == set(INSTRUMENTS)
    ]
    complete.sort(key=lambda row: (row["kit"], row["drummer"], row["groove"]))
    return complete


def select_tracks(groups: list[dict[str, Any]], maximum: int) -> list[dict[str, Any]]:
    """Greedily prefer distinct kits, drummers and grooves, with stable ties."""
    if maximum < 1 or maximum > 8:
        raise ValueError("max_tracks must be between 1 and 8")
    remaining = list(groups)
    selected: list[dict[str, Any]] = []
    kits: set[str] = set()
    drummers: set[str] = set()
    grooves: set[str] = set()
    while remaining and len(selected) < maximum:
        best = max(
            remaining,
            key=lambda row: (
                int(row["kit"] not in kits),
                int(row["drummer"] not in drummers),
                int(row["groove"] not in grooves),
                row["kit"], row["drummer"], row["groove"],
            ),
        )
        selected.append(best)
        remaining.remove(best)
        kits.add(best["kit"])
        drummers.add(best["drummer"])
        grooves.add(best["groove"])
    return selected


def _read_reference(path: Path) -> Any:
    import numpy as np
    import soundfile as sf

    samples, rate = sf.read(str(path), dtype="float32", always_2d=True)
    if rate != SAMPLE_RATE:
        raise ValueError(f"StemGMD reference must be 44.1 kHz: {path} has {rate}")
    if samples.shape[1] != CHANNELS:
        raise ValueError(f"StemGMD reference must be stereo: {path} has {samples.shape[1]} channels")
    if samples.shape[0] == 0 or not np.isfinite(samples).all():
        raise ValueError(f"StemGMD reference is empty or contains NaN/Inf: {path}")
    return samples


def download_subset(
    output_dir: Path,
    manifest_path: Path,
    report_path: Path,
    max_tracks: int = 4,
    archive_url: str = ARCHIVE_URL,
) -> dict[str, Any]:
    try:
        from remotezip import RemoteZip
        import numpy as np
        import soundfile as sf
    except ImportError as exc:
        raise RuntimeError(
            "StemGMD subset download requires remotezip, numpy and soundfile"
        ) from exc

    if not 1 <= max_tracks <= 8:
        raise ValueError("max_tracks must be between 1 and 8")
    output_dir.mkdir(parents=True, exist_ok=True)
    manifest_path.parent.mkdir(parents=True, exist_ok=True)
    report_path.parent.mkdir(parents=True, exist_ok=True)

    with RemoteZip(archive_url) as archive:
        groups = index_archive_members(archive.infolist())
        if not groups:
            raise RuntimeError(
                "No complete StemGMD groups found; archive layout differs from "
                "the verified Eval Session structure"
            )
        selected = select_tracks(groups, max_tracks)
        rows: list[dict[str, Any]] = []
        extracted_bytes = 0

        for group in selected:
            track_id = _track_id(group["drummer"], group["kit"], group["groove"])
            track_dir = output_dir / track_id
            reference_dir = track_dir / "references"
            reference_dir.mkdir(parents=True, exist_ok=True)
            local_paths: dict[str, Path] = {}

            for instrument in INSTRUMENTS:
                destination = reference_dir / f"{instrument}.wav"
                member = group["members"][instrument]
                with archive.open(member, "r") as source, destination.open("wb") as target:
                    shutil.copyfileobj(source, target, length=1024 * 1024)
                local_paths[instrument] = destination
                extracted_bytes += destination.stat().st_size

            references = {name: _read_reference(path) for name, path in local_paths.items()}
            frame_counts = {name: samples.shape[0] for name, samples in references.items()}
            if len(set(frame_counts.values())) != 1:
                raise ValueError(
                    f"StemGMD reference lengths are not aligned for {track_id}: {frame_counts}"
                )

            mixture = np.zeros_like(references[INSTRUMENTS[0]], dtype=np.float64)
            for instrument in INSTRUMENTS:
                mixture += references[instrument].astype(np.float64)
            mixture_path = track_dir / "mixture.wav"
            # FLOAT WAV preserves the unnormalized sum without PCM clipping.
            sf.write(str(mixture_path), mixture.astype(np.float32), SAMPLE_RATE, subtype="FLOAT")
            extracted_bytes += mixture_path.stat().st_size

            relative = lambda path: path.relative_to(manifest_path.parent).as_posix()
            row = {
                "track_id": track_id,
                "mixture": relative(mixture_path),
                "snare": relative(local_paths["snare"]),
                "toms": ";".join(relative(local_paths[name]) for name in TOM_INSTRUMENTS),
            }
            rows.append({
                **row,
                "drummer": group["drummer"],
                "kit": group["kit"],
                "groove": PurePosixPath(group["groove"]).stem,
                "source_stems": {
                    name: relative(local_paths[name]) for name in INSTRUMENTS
                },
                "sample_rate": SAMPLE_RATE,
                "channels": CHANNELS,
                "frames": int(next(iter(frame_counts.values()))),
                "duration_seconds": round(next(iter(frame_counts.values())) / SAMPLE_RATE, 6),
                "mixture_peak": round(float(np.max(np.abs(mixture))), 8),
            })

    with manifest_path.open("w", encoding="utf-8", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=("track_id", "mixture", "snare", "toms"))
        writer.writeheader()
        for row in rows:
            writer.writerow({key: row[key] for key in writer.fieldnames})

    payload = {
        "schema_version": 1,
        "dataset": "StemGMD Eval Session",
        "source": "https://zenodo.org/records/7882857",
        "license": "CC BY 4.0",
        "archive_url": archive_url,
        "download_method": "HTTP Range requests for selected ZIP members",
        "full_archive_downloaded": False,
        "audio_downloaded": True,
        "selected_tracks": len(rows),
        "selected_track_metadata": rows,
        "instrument_stems_per_track": len(INSTRUMENTS),
        "extracted_audio_files": len(rows) * (len(INSTRUMENTS) + 1),
        "extracted_bytes": extracted_bytes,
        "manifest": str(manifest_path),
        "mixture_construction": (
            "Unnormalized float64 sum of the nine aligned reference stems, "
            "written as stereo 44.1 kHz FLOAT WAV"
        ),
        "quality_evaluation": {
            "runner": "scripts/evaluate_mseparate_drum_heads.py",
            "targets": ["snare", "toms"],
            "toms_reference": "sum of hi_tom, mid_tom and low_tom",
            "status": "prepared_not_run",
        },
        "limitations": [
            "StemGMD uses sample-library-rendered acoustic kits, not live microphone recordings.",
            "The compact subset is a controlled technical benchmark, not a statistically representative evaluation.",
            "A CUDA-capable run is required for the GPU-only ZFTurbo 53-stem model; CPU fallback is not permitted.",
        ],
    }
    report_path.write_text(
        json.dumps(payload, indent=2, ensure_ascii=False),
        encoding="utf-8",
    )
    return payload


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--manifest", type=Path, required=True)
    parser.add_argument("--report", type=Path, required=True)
    parser.add_argument("--max-tracks", type=int, default=4)
    args = parser.parse_args()
    result = download_subset(
        args.output_dir, args.manifest, args.report, args.max_tracks
    )
    print(json.dumps({
        "selected_tracks": result["selected_tracks"],
        "extracted_audio_files": result["extracted_audio_files"],
        "extracted_mib": round(result["extracted_bytes"] / (1024 * 1024), 2),
        "full_archive_downloaded": result["full_archive_downloaded"],
        "manifest": result["manifest"],
        "report": str(args.report),
    }, indent=2, ensure_ascii=False))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
