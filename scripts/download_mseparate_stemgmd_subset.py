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
import os
import re
import shutil
import time
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
            "member_compressed_sizes": {},
        })
        if instrument in row["members"]:
            raise ValueError(f"Duplicate {instrument} member in StemGMD group {key}")
        row["members"][instrument] = info.filename
        row["member_compressed_sizes"][instrument] = int(
            getattr(info, "compress_size", info.file_size)
        )

    complete = [
        row for row in grouped.values()
        if set(row["members"]) == set(INSTRUMENTS)
    ]
    complete.sort(key=lambda row: (row["kit"], row["drummer"], row["groove"]))
    return complete


def rank_track_candidates(groups: list[dict[str, Any]]) -> list[dict[str, Any]]:
    """Balance target-activity priority with greedy groove/kit/drummer diversity.

    Compressed size is only a scan-order heuristic, never evidence that a
    decoded reference is active. Diversity bonuses are deliberately material:
    a slightly smaller candidate from a new groove should be scanned before
    another kit/drummer rendering of a groove already represented.
    """
    if not groups:
        return []

    def activity_bytes(row: dict[str, Any]) -> int:
        sizes = row.get("member_compressed_sizes", {})
        return sum(int(sizes.get(name, 0)) for name in ("snare", *TOM_INSTRUMENTS))

    maximum_activity = max(activity_bytes(row) for row in groups) or 1
    remaining = list(groups)
    ranked: list[dict[str, Any]] = []
    kits: set[str] = set()
    drummers: set[str] = set()
    grooves: set[str] = set()
    while remaining:
        def priority(row: dict[str, Any]) -> tuple[float, int, str, str, str]:
            activity = activity_bytes(row) / maximum_activity
            novelty = (
                0.20 * int(row["groove"] not in grooves)
                + 0.15 * int(row["kit"] not in kits)
                + 0.10 * int(row["drummer"] not in drummers)
            )
            # Activity retains the majority weight; novelty now affects
            # ordering instead of being consulted only for exact-size ties.
            score = 0.55 * activity + novelty
            return (score, activity_bytes(row), row["kit"], row["drummer"], row["groove"])

        best = max(remaining, key=priority)
        ranked.append(best)
        remaining.remove(best)
        kits.add(best["kit"])
        drummers.add(best["drummer"])
        grooves.add(best["groove"])
    return ranked


def select_tracks(groups: list[dict[str, Any]], maximum: int) -> list[dict[str, Any]]:
    """Return a compact, diverse prefix of the deterministic candidate order."""
    if maximum < 1 or maximum > 8:
        raise ValueError("max_tracks must be between 1 and 8")
    return rank_track_candidates(groups)[:maximum]


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


def _open_remote_archive(remotezip_module: Any, archive_url: str):
    """Open Zenodo ZIP with bounded HTTP retries and URL fallbacks.

    Range requests remain enabled: this never falls back to downloading the
    multi-gigabyte archive. A fresh RemoteZip instance is used per endpoint.
    """
    import requests
    from requests.adapters import HTTPAdapter
    from urllib3.util.retry import Retry

    retry = Retry(
        total=6,
        connect=6,
        read=6,
        status=6,
        backoff_factor=1.0,
        status_forcelist=(408, 429, 500, 502, 503, 504),
        allowed_methods=frozenset(("GET", "HEAD")),
        respect_retry_after_header=True,
        raise_on_status=True,
    )
    session = requests.Session()
    adapter = HTTPAdapter(max_retries=retry, pool_connections=4, pool_maxsize=4)
    session.mount("https://", adapter)
    session.mount("http://", adapter)

    urls = [archive_url]
    if "?download=1" in archive_url:
        urls.append(archive_url.replace("?download=1", ""))
    # Zenodo's record API exposes the same file via its content endpoint.
    if archive_url == ARCHIVE_URL:
        urls.append(
            "https://zenodo.org/api/records/7882857/files/"
            "StemGMD_eval_session.zip/content"
        )

    errors = []
    for url in dict.fromkeys(urls):
        try:
            return remotezip_module.RemoteZip(url, session=session)
        except Exception as exc:
            errors.append(f"{url}: {type(exc).__name__}: {exc}")
    session.close()
    raise RuntimeError(
        "Unable to open StemGMD archive through Zenodo endpoints after "
        "HTTP retries; no full-archive download was attempted. " + " | ".join(errors)
    )


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

    with _open_remote_archive(__import__("remotezip"), archive_url) as archive:
        groups = index_archive_members(archive.infolist())
        if not groups:
            raise RuntimeError(
                "No complete StemGMD groups found; archive layout differs from "
                "the verified Eval Session structure"
            )
        ranked = rank_track_candidates(groups)
        # Some grooves contain no Snare or no Tom hits. The strict SI-SDR
        # runner rejects silent references, so preflight only the four target
        # stems and continue through the complete ranked inventory until the
        # requested valid subset is found. There is deliberately no arbitrary
        # scan cutoff: sparse active grooves can occur late in the ZIP.
        rows: list[dict[str, Any]] = []
        rejected_tracks: list[dict[str, str]] = []
        extracted_bytes = 0
        candidates_checked = 0

        scan_started = time.monotonic()
        for group in ranked:
            if len(rows) >= max_tracks:
                break
            candidate_started = time.monotonic()
            candidates_checked += 1
            track_id = _track_id(group["drummer"], group["kit"], group["groove"])
            track_dir = output_dir / track_id
            reference_dir = track_dir / "references"
            reference_dir.mkdir(parents=True, exist_ok=True)
            local_paths: dict[str, Path] = {}

            # Fetch Snare and all three Tom references first. This avoids
            # downloading the other five stems for a groove with silent targets.
            for instrument in ("snare", *TOM_INSTRUMENTS):
                destination = reference_dir / f"{instrument}.wav"
                member = group["members"][instrument]
                with archive.open(member, "r") as source, destination.open("wb") as target:
                    shutil.copyfileobj(source, target, length=1024 * 1024)
                local_paths[instrument] = destination

            target_refs = {
                name: _read_reference(local_paths[name])
                for name in ("snare", *TOM_INSTRUMENTS)
            }
            target_frames = {samples.shape[0] for samples in target_refs.values()}
            if len(target_frames) != 1:
                raise ValueError(
                    f"StemGMD Snare/Toms references are not aligned for {track_id}"
                )
            snare_peak = float(np.max(np.abs(target_refs["snare"])))
            active_tom_components = [
                name for name in TOM_INSTRUMENTS
                if float(np.max(np.abs(target_refs[name]))) > 1e-8
            ]
            tom_reference = np.zeros_like(
                target_refs[TOM_INSTRUMENTS[0]], dtype=np.float64
            )
            for name in TOM_INSTRUMENTS:
                tom_reference += target_refs[name].astype(np.float64)
            toms_peak = float(np.max(np.abs(tom_reference)))
            if snare_peak <= 1e-8 or toms_peak <= 1e-8:
                reason = (
                    "silent_snare_reference" if snare_peak <= 1e-8
                    else "silent_toms_reference"
                )
                rejected_tracks.append({
                    "track_id": track_id,
                    "reason": reason,
                    "target_compressed_bytes": sum(
                        group.get("member_compressed_sizes", {}).get(name, 0)
                        for name in ("snare", *TOM_INSTRUMENTS)
                    ),
                })
                shutil.rmtree(track_dir, ignore_errors=True)
                outcome = f"rejected ({reason})"
                elapsed = time.monotonic() - candidate_started
                if candidates_checked == 1 or candidates_checked % 10 == 0:
                    print(
                        "[StemGMD] "
                        f"checked={candidates_checked}/{len(ranked)} "
                        f"selected={len(rows)}/{max_tracks} "
                        f"rejected={len(rejected_tracks)} "
                        f"candidate={track_id} outcome={outcome} "
                        f"candidate_seconds={elapsed:.1f} "
                        f"total_minutes={(time.monotonic() - scan_started) / 60:.1f}",
                        flush=True,
                    )
                continue

            # Only accepted target-bearing tracks need the remaining five
            # components to construct a full-kit mixture.
            for instrument in INSTRUMENTS:
                if instrument in local_paths:
                    continue
                destination = reference_dir / f"{instrument}.wav"
                member = group["members"][instrument]
                with archive.open(member, "r") as source, destination.open("wb") as target:
                    shutil.copyfileobj(source, target, length=1024 * 1024)
                local_paths[instrument] = destination

            references = {
                name: _read_reference(path) for name, path in local_paths.items()
            }
            frame_counts = {
                name: samples.shape[0] for name, samples in references.items()
            }
            if len(set(frame_counts.values())) != 1:
                raise ValueError(
                    f"StemGMD reference lengths are not aligned for {track_id}: {frame_counts}"
                )

            mixture = np.zeros_like(references[INSTRUMENTS[0]], dtype=np.float64)
            for instrument in INSTRUMENTS:
                mixture += references[instrument].astype(np.float64)
            mixture_rms = float(np.sqrt(np.mean(mixture ** 2)))
            snare_rms = float(np.sqrt(np.mean(references["snare"] ** 2)))
            tom_reference = np.zeros_like(mixture, dtype=np.float64)
            for name in TOM_INSTRUMENTS:
                tom_reference += references[name].astype(np.float64)
            toms_rms = float(np.sqrt(np.mean(tom_reference ** 2)))
            mixture_rms_db_floor = max(mixture_rms, 1e-12)
            toms_to_mixture_rms_db = 20.0 * np.log10(
                max(toms_rms, 1e-12) / mixture_rms_db_floor
            )
            toms_to_snare_rms_db = 20.0 * np.log10(
                max(toms_rms, 1e-12) / max(snare_rms, 1e-12)
            )
            mixture_path = track_dir / "mixture.wav"
            # FLOAT WAV preserves the unnormalized sum without PCM clipping.
            sf.write(str(mixture_path), mixture.astype(np.float32), SAMPLE_RATE, subtype="FLOAT")

            relative = lambda path: Path(os.path.relpath(path.resolve(), manifest_path.parent.resolve())).as_posix()
            row = {
                "track_id": track_id,
                "mixture": relative(mixture_path),
                "snare": relative(local_paths["snare"]),
                "toms": ";".join(relative(local_paths[name]) for name in TOM_INSTRUMENTS),
                "components": json.dumps(
                    {name: relative(path) for name, path in local_paths.items()},
                    separators=(",", ":"),
                ),
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
                "snare_peak": round(snare_peak, 8),
                "toms_peak": round(toms_peak, 8),
                "active_tom_components": active_tom_components,
                "active_tom_component_count": len(active_tom_components),
                "snare_rms": round(snare_rms, 8),
                "toms_rms": round(toms_rms, 8),
                "toms_to_snare_rms_db": round(float(toms_to_snare_rms_db), 4),
                "toms_to_mixture_rms_db": round(float(toms_to_mixture_rms_db), 4),
                "mixture_peak": round(float(np.max(np.abs(mixture))), 8),
                "target_compressed_bytes": sum(
                    group.get("member_compressed_sizes", {}).get(name, 0)
                    for name in ("snare", *TOM_INSTRUMENTS)
                ),
            })
            extracted_bytes += sum(path.stat().st_size for path in local_paths.values())
            extracted_bytes += mixture_path.stat().st_size
            outcome = "selected"

            elapsed = time.monotonic() - candidate_started
            if candidates_checked == 1 or candidates_checked % 10 == 0 or len(rows) >= max_tracks:
                print(
                    "[StemGMD] "
                    f"checked={candidates_checked}/{len(ranked)} "
                    f"selected={len(rows)}/{max_tracks} "
                    f"rejected={len(rejected_tracks)} "
                    f"candidate={track_id} outcome={outcome} "
                    f"candidate_seconds={elapsed:.1f} "
                    f"total_minutes={(time.monotonic() - scan_started) / 60:.1f}",
                    flush=True,
                )

        minimum_required = min(4, max_tracks)
        if len(rows) < minimum_required:
            raise RuntimeError(
                f"Found only {len(rows)} tracks with non-silent Snare and Tom references "
                f"after checking {candidates_checked} of {len(ranked)} complete groups; "
                f"at least {minimum_required} are required for this subset."
            )

    with manifest_path.open("w", encoding="utf-8", newline="") as stream:
        writer = csv.DictWriter(
            stream, fieldnames=("track_id", "mixture", "snare", "toms", "components")
        )
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
        "candidate_ordering": (
            "Greedy weighted ranking: 55% normalized compressed target size, "
            "20% new groove, 15% new kit and 10% new drummer relative to already "
            "ranked candidates; compressed size is only a prioritization heuristic "
            "and never replaces decoded-sample validation."
        ),
        "full_archive_downloaded": False,
        "audio_downloaded": True,
        "selected_tracks": len(rows),
        "candidate_groups_checked": candidates_checked,
        "rejected_silent_target_tracks": rejected_tracks,
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
            "The ZFTurbo 53-stem checkpoint has a verified CPU runtime; CPU inference is the required baseline and CUDA is optional acceleration.",
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
