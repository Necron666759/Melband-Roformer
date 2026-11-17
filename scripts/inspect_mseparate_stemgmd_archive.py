#!/usr/bin/env python3
"""Inspect StemGMD Eval Session ZIP metadata using HTTP byte ranges only.

This preflight reads the remote ZIP central directory. It does not download or
extract audio. It groups instrument references by matching groove and kit so
a later downloader can select exact mixture/Snare/Tom members.
"""
from __future__ import annotations

import argparse
import json
from collections import Counter, defaultdict
from pathlib import Path, PurePosixPath

ARCHIVE_URL = (
    "https://zenodo.org/records/7882857/files/"
    "StemGMD_eval_session.zip?download=1"
)
MIX_DIRS = {"mix", "mixes", "mixture", "mixtures", "full_mix", "full_mixture"}
SNARE_DIRS = {"snare"}
TOM_DIRS = {"hi_tom", "mid_tom", "low_tom", "floor_tom", "tom", "toms"}


def classify_wav_path(path: str) -> tuple[str | None, str | None]:
    parts = PurePosixPath(path).parts
    if len(parts) < 2:
        return None, None
    parent = parts[-2].lower().replace("-", "_").replace(" ", "_")
    if parent in MIX_DIRS:
        return "mixture", parent
    if parent in SNARE_DIRS:
        return "snare", parent
    if parent in TOM_DIRS:
        return "tom", parent

    # Some archive revisions encode component names in the filename instead
    # of the parent directory; retain this fallback but prefer path metadata.
    stem = PurePosixPath(path).stem.lower().replace("-", "_").replace(" ", "_")
    tokens = set(stem.split("_"))
    if stem in {"mix", "mixture", "full_mix", "full_mixture"} or "mixture" in tokens:
        return "mixture", "filename"
    if "snare" in tokens or stem in {"sd", "sn"}:
        return "snare", "filename"
    if "tom" in tokens or any(token.endswith("_tom") for token in tokens):
        return "tom", "filename"
    return None, None


def track_group_key(path: str) -> str:
    # StemGMD stores each instrument in its own sibling directory; the
    # filename identifies the shared groove/performance across those folders.
    parts = list(PurePosixPath(path).parts)
    if len(parts) >= 2:
        parts.pop(-2)
    return str(PurePosixPath(*parts).with_suffix(""))


def inspect_archive(report_path: Path) -> dict:
    try:
        from remotezip import RemoteZip
    except ImportError as exc:
        raise RuntimeError("Install remotezip>=0.12,<1 to inspect the archive") from exc

    # RemoteZip requires HTTP Range support and reads the central directory;
    # deliberately do not add a full-download fallback.
    with RemoteZip(ARCHIVE_URL) as archive:
        infos = archive.infolist()

    files = [info for info in infos if not info.is_dir()]
    wavs = [info for info in files if PurePosixPath(info.filename).suffix.lower() == ".wav"]
    basenames = Counter(PurePosixPath(info.filename).name for info in wavs)
    parent_dirs = Counter(PurePosixPath(info.filename).parent.name for info in wavs)
    samples = {"mixture_like": [], "snare_like": [], "tom_like": [], "wav_paths": []}
    groups: dict[str, dict] = defaultdict(
        lambda: {"mixture": [], "snare": [], "tom": defaultdict(list), "components": defaultdict(list)}
    )

    for info in wavs:
        path = info.filename
        category, component = classify_wav_path(path)
        group = groups[track_group_key(path)]
        parent = PurePosixPath(path).parent.name.lower().replace("-", "_").replace(" ", "_")
        group["components"][parent].append(path)
        if category:
            bucket = {
                "mixture": "mixture_like",
                "snare": "snare_like",
                "tom": "tom_like",
            }[category]
            if len(samples[bucket]) < 80:
                samples[bucket].append({
                    "path": path,
                    "uncompressed_bytes": info.file_size,
                })
            if category == "tom":
                group["tom"][component].append(path)
            else:
                group[category].append(path)
        if len(samples["wav_paths"]) < 120:
            samples["wav_paths"].append(path)

    matching_groups = []
    for key, group in sorted(groups.items()):
        if group["snare"] and group["tom"]:
            matching_groups.append({
                "track_key": key,
                "mixtures": sorted(group["mixture"]),
                "snare": sorted(group["snare"]),
                "tom_components": {
                    name: sorted(paths) for name, paths in sorted(group["tom"].items())
                },
                "all_component_directories": sorted(group["components"]),
                "can_reconstruct_mixture_by_sum": not bool(group["mixture"]),
            })
    component_counts = Counter(
        component for group in groups.values() for component in group["components"]
    )
    components_per_track = Counter(
        len(group["components"]) for group in groups.values()
    )

    payload = {
        "schema_version": 2,
        "dataset": "StemGMD Eval Session",
        "source": "https://zenodo.org/records/7882857",
        "license": "CC BY 4.0",
        "archive_url": ARCHIVE_URL,
        "download_method": "HTTP Range; ZIP central directory only",
        "audio_downloaded": False,
        "archive_members": len(infos),
        "file_members": len(files),
        "wav_members": len(wavs),
        "wav_uncompressed_bytes": sum(info.file_size for info in wavs),
        "wav_basename_counts": basenames.most_common(160),
        "wav_parent_directory_counts": parent_dirs.most_common(120),
        "matching_mixture_snare_tom_groups": sum(
            1 for group in matching_groups if group["mixtures"]
        ),
        "matching_reference_groups": len(matching_groups),
        "wav_component_directory_counts": component_counts.most_common(120),
        "component_directories_per_track_distribution": sorted(
            components_per_track.items()
        ),
        "matching_groups": matching_groups[:80],
        "sample_paths": samples,
        "limitations": [
            "This is metadata-only discovery; no audio bytes were extracted.",
            "Component names are read from archive directory structure, not guessed from filenames alone.",
            "A subsequent downloader must verify sample rate, channel count, duration and exact frame alignment after extracting selected members.",
        ],
    }
    report_path.parent.mkdir(parents=True, exist_ok=True)
    report_path.write_text(
        json.dumps(payload, indent=2, ensure_ascii=False),
        encoding="utf-8",
    )
    return payload


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--report", type=Path, required=True)
    args = parser.parse_args()
    result = inspect_archive(args.report)
    print(json.dumps({
        "dataset": result["dataset"],
        "archive_members": result["archive_members"],
        "wav_members": result["wav_members"],
        "wav_uncompressed_gib": round(
            result["wav_uncompressed_bytes"] / 1024**3, 3
        ),
        "audio_downloaded": result["audio_downloaded"],
        "matching_mixture_snare_tom_groups": result["matching_mixture_snare_tom_groups"],
        "matching_reference_groups": result["matching_reference_groups"],
        "wav_component_directory_counts": result["wav_component_directory_counts"],
        "component_directories_per_track_distribution": result["component_directories_per_track_distribution"],
        "matching_groups": result["matching_groups"][:12],
        "report": str(args.report),
        "sample_paths": result["sample_paths"],
    }, indent=2, ensure_ascii=False))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
