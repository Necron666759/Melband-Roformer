#!/usr/bin/env python3
"""Selectively fetch a small BabySlakh subset using HTTP range requests.

Only selected mixtures, their metadata, and isolated drum stems are fetched.
The 883 MB archive is never downloaded in full. ZIP member CRCs are checked
while each selected member is read. Audio remains on the ephemeral runner.
"""
from __future__ import annotations

import argparse
import json
import shutil
from pathlib import Path, PurePosixPath
from typing import Any

DATA_URL = "https://zenodo.org/records/4603844/files/babyslakh_16k.zip?download=1"
MIX_NAMES = {"mix.wav", "mix.flac", "mixture.wav", "mixture.flac"}


def _track_metadata_members(names: list[str]) -> dict[str, str]:
    result: dict[str, str] = {}
    for name in names:
        path = PurePosixPath(name)
        if path.name.lower() == "metadata.yaml":
            result[str(path.parent)] = name
    return dict(sorted(result.items()))


def _drum_stem_ids(metadata: dict[str, Any]) -> list[str]:
    stems = metadata.get("stems")
    if not isinstance(stems, dict):
        return []
    return sorted(
        str(stem_id)
        for stem_id, details in stems.items()
        if isinstance(details, dict) and details.get("is_drum") is True
    )


def _find_mix_member(track_dir: str, names: set[str]) -> str | None:
    prefix = track_dir.rstrip("/") + "/"
    candidates = [
        name for name in names
        if name.startswith(prefix) and PurePosixPath(name).name.lower() in MIX_NAMES
    ]
    return sorted(candidates)[0] if candidates else None


def _find_stem_member(track_dir: str, metadata: dict[str, Any], stem_id: str,
                      names: set[str]) -> str | None:
    audio_dir = str(metadata.get("audio_dir") or "stems").strip("/")
    if audio_dir == track_dir or audio_dir.startswith(track_dir + "/"):
        stem_dir = audio_dir
    else:
        stem_dir = f"{track_dir}/{audio_dir}"
    prefix = stem_dir.rstrip("/") + "/"
    candidates = [
        name for name in names
        if name.startswith(prefix)
        and PurePosixPath(name).stem.lower() == stem_id.lower()
        and PurePosixPath(name).suffix.lower() in {".wav", ".flac"}
    ]
    return sorted(candidates)[0] if candidates else None


def _safe_destination(root: Path, member: str) -> Path:
    relative = PurePosixPath(member)
    if relative.is_absolute() or ".." in relative.parts:
        raise ValueError(f"Unsafe ZIP member path: {member}")
    destination = (root / Path(*relative.parts)).resolve()
    resolved_root = root.resolve()
    if destination != resolved_root and resolved_root not in destination.parents:
        raise ValueError(f"ZIP member escapes subset directory: {member}")
    return destination


def _copy_member(archive: Any, member: str, root: Path) -> dict[str, Any]:
    destination = _safe_destination(root, member)
    destination.parent.mkdir(parents=True, exist_ok=True)
    with archive.open(member, "r") as source, destination.open("wb") as target:
        shutil.copyfileobj(source, target, length=1024 * 1024)
    return {"member": member, "path": destination.relative_to(root).as_posix(),
            "bytes": destination.stat().st_size}


def download_subset(destination: Path, manifest_path: Path, max_tracks: int = 8) -> dict[str, Any]:
    if max_tracks < 1:
        raise ValueError("max_tracks must be >= 1")
    try:
        from remotezip import RemoteZip
        import yaml
    except ImportError as exc:
        raise RuntimeError(
            "Selective BabySlakh download requires remotezip and PyYAML"
        ) from exc

    destination.mkdir(parents=True, exist_ok=True)
    manifest_path.parent.mkdir(parents=True, exist_ok=True)
    # RemoteZip reads the central directory and selected member byte ranges;
    # it does not fall back to downloading the complete archive.
    with RemoteZip(DATA_URL) as archive:
        names_list = archive.namelist()
        names = set(names_list)
        metadata_members = _track_metadata_members(names_list)
        candidates: list[dict[str, Any]] = []
        for track_dir, metadata_member in metadata_members.items():
            raw = archive.read(metadata_member)
            metadata = yaml.safe_load(raw.decode("utf-8")) or {}
            drum_ids = _drum_stem_ids(metadata)
            if not drum_ids:
                continue
            mix_member = _find_mix_member(track_dir, names)
            if not mix_member:
                continue
            drum_members = [
                _find_stem_member(track_dir, metadata, stem_id, names)
                for stem_id in drum_ids
            ]
            drum_members = [member for member in drum_members if member]
            if not drum_members:
                continue
            candidates.append({
                "track": PurePosixPath(track_dir).name,
                "track_dir": track_dir,
                "metadata_member": metadata_member,
                "mix_member": mix_member,
                "drum_stem_members": drum_members,
                "drum_source_ids": drum_ids,
            })

        selected = candidates[:max_tracks]
        if len(selected) < min(4, max_tracks):
            raise RuntimeError(
                f"BabySlakh subset contains only {len(selected)} tracks with "
                "both a mixture and at least one resolvable isolated drum stem; "
                "at least four are required for this evaluation."
            )

        rows = []
        extracted_members = []
        for item in selected:
            metadata_file = _copy_member(archive, item["metadata_member"], destination)
            mix_file = _copy_member(archive, item["mix_member"], destination)
            drum_files = [
                _copy_member(archive, member, destination)
                for member in item["drum_stem_members"]
            ]
            extracted_members.extend([metadata_file, mix_file, *drum_files])
            rows.append({
                "track": item["track"],
                "mix": mix_file["path"],
                "metadata": metadata_file["path"],
                "drum_stems": [entry["path"] for entry in drum_files],
                "drum_source_ids": item["drum_source_ids"],
                "ground_truth": {
                    "drums_present": True,
                    "drum_stem_count": len(drum_files),
                    "source_type": "sample-based MIDI-rendered drum source",
                },
            })

    payload = {
        "schema_version": 1,
        "dataset": "BabySlakh",
        "source": "https://zenodo.org/records/4603844",
        "license": "CC BY 4.0",
        "download_method": "HTTP range requests for selected ZIP members",
        "full_archive_downloaded": False,
        "selected_tracks": len(rows),
        "extracted_files": len(extracted_members),
        "extracted_bytes": sum(item["bytes"] for item in extracted_members),
        "tracks": rows,
        "limitations": [
            "BabySlakh is sample-based MIDI-rendered music, not live drum recordings.",
            "Its drum ground truth identifies a combined drum source, not isolated Snare/Toms components.",
            "The subset is a compact routing/source-presence diagnostic, not a perceptual-quality benchmark.",
        ],
    }
    manifest_path.write_text(json.dumps(payload, indent=2, ensure_ascii=False), encoding="utf-8")
    return payload


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--manifest", type=Path, required=True)
    parser.add_argument("--max-tracks", type=int, default=8)
    args = parser.parse_args()
    result = download_subset(args.output_dir, args.manifest, args.max_tracks)
    print(json.dumps({
        "selected_tracks": result["selected_tracks"],
        "extracted_files": result["extracted_files"],
        "extracted_mib": round(result["extracted_bytes"] / (1024 * 1024), 2),
        "full_archive_downloaded": result["full_archive_downloaded"],
    }, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
