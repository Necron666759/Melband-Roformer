#!/usr/bin/env python3
"""Inspect StemGMD Eval Session ZIP metadata using HTTP byte ranges only.

This preflight reads the remote ZIP central directory. It does not download or
extract audio. The resulting inventory is used to design a small, exact
Snare/Toms reference subset without fetching the 5 GB archive.
"""
from __future__ import annotations

import argparse
import json
from collections import Counter
from pathlib import Path, PurePosixPath

ARCHIVE_URL = (
    "https://zenodo.org/records/7882857/files/"
    "StemGMD_eval_session.zip?download=1"
)


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
    samples = {
        "mixture_like": [],
        "snare_like": [],
        "tom_like": [],
        "wav_paths": [],
    }
    for info in wavs:
        path = info.filename
        name = PurePosixPath(path).stem.lower().replace("-", "_").replace(" ", "_")
        tokens = set(name.split("_"))
        if name in {"mix", "mixture", "full_mix", "full_mixture"} or "mixture" in tokens:
            bucket = "mixture_like"
        elif "snare" in tokens or name in {"sd", "sn"}:
            bucket = "snare_like"
        elif "tom" in tokens or any(token.endswith("tom") for token in tokens):
            bucket = "tom_like"
        else:
            bucket = None
        if bucket and len(samples[bucket]) < 80:
            samples[bucket].append({
                "path": path,
                "uncompressed_bytes": info.file_size,
            })
        if len(samples["wav_paths"]) < 120:
            samples["wav_paths"].append(path)

    payload = {
        "schema_version": 1,
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
        "sample_paths": samples,
        "limitations": [
            "This is metadata-only discovery; no audio bytes were extracted.",
            "Filename patterns are provisional until the archive layout is reviewed.",
            "A subsequent downloader must select only exact mixture, Snare and Tom reference members.",
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
        "report": str(args.report),
        "sample_paths": result["sample_paths"],
    }, indent=2, ensure_ascii=False))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
