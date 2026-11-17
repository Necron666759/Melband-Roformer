"""Per-user mseparate data paths and atomic JSON reports."""
from __future__ import annotations

import json
import os
import tempfile
from pathlib import Path
from typing import Any

from .types import to_jsonable


def data_root() -> Path:
    override = os.environ.get("MSEPARATE_HOME")
    return Path(override).expanduser() if override else Path.home() / ".local" / "share" / "mseparate"


def ensure_layout(root: str | Path | None = None) -> dict[str, Path]:
    base = Path(root).expanduser() if root is not None else data_root()
    paths = {
        "root": base,
        "models": base / "models",
        "registry": base / "registry",
        "cache": base / "cache",
        "jobs": base / "jobs",
    }
    for path in paths.values():
        path.mkdir(parents=True, exist_ok=True)
    return paths


def write_json_atomic(path: str | Path, payload: Any) -> Path:
    target = Path(path)
    target.parent.mkdir(parents=True, exist_ok=True)
    encoded = json.dumps(to_jsonable(payload), ensure_ascii=False, indent=2, sort_keys=True) + "\n"
    fd, temporary = tempfile.mkstemp(prefix=f".{target.name}.", suffix=".part", dir=target.parent)
    try:
        with os.fdopen(fd, "w", encoding="utf-8") as stream:
            stream.write(encoded)
            stream.flush()
            os.fsync(stream.fileno())
        os.replace(temporary, target)
    except Exception:
        try:
            os.unlink(temporary)
        except OSError:
            pass
        raise
    return target
