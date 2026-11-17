"""Run-level observability helpers for mseparate."""
from __future__ import annotations

import threading
import time
import uuid
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, TextIO


def utc_now() -> str:
    return datetime.now(timezone.utc).isoformat(timespec="milliseconds")


class RunJournal:
    """Collect a compact, JSON-serializable timeline for one CLI invocation."""

    def __init__(self) -> None:
        self.run_id = uuid.uuid4().hex
        self.started_at = utc_now()
        self._started = time.perf_counter()
        self._active: dict[str, float] = {}
        self.events: list[dict[str, Any]] = []
        self.record("run", "started")

    def record(self, stage: str, status: str, **details: Any) -> None:
        event: dict[str, Any] = {
            "sequence": len(self.events) + 1,
            "run_id": self.run_id,
            "timestamp": utc_now(),
            "elapsed_seconds": round(time.perf_counter() - self._started, 6),
            "stage": stage,
            "status": status,
        }
        if details:
            event["details"] = details
        self.events.append(event)

    def begin(self, stage: str, **details: Any) -> None:
        self._active[stage] = time.perf_counter()
        self.record(stage, "started", **details)

    def end(self, stage: str, status: str = "completed", **details: Any) -> None:
        started = self._active.pop(stage, None)
        if started is not None:
            details["duration_seconds"] = round(time.perf_counter() - started, 6)
        self.record(stage, status, **details)

    def snapshot(self) -> dict[str, Any]:
        return {
            "run_id": self.run_id,
            "started_at": self.started_at,
            "elapsed_seconds": round(time.perf_counter() - self._started, 6),
            "events": list(self.events),
        }


class TeeTextIO:
    """Mirror text written to a terminal stream into a UTF-8 log file."""

    def __init__(self, terminal: TextIO, log: TextIO) -> None:
        self.terminal = terminal
        self.log = log
        self._lock = threading.Lock()

    def write(self, value: str) -> int:
        with self._lock:
            self.terminal.write(value)
            self.log.write(value)
        return len(value)

    def flush(self) -> None:
        with self._lock:
            self.terminal.flush()
            self.log.flush()

    def isatty(self) -> bool:
        return self.terminal.isatty()

    def fileno(self) -> int:
        return self.terminal.fileno()

    @property
    def encoding(self) -> str:
        return self.terminal.encoding or "utf-8"

    @property
    def errors(self) -> str:
        return self.terminal.errors or "replace"

    def writable(self) -> bool:
        return True

    def __getattr__(self, name: str) -> Any:
        return getattr(self.terminal, name)


def resolve_log_path(value: str, output_dir: str | Path) -> Path:
    path = Path(value).expanduser()
    if not path.is_absolute():
        path = Path(output_dir).expanduser() / path
    path.parent.mkdir(parents=True, exist_ok=True)
    return path
