"""Persistent, reversible candidate quarantine ledger.

The ledger records decisions only. It never moves, deletes, overwrites or
silently promotes audio files. A quarantine state means "hold for review",
not "proven bad separation".
"""
from __future__ import annotations

import json
import os
import tempfile
from pathlib import Path
from typing import Any


STATUSES = frozenset({"pending", "accepted", "quarantined", "rejected", "superseded"})
_TRANSITIONS = {
    "pending": {"accepted", "quarantined", "rejected"},
    "quarantined": {"pending", "accepted", "rejected"},
    "accepted": {"quarantined", "superseded"},
    "rejected": {"pending"},
    "superseded": {"pending"},
}


class QuarantineError(ValueError):
    """Invalid candidate or state transition."""


class QuarantineLedger:
    """Store candidate lineage and append-only decision history in JSON."""

    def __init__(self, path: str | Path):
        self.path = Path(path).expanduser()

    def _read(self) -> dict[str, Any]:
        if not self.path.exists():
            return {"schema_version": 1, "candidates": {}}
        try:
            data = json.loads(self.path.read_text(encoding="utf-8"))
        except (OSError, json.JSONDecodeError) as exc:
            raise QuarantineError(f"Cannot read quarantine ledger: {exc}") from exc
        if not isinstance(data, dict) or data.get("schema_version") != 1:
            raise QuarantineError("Unsupported or malformed quarantine ledger")
        if not isinstance(data.get("candidates"), dict):
            raise QuarantineError("Quarantine ledger candidates must be an object")
        return data

    def _write(self, data: dict[str, Any]) -> None:
        self.path.parent.mkdir(parents=True, exist_ok=True)
        fd, temp_name = tempfile.mkstemp(prefix=self.path.name + ".", suffix=".tmp",
                                         dir=str(self.path.parent))
        try:
            with os.fdopen(fd, "w", encoding="utf-8") as stream:
                json.dump(data, stream, ensure_ascii=False, indent=2, sort_keys=True)
                stream.write("\n")
                stream.flush()
                os.fsync(stream.fileno())
            os.replace(temp_name, self.path)
        finally:
            try:
                os.unlink(temp_name)
            except FileNotFoundError:
                pass

    def list_candidates(self) -> list[dict[str, Any]]:
        data = self._read()
        return [data["candidates"][key] for key in sorted(data["candidates"])]

    def add_candidate(
        self,
        candidate_id: str,
        *,
        stem: str,
        path: str | Path,
        model_id: str,
        parent_id: str | None = None,
        reason: str = "",
        analysis: dict[str, Any] | None = None,
    ) -> dict[str, Any]:
        if not candidate_id.strip() or not stem.strip() or not model_id.strip():
            raise QuarantineError("candidate_id, stem and model_id must be non-empty")
        data = self._read()
        if candidate_id in data["candidates"]:
            raise QuarantineError(f"Candidate already exists: {candidate_id}")
        if parent_id is not None and parent_id not in data["candidates"]:
            raise QuarantineError(f"Unknown parent candidate: {parent_id}")
        record = {
            "candidate_id": candidate_id,
            "stem": stem,
            "path": str(Path(path).expanduser()),
            "model_id": model_id,
            "parent_id": parent_id,
            "status": "pending",
            "reason": reason,
            "analysis": analysis or {},
            "history": [{"from": None, "to": "pending", "reason": "candidate created"}],
        }
        data["candidates"][candidate_id] = record
        self._write(data)
        return record

    def transition(self, candidate_id: str, status: str, *, reason: str) -> dict[str, Any]:
        if status not in STATUSES:
            raise QuarantineError(f"Unknown candidate status: {status!r}")
        if not reason.strip():
            raise QuarantineError("A transition reason is required")
        data = self._read()
        record = data["candidates"].get(candidate_id)
        if record is None:
            raise QuarantineError(f"Unknown candidate: {candidate_id}")
        previous = record["status"]
        if status not in _TRANSITIONS.get(previous, set()):
            raise QuarantineError(f"Invalid candidate transition: {previous} -> {status}")
        record["status"] = status
        record["reason"] = reason
        record["history"].append({"from": previous, "to": status, "reason": reason})
        self._write(data)
        return record
