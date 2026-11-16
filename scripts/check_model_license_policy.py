#!/usr/bin/env python3
"""Fail closed when changed model weights lack a complete redistribution review.

The check is intentionally conservative. It validates the review record and
its declared evidence; it does not pretend that a URL or SPDX label proves
copyright ownership. Human review of the cited primary sources remains required.
"""
from __future__ import annotations

import argparse
import datetime as dt
import json
import re
import subprocess
import sys
from pathlib import Path
from typing import Any

WEIGHT_SUFFIXES = {
    ".ckpt", ".pt", ".pth", ".safetensors", ".onnx", ".bin", ".h5",
    ".hdf5", ".model", ".tflite",
}
URL_RE = re.compile(r"^https?://\\S+$")
SPDX_RE = re.compile(r"^[A-Za-z0-9][A-Za-z0-9.+-]*$")


def _read_models(path: Path) -> dict[str, dict[str, Any]]:
    raw = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(raw, list):
        raise ValueError(f"{path}: model registry root must be an array")
    result: dict[str, dict[str, Any]] = {}
    for item in raw:
        if not isinstance(item, dict) or not isinstance(item.get("model_id"), str):
            raise ValueError(f"{path}: each model entry must have a string model_id")
        if item["model_id"] in result:
            raise ValueError(f"{path}: duplicate model_id {item['model_id']!r}")
        result[item["model_id"]] = item
    return result


def _urls(value: Any) -> bool:
    return (
        isinstance(value, list)
        and bool(value)
        and all(isinstance(item, str) and URL_RE.fullmatch(item) for item in value)
    )


def validate_review(model: dict[str, Any]) -> list[str]:
    """Validate a single model's legal-review state without making legal conclusions."""
    model_id = model.get("model_id", "<unknown>")
    review = model.get("license_review")
    if not isinstance(review, dict):
        return [f"{model_id}: missing license_review; new/changed weights require an explicit review record"]

    state = review.get("status")
    license_status = model.get("license_status")
    model_status = model.get("status")
    if state == "blocked":
        problems = []
        if license_status == "verified":
            problems.append(f"{model_id}: blocked review cannot have license_status=verified")
        if model_status == "available":
            problems.append(f"{model_id}: blocked review cannot have status=available")
        if review.get("training_data_status") not in {"unknown", "restricted", "not_cleared"}:
            problems.append(f"{model_id}: blocked review must state why training data is not cleared")
        return problems

    if state == "rejected":
        if license_status != "rejected" or model_status != "disabled":
            return [f"{model_id}: rejected review requires license_status=rejected and status=disabled"]
        return []

    if state != "approved":
        return [f"{model_id}: license_review.status must be approved, blocked, or rejected"]

    problems: list[str] = []
    license_id = model.get("license")
    if not isinstance(license_id, str) or not SPDX_RE.fullmatch(license_id):
        problems.append(f"{model_id}: approved model requires an explicit SPDX license identifier")
    if isinstance(license_id, str) and ("NC" in license_id.upper() or "NONCOMMERCIAL" in license_id.upper()):
        problems.append(f"{model_id}: non-commercial license is not accepted for the free-software distribution policy")
    if review.get("weight_license") != license_id:
        problems.append(f"{model_id}: reviewed weight_license must exactly match manifest license")
    if not _urls(review.get("weight_license_evidence")):
        problems.append(f"{model_id}: approved model requires primary-source weight_license_evidence URLs")
    if review.get("redistribution_permission") != "granted":
        problems.append(f"{model_id}: approved model requires explicit redistribution_permission=granted")
    if review.get("training_data_status") != "cleared":
        problems.append(f"{model_id}: training_data_status must be cleared; unknown/restricted data blocks approval")
    if not _urls(review.get("training_data_evidence")):
        problems.append(f"{model_id}: approved model requires training_data_evidence URLs")
    if review.get("compatibility_status") != "compatible":
        problems.append(f"{model_id}: license compatibility with the project must be explicitly approved")
    if license_status != "verified" or model_status != "available":
        problems.append(f"{model_id}: approved review requires license_status=verified and status=available")
    if not isinstance(review.get("reviewed_by"), str) or not review["reviewed_by"].strip():
        problems.append(f"{model_id}: reviewed_by is required")
    reviewed_on = review.get("reviewed_on")
    try:
        dt.date.fromisoformat(reviewed_on)
    except (TypeError, ValueError):
        problems.append(f"{model_id}: reviewed_on must be an ISO date (YYYY-MM-DD)")
    if not isinstance(review.get("rationale"), str) or len(review["rationale"].strip()) < 30:
        problems.append(f"{model_id}: rationale must document the scope and reasoning")
    return problems


def _changed_paths(base_ref: str, head_ref: str) -> list[str]:
    result = subprocess.run(
        ["git", "diff", "--name-only", f"{base_ref}...{head_ref}"],
        check=True, capture_output=True, text=True,
    )
    return [line.strip() for line in result.stdout.splitlines() if line.strip()]


def _base_models_from_git(base_ref: str, registry_path: str) -> dict[str, dict[str, Any]] | None:
    result = subprocess.run(
        ["git", "show", f"{base_ref}:{registry_path}"],
        capture_output=True, text=True,
    )
    if result.returncode:
        return None
    raw = json.loads(result.stdout)
    if not isinstance(raw, list):
        raise ValueError("Base model registry root must be an array")
    return {item["model_id"]: item for item in raw}


def check_policy(
    base_models: dict[str, dict[str, Any]] | None,
    head_models: dict[str, dict[str, Any]],
    changed_paths: list[str],
) -> list[str]:
    problems: list[str] = []
    if base_models is not None:
        changed_ids = {
            model_id for model_id, model in head_models.items()
            if model_id not in base_models or model != base_models[model_id]
        }
        for model_id in sorted(changed_ids):
            problems.extend(validate_review(head_models[model_id]))

    # A weight file may not be committed unless the exact path is registered
    # and its model has passed the complete approval gate.
    for raw_path in changed_paths:
        path = Path(raw_path)
        if path.suffix.lower() not in WEIGHT_SUFFIXES:
            continue
        owners = [
            model for model in head_models.values()
            if any(
                asset.get("relative_path") == raw_path
                or raw_path.endswith("/" + str(asset.get("relative_path", "")))
                for asset in model.get("assets", [])
                if isinstance(asset, dict)
            )
        ]
        if not owners:
            problems.append(f"{raw_path}: weight file is not mapped to a model manifest asset")
            continue
        if not any(
            isinstance(model.get("license_review"), dict)
            and model["license_review"].get("status") == "approved"
            for model in owners
        ):
            problems.append(f"{raw_path}: weight file cannot be committed without an approved license_review")

    return problems


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--base-ref", required=True)
    parser.add_argument("--head-ref", required=True)
    parser.add_argument("--registry", default="app/mseparate/data/models.json")
    args = parser.parse_args()

    head_models = _read_models(Path(args.registry))
    base_models = _base_models_from_git(args.base_ref, args.registry)
    changed_paths = _changed_paths(args.base_ref, args.head_ref)
    problems = check_policy(base_models, head_models, changed_paths)

    if base_models is None:
        print(
            "NOTICE: no model registry exists at the PR base; treating current entries as the initial baseline. "
            "The strict changed-entry gate applies to every subsequent model-registry change."
        )
    if problems:
        print("MODEL LICENSE POLICY: BLOCKED", file=sys.stderr)
        for problem in problems:
            print(f" - {problem}", file=sys.stderr)
        return 1
    print("MODEL LICENSE POLICY: PASS")
    print("Changed model records and added weight files satisfy the declared review gate.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
