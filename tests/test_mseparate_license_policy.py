"""Fail-closed tests for model weight redistribution review."""
from __future__ import annotations

import importlib.util
from pathlib import Path

_SCRIPT = Path(__file__).resolve().parents[1] / "scripts" / "check_model_license_policy.py"
_SPEC = importlib.util.spec_from_file_location("model_license_policy", _SCRIPT)
_POLICY = importlib.util.module_from_spec(_SPEC)
assert _SPEC.loader is not None
_SPEC.loader.exec_module(_POLICY)


def _approved_model():
    return {
        "model_id": "model-a",
        "license": "MIT",
        "license_status": "verified",
        "status": "available",
        "assets": [{"role": "checkpoint", "relative_path": "model-a/model.ckpt"}],
        "license_review": {
            "status": "approved",
            "weight_license": "MIT",
            "weight_license_evidence": ["https://example.org/weights-license"],
            "redistribution_permission": "granted",
            "training_data_status": "cleared",
            "training_data_evidence": ["https://example.org/dataset-rights"],
            "compatibility_status": "compatible",
            "reviewed_by": "maintainer",
            "reviewed_on": "2026-11-16",
            "rationale": "The weight license and all training-data rights were checked against primary evidence.",
        },
    }


def test_blocked_candidate_with_unknown_training_data_is_not_approved_but_valid_hold():
    model = {
        "model_id": "mega53-synth",
        "license": None,
        "license_status": "review_required",
        "status": "candidate",
        "assets": [{"role": "checkpoint", "relative_path": "synth.ckpt"}],
        "license_review": {
            "status": "blocked",
            "weight_license": "MIT",
            "weight_license_evidence": ["https://example.org/upstream-owner-statement"],
            "training_data_status": "unknown",
            "reviewed_by": "maintainer",
            "reviewed_on": "2026-11-16",
            "rationale": "Upstream license grant exists, but training data rights remain undocumented and blocked.",
        },
    }
    assert _POLICY.validate_review(model) == []


def test_maintainer_acceptance_removes_block_without_claiming_dataset_clearance():
    model = {
        "model_id": "mega53-synth",
        "license": "MIT",
        "license_status": "verified",
        "status": "available",
        "assets": [],
        "license_review": {
            "status": "accepted_by_maintainer",
            "weight_license": "MIT",
            "weight_license_evidence": ["https://example.org/original-author-mit-confirmation"],
            "redistribution_permission": "author_confirmed_for_original_53_stem_weights_and_conversions",
            "training_data_status": "unknown",
            "acceptance_basis": "Maintainer explicitly accepts the original author's MIT grant despite unresolved training-data provenance.",
            "reviewed_by": "repository maintainer",
            "reviewed_on": "2026-11-16",
            "rationale": "The MIT grant is accepted as sufficient to remove the license block; training-data provenance remains unknown."
        }
    }
    assert _POLICY.validate_review(model) == []


def test_approved_model_requires_both_weight_and_dataset_rights():
    assert _POLICY.validate_review(_approved_model()) == []


def test_unknown_training_data_blocks_an_otherwise_mit_licensed_model():
    model = _approved_model()
    model["license_review"]["training_data_status"] = "unknown"
    assert any("training_data_status must be cleared" in issue
               for issue in _POLICY.validate_review(model))


def test_noncommercial_license_is_rejected_for_free_software_distribution():
    model = _approved_model()
    model["license"] = "CC-BY-NC-SA-4.0"
    model["license_review"]["weight_license"] = "CC-BY-NC-SA-4.0"
    issues = _POLICY.validate_review(model)
    assert any("non-commercial license" in issue for issue in issues)


def test_new_or_changed_model_without_review_fails_policy():
    base = {}
    head = {"model-a": {"model_id": "model-a", "status": "candidate"}}
    issues = _POLICY.check_policy(base, head, [])
    assert any("missing license_review" in issue for issue in issues)


def test_unchanged_legacy_model_is_not_reaudited_by_every_pr():
    model = {"model_id": "legacy", "status": "available"}
    assert _POLICY.check_policy({"legacy": model.copy()}, {"legacy": model.copy()}, []) == []


def test_added_weight_file_requires_approved_manifest_mapping():
    model = _approved_model()
    issues = _POLICY.check_policy({}, {"model-a": model}, ["model-a/model.ckpt"])
    assert issues == []


def test_unmapped_or_blocked_weight_file_is_rejected():
    model = {
        "model_id": "model-a",
        "assets": [{"relative_path": "model-a/model.ckpt"}],
        "license_review": {"status": "blocked"},
    }
    issues = _POLICY.check_policy({}, {"model-a": model}, ["model-a/model.ckpt"])
    assert any("without an approved license_review" in issue for issue in issues)
