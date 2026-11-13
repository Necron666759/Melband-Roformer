# mseparate model registry

Status: schema requirements; this file does not certify or enable any model weights.

## Required model record

| Field | Meaning |
|---|---|
| `model_id` | Stable unique identifier |
| `name` | Human-readable model name |
| `architecture` | Architecture family/version |
| `backend` | Adapter identifier |
| `source_url` | Canonical project/model source |
| `license` | SPDX identifier or explicit unknown |
| `license_url` | License evidence |
| `license_status` | `verified`, `review_required`, or `rejected` |
| `assets` | Required checkpoint/config assets |
| `native_stems` | Actual trained/native output stems |
| `target_stems` | Meaningful targets the model can produce |
| `dummy_stems` | Placeholder/derived outputs that are not independent estimates |
| `refinement_targets` | Stems eligible for a subsequent refinement |
| `sample_rate` | Required/expected sample rate, if constrained |
| `channels` | Supported channel policy |
| `segment_parameters` | Segment size, overlap and model-specific limits |
| `resource_profile` | Measured VRAM/runtime plus measurement provenance |
| `status` | `candidate`, `available`, `incompatible`, `unavailable`, or `disabled` |

Each asset record includes relative local path, URL, expected byte size where known, SHA-256, asset role and required/optional status. A missing checksum or unknown license is not silently treated as verified.

## Eligibility rules

1. Reject disabled, incompatible, unavailable or license-rejected entries.
2. Require verified license status before automatic use.
3. Require all mandatory assets to be present and checksum-valid before planning inference.
4. Verify the backend adapter and runtime compatibility.
5. Distinguish native, dummy and derived outputs in both planning and user-facing reporting.
6. Keep hardware measurements separate from estimates; include device, driver, CUDA/PyTorch and test conditions.
7. Never infer a model's output capabilities from its architecture name alone; inspect the checkpoint/config and authoritative model documentation.

## Candidate families to investigate

Potential research candidates include MelBand RoFormer, BS-RoFormer, RoFormer, HTDemucs, Demucs, MDX23C, SCNet, Swin Upernet, BandIt, Apollo, BSMamba, Conformer and DTTNet.

This is a research list, not a support statement. Each checkpoint is a separate review item because licenses, configuration formats, output stems, weights availability and runtime constraints can differ even within one architecture family.
