# Model weight licensing audit

Review date: 2026-11-16  
Scope: MVSep Mega 53 Synth and Keys checkpoints currently referenced by the model registry.

## Decision

**BLOCKED — do not download, bundle, redistribute, or route these weights.**

This is a conservative repository-maintenance decision, not a finding that the weights are infringing. The public evidence confirms a permissive license grant for the original Mega 53 weights, but does not establish the rights status of all training audio or provide dataset provenance. The repository's requirement is stronger: we must have enough evidence to distribute weights in a public free-software project without knowingly carrying unresolved rights risk.

## Evidence reviewed

1. **Original author's statement — weight license.** In [ZFTurbo/Music-Source-Separation-Training issue #245, comment by ZFTurbo](https://github.com/ZFTurbo/Music-Source-Separation-Training/issues/245#issuecomment-5834608838), the original author states that the 53-stem model weights are released under MIT and that the license permits redistribution and conversion/optimization. This is direct evidence for the original Mega 53 weights.
2. **Explicit training-data disclaimer.** In the same statement, the author says they do not hold copyright to all audio used for training and cannot provide legal guarantees or indemnification for potential training-data claims. The MIT "AS IS" disclaimer does not resolve third-party dataset rights.
3. **Unanswered provenance question.** [Issue #255](https://github.com/ZFTurbo/Music-Source-Separation-Training/issues/255) asks which datasets were used and whether commercial permissions/restrictions were considered. The retrieved discussion contains no answer from the original author. A third-party comment is not authoritative provenance evidence.
4. **Release scope.** [Upstream release v1.0.21](https://github.com/ZFTurbo/Music-Source-Separation-Training/releases/tag/v1.0.21) identifies the 53-stem model and includes both Synth and Keys among its outputs.
5. **Exact mirror used by this project.** [noblebarkrr/BS-Roformer-MVSep-Mega-53-stems](https://huggingface.co/noblebarkrr/BS-Roformer-MVSep-Mega-53-stems) identifies ZFTurbo's v1.0.21 release as the original model and lists the individual stems, but its model card has no SPDX/license metadata. It does not provide training-data provenance or independent rights evidence for the per-stem files.

## Assessment

- The original author has made a clear MIT grant for the original 53-stem weights, including redistribution and conversion.
- The model-card mirror itself does not declare a license. The upstream statement is useful evidence, but we still need an auditable chain linking the exact mirrored per-stem checkpoint files and their conversion/extraction process to the upstream grant.
- More importantly, training-data provenance and the rights for the training recordings are unknown. The original author expressly disclaims holding all those rights.
- A permissive weight license does not itself prove that every training recording was licensed for the relevant use. We will not infer dataset clearance from silence, the inference-code license, a mirror's metadata, or a third party's comment.

## Required evidence to unblock

All of the following are required before changing the review to 'approved':

1. Primary-source confirmation from the original author/rightsholder that the exact v1.0.21 source checkpoint and the specific extracted Synth/Keys derivative files are covered by the MIT grant and may be redistributed.
2. A documented training-data inventory, with primary-source license/permission evidence for each dataset used, sufficient to establish compatibility with public redistribution in this project; or an authoritative explanation of why the training data creates no redistribution restriction.
3. A documented compatibility review of the weight license, dataset terms, project license and notices/attribution requirements.
4. A named maintainer review with date, rationale and immutable evidence URLs in the model manifest.

Until then, 'license_status=review_required', 'status=candidate', and 'license_review.status=blocked' are mandatory. The asset manager now refuses to download any candidate or license-unverified model.

## Alternative search direction

If the missing provenance cannot be obtained, do not wait indefinitely or weaken the gate. Search for:
- a specialist model whose author publishes an explicit weight license and training-data provenance compatible with public redistribution;
- a model trained solely on appropriately licensed synthetic/MIDI-rendered stems or datasets with documented redistribution permissions;
- a new in-house specialist trained from datasets whose licenses and contributor permissions are recorded before training.

An alternative is not accepted merely because its model card says MIT; it must pass the same weight, data, derivative, notice and compatibility review.

## Automated controls

- 'scripts/check_model_license_policy.py' validates every changed model record in a PR against its legal-review record.
- '.github/workflows/model-license-policy.yml' runs before merge when model manifests or model-weight files change.
- Adding binary model weights requires a manifest mapping and an approved review.
- 'ensure_model_assets' refuses to download model assets unless the model is both 'available' and 'license_status=verified'.
- 'eligible_models' already requires verified license status before planner selection.

The CI check is a fail-closed evidence/schema gate, not an automated legal opinion. A human maintainer must review the primary sources. Existing model entries present before this policy was introduced are a legacy baseline and are not silently certified by this new check; they require a separate audit before their license status is represented as fully cleared.
