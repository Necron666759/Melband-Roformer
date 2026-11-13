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
| `planner_priority` | Explicit operator routing preference used only after specialist matching; it is not a model-quality score |

Each asset record includes relative local path, URL, expected byte size where known, SHA-256, asset role and required/optional status. A missing checksum or unknown license is not silently treated as verified.

## Eligibility rules

1. Reject disabled, incompatible, unavailable or license-rejected entries.
2. Require verified license status before automatic use.
3. Require all mandatory assets to be present and checksum-valid before planning inference.
4. Verify the backend adapter and runtime compatibility.
5. Distinguish native, dummy and derived outputs in both planning and user-facing reporting.
6. Keep hardware measurements separate from estimates; include device, driver, CUDA/PyTorch and test conditions.
7. Never infer a model's output capabilities from its architecture name alone; inspect the checkpoint/config and authoritative model documentation.

## Enabled checkpoint records

The bundled registry currently includes the following newly reviewed vocal models:

| Model ID | Architecture | Backend | Weight license | Checkpoint SHA-256 |
|---|---|---|---|---|
| `bs-roformer-anvuew-vocals` | BS-RoFormer | `bs-roformer-pymss` | GPL-3.0 | `8d0f72ad0ac4154f5f4c3cdf230880bbb9a5bd01dc610241a2abfe787f0d7784` |
| `mdx23c-zfturbo-vocals` | MDX23C | `mdx23c-pymss` | CC-BY-NC-SA-4.0 | `49d51472769e34a2501cd1da782346a3212555c3a5619fc2c53507445528d816` |

Both models expose only their native vocals estimate. Their secondary other/instrument estimate is not relabelled as a clean instrumental stem. The BS-RoFormer checkpoint/config are pinned to Hugging Face revision 7e4e5669cad8a8810983e02fa25294229246d374; the MDX23C checkpoint is pinned to Sucial/MSST-WebUI revision 55155c4e2f99d0ca94d24fce88d6dc25711a9252, with its matching configuration pinned to ZFTurbo commit 6190c903f353715b9d2f53bb40b0ff7145a13b72.

The MDX23C model repository declares CC-BY-NC-SA-4.0. It is non-commercial: commercial use is not permitted; attribution and share-alike requirements apply. The BS-RoFormer model repository declares GPL-3.0. These are model-asset licenses, distinct from the MIT license of the pymss inference code.

## Candidate families to investigate

Potential future families include additional BS/MelBand RoFormer variants, SCNet, Swin Upernet, BandIt, Apollo, BSMamba, Conformer and DTTNet.

This remains a research list, not a support statement. Each checkpoint is a separate review item because licenses, configuration formats, output stems, weight provenance and runtime constraints can differ even within one architecture family.


## Asset manager behaviour

The first asset manager implementation is in `app/mseparate/assets.py` and uses only the Python standard library.

- A checkpoint is downloaded only when a manifest supplies a pinned SHA-256.
- Downloads stream in bounded 1 MiB chunks to a sibling `.part` file.
- Existing partial files resume with HTTP Range when the server returns a valid 206/Content-Range response. If the server ignores Range and returns 200, the partial is overwritten from byte zero.
- A completed partial file is re-hashed and size-checked before atomic replacement into its final name. A verified partial left by a previous interrupted process can be promoted without another download.
- A verified final file is reused. A corrupt pre-existing final file is preserved with a `.corrupt-<id>` suffix before replacement.
- Each asset is handled independently; failures are recorded and do not prevent checking other assets.
- `--prepare-models` explicitly starts model asset preparation. `--no-download` restricts it to local verification.
- Only HTTP(S) URLs are accepted. Downloaded files are never executed.
- The current implementation uses an exclusive lock file to avoid concurrent writers to the same asset.

The asset manager now has pinned, license-reviewed model records for MelBand RoFormer, HT-Demucs, BS-RoFormer Anvuew vocals and MDX23C ZFTurbo vocals. Each newly enabled checkpoint and config is SHA-256/size verified before backend loading. Additional models remain candidates until their own license, URL, checksum, output semantics and backend compatibility are reviewed.
