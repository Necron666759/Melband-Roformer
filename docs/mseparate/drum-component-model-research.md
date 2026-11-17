# Drum component model research

## Decision summary

This stage enables acoustic Snare and Toms through the already integrated ZFTurbo 53-stem BS-RoFormer, because its manifest explicitly declares both native outputs and the pinned checkpoint/config have already passed strict-load and real inference validation in the project. The dedicated smoke test added in this branch additionally requests `snare` and `toms` from that exact checkpoint.

Cymbals remain on the combined Drums route. No currently reviewed Cymbals candidate has both a sufficiently clear weight-license grant and a verified integration path in this repository.

## Existing project model: ZFTurbo BS-RoFormer 53-stem

- Model: `zfturbo-bs-roformer-53stem-candidate`
- Backend: `bs-roformer-pymss`, already supported by the project.
- Native outputs include `snare` and `toms`; there is no `cymbals` output.
- Checkpoint: 1,368,919,887 bytes, SHA-256 `c62820893bbf86d4e734f966bd142d9157cfc8bb8e79e9d8f9ea553f3ff3519f`.
- Config: 4,184 bytes, SHA-256 `7e198062a251587088adb91215a4f44ab59e67bd62fcc805cf54d6e7dfc51103`.
- Upstream release: [ZFTurbo v1.0.21](https://github.com/ZFTurbo/Music-Source-Separation-Training/releases/tag/v1.0.21).
- The project registry records the checkpoint/config as MIT-licensed with the original author's redistribution confirmation. Training-data provenance remains unknown and is not represented as separately cleared.
- Existing CUDA validation of the 10-second chunk profile was performed on an RTX 3060 Ti. The strict VRAM admission policy remains unchanged: the low-VRAM profile still requires at least 7 GiB total VRAM and 6 GiB currently free. No CPU fallback is introduced for GPU-only use.
- The previous real-model smoke selected only bass, keys, synth and vocal. This branch adds Snare and Toms to the smoke output list so those two output heads are explicitly requested and checked for finite samples, shape and sample rate.

Automatic routing is limited to acoustic source evidence:
- `acoustic_snare` -> native `snare`
- `toms` / `acoustic_toms` -> native `toms`
- Electronic snare/tom evidence does not activate these acoustic heads.
- When acoustic Snare/Toms are routed through the 53-stem model, the broad Mega 53 Drums specialist is suppressed for the same live-kit-only evidence to avoid duplicate extraction. Explicit separate electronic-drums or breakbeat evidence can still be routed independently.

## DrumSep candidates for cymbals and further component separation

MVSep documents DrumSep as a second-stage model that operates on an isolated drum track, not on a full song mix. Its normal workflow first extracts the Drums stem, then applies DrumSep. MVSep lists several model families and reports DrumSep validation results on a dataset containing both acoustic and electronic drum kits.

### MDX23C DrumSep 6-stem

- Outputs: `kick`, `snare`, `toms`, `hh`, `ride`, `crash`.
- Checkpoint: `aufr33-jarredou_DrumSep_model_mdx23c_ep_141_sdr_10.8059.ckpt`.
- Checkpoint SHA-256: `d2a4aa53eb584d21eead358a4e66d1882ad182911be018f052b5da73be9096d0`.
- Config SHA-256: `17d1649a227f841165bdb4c11a42082898192a1ea3ceab7e7e0b9293d6589dd6`.
- Current mirror: [OpenMIRLab mdxnet-infer](https://github.com/openmirlab/mdxnet-infer), release `weights-drumsep-v1`.
- The MDX23C TFC-TDF inference architecture is MIT-licensed and can be implemented using the existing `mdx23c-pymss` backend family, but the model weights are a separate matter.
- The original authors (aufr33/jarredou) did not publish explicit weight-license terms. The mirror records this as unknown and says non-commercial-safe only pending author confirmation. This project therefore does not add the checkpoint as an enabled/downloadable model. No license permission is inferred from the MIT license of the inference code.
- The original 5-stem checkpoint (kick/snare/toms/hh/cymbals) is reported lost upstream; the available 6-stem checkpoint is not a drop-in replacement for it. Ride and Crash remain separate native outputs. The upstream inference package has an explicit `combine-cymbals` option, so a future integration could expose a derived Cymbals stem only after preserving and testing that explicit merge operation; it must not relabel either output as a native Cymbals head.
- Integration also requires a parent/child execution path: first a verified Drums model must isolate the drum track, then DrumSep may process that track. Running this model directly on a full mix is outside its documented input assumptions.

### Other MVSep DrumSep families

MVSep's Ensemble All-In documents a seven-output DrumSep stage: Kick, Toms, Snare, Cymbals, Hi-Hat, Crash and Ride. Its page explicitly says the extra instrument stems use filtered inputs from the earlier stems and that the overall pipeline is slow. This is a useful reference for the intended staged architecture, not a locally reproducible checkpoint manifest.

MVSep also lists SCNet XL 4/5/6-stem, MelBand RoFormer 4/6-stem and BS-RoFormer 5/10-stem variants. Public benchmark results show these as relevant candidates, but this repository has not established a pinned checkpoint/config pair, weight-license grant, and compatible runtime validation for those variants. They are research leads, not enabled routes.

### Hybrid Demucs DrumSep

The `inagoy/drumsep` project contains MIT-licensed Hybrid Demucs inference code with Kick, Snare, Cymbals and Toms outputs. The repository's code license is not treated as a separate model-weight license grant; weight provenance/redistribution terms need independent review. It expects a drum recording as input. This repository deliberately excludes its existing HT-Demucs ONNX model from normal routing because of measured inference cost; the inagoy project is a distinct implementation and would require its own backend, checkpoint provenance review, performance evaluation and VRAM admission tests before consideration. It is not enabled by this change.

## Validation boundaries

The dedicated real-model smoke validates strict checkpoint loading and that the requested Snare/Toms outputs are returned with finite audio arrays of the expected shape and sample rate. It does not measure separation quality, bleed, SDR/SI-SDR, or quality on real drum recordings. Those require a ground-truth drum-component evaluation set and separate reporting.

## Research sources

- [ZFTurbo pretrained model catalogue](https://github.com/ZFTurbo/Music-Source-Separation-Training/blob/main/docs/pretrained_models.md) — lists the DrumSep MDX23C 5/6-stem models and checkpoint/config resources.
- [MVSep algorithms and DrumSep description](https://www.mvsep.com/en/algorithms) — documents the drum-only input requirement, staged Drums-then-DrumSep workflow, model families and benchmark tables.
- [OpenMIRLab MDX23C inference repository](https://github.com/openmirlab/mdxnet-infer) — documents the mirrored 6-stem checkpoint, hashes, missing 5-stem checkpoint and unresolved weight licensing.
- [inagoy/drumsep](https://github.com/inagoy/drumsep) — MIT-licensed Hybrid Demucs implementation and four drum outputs.
