# Electronic-music model research

Research notes for expanding the local `mseparate` model catalogue. A model is
not executable merely because a model card or hosted service describes it:
execution requires a supported local backend, immutable asset URLs, verified
SHA-256 hashes, a confirmed weight license, and output-stem mapping.

## High-value candidates

### MVSep Synth

- Reference: https://mvsep.com/en/algorithms
- Described outputs: `synth` and `other`.
- The service describes the synth family as including synth/synthesizer,
  synth pad, synth bass, synth vocals, synth strings, synth percussion,
  synth FX, synth keys, synth brass, synth guitar, synth flute and ambient
  synth material.
- This is the most directly relevant specialist discovered for electronic
  music. However, no public downloadable checkpoint, matching configuration,
  local inference backend or redistribution license was verified.
- The hosted MVSep Synth algorithm is not the same artifact as the public
  Mega 53 `synth` checkpoint. Registry ID:
  `mvsep-synth-specialist-candidate`; deliberately disabled.

### MVSep Mega 53 single-stem derivatives

- Model catalogue: https://huggingface.co/noblebarkrr/BS-Roformer-MVSep-Mega-53-stems/tree/main/v1
- Resource index: https://huggingface.co/noblebarkrr/mvsepless_resources
- This is more useful than treating the 53-stem checkpoint as one monolithic
  inference. The public resource index exposes separate BS-RoFormer checkpoints
  and YAML files for individual targets, each returning the requested stem and
  `other` residual.
- Electronic-production-relevant targets include `synth`, `keys`, `bass`,
  `kick`, `drums`, `organ`, `digital-piano`, `percussion` and `hh`.
  This lets the planner request only specialists that audio analysis considers
  plausible rather than load all 53 models.
- The public model folder lists the individual checkpoint files at about
  77.6 MB each, plus small YAML configs. These file sizes do not predict
  inference speed; actual performance must be benchmarked on the target GPU.
- An independent registry snapshot pins both artifacts to revision
  `0677941f9cdfd891a6bc3336229244e197035327` and publishes checkpoint and
  config SHA-256 values. The Synth and Keys checkpoint hashes are also
  corroborated by a separate `SHA256SUMS` manifest. The bundled registry now
  records these published digests and the PyMSS catalog's checkpoint byte sizes.
- The derivative repository's original license metadata was incomplete. The
  upstream Mega 53 author later confirmed MIT licensing and redistribution
  permission for the original 53-stem weights and conversions. The maintainer
  accepted that evidence for the bundled Synth/Keys derivatives, while
  separately recording that training-data provenance remains unknown. Their
  current license and runtime gates are represented in the bundled registry.

### ZFTurbo BS-RoFormer 53-stem parent model

- Reference: https://github.com/ZFTurbo/Music-Source-Separation-Training/releases
- The upstream release describes 53 output classes, including `synth`,
  `bass`, `drums`, `kick`, `snare`, `hh`, `keys`, `organ`,
  `digital-piano`, vocals and a wide range of acoustic instruments.
- The release notes warn that the full multi-stem model is memory-intensive and
  recommend at least 16 GB VRAM. Individual stems may be weaker than dedicated
  specialists, and stems may overlap rather than sum to the original mix.
- This is a broad multi-instrument model, not a model trained exclusively on
  electronic genres. Its single `synth` stem does not separate pads, leads,
  arpeggios and synth bass individually.
- The parent model's assets, license review and runtime validation are now
  recorded in the bundled registry. It remains outside indiscriminate routing:
  the planner schedules it only for detected acoustic instruments and requests
  only selected native output heads, subject to CUDA/VRAM admission.

### BS-RoFormer SW Fixed (6 stems)

- Reference: https://huggingface.co/FractalSurfer/BS-RoFormer-SW-6stem
- Output order: bass, drums, other, vocals, guitar and piano.
- A useful six-stem general-purpose candidate, but not an electronic-music
  specialist. The public mirror currently reports an unknown license.
- Registry ID: `bs-roformer-sw-6stem-candidate`; deliberately disabled pending
  rights and asset verification.

## Implemented analysis changes

The CLAP analysis catalogue now separates broad source-family prompts from
electronic-genre prompts. Source families include electronic drums, breakbeats,
synth bass, synth leads, pads, arpeggios, keys, strings, brass, vocoder-like
vocals, synth FX and samples. Genre prompts include techno, house, trance,
drum and bass, dubstep, breakbeat, ambient electronic, synthwave, industrial
electronic and electro.

Genre matches are contextual metadata only. They do not prove a source exists
and cannot independently schedule a separation model. Planner routing uses
source-family predictions only, and model categories can explicitly identify
synth specialists.

The hosted MVSep Synth service remains a non-executable candidate because no
local checkpoint/backend/license package has been verified. BS-RoFormer SW
Fixed remains disabled pending rights and asset verification. The ZFTurbo
53-stem parent and the reviewed Mega 53 single-target derivatives have separate
eligibility states in the bundled registry; they are not enabled as a blanket
53-output batch.


## Synth and Keys single-stem candidates

The MVSep Mega 53 resource catalog declares dedicated BS-RoFormer single-target checkpoints for both `synth` and `keys`, each with a matching YAML config and a complementary `other` output. The corresponding third-party Hugging Face repository describes each file as a single-stem derivative of ZFTurbo's Mega 53 release. The repository's upload revision is pinned in the registry to `0677941f9cdfd891a6bc3336229244e197035327`.

The initial registry snapshot kept Synth and Keys disabled while weight provenance, hashes and runtime compatibility were unresolved. This status was superseded by the later checksum and real-checkpoint validation work recorded in the bundled registry: the Mega 53 Synth and Keys single-target derivatives are now license-reviewed, hash-pinned, runtime-validated and eligible for automatic source-aware routing. The separate hosted MVSep Synth service remains a different artifact and is still not treated as a locally executable checkpoint.

The shared `PymssAdapter` accepts arbitrary manifest-declared native stems, rather than requiring every BS-RoFormer model to output vocals. Vocal models retain their existing mixture-minus-vocals instrumental residual behavior; a non-vocal model cannot declare that residual without a native vocals estimate. CPU contract tests cover the adapter contract with a fake separator. Separate real-checkpoint smoke validation is recorded in the Synth/Keys manifest entries; those entries have CPU runtime validation, while CUDA performance for each derivative is not claimed by that evidence.


## Synth and Keys asset verification status

The exact Hugging Face revision is pinned to
`0677941f9cdfd891a6bc3336229244e197035327`.

| Asset | Synth SHA-256 | Keys SHA-256 |
|---|---|---|
| Checkpoint | `04749df7e2fab770a0d26bf9aa4850c1a6cd89547fc8bfe9c9d223602c0afb82` | `4bc2bde8c5b26d5e22720304deace45dfeaffb2d02dc07922a8f2be17d82c844` |
| YAML config | `3a0489996305cedf1e44f086f9b0fe9c387c86104f972092b4c9ba5ac8e15b42` | `ca9d5cb14ed0af13efd91e2e0e6ed3b81c2391b013c720f09c62ce6f2ebdfd89` |

Checkpoint sizes reported by the pinned PyMSS model catalog:
- Synth: 77,616,988 bytes.
- Keys: 77,616,219 bytes.

Digest provenance:
- [HAGerox/Stem-Separator-Models registry snapshot](https://github.com/HAGerox/Stem-Separator-Models/blob/6c87b47412d4faf8d9e8c530d12b4d66764f2cdc/registry.json) publishes all four hashes and the immutable Hugging Face revision.
- [YacobiHime/stem-splitter SHA256SUMS](https://github.com/YacobiHime/stem-splitter/blob/c474faf52eef3e9be1b5ceb2f2b31669ceff0acd/models/SHA256SUMS.txt) independently corroborates both checkpoint hashes.
- [PyMSS model catalog](https://github.com/pymss-project/pymss/blob/c78bc27265d1d2bad20e429f615c063239837be2/pymss/resources/model_catalog.json) lists both checkpoints as supported `bs_roformer` assets with the byte sizes above.

These are independently published reference digests; the local runtime smoke validates the downloaded bytes against the pinned SHA-256 and size before loading. The real-model smoke workflow loads each asset through the project's pinned `pymss==2.3.5` adapter and runs a 20-second stereo forward pass. PyMSS' current BS-RoFormer loader calls `model.load_state_dict(state_dict, assign=True)` without overriding PyTorch's default `strict=True`; a successful constructor therefore checks the checkpoint key set strictly. The bundled registry marks the Synth and Keys CPU runtime smoke as passed.

The independent external smoke evidence is for `python-audio-separator`, not pymss. The PyMSS catalog's `supported: true` is only a compatibility declaration; the exact-checkpoint mseparate smoke result is recorded separately in each manifest's runtime-validation details.

The initial third-party registry reported the derivative weight license as unknown. The later author confirmation and maintainer review are recorded in the bundled Synth/Keys manifests as accepted MIT weight licensing, with training-data provenance still unknown. Both entries are now eligible after checksum and CPU runtime validation, but automatic routing remains source-gated and does not run on unrelated audio.



## Current style-aware routing policy

The planner now classifies a provisional music-style family from CLAP genre
rankings and separately detects source families and acoustic instruments.
Electro-industrial, industrial electronic, techno, house, trance, synthwave,
electro and related styles form the electronic family. Classical, orchestral,
chamber and baroque labels form an orchestral family. Ambiguous genre rankings
are explicitly marked uncertain; style labels are never treated as calibrated
probabilities.

For electronic material, the verified Mega 53 Synth and Keys specialists are
considered when their corresponding synth-source evidence is present. Drum,
kick, bass, hi-hat and other single-target specialists are gated by their own
source evidence. A genre label by itself does not trigger a separator.

For acoustic instruments such as violin, viola, cello, harp, flute, clarinet,
oboe, bassoon, saxophone, trombone, trumpet, tuba, French horn, guitars,
harpsichord, piano and related sources, the planner may use the verified
ZFTurbo 53-stem checkpoint. It requests only the detected native heads in one
inference task (up to twelve), not all 53 output files. Acoustic/synthetic
prompt pairs are compared so that a close or stronger synth-string, synth-brass,
synth-flute or similar match suppresses an unsupported acoustic route. Existing
single-target Mega 53 specialists take precedence for categories already
covered by dedicated weights, avoiding duplicate versions of the same stem.

The 53-stem parent model remains subject to the existing CUDA and VRAM policy.
If the device does not pass the standard or verified low-VRAM admission gate,
the acoustic-instrument pass is skipped rather than silently retried on CPU.
