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
- The resource index provides direct checkpoint/config URLs, but the exact
  weight-specific license, immutable file revision, SHA-256 checksums and
  compatibility with the project's installed pymss version still need
  verification. The corresponding registry entries are candidates and remain
  disabled until those gates are completed.

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
- The parent model remains a disabled candidate pending asset and runtime
  verification.

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

All three external model entries remain candidates and are excluded from
automatic execution until the missing provenance, licensing, checksums and
runtime requirements are resolved.
