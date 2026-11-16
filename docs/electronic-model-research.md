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
- Registry ID: `mvsep-synth-specialist-candidate`; deliberately disabled.

### ZFTurbo BS-RoFormer 53-stem

- Reference: https://github.com/ZFTurbo/Music-Source-Separation-Training/releases
- The upstream release describes 53 output classes, including `synth`,
  `bass`, `drums`, `kick`, `snare`, `hh`, `keys`, `organ`,
  `digital-piano`, vocals and a wide range of acoustic instruments.
- The release notes warn that the model is memory-intensive and recommend at
  least 16 GB VRAM. Individual stems may be weaker than dedicated specialists,
  and stems may overlap rather than sum to the original mix.
- This is a broad multi-instrument model, not a model trained exclusively on
  electronic genres. Its single `synth` stem does not separate pads, leads,
  arpeggios and synth bass individually.
- Exact asset provenance, weight license, config/checkpoint hashes and local
  runtime compatibility still need verification. Registry ID:
  `zfturbo-bs-roformer-53stem-candidate`; deliberately disabled.

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
