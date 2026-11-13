# GPU multi-stem separation with mseparate

The AppImage includes a six-stem HT-Demucs ONNX backend and a CLAP-based
semantic source-analysis stage. CLAP is loaded only for --execute; model asset
preparation does not require it. During --execute, mseparate analyzes the mix
in overlapping 10-second windows and ranks broad source families: voice,
drums, bass, guitar, piano, synthesizer, strings, brass, woodwinds and
electronic sound effects.

CLAP is a zero-shot audio/text similarity model: scores are relative rankings,
not calibrated probabilities or proof that a source exists. It cannot reliably
name every individual synth patch or guarantee a clean stem. The planner can
only select models actually present in the registry; a source label without a
compatible specialist model remains part of a broader stem such as other.

The registry contains four enabled models: the existing MelBand RoFormer
vocal model, six-stem HT-Demucs ONNX, a vocal-focused BS-RoFormer checkpoint,
and a vocal-focused MDX23C checkpoint. The two new PyTorch models use separate
adapters backed by the pinned `pymss==2.3.5` runtime and the same sequential
load/infer/unload lifecycle as the existing backends.

| Model | Weights | Declared model license | SHA-256 |
|---|---:|---|---|
| BS-RoFormer Anvuew vocals | 204,485,563 bytes | GPL-3.0 | `8d0f72ad0ac4154f5f4c3cdf230880bbb9a5bd01dc610241a2abfe787f0d7784` |
| MDX23C ZFTurbo vocals | 448,101,203 bytes | CC-BY-NC-SA-4.0 | `49d51472769e34a2501cd1da782346a3212555c3a5619fc2c53507445528d816` |

The BS-RoFormer checkpoint and config are pinned to Hugging Face revision
`7e4e5669cad8a8810983e02fa25294229246d374`. Its model repository declares
GPL-3.0. The MDX23C checkpoint is pinned to Sucial/MSST-WebUI revision
`55155c4e2f99d0ca94d24fce88d6dc25711a9252`; that model repository declares
CC-BY-NC-SA-4.0. The latter license requires attribution, prohibits commercial
use, and applies share-alike terms to adaptations/redistributions. The MDX23C
config is pinned to the ZFTurbo `v1.0.0` commit `6190c903f353715b9d2f53bb40b0ff7145a13b72`. The registry also pins
config SHA-256 and byte size for both models.
The CC-BY-NC-SA-4.0 declaration is from the Sucial/MSST-WebUI mirror hosting the checkpoint; a separate weight-specific license grant from the original ZFTurbo release author was not located. The application treats the mirror's declaration as a restriction, but this is not independent legal confirmation of the uploader's rights. Avoid redistributing the MDX23C checkpoint unless its provenance is confirmed.

Both vocal models can also return an `instrumental.wav` residual computed exactly as input mixture minus the model's vocals estimate. The manifest marks this as a derived stem, not a native network prediction. When `--prefer-model` selects BS-RoFormer or MDX23C, the same model produces both the vocals and its complementary residual.

The inference library code (`pymss`) is MIT-licensed, but its code license does
not replace the separate licenses of the downloaded model weights. Review the
model license before using or redistributing the corresponding checkpoint.
The pinned `pymss==2.3.5` release is classified as Alpha upstream. Its API and adapter contract are covered by CPU tests, but real checkpoint loading, VRAM use and GPU inference remain unverified until the hardware integration tests are run.
The BS-RoFormer and MDX23C checkpoints have not yet undergone real-hardware
inference in this project.

For overlapping output capabilities, the planner first prefers a declared
specialist, then the explicit `planner_priority` in the manifest, and finally
the source-salience score and stable model ID. The priority is an operator-set
routing preference, not a measured quality score. Defaults keep the existing
MelBand backend ahead of the newly added alternatives until comparable
benchmarking has been performed. CLAP/acoustic scores are not estimates of
model quality.

Use `--prefer-model MODEL_ID` to explicitly select one of the vocal alternatives
for overlapping routes. For example:

```bash
mseparate input.wav --plan-only --prefer-model bs-roformer-anvuew-vocals
mseparate input.wav --execute --device cuda --prefer-model mdx23c-zfturbo-vocals
```

The preference changes routing only; it does not change the model's license or
skip asset verification. Use `mseparate --list-models` to see model IDs and
their declared licenses. The default priority keeps the established MelBand
route first until equivalent reference-set benchmarks are available.

The current general-purpose model separates a supported audio file into:

- `vocals.wav`
- `drums.wav`
- `bass.wav`
- `other.wav`
- `guitar.wav`
- `piano.wav`

The guitar and piano stems are additional estimates, not guaranteed isolated
sources. In particular, the upstream HT-Demucs documentation describes piano
as experimental and prone to bleed/artifacts.

## Build requirements

Rebuild the AppImage after pulling this branch. The build installs:

- `demucs-onnx==0.3.4` without its generic CPU `onnxruntime` dependency;
- `onnxruntime-gpu[cuda,cudnn]` (CUDA 13 / cuDNN 9) and its NVIDIA runtime
  wheels;
- Hugging Face Hub and SOXR;
- Transformers and librosa for on-demand CLAP source recognition.
- `pymss==2.3.5` and `pymss-core==0.1.15` for BS-RoFormer and MDX23C inference; its `torch>=2.7.1` requirement is satisfied by the existing pinned PyTorch 2.13.0 build.

The ONNX Runtime CUDA 13 build is selected to match this project's
PyTorch 2.13 CUDA 13.0 stack. The NVIDIA driver must support CUDA 13.0;
the RTX 3060 and RTX 3060 Ti GPU architectures are supported.

## Run

Quote the input path when it contains spaces:

```bash
mseparate "/home/clerik/Музыка/Servus Machina/Servus_Machina_capture.wav" \
  --execute \
  --device cuda \
  --output-dir "/home/clerik/Музыка/Servus Machina/Separated" \
  --report
```

The first execution downloads the CLAP source-analysis model and the selected
ONNX model from Hugging Face. CLAP is cached under
`~/.local/share/mseparate/models/source-analysis/`; ONNX files are managed by
demucs-onnx. Later runs reuse their caches. `--no-download` requires both
model families to be cached already.

To explicitly download and prewarm every currently eligible model before
inference, run the same command with `--prepare-models` and without
`--execute`; then run it again with `--execute`.

`--device cuda` is strict: if ONNX Runtime does not expose
`CUDAExecutionProvider`, execution fails with a diagnostic. In strict mode
mseparate passes the explicit `CUDAExecutionProvider` name rather than the
upstream `cuda` alias, because that alias appends `CPUExecutionProvider` as a
fallback. Before inference, mseparate prewarms the graph, verifies that the
session registered only `CUDAExecutionProvider`, and disables ONNX Runtime's
retry fallback. If CUDA session creation or inference fails, the task fails
instead of retrying on CPU. `--device auto` may use CPU if CUDA is unavailable
and may retain CPU as an ONNX Runtime fallback provider.

The six-stem model uses the `fp16weights` download variant to reduce cache
size. This only reduces download/storage size; the upstream package states
that runtime precision, latency, and memory use are unchanged.

## Output layout

```text
Separated/
  mseparate-stems/
    separate-01/
      drums.wav
      bass.wav
      other.wav
      vocals.wav
      guitar.wav
      piano.wav
    separate-02/
      ... optional outputs from other eligible models ...
  mseparate-report.json
```

Task numbering is generated by the planner and can vary when the eligible
model registry changes. Check the execution summary and JSON report to map
each task to its model.

## VRAM notes for RTX 3060 / 3060 Ti

The six-stem model is a single ONNX graph and runs one model at a time. The
8 GB RTX 3060 Ti and 12 GB RTX 3060 use the same model. If CUDA reports an
out-of-memory error, use `--device cpu` as a fallback; do not expect the
`fp16weights` file to reduce inference VRAM because it changes only the
stored weights.

## Optional real-model integration tests

The default CI suite uses mocked ONNX Runtime modules and does not download
model weights. On a machine with the package installed and network access, the
real CPU integration test downloads the six-stem fp16weights graph and checks
that all six WAV files are produced with the expected channel count, sample
rate and duration:

```bash
MSEPARATE_RUN_DEMUCS_ONNX_INTEGRATION=1 \
  python -m pytest -q tests/test_mseparate_demucs_onnx.py -k RUN_MODEL
```

On a host with a working NVIDIA driver and ONNX Runtime CUDA provider, run the
same real-model check through the strict CUDA path:

```bash
MSEPARATE_RUN_DEMUCS_ONNX_GPU=1 \
  python -m pytest -q tests/test_mseparate_demucs_onnx.py -k RUN_GPU
```

These tests are opt-in because they download model weights and the GPU test
requires compatible hardware. A successful CPU test validates the pinned
Python API and output contract; only the GPU test validates CUDA execution.

## Adding more instrument families

The current general-purpose model exposes six stems, while CLAP adds broad
source-family ranking but does not itself separate audio. Further splits such as
kick/snare/toms/cymbals or individual string/synth parts require additional
specialist models with verified model files, licenses, input/output
contracts, and quality checks. They should be added as separate sequential
backends/tasks rather than relabeling the existing `other` residual as
specific instruments.


## Sample-rate conversion

Input WAV files at 48, 88.2, 96, 176.4 or 192 kHz can be used with models
whose manifest declares a 44.1 kHz working rate. The original file is never
modified. Conversion uses SOXR before model inference, and every output stem
is converted back to the original sample rate before quality validation and
reporting. This preserves the output file's declared sample rate and duration,
but it cannot restore ultrasonic content removed by downsampling; the actual
separation is performed at the model's supported working rate.
