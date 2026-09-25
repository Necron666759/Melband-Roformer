"""Automated tests for the melband-roformer wrapper.

Run inside the built venv:
    /usr/lib/melband-roformer/venv/bin/python -m pytest tests/ -v

Tests are split into two tiers:

  * always-run: import, argument parsing, --version, --info, CUDA
    detection logic -- these must pass on any machine (CPU-only included)
    and do not require the model to be downloaded or a GPU to be present.

  * gated (skipped unless explicitly enabled): full separation on a real
    audio file, and the GPU-forced path. Gated because they need the
    ~913 MB model downloaded (MELBAND_ROFORMER_RUN_MODEL_TESTS=1) and/or an
    actual NVIDIA GPU (MELBAND_ROFORMER_RUN_GPU_TESTS=1) -- neither is
    assumed to exist in every CI/build environment, matching the task's
    own instruction not to make GPU a hard requirement for building the
    .deb.
"""
from __future__ import annotations

import os
import subprocess
import sys
import wave
from pathlib import Path

import pytest

from melband_roformer_wrapper import __version__
from melband_roformer_wrapper import gpuinfo
from melband_roformer_wrapper.cli import build_parser, main


# --------------------------------------------------------------------------
# Always-run tier
# --------------------------------------------------------------------------

def test_import():
    import melband_roformer_wrapper  # noqa: F401


def test_version_string():
    assert __version__.count(".") == 2


def test_cli_version(capsys):
    rc = main(["--version"])
    assert rc == 0
    out = capsys.readouterr().out
    assert "melband-roformer" in out
    assert __version__ in out


def test_cli_info_runs_without_crashing(capsys):
    # Must succeed even with no GPU / no torch import path broken --
    # detect() is defensive.
    rc = main(["--info"])
    assert rc == 0
    out = capsys.readouterr().out
    for field in ("Application version:", "Python:", "PyTorch:",
                  "CUDA available:", "Selected device:"):
        assert field in out


def test_cli_no_args_prints_usage_and_exits_nonzero(capsys):
    rc = main([])
    assert rc == 2
    err = capsys.readouterr().err
    assert "No input file" in err


def test_parser_accepts_expected_flags():
    parser = build_parser()
    ns = parser.parse_args([
        "a.wav", "b.wav",
        "--output-dir", "./out",
        "--device", "cpu",
        "--model", "melband-roformer-kim-vocals",
    ])
    assert ns.inputs == ["a.wav", "b.wav"]
    assert ns.output_dir == "./out"
    assert ns.device == "cpu"
    assert ns.model == "melband-roformer-kim-vocals"


def test_missing_input_file_errors_cleanly(tmp_path, capsys):
    missing = tmp_path / "does_not_exist.wav"
    rc = main([str(missing)])
    assert rc == 2
    err = capsys.readouterr().err
    assert "not found" in err


def test_sitecustomize_patches_deprecated_sdp_kernel():
    """The venv-wide sitecustomize.py (see its own docstring for why it
    has to be a top-level module rather than living inside
    melband_roformer_wrapper) must replace the deprecated
    torch.backends.cuda.sdp_kernel() with a shim built on
    torch.nn.attention.sdpa_kernel(), so upstream's own call to it no
    longer raises FutureWarning. CPU-only, no GPU/model required: the
    context manager just needs to be entered, not actually run attention
    on a real device.

    Deliberately does NOT rely on this *test process*'s own ambient
    `import sitecustomize` resolution (i.e. whatever Python's `site`
    machinery happens to auto-import at this interpreter's own startup).
    That resolution depends on this host's sys.path ordering, which is
    not guaranteed -- e.g. some other installed software can ship its
    own sitecustomize.py that wins the race, in which case this test
    process's own `torch.backends.cuda.sdp_kernel` is never patched, even
    though nothing about the actual product is broken. What production
    code (cli.py/selftest.py) actually relies on for correctness is
    `subprocess_env()`'s explicit PYTHONPATH override for the *real*
    upstream subprocess -- so that is what this test exercises too: a
    fresh interpreter, launched exactly the way cli.py/selftest.py launch
    the real `melband-roformer-infer` subprocess.
    """
    import subprocess
    import sys

    from melband_roformer_wrapper import subprocess_env

    code = (
        "import warnings, torch\n"
        "with warnings.catch_warnings():\n"
        "    warnings.simplefilter('error', FutureWarning)\n"
        "    with torch.backends.cuda.sdp_kernel(enable_flash=True, "
        "enable_math=True, enable_mem_efficient=True):\n"
        "        pass\n"
        "print('SDP_KERNEL_OK')\n"
    )
    result = subprocess.run(
        [sys.executable, "-c", code],
        env=subprocess_env(), capture_output=True, text=True, timeout=60,
    )
    assert result.returncode == 0, (
        f"subprocess failed (stdout={result.stdout!r} stderr={result.stderr!r})"
    )
    assert "SDP_KERNEL_OK" in result.stdout, (
        f"unexpected output (stdout={result.stdout!r} stderr={result.stderr!r})"
    )


def test_sitecustomize_patches_deprecated_cuda_amp_autocast():
    """Same coverage as test_sitecustomize_patches_deprecated_sdp_kernel
    above, for the second deprecated symbol sitecustomize.py patches:
    torch.cuda.amp.autocast(), used by upstream mel_band_roformer/utils.py.
    CPU-only: the shim's __init__/__enter__/__exit__ are exercised without
    needing a real CUDA device -- torch.amp.autocast('cuda', ...) builds
    fine on a CPU-only host, it just wouldn't autocast anything useful at
    runtime, which is irrelevant to what's being tested here (that the
    deprecated call site no longer raises FutureWarning).

    Also does NOT rely on this test process's own ambient `import
    sitecustomize` for the same reason as the sdp_kernel test: goes
    through subprocess_env() so it exercises the exact mechanism
    cli.py/selftest.py rely on for the real upstream subprocess.
    """
    import subprocess
    import sys

    from melband_roformer_wrapper import subprocess_env

    code = (
        "import warnings, torch\n"
        "with warnings.catch_warnings():\n"
        "    warnings.simplefilter('error', FutureWarning)\n"
        "    with torch.cuda.amp.autocast():\n"
        "        pass\n"
        "    with torch.cuda.amp.autocast(enabled=True, dtype=torch.bfloat16, "
        "cache_enabled=False):\n"
        "        pass\n"
        "print('AUTOCAST_OK')\n"
    )
    result = subprocess.run(
        [sys.executable, "-c", code],
        env=subprocess_env(), capture_output=True, text=True, timeout=60,
    )
    assert result.returncode == 0, (
        f"subprocess failed (stdout={result.stdout!r} stderr={result.stderr!r})"
    )
    assert "AUTOCAST_OK" in result.stdout, (
        f"unexpected output (stdout={result.stdout!r} stderr={result.stderr!r})"
    )


def test_gpu_detect_never_raises():
    # Whatever the host looks like (no GPU, broken driver, no torch),
    # detect() must not raise -- that's the whole point of the module.
    status = gpuinfo.detect()
    assert isinstance(status.cuda_available, bool)


def test_resolve_device_cpu_always_ok():
    assert gpuinfo.resolve_device("cpu") == "cpu"


def test_resolve_device_cuda_forced_raises_if_unavailable():
    status = gpuinfo.detect()
    if status.cuda_available:
        pytest.skip("CUDA is available on this machine; nothing to assert here")
    with pytest.raises(gpuinfo.GPURequiredError):
        gpuinfo.resolve_device("cuda", status)


def _make_silence_wav(path: Path, seconds: float = 1.0, sr: int = 44100) -> None:
    n_frames = int(seconds * sr)
    with wave.open(str(path), "w") as w:
        w.setnchannels(1)
        w.setsampwidth(2)
        w.setframerate(sr)
        w.writeframes(b"\x00\x00" * n_frames)


def test_self_test_cli_flag_runs(capsys):
    from melband_roformer_wrapper.cli import main as cli_main
    rc = cli_main(["--self-test"])
    out = capsys.readouterr().out
    assert "Python" in out
    assert rc in (0, 1)  # 1 if model/GPU missing on this machine -- not a bug


# --------------------------------------------------------------------------
# Gated tier: needs the real ~913MB model downloaded
# --------------------------------------------------------------------------

RUN_MODEL_TESTS = os.environ.get("MELBAND_ROFORMER_RUN_MODEL_TESTS") == "1"
RUN_GPU_TESTS = os.environ.get("MELBAND_ROFORMER_RUN_GPU_TESTS") == "1"


@pytest.mark.skipif(not RUN_MODEL_TESTS, reason="set MELBAND_ROFORMER_RUN_MODEL_TESTS=1 "
                     "to run (downloads/uses the real ~913MB model)")
def test_end_to_end_separation_produces_two_files(tmp_path):
    input_wav = tmp_path / "silence.wav"
    _make_silence_wav(input_wav, seconds=2.0)
    out_dir = tmp_path / "out"

    rc = main([str(input_wav), "--output-dir", str(out_dir), "--device",
               "cpu", "--yes"])
    assert rc == 0
    vocals = list(out_dir.glob("*_vocals.wav"))
    instrumental = list(out_dir.glob("*_instrumental.wav"))
    assert len(vocals) == 1, "expected exactly one *_vocals.wav"
    assert len(instrumental) == 1, "expected exactly one *_instrumental.wav"
    assert vocals[0].stat().st_size > 0
    assert instrumental[0].stat().st_size > 0


@pytest.mark.skipif(not RUN_MODEL_TESTS, reason="needs the real model downloaded")
def test_music_plus_spoken_voice_case(tmp_path):
    """The task's key real-world scenario: a short spoken monologue over an
    industrial-music intro. We cannot assert perceptual separation quality
    automatically (no ground truth, no reference stem) -- this test only
    asserts the pipeline *runs end-to-end* on such audio and produces
    non-empty, non-identical output files. Manual listening is required to
    judge actual separation quality; see README.md "Known limitations"."""
    import numpy as np
    import soundfile as sf

    sr = 44100
    seconds = 5.0
    t = np.linspace(0, seconds, int(sr * seconds), endpoint=False)
    # Crude synthetic stand-in for "industrial music bed": layered
    # low-frequency pulse + noise burst, NOT a substitute for a real test
    # fixture -- replace with an actual short royalty-free clip for
    # meaningful manual QA.
    music_bed = 0.2 * np.sin(2 * np.pi * 60 * t) + 0.05 * np.random.randn(len(t))
    spoken_proxy = 0.15 * np.sin(2 * np.pi * 180 * t) * (t % 1.0 < 0.4)
    mix = (music_bed + spoken_proxy).astype("float32")

    input_wav = tmp_path / "industrial_intro_with_voice.wav"
    sf.write(input_wav, mix, sr)
    out_dir = tmp_path / "out"

    rc = main([str(input_wav), "--output-dir", str(out_dir), "--device",
               "cpu", "--yes"])
    assert rc == 0
    vocals = list(out_dir.glob("*_vocals.wav"))
    assert len(vocals) == 1
    assert vocals[0].stat().st_size > 0


@pytest.mark.skipif(not RUN_GPU_TESTS, reason="set MELBAND_ROFORMER_RUN_GPU_TESTS=1 "
                     "to run on a machine with a real NVIDIA GPU")
def test_gpu_forced_path_end_to_end(tmp_path):
    input_wav = tmp_path / "silence.wav"
    _make_silence_wav(input_wav, seconds=2.0)
    out_dir = tmp_path / "out"
    rc = main([str(input_wav), "--output-dir", str(out_dir), "--device",
               "cuda", "--yes"])
    assert rc == 0
    assert list(out_dir.glob("*_vocals.wav"))


def test_cli_entrypoint_script_is_on_path_after_install():
    """Smoke check for the packaged console-script entrypoint itself,
    meant to be run against the *installed* venv (see test-deb.sh), not
    plain `pytest` from a source checkout."""
    exe = Path(sys.prefix) / "bin" / "melband-roformer"
    if not exe.exists():
        pytest.skip("not running inside the installed venv")
    result = subprocess.run([str(exe), "--version"], capture_output=True, text=True)
    assert result.returncode == 0
    assert "melband-roformer" in result.stdout
