from __future__ import annotations

import subprocess
import sys
import tempfile
from pathlib import Path

from . import gpuinfo
from . import subprocess_env
from .modeldl import _model_is_cached  # internal, but this is our own package


def _status(ok: bool, label: str, detail: str = "") -> bool:
    tag = "[OK]  " if ok else "[FAIL]"
    suffix = f" ({detail})" if detail and not ok else ""
    print(f"{tag} {label}{suffix}")
    return ok


def run_self_test() -> int:
    all_ok = True

    all_ok &= _status(True, "Python", "")  # we're running, so this always passes

    try:
        import torch  # noqa: F401
        ok = True
        detail = f"v{torch.__version__}"
    except Exception as exc:
        ok, detail = False, str(exc)
    all_ok &= _status(ok, "PyTorch import", detail)

    status = gpuinfo.detect()
    all_ok &= _status(status.cuda_available, "CUDA available",
                       "no GPU detected / driver mismatch, see --info")

    gpu_ok = status.cuda_available and bool(status.device_name)
    if status.cuda_available:
        _status(True, f"GPU: {status.device_name}")
    else:
        print("[SKIP]  GPU (no CUDA device to report)")

    from mel_band_roformer import DEFAULT_MODEL
    slug = DEFAULT_MODEL
    cached = _model_is_cached(slug)
    all_ok &= _status(cached, f"Model present: {slug}",
                       "not downloaded; run `melband-roformer --download-model`")

    if not cached:
        print("[SKIP]  Model loads (model not downloaded)")
        print("[SKIP]  Inference on test tone (model not downloaded)")
        print()
        print("Self-test incomplete: download the model first, then re-run "
              "--self-test." if all_ok else
              "Mel-Band RoFormer is NOT fully ready (see FAILs above).")
        return 0 if all_ok else 1

    # Model is cached: attempt an actual tiny inference on a synthetic
    # 2-second test tone via the real upstream CLI, exactly like a real run.
    load_ok = True
    infer_ok = False
    try:
        with tempfile.TemporaryDirectory(prefix="melband-roformer-selftest-") as tmp:
            tmp_path = Path(tmp)
            in_dir = tmp_path / "in"
            out_dir = tmp_path / "out"
            in_dir.mkdir()
            out_dir.mkdir()

            import numpy as np
            import soundfile as sf
            sr = 44100
            t = np.linspace(0, 2.0, int(sr * 2.0), endpoint=False)
            tone = 0.1 * np.sin(2 * 3.14159265 * 220.0 * t).astype("float32")
            sf.write(in_dir / "selftest_tone.wav", tone, sr)

            import shutil as _shutil
            exe = _shutil.which("melband-roformer-infer") or str(
                Path(sys.executable).parent / "melband-roformer-infer"
            )
            result = subprocess.run(
                [exe, "--input_folder", str(in_dir), "--store_dir", str(out_dir),
                 "--model", slug],
                capture_output=True, text=True, timeout=300,
                env=subprocess_env(),
            )
            produced = list(out_dir.glob("*_vocals.wav"))
            infer_ok = result.returncode == 0 and bool(produced)
            if not infer_ok:
                print(result.stderr[-1500:] if result.stderr else "(no stderr)")
    except Exception as exc:
        load_ok = False
        print(f"  exception: {exc}")

    all_ok &= _status(load_ok, "Model loads")
    all_ok &= _status(infer_ok, "Inference on 2s test tone")

    print()
    if all_ok:
        print("Mel-Band RoFormer is ready.")
        return 0
    print("Mel-Band RoFormer is NOT fully ready (see FAILs above).")
    return 1
