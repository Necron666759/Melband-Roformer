"""Shared adapter for MSST-format models through the pinned pymss runtime."""
from __future__ import annotations

import gc
from pathlib import Path
from time import time

import numpy as np
import soundfile as sf
from tqdm.auto import tqdm

from .assets import verify_asset
from .execution import BackendInfo, StemArtifact
from .types import ModelManifest


def _format_progress_time(value: float) -> str:
    seconds = max(0, int(round(value or 0)))
    hours, remainder = divmod(seconds, 3600)
    minutes, seconds = divmod(remainder, 60)
    if hours:
        return f"{hours:d}:{minutes:02d}:{seconds:02d}"
    return f"{minutes:d}:{seconds:02d}"


class _PymssAudioProgressBar(tqdm):
    """tqdm bar that retains pymss' elapsed/total audio and RTF display."""

    @property
    def format_dict(self):
        data = super().format_dict
        done = float(self.n or 0)
        elapsed = max(0.0, time() - getattr(self, "start_t", time()))
        rtf = "--" if done <= 0 else f"{elapsed / done:.2f}"
        data["audio_progress"] = (
            f"{_format_progress_time(done)}/{_format_progress_time(self.total)}, RTF={rtf}"
        )
        return data


class _PymssProgressCallback:
    """Render pymss callback updates without creating an empty 0% bar."""

    def __init__(self) -> None:
        self._bar = None
        self._message = None
        self._total = None

    def __call__(self, done, total, message) -> None:
        total = max(1, int(total or 1))
        done = max(0, min(int(done), total))
        changed = self._message != message or self._total != total
        reset = self._bar is not None and done < self._bar.n
        if changed or reset:
            self.close()
        self._message = message
        self._total = total

        # pymss emits callback(0, total, ...) before the first model chunk.
        # Wait for a useful non-zero update, then create the bar at that value.
        if self._bar is None:
            if done <= 0:
                return
            self._bar = _PymssAudioProgressBar(
                total=total,
                initial=done,
                desc=message,
                leave=False,
                unit="",
                bar_format="{l_bar}{bar}| {audio_progress}",
                mininterval=0,
                miniters=1,
            )
            self._bar.refresh()
            return

        if done > self._bar.n:
            self._bar.update(done - self._bar.n)
            self._bar.refresh()

    def close(self) -> None:
        if self._bar is not None:
            self._bar.close()
            self._bar = None

class PymssAdapter:
    """Run one MSST-format model through pymss with explicit local assets."""

    BACKEND = ""
    MODEL_TYPE = ""

    def __init__(self) -> None:
        self._separator = None
        self._progress: _PymssProgressCallback | None = None
        self._model_id: str | None = None
        self._device: str | None = None
        self._native_stems: tuple[str, ...] = ()
        self._derived_stems: tuple[str, ...] = ()

    def compatibility(self, manifest: ModelManifest) -> tuple[bool, str]:
        if manifest.backend != self.BACKEND:
            return False, f"Manifest backend must be {self.BACKEND!r}"
        if "vocals" not in manifest.native_stems:
            return False, "MSST vocal adapter requires vocals as a native output"
        supported = set(manifest.native_stems) | set(manifest.derived_stems)
        if set(manifest.target_stems) - supported:
            return False, "MSST adapter target stems must be native or explicitly derived"
        if set(manifest.derived_stems) - {"instrumental"}:
            return False, "MSST adapter currently supports only the mixture-minus-vocals instrumental residual"
        assets = {asset.role: asset for asset in manifest.assets}
        missing = {"checkpoint", "config"} - set(assets)
        if missing:
            return False, "Manifest is missing required assets: " + ", ".join(sorted(missing))
        if any(not assets[role].sha256 for role in ("checkpoint", "config")):
            return False, "Checkpoint and config must both have pinned SHA-256 digests"
        return True, ""

    def load(self, manifest: ModelManifest, model_dir: Path, device: str) -> None:
        compatible, reason = self.compatibility(manifest)
        if not compatible:
            raise RuntimeError(reason)
        if device not in {"auto", "cuda", "cpu"}:
            raise ValueError(f"Unsupported device: {device!r}")

        assets = {asset.role: model_dir / asset.relative_path for asset in manifest.assets}
        for role in ("checkpoint", "config"):
            asset = next(item for item in manifest.assets if item.role == role)
            if not verify_asset(assets[role], asset):
                raise RuntimeError(
                    f"{role} asset is missing or failed SHA-256/size verification: {assets[role]}"
                )

        try:
            from pymss import MSSeparator
        except ImportError as exc:
            raise RuntimeError(
                "The MSST model runtime is unavailable. Install pymss==2.3.5 "
                "in the mseparate environment."
            ) from exc

        # Explicit local paths prevent pymss from consulting or downloading
        # another model catalog entry. Model bytes are verified by mseparate
        # before the runtime is allowed to deserialize them.
        self._progress = _PymssProgressCallback()
        self._separator = MSSeparator(
            model_type=self.MODEL_TYPE,
            model_path=str(assets["checkpoint"]),
            config_path=str(assets["config"]),
            device=device,
            device_ids=[0],
            output_format="wav",
            use_tta=False,
            store_dirs={},
            debug=True,
            progress_callback=self._progress,
            inference_params={"batch_size": 1, "normalize": False},
        )
        self._model_id = manifest.model_id
        self._device = str(getattr(self._separator, "device", device))
        self._native_stems = manifest.native_stems
        self._derived_stems = manifest.derived_stems

    def infer(
        self, input_audio: Path, output_dir: Path, target_stems: tuple[str, ...]
    ) -> list[StemArtifact]:
        if self._separator is None:
            raise RuntimeError("MSST adapter is not loaded")
        if not input_audio.is_file():
            raise FileNotFoundError(input_audio)

        mixture, sample_rate = sf.read(
            str(input_audio), dtype="float32", always_2d=True
        )
        # pymss expects channel-major input and returns sample-major outputs.
        native_targets = [stem for stem in target_stems if stem in self._native_stems]
        # Instrumental is deliberately a derived residual, never presented as
        # a native model output: mixture - vocals gives an exactly complementary
        # pair for the selected model's vocal estimate.
        if "instrumental" in target_stems and "vocals" not in native_targets:
            native_targets.append("vocals")
        try:
            # pymss' callback reports audio-time progress. The callback skips
            # its initial zero update and starts the bar at the first real
            # inference update, without redirecting stdout/stderr or hiding
            # runtime warnings.
            separated = self._separator.separate(
                mixture.T, pbar=False, stems=native_targets
            )
        finally:
            if self._progress is not None:
                self._progress.close()
        output_dir.mkdir(parents=True, exist_ok=True)
        artifacts: list[StemArtifact] = []
        for stem in target_stems:
            if stem == "instrumental" and stem in self._derived_stems:
                if "vocals" not in separated:
                    raise RuntimeError(
                        f"{self.MODEL_TYPE} did not return vocals needed for the instrumental residual"
                    )
                vocals = np.asarray(separated["vocals"], dtype=np.float32)
                if vocals.ndim == 1:
                    vocals = vocals[:, None]
                elif (
                    vocals.ndim == 2
                    and vocals.shape[0] in (1, 2)
                    and vocals.shape[1] == mixture.shape[0]
                ):
                    vocals = vocals.T
                if vocals.shape != mixture.shape:
                    raise RuntimeError(
                        f"{self.MODEL_TYPE} vocals shape {vocals.shape} does not match mixture {mixture.shape}"
                    )
                samples = mixture - vocals
            else:
                if stem not in separated:
                    raise RuntimeError(f"{self.MODEL_TYPE} did not return requested stem {stem!r}")
                samples = np.asarray(separated[stem], dtype=np.float32)
                if samples.ndim == 1:
                    samples = samples[:, None]
                elif (
                    samples.ndim == 2
                    and samples.shape[0] in (1, 2)
                    and samples.shape[1] == mixture.shape[0]
                ):
                    samples = samples.T
            if samples.ndim != 2 or samples.shape[1] not in (1, 2):
                raise RuntimeError(
                    f"{self.MODEL_TYPE} returned an invalid {stem!r} array shape {samples.shape}"
                )
            if samples.shape != mixture.shape:
                raise RuntimeError(
                    f"{self.MODEL_TYPE} output {stem!r} has shape {samples.shape}; "
                    f"expected {mixture.shape}"
                )
            path = output_dir / f"{stem}.wav"
            sf.write(str(path), samples, sample_rate, subtype="FLOAT")
            artifacts.append(StemArtifact(stem, path))
        return artifacts

    def unload(self) -> None:
        separator = self._separator
        self._separator = None
        self._model_id = None
        self._device = None
        self._native_stems = ()
        self._derived_stems = ()
        if separator is not None:
            separator.close()
        del separator
        gc.collect()

    def info(self) -> BackendInfo:
        return BackendInfo(
            name=f"pymss-{self.MODEL_TYPE}",
            device=self._device,
            details={"model_id": self._model_id or ""},
        )
