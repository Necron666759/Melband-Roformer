"""Isolated adapter for the repository's existing MelBand wrapper CLI.

The wrapper is invoked as a child process so its PyTorch model lifetime ends
when inference exits. This avoids importing Torch into mseparate's coordinator
and gives the execution manager a hard process boundary for model cleanup.
"""
from __future__ import annotations

import shutil
import subprocess
from pathlib import Path

from .execution import BackendInfo, StemArtifact
from .types import ModelManifest


class MelBandCLIAdapter:
    """Run the already-installed melband-roformer wrapper for one input stem."""

    def __init__(self, executable: str = "melband-roformer", timeout: float | None = None):
        self.executable = executable
        self.timeout = timeout
        self._command: str | None = None
        self._checkpoint: Path | None = None
        self._config: Path | None = None
        self._device = "auto"
        self._model_id: str | None = None

    def compatibility(self, manifest: ModelManifest) -> tuple[bool, str]:
        if manifest.backend != "melband-roformer":
            return False, "Manifest backend must be 'melband-roformer'"
        roles = {asset.role for asset in manifest.assets}
        missing = {"checkpoint", "config"} - roles
        if missing:
            return False, "Manifest is missing required asset roles: " + ", ".join(sorted(missing))
        if not manifest.native_stems:
            return False, "Manifest must declare native_stems before this adapter can run"
        return True, ""

    def load(self, manifest: ModelManifest, model_dir: Path, device: str) -> None:
        compatible, reason = self.compatibility(manifest)
        if not compatible:
            raise RuntimeError(reason)
        command = shutil.which(self.executable)
        if command is None:
            raise FileNotFoundError(
                f"Cannot find {self.executable!r}; install the melband-roformer wrapper in this environment"
            )
        assets = {asset.role: model_dir / asset.relative_path for asset in manifest.assets}
        checkpoint = assets["checkpoint"]
        config = assets["config"]
        for role, path in (("checkpoint", checkpoint), ("config", config)):
            if not path.is_file():
                raise FileNotFoundError(f"Verified {role} asset is not present: {path}")
        if device not in {"auto", "cuda", "cpu"}:
            raise ValueError(f"Unsupported device: {device!r}")
        self._command = command
        self._checkpoint = checkpoint
        self._config = config
        self._device = device
        self._model_id = manifest.model_id

    def infer(self, input_audio: Path, output_dir: Path,
              target_stems: tuple[str, ...]) -> list[StemArtifact]:
        if not self._command or not self._checkpoint or not self._config:
            raise RuntimeError("MelBand adapter is not loaded")
        if not input_audio.is_file():
            raise FileNotFoundError(input_audio)
        output_dir.mkdir(parents=True, exist_ok=True)
        args = [
            self._command,
            str(input_audio),
            "--output-dir", str(output_dir),
            "--device", self._device,
            "--config-path", str(self._config),
            "--model-path", str(self._checkpoint),
            "--yes",
        ]
        completed = subprocess.run(
            args, check=False, timeout=self.timeout, text=True,
        )
        if completed.returncode != 0:
            raise RuntimeError(
                f"melband-roformer exited with status {completed.returncode}"
            )
        artifacts = []
        for stem in target_stems:
            output = output_dir / f"{input_audio.stem}_{stem}.wav"
            if output.is_file():
                artifacts.append(StemArtifact(stem, output))
        return artifacts

    def unload(self) -> None:
        # Inference runs in a child process. A successful/failed subprocess.run
        # has already reaped it; clearing these paths releases adapter state.
        self._command = None
        self._checkpoint = None
        self._config = None
        self._model_id = None

    def info(self) -> BackendInfo:
        return BackendInfo(
            name="melband-roformer-cli",
            details={"isolation": "subprocess", "model_id": self._model_id or ""},
        )
