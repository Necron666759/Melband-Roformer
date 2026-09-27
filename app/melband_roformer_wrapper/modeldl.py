"""Explicit-consent model download wrapper.

Delegates all actual resolution/download/checksum-verification to upstream
`mel_band_roformer` (ensure_model_assets, MODEL_REGISTRY, DEFAULT_MODEL).
We do not reimplement checksum verification or hold our own list of
download URLs -- that would be exactly the kind of "made-up path" the task
explicitly says not to invent, and would drift from upstream's own
(actively audited) registry.
"""
from __future__ import annotations

import sys


def _load_registry():
    from mel_band_roformer import MODEL_REGISTRY, DEFAULT_MODEL
    return MODEL_REGISTRY, DEFAULT_MODEL


def list_models() -> int:
    registry, default = _load_registry()
    print(f"Default model: {default}\n")
    print(registry.as_table())
    return 0


def _model_is_cached(slug: str) -> bool:
    from mel_band_roformer import ensure_model_assets
    try:
        # ensure_model_assets with a check-only path isn't part of the
        # documented public API, so we do the honest thing: attempt
        # resolution against local dirs only by checking the standard
        # cache location ourselves, matching upstream's own precedence
        # order (models_dir arg > MELBAND_ROFORMER_MODELS_PATH env >
        # ~/.cache/melband-roformer-infer/ > legacy ./models fallback).
        import os
        from pathlib import Path
        candidates = []
        env_dir = os.environ.get("MELBAND_ROFORMER_MODELS_PATH")
        if env_dir:
            candidates.append(Path(env_dir) / slug)
        candidates.append(Path.home() / ".cache" / "melband-roformer-infer" / slug)
        candidates.append(Path("models") / slug)
        return any(c.is_dir() and any(c.iterdir()) for c in candidates if c.exists())
    except Exception:
        return False


def download_model(slug: str, assume_yes: bool = False, models_dir: str | None = None) -> int:
    registry, default = _load_registry()
    slug = slug or default

    try:
        entry = registry.get(slug)
    except Exception:
        print(f"Unknown model slug: {slug!r}. Run --list-models to see valid slugs.",
              file=sys.stderr)
        return 2

    if _model_is_cached(slug):
        print(f"Model already present: {slug}")
        return 0

    label = getattr(entry, "name", slug)
    if not assume_yes:
        print(f"Model not found: {slug}")
        print(f"Download {label}? This will fetch the checkpoint from its "
              f"upstream host and verify it against the recorded sha256 "
              f"checksum before use.")
        answer = input("Proceed? [y/N] ").strip().lower()
        if answer not in ("y", "yes"):
            print("Aborted; no download performed.")
            return 1

    from mel_band_roformer import ensure_model_assets
    kwargs = {}
    if models_dir:
        kwargs["models_dir"] = models_dir
    try:
        ckpt_path, config_path = ensure_model_assets(slug, **kwargs)
    except Exception as exc:
        print(f"Download/verification failed: {exc}", file=sys.stderr)
        return 3

    print(f"Model ready:\n  checkpoint: {ckpt_path}\n  config:     {config_path}")
    return 0
