"""Debian-packaging wrapper around the upstream `melband-roformer-infer` project.

This package does not implement Mel-Band RoFormer itself. It provides:
  * a friendlier CLI (`melband-roformer`) that maps to upstream's real
    `melband-roformer-infer` / `melband-roformer-download` entry points and
    the real `mel_band_roformer` Python API,
  * GPU/CUDA detection with clear, non-crashing fallback,
  * an explicit-consent model download flow with sha256 verification
    (delegated to upstream's own checksum registry, not reimplemented here),
  * `--info` / `--self-test` diagnostics,
  * an optional thin GTK4 GUI.
"""

__version__ = "0.1.0"


def subprocess_env(base_env: "dict[str, str] | None" = None) -> "dict[str, str]":
    """Build an environment dict for subprocess calls that exec the real
    upstream `melband-roformer-infer` console-script (cli.py's
    `_run_separation`, selftest.py's inference check).

    Normally this venv's own sitecustomize.py (see that file's docstring)
    is found automatically, because a venv's own site-packages precedes
    the system's on sys.path. But `sitecustomize` resolution is a plain
    top-level import against whatever `sys.path` happens to contain on
    the target machine, and *some* systems place a sitecustomize.py
    somewhere unusual enough to win that race regardless (observed in
    practice: this exact ambiguity broke this venv's own build-time test
    suite on at least one real machine). Explicitly prepending this
    venv's own sitecustomize.py directory onto PYTHONPATH removes that
    ambiguity: PYTHONPATH entries are searched before a venv's
    installation-default paths, so our copy is found first no matter
    what else is on the host.

    Deliberately does NOT locate that directory via `import
    sitecustomize` -- that ambient resolution is exactly the mechanism
    this function exists to route around, so using it here would just
    relocate the same bug: on a host where ambient resolution finds
    someone else's sitecustomize.py, `import sitecustomize` in *this*
    process would return that wrong file too, and we'd faithfully
    compute the wrong directory to prepend. Instead this is derived
    purely from this package's own on-disk location: sitecustomize.py
    and the melband_roformer_wrapper package are installed by the same
    wheel into the same site-packages directory (see app/pyproject.toml's
    `py-modules`), so sitecustomize.py always lives exactly one directory
    above this file's own package directory, regardless of sys.path.
    """
    import os

    env = dict(base_env if base_env is not None else os.environ)

    package_dir = os.path.dirname(os.path.abspath(__file__))
    site_dir = os.path.dirname(package_dir)
    if not os.path.isfile(os.path.join(site_dir, "sitecustomize.py")):
        return env  # unexpected layout (e.g. an editable/dev checkout); nothing to add

    existing = env.get("PYTHONPATH", "")
    env["PYTHONPATH"] = os.pathsep.join([site_dir, existing] if existing else [site_dir])
    return env
