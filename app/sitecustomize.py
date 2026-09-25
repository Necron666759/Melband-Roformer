"""Venv-wide startup patch: silence/eliminate upstream FutureWarnings from
deprecated torch APIs -- currently `torch.backends.cuda.sdp_kernel()` and
`torch.cuda.amp.autocast()`.

WHY THIS FILE LIVES HERE (top-level module, not inside
melband_roformer_wrapper/):
Python's `site` module automatically does ``import sitecustomize`` once,
near the end of interpreter start-up, for *every* process that runs with
this venv's interpreter -- unconditionally, before any application code
executes. Because this venv is built with --system-site-packages (see
debian/rules), its own site-packages directory is searched, and takes
priority, ahead of the system one for anything installed in both (same
comment in debian/rules: "venv site-packages always takes priority for
same-named packages"), so a `sitecustomize.py` shipped as a top-level
module of *our own* package (via `[tool.setuptools] py-modules` in
app/pyproject.toml) ends up on that search path and gets picked up first.

That matters because the actual separation work is NOT done in-process by
melband_roformer_wrapper: cli.py's `_run_separation()` execs the real
upstream console-script, `melband-roformer-infer` (from the
`melband-roformer-infer` PyPI package, see app/requirements.txt), as a
*separate* subprocess/interpreter -- selftest.py does the same for its
inference check. A monkeypatch placed only inside
melband_roformer_wrapper/__init__.py would apply only to our own wrapper
process and would never reach that subprocess, since it is a fresh
interpreter importing upstream's own code, not ours. A `sitecustomize.py`
is the one hook that runs automatically in *every* interpreter start in
this venv -- our CLI, our GUI, the `melband-roformer-infer` /
`melband-roformer-download` console-scripts upstream ships, and even a
user invoking `.../venv/bin/python3` directly -- without upstream needing
to import anything from us.

WHAT IS BEING PATCHED, AND WHY (#1: torch.backends.cuda.sdp_kernel):
Upstream's attention code still opens its backend selection with the
now-deprecated call

    with torch.backends.cuda.sdp_kernel(enable_flash=..., enable_math=...,
                                         enable_mem_efficient=...):
        ...

which, on the PyTorch version this package pins (see
app/requirements.txt), prints on every such call:

    FutureWarning: `torch.backends.cuda.sdp_kernel()` is deprecated. In
    the future, this context manager will be removed. Please see
    `torch.nn.attention.sdpa_kernel()` for the new context manager, with
    updated signature.

We do not hand-edit upstream's installed .py file directly under
/usr/lib/melband-roformer/venv: that file is reproduced verbatim every
time this package is rebuilt against upstream (`pip install
melband-roformer-infer==...` in debian/rules), so an edit made straight
to the installed copy would silently disappear on the next rebuild, and
it is not something we can carry as a source-tree diff since upstream's
source is not vendored in this repository (it's a pinned PyPI dependency,
see app/requirements.txt / app/pyproject.toml) -- there is nothing under
version control here to patch.

Instead, we replace `torch.backends.cuda.sdp_kernel` itself, at
interpreter start-up, with a drop-in shim that has the exact same call
signature upstream already uses, but is implemented purely in terms of
the new, non-deprecated `torch.nn.attention.sdpa_kernel`. Upstream's own
`with torch.backends.cuda.sdp_kernel(...):` line therefore keeps working
completely unmodified -- it just no longer touches the deprecated code
path internally, so the warning does not merely get suppressed, it no
longer fires at all.

WHAT IS BEING PATCHED, AND WHY (#2: torch.cuda.amp.autocast):
Separately, upstream's `mel_band_roformer/utils.py` opens its forward
pass with the also-deprecated

    with torch.cuda.amp.autocast():
        ...

which prints, on every call:

    FutureWarning: `torch.cuda.amp.autocast(args...)` is deprecated.
    Please use `torch.amp.autocast('cuda', args...)` instead.

Same reasoning as #1 applies (upstream file is not ours to hand-edit, and
would be overwritten on the next rebuild against PyPI anyway), so we
replace `torch.cuda.amp.autocast` itself with a drop-in, same-signature
shim built on the new, non-deprecated `torch.amp.autocast('cuda', ...)`.
Upstream's `with torch.cuda.amp.autocast():` line keeps working
unmodified; it just no longer touches the deprecated code path.
"""
from __future__ import annotations

import contextlib


def _patch_sdp_kernel() -> None:
    try:
        import torch
    except ImportError:
        # torch isn't installed yet -- e.g. this venv is still mid-assembly
        # inside `pip install -r requirements.txt` at package *build* time
        # (see debian/rules' override_dh_auto_build, which runs `pip
        # install --upgrade pip wheel` in this same venv before torch is
        # present). Nothing to patch yet, and nothing downstream needs it
        # yet either.
        return

    cuda_backend = getattr(torch.backends, "cuda", None)
    if cuda_backend is None or not hasattr(cuda_backend, "sdp_kernel"):
        return  # unexpected/future torch build with a different API shape

    if getattr(cuda_backend.sdp_kernel, "_melband_roformer_shim", False):
        return  # already patched (defensive, in case of a double import)

    try:
        from torch.nn.attention import SDPBackend, sdpa_kernel
    except ImportError:
        # Older torch without the replacement API: sdp_kernel() isn't
        # deprecated there in the first place, so leave it as-is.
        return

    @contextlib.contextmanager
    def _sdp_kernel_shim(
        enable_flash: bool = True,
        enable_math: bool = True,
        enable_mem_efficient: bool = True,
        enable_cudnn: bool = True,
        **_ignored_kwargs,
    ):
        """Same-signature replacement for the deprecated
        torch.backends.cuda.sdp_kernel(), re-implemented on top of
        torch.nn.attention.sdpa_kernel() so upstream's existing call site
        keeps working, minus the FutureWarning."""
        backends = []
        if enable_flash:
            backends.append(SDPBackend.FLASH_ATTENTION)
        if enable_mem_efficient:
            backends.append(SDPBackend.EFFICIENT_ATTENTION)
        if enable_cudnn and hasattr(SDPBackend, "CUDNN_ATTENTION"):
            backends.append(SDPBackend.CUDNN_ATTENTION)
        if enable_math:
            backends.append(SDPBackend.MATH)
        if not backends:
            # sdpa_kernel() requires at least one backend; the old
            # sdp_kernel() let PyTorch fall back to its own default in
            # this case -- MATH is the always-available equivalent.
            backends = [SDPBackend.MATH]
        with sdpa_kernel(backends):
            yield

    _sdp_kernel_shim._melband_roformer_shim = True
    cuda_backend.sdp_kernel = _sdp_kernel_shim


def _patch_cuda_amp_autocast() -> None:
    try:
        import torch
    except ImportError:
        # Same build-time-only window as _patch_sdp_kernel() above: this
        # venv can be mid-assembly (pip installing its own dependencies)
        # before torch itself is present yet.
        return

    cuda_ns = getattr(torch, "cuda", None)
    amp_ns = getattr(cuda_ns, "amp", None)
    if amp_ns is None or not hasattr(amp_ns, "autocast"):
        return  # unexpected/future torch build with a different API shape

    if getattr(amp_ns.autocast, "_melband_roformer_shim", False):
        return  # already patched (defensive, in case of a double import)

    try:
        from torch.amp import autocast as _new_autocast
    except ImportError:
        # Older torch without the replacement API: torch.cuda.amp.autocast
        # isn't deprecated there in the first place, so leave it as-is.
        return

    class _CudaAmpAutocastShim:
        """Same-signature replacement for the deprecated
        torch.cuda.amp.autocast(), re-implemented on top of the new
        torch.amp.autocast('cuda', ...) so upstream's existing call site
        (``with torch.cuda.amp.autocast():`` etc., with or without
        explicit enabled=/dtype=/cache_enabled=) keeps working, minus the
        FutureWarning. Usable both as a context manager and, like the
        original, as a function decorator -- torch.amp.autocast already
        supports both, so we just delegate.
        """

        _melband_roformer_shim = True

        def __init__(
            self,
            enabled: bool = True,
            dtype=None,
            cache_enabled: bool = True,
            **_ignored_kwargs,
        ):
            # torch.cuda.amp.autocast's own default dtype is float16;
            # torch.amp.autocast(device_type=...) needs it made explicit
            # rather than left as None, to reproduce that same default
            # instead of picking a possibly different one for "cuda".
            if dtype is None:
                dtype = torch.float16
            self._ctx = _new_autocast(
                "cuda", enabled=enabled, dtype=dtype, cache_enabled=cache_enabled,
            )

        def __enter__(self):
            return self._ctx.__enter__()

        def __exit__(self, *exc_info):
            return self._ctx.__exit__(*exc_info)

        def __call__(self, func):
            # Decorator usage: @torch.cuda.amp.autocast()
            return self._ctx(func)

    amp_ns.autocast = _CudaAmpAutocastShim


_patch_sdp_kernel()
_patch_cuda_amp_autocast()
