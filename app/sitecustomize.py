"""Venv-wide startup patch: silence/eliminate noisy-but-harmless upstream
console output -- currently FutureWarnings from deprecated torch APIs
(`torch.backends.cuda.sdp_kernel()`, `torch.cuda.amp.autocast()`), a benign
UserWarning from an internal torch.stft() shape-probe call, and a short
README pointer appended after upstream's one-line GPU-attention-backend
banner (see README.md, "GPU attention backend", for the user-facing
explanation of what that banner and the suppressed warning mean).

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
import warnings
from functools import wraps


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


def _silence_benign_stft_window_warning() -> None:
    """Silence the UserWarning torch.stft() emits when called without an
    explicit `window=` argument:

        UserWarning: A window was not provided. A rectangular window will
        be applied,which is known to cause spectral leakage. Other windows
        such as torch.hann_window or torch.hamming_window are recommended
        to reduce spectral leakage. To suppress this warning and use a
        rectangular window, explicitly set `window=torch.ones(n_fft,
        device=<device>)`. (Triggered internally at .../SpectralOps.cpp.)

    Confirmed directly against the installed upstream wheel (melband-
    roformer-infer==0.1.5): mel_band_roformer/mel_band_roformer.py, inside
    MelBandRoformer.__init__, has exactly one torch.stft() call with no
    `window=` kwarg --

        freqs = torch.stft(torch.randn(1, 4096), **self.stft_kwargs,
                            return_complex=True).shape[1]

    -- a shape probe against dummy random noise, run once per model
    construction purely to count how many frequency bins the configured
    n_fft/hop_length produce. It is not a spectral transform of real audio:
    every *other* torch.stft()/istft() call in that same file (the ones
    that actually process the input signal) already passes an explicit
    `window=`, so the "spectral leakage" this warning describes cannot
    apply to the call that triggers it here -- there is no real signal
    being windowed, and the resulting `freqs` count does not depend on
    which window function would have been used.

    We do not hand-edit upstream's installed .py file for the same reason
    given in the sdp_kernel/autocast shims above (pinned PyPI dependency,
    not vendored in this repo, would be silently overwritten on the next
    rebuild). Filtering the exact message text (rather than blanket-
    silencing every UserWarning torch can raise) keeps any *other*,
    potentially meaningful torch.stft()/istft() warning visible.
    """
    warnings.filterwarnings(
        "ignore",
        message=r"A window was not provided\. A rectangular window will be applied",
        category=UserWarning,
    )


# Exact text of the two (mutually exclusive, printed-at-most-once-per-
# process) messages upstream's mel_band_roformer/attend.py prints via its
# own `once(print)` wrapper, at Attend.__init__ time, to announce which
# SDPA backend it selected based on GPU compute capability. See README.md,
# "GPU attention backend", for the user-facing explanation this pointer
# refers to.
_ATTEND_GPU_BACKEND_BANNERS = frozenset((
    "A100 GPU detected, using flash attention if input tensor is on cuda",
    "Non-A100 GPU detected, using math or mem efficient attention if "
    "input tensor is on cuda",
))


def _patch_attend_gpu_banner_readme_pointer() -> None:
    """Append a one-line "see README" pointer immediately after upstream's
    GPU-attention-backend banner (one of the two exact strings in
    `_ATTEND_GPU_BACKEND_BANNERS` above), instead of leaving that single
    line to stand alone with nowhere to point a user confused by it.

    This is a plain print() in upstream's mel_band_roformer/attend.py
    (`print_once = once(print)`, called from Attend.__init__), not a
    warnings.warn() -- so, unlike the stft warning above, there is no
    warnings.filterwarnings() lever for it; it needs an actual monkeypatch
    of the function upstream calls. Same "not ours to hand-edit" reasoning
    as the other shims in this file applies (pinned PyPI dependency, not
    vendored here).

    Upstream's `once()` helper (mel_band_roformer/attend.py) makes the
    underlying print() fire at most once per interpreter, via a closure
    flag that isn't observable from outside -- so this shim keeps its own,
    independent "already annotated" flag rather than trying to inspect
    upstream's, to guarantee the pointer line also prints exactly once
    (not once per Attend() construction) regardless of how many times
    this process constructs a model.
    """
    try:
        from mel_band_roformer import attend as _attend
    except ImportError:
        # mel_band_roformer isn't on the import path in this interpreter
        # yet/at all (e.g. build-time pip bootstrap, or a process that
        # never imports the model). Nothing to patch.
        return

    original_print_once = getattr(_attend, "print_once", None)
    if original_print_once is None:
        return  # unexpected/future upstream with a different API shape

    if getattr(original_print_once, "_melband_roformer_shim", False):
        return  # already patched (defensive, in case of a double import)

    already_annotated = False

    @wraps(original_print_once)
    def _print_once_shim(message):
        nonlocal already_annotated
        result = original_print_once(message)
        if not already_annotated and message in _ATTEND_GPU_BACKEND_BANNERS:
            already_annotated = True
            print('See README.md ("GPU attention backend") for what this means.')
        return result

    _print_once_shim._melband_roformer_shim = True
    _attend.print_once = _print_once_shim


_patch_sdp_kernel()
_patch_cuda_amp_autocast()
_silence_benign_stft_window_warning()
_patch_attend_gpu_banner_readme_pointer()
