# AppImage builder notes

Builds `melband-roformer-<version>-x86_64.AppImage`: a single-purpose,
self-contained AppImage whose **only** entry point is a restricted
terminal that runs melband-roformer commands and nothing else. This is
not "an AppImage that happens to have a terminal in it" -- the terminal
*is* the application, and it cannot run anything outside melband-roformer.

```bash
./scripts/build-appimage.sh
```

Output: `./out/melband-roformer-<version>-x86_64.AppImage`. Run it like
any AppImage (`chmod +x` then double-click, or
`--appimage-extract-and-run` in a container/CI without FUSE).

## What happens when you run it

- **Launched from a file manager / app launcher (no controlling
  terminal):** this AppImage's own bundled terminal opens -- a small
  `tkinter` + `pty` + `pyte` terminal emulator
  (`appimage/melband-roformer-hud-term.py`) styled after the
  Terminator films' cyborg "HUD" vision (monochrome phosphor red on
  near-black, large type) -- running nothing but
  `melband-roformer-shell.py`. Closing that one program (`exit`, `quit`,
  Ctrl-D, or closing the window) closes the AppImage. No other program
  ever attaches to that terminal, and no general-purpose terminal
  emulator (xterm or otherwise) is bundled or required any more.
- **Launched from an existing terminal** (`./melband-roformer.AppImage`
  from a shell you already have open): skips opening a second, redundant
  terminal window and runs the restricted shell directly in the one you're
  already in.

Either way, the thing you end up talking to is the same restricted REPL,
never a real shell.

### The bundled HUD terminal

`appimage/melband-roformer-hud-term.py` is this AppImage's entire
windowed-terminal UI. It:

- opens a pseudo-terminal (`pty`) and execs a fixed, hardcoded argv into
  it (the same one `xterm -e` used to receive) -- never a shell, never a
  user-supplied command line;
- feeds the bytes read back from that pty through
  [`pyte`](https://pypi.org/project/pyte/), a pure-Python VT100/xterm
  terminal-state emulator, so escape sequences (cursor moves, colors,
  and -- importantly -- carriage-return line overwrites, the mechanism
  progress bars like `tqdm` use to animate a single line in place) are
  interpreted correctly rather than dumped as literal text -- except for
  *window resizing*, which goes through this program's own
  `_resize_screen()` instead of pyte 0.8.2's own `Screen.resize()`,
  working around a couple of real bugs in the latter (buffer content
  going stale, and the cursor drifting away from the text it was next
  to) that were easy to trigger just by dragging this window smaller
  after having made it bigger -- see that function's docstring for the
  full explanation;
- composites that screen buffer itself, once per frame, into a plain RGB
  image -- glyphs rasterized by [`Pillow`](https://pypi.org/project/pillow/)
  (using its own statically-linked FreeType, not Tk's font engine) from
  the DejaVu Sans Mono TrueType font bundled at
  `usr/share/melband-roformer/fonts/` -- and hands that one image to a
  plain `tkinter.Canvas` to display. Monospace, clamped to a minimum of
  14pt (default 16pt, configurable via `$MELBAND_ROFORMER_HUD_FONT_SIZE`,
  but never smaller);
- supports the host clipboard directly through `tkinter`'s own
  `clipboard_get`/`clipboard_append`, plus its own cell-based mouse-drag
  selection (Ctrl+Shift+C to copy the current selection, Ctrl+Shift+V or
  Shift+Insert to paste, middle-click to paste the X11 `PRIMARY`
  selection, or the right-click context menu) -- no external clipboard
  tool (`xclip`, `xsel`, ...) is shelled out to or bundled. The
  Ctrl+Shift+C/V chords are matched by physical key position
  (`event.keycode`), not by the symbol the active keyboard layout maps
  that key to, so they keep working under non-Latin layouts (e.g.
  Cyrillic) where the C/V keys report different keysyms.

`tkinter` ships as part of the bundled Python interpreter (see
"Portability" below) and needs no extra shared libraries bundled for it
on Linux: the interpreter's `_tkinter`/Tcl/Tk are self-contained (no
`libX11`/`libxcb`/`libXft` dependency -- verified with `ldd`/`objdump -p`
against the fetched interpreter). One consequence of there being no
`libXft`/fontconfig linkage: **Tk itself** can only ever resolve to the
host X server's own core bitmap fonts (typically `fixed`, `courier`,
`helvetica`, `times`) -- non-antialiased, and only at a handful of fixed
design sizes, which is why earlier builds of this HUD terminal (which
drew text with a plain `tkinter.Text` widget) looked visibly aliased/
blocky at the larger sizes this HUD uses. The terminal grid no longer
asks Tk to draw text at all, specifically to route around that: Pillow
(a real runtime dependency now, see `app/requirements.txt`-adjacent
install step in `scripts/build-appimage.sh`) rasterizes every glyph
itself and only the resulting bitmap image is ever handed to Tk. This
needs no new shared-library dependency -- Pillow's official wheels
statically link their own FreeType/zlib -- and keeps working regardless
of what fonts, if any, are installed on the host. The header/footer HUD
status lines, and the right-click context menu, are rendered the same
way -- including the menu, which is *not* a native `tkinter.Menu` (that
widget draws its own item labels internally, with no way to hand it a
pre-rasterized one instead). It is `PillowMenu` in
`melband-roformer-hud-term.py`: a borderless `Toplevel` holding a single
`Canvas` that displays one composited image of the whole menu, redrawn on
hover for the highlight. Tk does not draw a single glyph anywhere in this
program any more.

## Why a restricted shell, and how it's enforced

`appimage/melband-roformer-shell.py` is a `cmd.Cmd` subclass with an
explicit whitelist of subcommands (`separate`, `list-models`,
`download-model`, `info`, `self-test`, `version`, `clear`, `help`,
`exit`/`quit`), one per melband-roformer CLI flag documented in the main
`README.md` -- with the single exception of `clear`, which has no CLI
flag counterpart: it never reaches `melband_roformer_wrapper.cli.main()`
at all, and instead writes a raw ANSI clear-screen escape sequence
directly to stdout (the same kind of cursor-control byte this terminal's
`pyte` screen already interprets for `progress.py`'s own bar). Each
accepted line is:

1. Tokenized with `shlex.split` -- which only splits quoted words. It does
   **not** interpret `;`, `&&`, `|`, backticks, `$()`, or shell globbing;
   those are just literal characters to it, same as any other.
2. Mapped to a fixed, hardcoded argv prefix (e.g. `list-models` always
   becomes exactly `["--list-models", *your extra args]`) and passed
   in-process to `melband_roformer_wrapper.cli.main()` -- the same
   function the `.deb`'s `melband-roformer` command calls. No shell is
   spawned to run it.
3. Anything not matching a whitelisted subcommand -- including
   `cmd.Cmd`'s own built-in `!`-prefixed shell-escape, which is explicitly
   overridden to print a refusal instead of running anything -- is
   rejected by `default()` with a message, never executed.

The only subprocess this ever starts, transitively, is
`melband-roformer-infer` inside this AppImage's own bundled venv (see
`cli.py`'s `_run_separation`), invoked with an argv list built from parsed
flags -- never a raw string. There is no `cd`, `ls`, or other filesystem
browsing command: this terminal exists exclusively for melband-roformer,
per the task's own requirement, not as a general restricted shell that
happens to default-deny.

This was manually tested against injection attempts during development
(`; rm -rf /`, `$(reboot)`, `!ls -la /`, `cd /`, bare `ls`) -- all
rejected without executing anything; legitimate commands (`version`,
`list-models`, `separate --help`) dispatch correctly. If you change
`melband-roformer-shell.py`, re-run that check by hand; there is
currently no automated test for it (a good thing to add if this shell
grows more commands).

**Prompt note:** `cmd.Cmd`'s prompt is written through the builtin
`input()`, which -- when the `readline` module is loaded, as it is here
-- echoes the prompt through its own line-editing logic rather than
writing it to the terminal as-is. Readline needs any non-printing bytes
in a prompt (here: the ANSI color codes around `[melband-roformer]>`)
wrapped in its `\001`/`\002` zero-width markers, or it corrupts them --
concretely, it drops the leading ESC byte of each escape sequence,
leaving the rest of the sequence (e.g. `[1m`) behind as literal, visible
text in front of the prompt. This was hit and fixed during this
terminal's development (`_rl_invisible()` in
`melband-roformer-shell.py`); it affects the prompt in *any* real
terminal this shell runs in, not just the bundled HUD one.

## Why the venv is built differently than the `.deb`'s

`debian/rules` builds with `--system-site-packages` and then rewrites
every venv shebang to the package's one fixed final install path
(`/usr/lib/melband-roformer/venv/...`) because a `.deb` always lands at
that exact path. An AppImage has no fixed final path -- it's mounted (or
extracted) at a different, randomly-chosen location every time it runs --
so `build-appimage.sh` does things differently:

- Builds a fully isolated interpreter + venv (no `--system-site-packages`
  and no dependency on the build host's own Python at all): the AppImage
  should never depend on anything from the host's Python installation,
  which is the entire point of the format. There's no optional GTK4 GUI
  in this image to want to see the host's `python3-gi` for.
- After `pip install`, rewrites every script's shebang from the build-time
  path to `#!/usr/bin/env python3`, and `AppRun` puts this AppImage's own
  `venv/bin` first on `$PATH`. That makes `env python3` resolve to the
  bundled interpreter correctly regardless of where the AppImage happens
  to be mounted at runtime -- no baked-in absolute path anywhere. (`pip`'s
  own `pip3`/`pip3.12` scripts don't need this: they already re-locate
  themselves relative to their own path at startup, which is why their
  shebang is `#!/bin/sh`, not a `python3` path, to begin with.)
- A consequence of the fully isolated venv: it also can't borrow a
  host-installed `python3-pytest` the way `debian/rules`' venv does via
  `--system-site-packages`, and `pytest` is deliberately not in
  `app/requirements.txt` either (it must not ship inside the AppImage).
  So `build-appimage.sh` installs `pytest` into the venv for the
  build-time test run only, then removes it again -- along with exactly
  whatever it pulled in as its own dependencies (diffed via `pip freeze`
  before/after, not a hardcoded package list) -- before the shebang
  rewrite and packaging steps, so none of it ends up in the shipped
  image. `pyte` (see "The bundled HUD terminal" above) is a genuine
  runtime dependency of the terminal UI, not a build/test-only tool, so
  it is installed earlier and is unaffected by that cleanup.

## Portability: build on any current Debian/Ubuntu, not just the oldest one

An earlier version of this builder used the build host's own `python3` to
create the venv (`python3 -m venv --copies`), which meant the AppImage
inherited *that machine's* glibc floor (glibc is backward- but not
forward-compatible: a binary built against an older glibc runs on a
newer one, never the reverse). That would have meant building on your
*oldest* supported target, same as the general AppImage guidance for
compiled C/C++ apps.

This builder avoids that instead of just documenting around it: the
base interpreter is a prebuilt, relocatable CPython from
[astral-sh/python-build-standalone](https://github.com/astral-sh/python-build-standalone),
fetched at build time, not the build host's own `python3`. Upstream
documents these `*-unknown-linux-gnu` builds as referencing glibc symbols
no newer than **2.17** (released 2014; compatible with Debian 8+/Ubuntu
14.04+) regardless of what machine builds with them.

This was verified directly while building this script, not assumed from
the docs:

```bash
$ objdump -T ./portable-python/python/bin/python3.13 | grep -oP 'GLIBC_\K[0-9.]+' | sort -V | uniq | tail
2.13
2.14
2.15
2.16
2.17
```

...run on this repo's actual fetched interpreter (`cpython-3.13.15+20260825`,
on an Ubuntu 24.04 host, glibc 2.39): the bundled `python3` only references
glibc up to 2.17 -- unchanged from the floor verified against the previous
pinned version (3.12.14). **Practical effect: build this AppImage on Debian
13, Ubuntu 24.04, or any other current release, and it carries the same
glibc floor (2.17) either way** -- there is no "build on the oldest target"
requirement to follow here.

This check should be re-run (same one-liner, just against
`portable-python/python/bin/python3.<X>`) any time `PBS_ASSET` in
`scripts/build-appimage.sh` is bumped to a different CPython version or
release tag -- the glibc floor is a property of that specific build, not
a permanent guarantee of the project as a whole.

What this does *not* cover: the pip-installed compiled extensions
(PyTorch, numpy, scipy) are pre-built wheels pulled unmodified from
PyPI/the PyTorch index, built by their own maintainers to the
`manylinux` standard (itself an old, portable glibc floor by design) --
they were never tied to your build host's glibc in the first place, with
or without this fix. The one thing that *was* tied to the build host was
the interpreter binary itself, and that's what's fixed above.

If you fork this script to build for a musl-based or otherwise unusual
target, `python-build-standalone` also publishes fully static
`*-unknown-linux-musl` builds with no glibc dependency at all -- see
their release notes if you need that instead.

### Interpreter placement: a copy of the whole tree, not a `venv`

`build-appimage.sh` does **not** run `python3 -m venv --copies` to build
the thing that ships inside the AppImage. A `venv` only copies the
interpreter *binary*; the standard library it needs at startup (`Lib/`,
`lib-dynload/`, and -- relevant here -- `tkinter`'s own Tcl/Tk script
libraries) stays referenced from the *base* interpreter's original
location, recorded as an absolute path in the venv's `pyvenv.cfg`. That
location, for this build, is this script's own ephemeral
`.appimage-build/portable-python/` directory -- which does not exist on
an end user's machine. A `venv --copies`-based AppImage would therefore
fail to even start Python on any machine other than the one it was built
on (`ModuleNotFoundError: encodings` or similar, immediately on launch).
This was reproduced directly: moving the base interpreter out from under
such a venv and trying to run it fails exactly that way.

python-build-standalone's interpreters are, true to their name, actually
relocatable: they locate their own standard library relative to their
*own binary's* path, not via any baked-in absolute path or `pyvenv.cfg`.
So this script instead copies the entire fetched interpreter tree
(`bin/`, `lib/`, `include/`, `share/` -- which is also how tkinter's Tcl/Tk
script libraries travel along automatically, with no separate copying
step needed for them) directly into place under `$APPDIR`, and installs
packages into that copy with its own bundled `pip`. This was also
verified directly: copying the tree to an arbitrary new path and
importing `os`/`tkinter` out of the copy, with the original extraction
directory removed first, both succeed unmodified.

## Size

Same tradeoff as the `.deb` (see `README.debian.md`): the bundled CUDA
PyTorch wheel dominates the size, so the resulting `.AppImage` is on the
same order (multi-GB) as the `.deb`. See the GitHub-release discussion in
this repo's history for why that argues for distributing build
instructions rather than the built artifact itself.

## Build progress / what's downloading right now

`./scripts/build-appimage.sh` prints an overall `[n/5]` step counter and,
for the long silent stretches that used to give zero feedback:

- **curl** (fetching `appimagetool` and the portable CPython archive): no
  longer run with `-s` (silent). curl's own transfer meter shows live, in
  place -- current file, percentage, transfer size, and actual download
  speed. That flag was hiding exactly the information that was missing;
  the fix is to stop hiding it, not to reimplement it.
- **pip** (installing `app/requirements.txt`, dominated by the multi-GB
  CUDA PyTorch wheels): no longer run with `-q` (quiet), for the same
  reason -- pip's own per-package `Downloading ... |████| N.N MB N.N MB/s`
  bars show live instead.
- **copying the portable interpreter into place**, the one step with
  genuinely no output of its own, gets a small animated spinner with an
  elapsed-time counter (`⠋ copying portable interpreter into place
  (00:07)`) instead of sitting silent for several seconds.

`appimagetool` was never silenced (its own output already streamed
live); it just gets a step header now too, so it's clear which of the
five stages is currently running.

All of this degrades to plain, static lines with no carriage-return
animation when stdout isn't a terminal (piped to a log file, run under
CI), so redirected build logs stay clean and grep-able.

## Build-tool provenance

`scripts/build-appimage.sh` downloads `appimagetool` from its upstream
GitHub Releases (`AppImage/AppImageKit`, `continuous` tag), and the
portable CPython interpreter from `astral-sh/python-build-standalone`
(pinned release tag, not "latest" -- see the portability section above),
into `.appimage-build/tools/` and `.appimage-build/portable-python/`,
running the AppImage-packaged tool with `--appimage-extract-and-run` so
the build also works in containers/CI without `/dev/fuse` available. Pin
`appimagetool` to a specific release tag instead of `continuous` too if
you want fully reproducible builds.

`pyte` (the terminal-state emulation library behind the bundled HUD
terminal, see "The bundled HUD terminal" above) is pinned to an exact
version (`pyte==0.8.2`) in `scripts/build-appimage.sh`, the same way
`app/requirements.txt` pins everything else that ships inside the image.

This AppImage no longer bundles or depends on `xterm`, or on
`linuxdeploy` (previously used only to bundle `xterm`'s own X11/Xft
shared libraries) -- the bundled terminal's `tkinter`/Tcl/Tk have no such
external shared-library dependencies to bundle in the first place (see
"The bundled HUD terminal" above).
