"""Terminal progress rendering for the wrapped `melband-roformer-infer` subprocess.

Upstream (`mel_band_roformer/inference.py`, pinned melband-roformer-infer==0.1.5)
prints two separate, uncoordinated progress signals to stdout:

  1. A `tqdm(paths, desc="Tracks", unit="track")` bar wrapping the *outer*
     for-loop over input files. tqdm only advances this bar once per
     completed *file*, so with a single input file it is emitted at 0%
     when the loop starts and jumps straight to 100% when the whole run
     is done -- it never animates and carries no information about how
     far into that one file processing actually is.
  2. A `\r`-overwritten "Estimated time remaining: X seconds" text line,
     recomputed per audio chunk inside `demix_track`, which *does* update
     continuously but is plain text with no bar and no notion of the
     batch as a whole.

Neither signal is weighted by how much audio there actually is to
process, and the two are never combined into one meaningful indicator.

This module runs that subprocess behind a pty (so upstream's tqdm still
believes it has a real terminal and keeps drawing full-width bars), parses
its plain `\r`/`\n`-delimited output (upstream never emits ANSI escapes,
so a full terminal emulator is not needed here), and replaces both of the
above with a single bar:

  * sized by total audio duration across all input files (seconds), not
    track count, so multi-file batches reflect real proportional progress;
  * filled in real time from the same "Estimated time remaining" figures
    upstream already computes, so it animates smoothly during a single
    track instead of jumping straight from 0% to 100%;
  * finished in the shortened form the terminal UI should show:
    ``Tracks: 100%|<bar>| 1/1 [24.20s/track]`` -- no ``N/N [elapsed<eta, rate]``
    clutter.

Every other line the subprocess prints (device/CUDA banner, "Processing
track i/N: ...", warnings, tracebacks, the final "Elapsed time: ..." line)
is passed through to our own real stdout unchanged.
"""

from __future__ import annotations

import codecs
import fcntl
import os
import pty
import re
import shutil
import struct
import subprocess
import sys
import termios
import time
from pathlib import Path

from tqdm import tqdm as _tqdm

_TRACKS_BAR_RE = re.compile(r"^Tracks:\s+\d+%\|.*\|\s*\d+/\d+\s*\[.*\]$")
_PROCESSING_TRACK_RE = re.compile(r"^Processing track (\d+)/(\d+): (.+)$")
_ESTIMATED_TOTAL_RE = re.compile(
    r"^Estimated total processing time for this track: ([\d.]+) seconds$"
)
_ESTIMATED_REMAINING_RE = re.compile(
    r"^Estimated time remaining: ([\d.]+) seconds$"
)


def track_durations(paths: "list[Path]") -> "dict[str, float]":
    """Best-effort audio duration (seconds) per input filename.

    Keyed by filename (not full path): upstream re-globs its own temp
    input folder and iterates it in sorted-by-name order, so the
    "Processing track i/N: <name>" lines it prints are matched back to
    a duration by name, not by our original argv order/paths.
    """
    import soundfile as sf

    durations: "dict[str, float]" = {}
    for p in paths:
        try:
            info = sf.info(p)
            if info.samplerate:
                durations[p.name] = info.frames / float(info.samplerate)
        except Exception:
            # Unreadable/non-audio input: fall back to equal weighting
            # for this file rather than aborting the whole progress bar.
            continue
    return durations


def _terminal_width(fallback: int = 100) -> int:
    try:
        return shutil.get_terminal_size(fallback=(fallback, 24)).columns
    except Exception:
        return fallback


def _render(n: float, total: float, elapsed: float, completed: int, total_units: int,
            track_seconds: float) -> str:
    """One frame of the combined bar, via tqdm's own bar-drawing code so it
    looks identical to a native tqdm bar (same block characters, same
    ncols behavior) -- only the surrounding text differs from upstream's.

    `track_seconds` is the real, wall-clock stopwatch reading for
    whichever single track is currently being timed (see `track_start_time`
    in `run_with_track_progress`) -- NOT an average pace across every
    track completed so far. It starts at (approximately) 0 the moment
    that track's "Processing track ..." line is printed and counts up in
    real time for the *entire* track: through its model-inference chunk
    loop AND whatever silent file I/O (reading the input, writing the
    output stem(s)) upstream does around it, all the way until this
    track's row is finalized -- either the next "Processing track ..."
    line, or the final "Elapsed time: ..." line for the last track (see
    the `draw()` call added just before `complete_current_track()` in
    both of those branches of `handle_line`). So it reflects this
    track's true share of the run, not just the part upstream happens to
    print ETA updates for.
    """
    bar_format = "{l_bar}{bar}| " + f"{completed}/{total_units} [{track_seconds:.2f}s/track]"
    return _tqdm.format_meter(
        n=max(0.0, min(n, total)),
        total=total,
        elapsed=elapsed,
        ncols=_terminal_width(),
        prefix="Tracks",
        unit="s",
        bar_format=bar_format,
    )


def run_with_track_progress(cmd: "list[str]", env: "dict[str, str]",
                             inputs: "list[Path]") -> int:
    """Run `cmd` (the real `melband-roformer-infer` invocation) behind a
    pty, replacing its track-count/ETA-text output with one duration-
    weighted progress bar. Returns the subprocess's exit code.
    """
    durations = track_durations(inputs)
    total_tracks = len(inputs)
    fallback_duration = (
        sum(durations.values()) / len(durations) if durations else 1.0
    )
    total_duration = sum(durations.get(p.name, fallback_duration) for p in inputs) or 1.0

    master_fd, slave_fd = pty.openpty()
    try:
        cols = _terminal_width()
        try:
            fcntl.ioctl(slave_fd, termios.TIOCSWINSZ,
                        struct.pack("HHHH", 24, cols, 0, 0))
        except OSError:
            pass

        proc = subprocess.Popen(
            cmd, env=env, stdout=slave_fd, stderr=slave_fd, close_fds=True,
        )
        os.close(slave_fd)
        slave_fd = -1

        start_time = time.time()
        # Real, wall-clock stopwatch start for whichever track is
        # currently being processed. Reset to `time.time()` every time a
        # new "Processing track ..." line arrives (see handle_line
        # below), so the "[X.XXs/track]" figure always counts up from
        # (approximately) 0 for that one file, instead of being an
        # average pace blended across every track completed so far.
        track_start_time = start_time
        completed_tracks = 0
        current_duration = fallback_duration
        current_estimated_total: "float | None" = None
        current_fraction = 0.0
        processed_duration = 0.0
        bar_is_live = False
        # True from the "Processing track ..." line until that track's
        # completion blank line. Needed because upstream's own
        # `print(f"\nProcessing track {n}/{total}: ...")` call *also*
        # emits a blank line (its leading "\n") that is otherwise
        # indistinguishable from the blank line that marks a track as
        # finished -- only tracking "are we mid-track right now" tells
        # them apart.
        in_track = False

        def draw(final: bool = False) -> None:
            nonlocal bar_is_live
            n = processed_duration + current_duration * current_fraction
            elapsed = time.time() - start_time
            track_seconds = time.time() - track_start_time
            line = _render(n, total_duration, elapsed, completed_tracks if not final
                            else total_tracks, total_tracks, track_seconds)
            sys.stdout.write("\r" + line + ("\n" if final else ""))
            sys.stdout.flush()
            bar_is_live = not final

        def complete_current_track() -> None:
            nonlocal completed_tracks, processed_duration, current_fraction
            nonlocal current_estimated_total, in_track
            if not in_track:
                return
            processed_duration += current_duration
            completed_tracks = min(completed_tracks + 1, total_tracks)
            current_fraction = 0.0
            current_estimated_total = None
            in_track = False

        def handle_line(line: str) -> None:
            nonlocal completed_tracks, current_duration, current_estimated_total
            nonlocal current_fraction, processed_duration, bar_is_live, in_track
            nonlocal track_start_time

            if not line:
                # A blank line on its own is not a reliable completion
                # signal here: the pty's line discipline turns every
                # plain `print()` newline into "\r\n", but a lone,
                # content-free "\r" immediately following an already-
                # committed line (e.g. right before upstream's own
                # "\rEstimated time remaining: ..." overwrites start)
                # produces the exact same empty string. Track completion
                # is instead detected structurally below: when the next
                # track's "Processing track ..." line appears, or when
                # the run's final "Elapsed time: ..." line appears.
                return

            if _TRACKS_BAR_RE.match(line):
                return  # upstream's own per-track-count bar: superseded, drop it

            m = _PROCESSING_TRACK_RE.match(line)
            if m:
                if in_track:
                    # Refresh the just-finished track's row with its true,
                    # full wall-clock reading *before* scrolling it off
                    # (the "\n" below): upstream prints nothing at all
                    # while it writes that track's output .wav file(s) to
                    # disk (and re-reads the input file to compute the
                    # instrumental) after the chunk loop's last ETA
                    # update, so without this the row left behind on
                    # screen would understate that track's real duration
                    # by exactly that silent gap.
                    draw()
                complete_current_track()
                if bar_is_live:
                    sys.stdout.write("\n")
                name = m.group(3).strip()
                current_duration = durations.get(name, fallback_duration)
                current_estimated_total = None
                current_fraction = 0.0
                in_track = True
                track_start_time = time.time()
                print(line)
                draw()
                return

            if line.startswith("Elapsed time:"):
                # Printed exactly once, right after the whole batch
                # finishes -- the reliable signal that the last track
                # is done (there is no further "Processing track ..."
                # line to trigger completion the usual way).
                if in_track:
                    draw()  # same final refresh as above, for the last track
                complete_current_track()
                if bar_is_live:
                    sys.stdout.write("\n")
                    bar_is_live = False
                print(line)
                return

            m = _ESTIMATED_TOTAL_RE.match(line)
            if m:
                current_estimated_total = float(m.group(1))
                draw()
                return

            m = _ESTIMATED_REMAINING_RE.match(line)
            if m and current_estimated_total:
                remaining = float(m.group(1))
                current_fraction = max(
                    0.0, min(1.0, 1.0 - remaining / current_estimated_total)
                )
                draw()
                return

            # Anything else (banners, warnings, tracebacks, the final
            # "Elapsed time: ..." line): pass through untouched.
            if bar_is_live:
                sys.stdout.write("\n")
                bar_is_live = False
            print(line)

        decoder = codecs.getincrementaldecoder("utf-8")(errors="replace")
        current_line = ""
        # True when the previous character was a bare "\r" whose meaning
        # (a real progress-bar overwrite vs. just the first half of a
        # "\r\n" line ending) isn't known until the next character
        # arrives. The pty's line discipline applies ONLCR by default,
        # so every plain `print(...)`/trailing "\n" the child writes
        # actually arrives here as "\r\n" -- if "\r" and "\n" were each
        # treated as their own terminator, every ordinary printed line
        # would fire handle_line() an extra, spurious time with an
        # empty string. Collapsing "\r\n" into a single terminator (like
        # universal-newlines) avoids that, while still treating a
        # genuine standalone "\r" (upstream's own in-place ETA-text
        # overwrites, and upstream's tqdm bar) as its own overwrite
        # event.
        pending_cr = False
        while True:
            try:
                chunk = os.read(master_fd, 4096)
            except OSError:
                break
            if not chunk:
                break
            text = decoder.decode(chunk)
            for ch in text:
                if pending_cr:
                    pending_cr = False
                    if ch == "\n":
                        handle_line(current_line)
                        current_line = ""
                        continue
                    # else: the previous "\r" was a real standalone
                    # overwrite (tqdm bar / ETA text) -- fire it now,
                    # then fall through to process `ch` normally.
                    handle_line(current_line)
                    current_line = ""
                if ch == "\r":
                    pending_cr = True
                elif ch == "\n":
                    handle_line(current_line)
                    current_line = ""
                else:
                    current_line += ch
        if pending_cr:
            handle_line(current_line)
            current_line = ""
        if current_line:
            handle_line(current_line)

        returncode = proc.wait()
        if returncode == 0:
            # The "Elapsed time: ..." line handler above already closes
            # out the last track and prints its own final newline in the
            # normal case, so re-drawing here would just print a second,
            # redundant "N/N [.. s/track]" line duplicating the one
            # already shown before "Elapsed time: ...". Only fall back to
            # drawing here if that didn't happen (e.g. some unanticipated
            # output-parsing edge case left the last track uncompleted).
            if completed_tracks < total_tracks:
                complete_current_track()
                draw(final=True)
        elif bar_is_live:
            sys.stdout.write("\n")
        return returncode
    finally:
        os.close(master_fd)
        if slave_fd != -1:
            try:
                os.close(slave_fd)
            except OSError:
                pass
