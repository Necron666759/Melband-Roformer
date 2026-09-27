#!/usr/bin/env python3
"""melband-roformer restricted shell.

This is the ONLY program the AppImage's embedded terminal ever runs. It is
a locked-down REPL, not a general-purpose shell: every line of input is
tokenized with `shlex` (which only splits quoted words -- it does not
interpret `;`, `&&`, `|`, backticks, `$()`, `*` globbing via a shell, or
any other shell metacharacter) and matched against an explicit whitelist
of melband-roformer subcommands. Anything else is rejected with a message,
never executed, never eval'd, never handed to os.system()/a real shell.

Design constraints (do not relax these when editing this file):
  - No `os.system`, no `subprocess.run(..., shell=True)`, no `eval`/`exec`
    of user input, ever.
  - The only subprocess this program (transitively, via
    melband_roformer_wrapper.cli) ever starts is the bundled
    `melband-roformer-infer` binary inside this AppImage's own venv, with
    an argv list built entirely from parsed, typed arguments -- never a
    raw string from the user.
  - No command that reads/writes/lists the filesystem outside of what
    melband-roformer's own CLI already exposes (input files, --output-dir,
    --models-dir). No `cd`, no `ls`, no general file browsing: this
    terminal exists exclusively for melband-roformer, nothing else.
  - `default()` (cmd.Cmd's catch-all for unrecognized input) must always
    print-and-refuse, never fall through to anything that executes.
"""
from __future__ import annotations

import cmd
import shlex
import sys

# ---------------------------------------------------------------------------
# HUD styling ("cyborg vision" look: monochrome red, bracketed readouts).
# ANSI escapes only -- no external dependency, and it degrades to plain
# text automatically when stdout isn't a real terminal (piped output,
# logging, etc.), so nothing here can corrupt a non-interactive run.
# AppRun sets the bundled HUD terminal's own colors/font size separately;
# these codes keep the same look even when this script is run inside a
# terminal AppRun doesn't own (the "already have a controlling terminal"
# path).
# ---------------------------------------------------------------------------
_COLOR = sys.stdout.isatty()
_RED = "\033[38;5;196m" if _COLOR else ""
_DIM_RED = "\033[38;5;124m" if _COLOR else ""
_BOLD = "\033[1m" if _COLOR else ""
_RESET = "\033[0m" if _COLOR else ""


def _hud(text: str) -> str:
    """Wrap text in the HUD red/bold color, reset after."""
    return f"{_BOLD}{_RED}{text}{_RESET}"


def _rl_invisible(escape_codes: str) -> str:
    """Wrap non-printing bytes (here: our ANSI color codes) in GNU
    readline's zero-width markers (\\001 .. \\002 / RL_PROMPT_START_IGNORE
    .. RL_PROMPT_END_IGNORE).

    This matters ONLY for `cmd.Cmd.prompt`, which is handed to the
    builtin `input()` and, when the readline module is loaded (as it is
    here, transitively, by `cmd`), gets echoed through readline's own
    line-editing logic rather than written to the terminal as-is.
    Without these markers, readline doesn't know the color codes are
    zero-width and corrupts them while computing cursor position --
    concretely, it silently drops the leading ESC (\\x1b) byte of each
    sequence, leaving the *rest* of the escape code (e.g. "[1m") behind
    as literal, visible garbage in front of the prompt instead of an
    invisible color change. Verified against this project's own restricted
    shell (see README.appimage.md / the AppImage build notes) before this
    fix was added.

    Never use this outside of `cmd.Cmd.prompt` / other `input()` prompts:
    plain `print()`/`sys.stdout.write()` output is never touched by
    readline, so wrapping it here would instead leave the literal \\001
    and \\002 bytes visible as stray artifacts.
    """
    return f"\001{escape_codes}\002" if escape_codes else escape_codes


BANNER = _hud(r"""
+============================================================================+
|                                                                            |
|       /\                                                                   |
|      /  \        M E L B A N D - R O F O R M E R                           |
|     / () \          A U D I O   A N A L Y S I S   U N I T                  |
|      \  /                                                                  |
|       \/         >>> VOCAL / INSTRUMENTAL SEPARATION SYSTEM <<<            |
|                                                                            |
+============================================================================+""") + \
    f"\n{_DIM_RED}  [ SYS ]{_RESET} restricted terminal -- runs melband-roformer only.\n" \
    f"{_DIM_RED}  [ SYS ]{_RESET} type 'help' for commands, 'exit' to quit.".rstrip("\n")

# The exact set of subcommands this shell understands. Kept as a plain
# tuple (not derived "cleverly") so the whitelist is easy to audit by eye.
_KNOWN_COMMANDS = (
    "separate", "list-models", "download-model", "info", "self-test",
    "version", "clear", "help", "exit", "quit",
)

# ANSI "Erase in Display" (mode 2: whole screen) + "Cursor Position" (row
# 1, col 1, the default/home position). Written directly to stdout, the
# same as the HUD color codes above -- this is NOT the external `clear(1)`
# binary (no subprocess, no shell, nothing new to whitelist at the OS
# level): it is just the raw escape sequence the bundled HUD terminal's
# `pyte` screen already knows how to interpret (see
# `Screen.erase_in_display`/`Screen.cursor_position` upstream), the same
# way it already interprets every other cursor-movement/overwrite
# sequence this project's own tqdm-replacement progress bar relies on
# (`progress.py`). Guarded behind `isatty()` like the color codes just
# above: on a non-interactive stdout (piped/logged output) it would just
# be noise, not a visible clear.
_CLEAR_SCREEN = "\033[2J\033[H" if _COLOR else ""


def _import_cli():
    """Imported lazily so `--help`-speed startup and clearly-wrong-input
    rejection don't pay the cost of importing torch et al."""
    from melband_roformer_wrapper import cli
    return cli


class MelbandRoformerShell(cmd.Cmd):
    intro = BANNER
    prompt = (f"{_rl_invisible(_BOLD + _RED)}[melband-roformer]>"
              f"{_rl_invisible(_RESET)} ")
    # cmd.Cmd's own '!' shell-escape and '?'-as-help are the two built-in
    # features most likely to be mistaken for a real shell -- both are
    # explicitly neutralized below rather than relied upon to be absent.

    def _dispatch(self, subcommand: str, rest: str) -> None:
        try:
            extra = shlex.split(rest)
        except ValueError as exc:
            print(f"Could not parse arguments: {exc}")
            return
        cli = _import_cli()
        argv = _build_argv(subcommand, extra)
        try:
            rc = cli.main(argv)
        except SystemExit as exc:
            # argparse calls sys.exit() on bad flags; keep it inside the
            # REPL instead of killing the whole shell.
            rc = exc.code
        except Exception as exc:  # noqa: BLE001 - last-resort safety net
            print(f"melband-roformer error: {exc}")
            rc = 1
        if rc:
            print(f"(exit code {rc})")

    # -- whitelisted commands -------------------------------------------------
    def do_separate(self, arg):
        "separate <file(s)> [--output-dir DIR] [--device auto|cuda|cpu] [--model SLUG]\n" \
        "    Separate vocals/instrumental from one or more audio files."
        self._dispatch("separate", arg)

    def do_list_models(self, arg):
        "list-models\n    List models available in the upstream registry."
        self._dispatch("list-models", arg)

    def do_download_model(self, arg):
        "download-model [--model SLUG] [--yes]\n    Download (with checksum verification) a model."
        self._dispatch("download-model", arg)

    def do_info(self, arg):
        "info\n    Print environment/GPU/model diagnostic info."
        self._dispatch("info", arg)

    def do_self_test(self, arg):
        "self-test\n    Run Python/PyTorch/CUDA/model/inference checks."
        self._dispatch("self-test", arg)

    def do_version(self, arg):
        "version\n    Print the melband-roformer version."
        self._dispatch("version", arg)

    def do_clear(self, arg):
        "clear\n    Clear the terminal screen."
        # Handled entirely in this process via a raw ANSI escape sequence
        # -- never dispatched to melband_roformer_wrapper.cli, since there
        # is no matching CLI flag for it (see _build_argv: every other
        # whitelisted command maps to one). Takes no arguments; anything
        # after "clear" is silently ignored, same as real terminals' own
        # `clear` command.
        if _CLEAR_SCREEN:
            sys.stdout.write(_CLEAR_SCREEN)
        # Re-print the banner (logo + "[ SYS ]" lines) right after
        # clearing. `cmd.Cmd` only ever prints `self.intro` once, before
        # the very first prompt of the session (see `Cmd.cmdloop()`'s own
        # source) -- so without this, "clear" would wipe it off the
        # screen for good, leaving a bare prompt with no way to bring it
        # back short of restarting the whole terminal.
        print(self.intro)
        if _CLEAR_SCREEN:
            sys.stdout.flush()

    # cmd.Cmd maps "list-models" etc. (with a hyphen) to do_list_models via
    # the alias table below, since Python identifiers can't contain '-'.
    def default(self, line):
        cmd_word = line.split(None, 1)[0] if line.strip() else ""
        alias = {"list-models": self.do_list_models,
                  "download-model": self.do_download_model,
                  "self-test": self.do_self_test}.get(cmd_word)
        if alias is not None:
            rest = line.split(None, 1)[1] if " " in line else ""
            return alias(rest)
        print(_hud(f"  [ ERR ] unrecognized input: {line.strip()!r}"))
        print(_hud("  [ ERR ] this terminal only runs melband-roformer commands. Type 'help'."))
        return None

    # -- exit -------------------------------------------------------------
    def do_exit(self, arg):
        "exit\n    Close this terminal."
        return True

    def do_quit(self, arg):
        "quit\n    Same as 'exit'."
        return True

    def do_EOF(self, arg):
        print()
        return True

    # -- neutralize cmd.Cmd features that could be mistaken for a real shell --
    def do_shell(self, arg):  # would normally back the '!' shell-escape
        print("This terminal does not run shell commands -- melband-roformer only.")

    def emptyline(self):
        pass  # never repeat the last command on a blank line


def _build_argv(subcommand: str, extra: list[str]) -> list[str]:
    if subcommand == "separate":
        return list(extra)
    if subcommand == "list-models":
        return ["--list-models", *extra]
    if subcommand == "download-model":
        return ["--download-model", *extra]
    if subcommand == "info":
        return ["--info", *extra]
    if subcommand == "self-test":
        return ["--self-test", *extra]
    if subcommand == "version":
        return ["--version", *extra]
    raise AssertionError(f"unreachable: unlisted subcommand {subcommand!r}")


def main() -> int:
    shell = MelbandRoformerShell()
    try:
        shell.cmdloop()
    except KeyboardInterrupt:
        print()
    return 0


if __name__ == "__main__":
    sys.exit(main())
