#!/usr/bin/env python3
"""melband-roformer HUD terminal.

A small, self-contained terminal emulator styled after the "cyborg vision"
heads-up displays from the Terminator films: monochrome phosphor-red text
on a near-black background, large type, and a bracketed HUD status line.
It replaces the previously bundled `xterm` as this AppImage's windowed
terminal, so the AppImage no longer needs to ship (or find on the host) a
general-purpose terminal emulator at all.

This program is the *entire* terminal UI: it owns the window, draws the
screen, and is the only place AppRun exec's a fixed, hardcoded child
command into (see AppRun -- exactly as it previously did for `xterm -e`,
no user-supplied command line ever reaches this program's argv). It never
interprets its child's output as anything other than terminal bytes to
render; it never runs a shell of its own.

Architecture:
  - `pty`      opens a pseudo-terminal and forks the child (argv[1:]) into
               its slave side, so the child sees a real, resizable tty.
  - `pyte`     is a pure-Python VT100/xterm-ish terminal state machine. It
               turns the raw bytes read from the pty master into a screen
               buffer (rows x cols of styled characters, cursor position),
               handling the escape sequences real terminal programs use --
               including carriage-return line overwrites (`\\r`), which is
               how progress bars/spinners like tqdm animate a single line.
               Without this, a progress bar would print one line per
               update instead of overwriting itself in place. Window
               *resizing*, however, goes through `_resize_screen()` in
               this file instead of pyte's own `Screen.resize()`, which
               has a couple of bugs around shrinking the row count (see
               that function's docstring) that are easy to trigger just
               by dragging this window smaller after having made it
               bigger.
  - `tkinter`  (standard library) still owns the windows, event handling,
               and the host clipboard -- but it no longer draws a single
               glyph anywhere in this program, including its own
               right-click context menu (see `PillowMenu` below). tkinter
               ships as part of this AppImage's bundled Python and, on
               Linux, needs nothing beyond glibc to run; but the bundled
               Tcl/Tk has no libXft/fontconfig linkage, so left to its own
               devices it can only ever render the X server's ancient,
               non-antialiased core bitmap fonts (see "Why not just use a
               Tk font" below) -- which is also why it is never asked to.
  - `Pillow`   (PIL, a real runtime dependency -- see build-appimage.sh)
               rasterizes every glyph anywhere in this window, using its
               own statically linked FreeType, completely independent of
               Tk/X11/Xft. The terminal screen buffer is composited into
               one RGB image per frame (a `tkinter.Canvas` just displays
               that image); the header/footer status lines and the
               right-click context menu (`PillowMenu`, a borderless
               `tkinter.Toplevel` holding nothing but a Canvas showing one
               composited image, redrawn on hover) work the same way.
               All of it uses the DejaVu Sans Mono TrueType font bundled
               alongside this script (appimage/fonts/). This is what
               actually fixes the "aliased/blocky" look: antialiasing was
               never available through Tk's own font engine on this
               interpreter, no matter what font *name* was requested.

Why not just use a Tk font:
  This AppImage's bundled CPython (python-build-standalone) ships a Tcl/Tk
  built with no libX11/libxcb/libXft dependency (verified via `ldd`/
  `objdump -p`; see README.appimage.md). That is deliberate upstream, for
  portability -- but the side effect is that tkinter.font can only ever
  resolve to the X server's own core bitmap fonts (typically "fixed",
  "courier", "helvetica", "times"), never a host-installed TrueType font,
  and X core bitmap fonts have no antialiasing at all: at the sizes this
  HUD uses (14pt+) they are naively scaled from a small fixed set of
  design sizes, which is exactly the jagged/"ribbed" look this rewrite
  replaces. Rendering glyphs ourselves with Pillow/FreeType sidesteps the
  bundled Tk's font engine entirely, so it stays fixed no matter what
  fonts the *host* has installed, with no new shared-library dependency
  (Pillow's official wheels statically link their own FreeType/zlib).

Design constraints (do not relax these when editing this file):
  - No shell is ever spawned by this program. It only ever execs the exact
    argv it was given on its own command line (by AppRun), same as xterm's
    old hardcoded `-e "$VENV_BIN/python3" "$SHELL_SCRIPT"`.
  - All child output is treated as opaque terminal bytes, never executed,
    evaluated, or interpreted as anything other than screen content.
"""
from __future__ import annotations

import fcntl
import os
import queue
import select
import signal
import struct
import sys
import termios
import threading
import tkinter as tk

import pyte
from PIL import Image, ImageDraw, ImageFont, ImageTk

# ---------------------------------------------------------------------------
# HUD palette / sizing. Overridable via the same environment variables
# AppRun has always exposed, so a user who dislikes the look (or wants a
# specific size) doesn't need to edit any file to change it.
# ---------------------------------------------------------------------------
HUD_FG = os.environ.get("MELBAND_ROFORMER_HUD_FG", "#ff2a00")       # phosphor red
HUD_FG_BRIGHT = os.environ.get("MELBAND_ROFORMER_HUD_FG_BRIGHT", "#ff6a40")
HUD_BG = os.environ.get("MELBAND_ROFORMER_HUD_BG", "#0a0000")       # near-black, faint red tint
HUD_FRAME = os.environ.get("MELBAND_ROFORMER_HUD_FRAME", "#3a0d00")

# Points; the task this terminal was built for requires readable, large
# type, so 14 is a hard floor regardless of what a user requests via the
# environment variable -- see _resolve_font_size(). Kept in points (not
# pixels) for backwards compatibility with existing overrides of
# $MELBAND_ROFORMER_HUD_FONT_SIZE; _pt_to_px() converts for Pillow, which
# only understands pixel sizes.
MIN_FONT_SIZE = 14
DEFAULT_FONT_SIZE = 16

# Explicit overrides, for anyone who wants to swap in their own TrueType
# font instead of the bundled DejaVu Sans Mono.
FONT_PATH_ENV = "MELBAND_ROFORMER_HUD_FONT_PATH"
FONT_BOLD_PATH_ENV = "MELBAND_ROFORMER_HUD_FONT_BOLD_PATH"
BUNDLED_FONT_REGULAR = "DejaVuSansMono.ttf"
BUNDLED_FONT_BOLD = "DejaVuSansMono-Bold.ttf"

DEFAULT_COLS = 100
DEFAULT_ROWS = 30

# Target number of scrollback lines one mouse-wheel notch moves (see the
# HistoryScreen construction in HudTerminal.__init__ for how this turns
# into pyte's own page-ratio parameter).
_SCROLLBACK_LINES_PER_NOTCH = 3


def _resolve_font_size() -> int:
    raw = os.environ.get("MELBAND_ROFORMER_HUD_FONT_SIZE", str(DEFAULT_FONT_SIZE))
    try:
        size = int(raw)
    except ValueError:
        size = DEFAULT_FONT_SIZE
    return max(MIN_FONT_SIZE, size)


def _pt_to_px(pt: int) -> int:
    """Point -> pixel size at a fixed 96dpi. Pillow's truetype loader only
    takes a pixel size; there is no host display DPI to query for the
    bundled, Xft-less Tk, so this keeps sizing predictable everywhere."""
    return max(1, round(pt * 96 / 72))


def _find_font_dir() -> str | None:
    """Locate the directory holding the bundled .ttf files, in priority
    order: next to this script (AppImage layout, and also the layout when
    running straight out of the source checkout), then relative to
    $APPDIR (belt-and-braces for alternate packaging layouts)."""
    here = os.path.dirname(os.path.abspath(__file__))
    candidates = [os.path.join(here, "fonts")]
    appdir = os.environ.get("APPDIR")
    if appdir:
        candidates.append(os.path.join(appdir, "usr", "share", "melband-roformer", "fonts"))
        candidates.append(os.path.join(appdir, "usr", "bin", "fonts"))
    for d in candidates:
        if os.path.isfile(os.path.join(d, BUNDLED_FONT_REGULAR)):
            return d
    return None


def _resolve_font_paths() -> tuple[str | None, str | None]:
    reg = os.environ.get(FONT_PATH_ENV)
    bold = os.environ.get(FONT_BOLD_PATH_ENV)
    if reg and bold and os.path.isfile(reg) and os.path.isfile(bold):
        return reg, bold
    font_dir = _find_font_dir()
    if font_dir:
        return (os.path.join(font_dir, BUNDLED_FONT_REGULAR),
                os.path.join(font_dir, BUNDLED_FONT_BOLD))
    return None, None


def _hex_to_rgb(value: str) -> tuple[int, int, int]:
    value = value.lstrip("#")
    return (int(value[0:2], 16), int(value[2:4], 16), int(value[4:6], 16))


def _rgb_to_hex(rgb: tuple[int, int, int]) -> str:
    return "#{:02x}{:02x}{:02x}".format(*rgb)


def _measure_font(font: ImageFont.FreeTypeFont) -> tuple[int, int]:
    probe = Image.new("L", (1, 1))
    draw = ImageDraw.Draw(probe)
    width = draw.textlength("M", font=font)
    ascent, descent = font.getmetrics()
    return max(1, int(round(width))), max(1, ascent + descent + 2)


def _render_label_image(text: str, font: ImageFont.ImageFont,
                         fg_rgb: tuple[int, int, int],
                         bg_rgb: tuple[int, int, int],
                         padx: int = 8, pady: int = 4) -> ImageTk.PhotoImage:
    """Render a single line of static HUD text (header/footer) as an
    antialiased image, same rationale as the main grid below."""
    probe = Image.new("RGB", (1, 1))
    draw = ImageDraw.Draw(probe)
    left, top, right, bottom = draw.textbbox((0, 0), text, font=font)
    width = (right - left) + 2 * padx
    height = (bottom - top) + 2 * pady
    img = Image.new("RGB", (max(1, width), max(1, height)), bg_rgb)
    draw = ImageDraw.Draw(img)
    draw.text((padx - left, pady - top), text, font=font, fill=fg_rgb)
    return ImageTk.PhotoImage(img)


def _resize_screen(screen: "pyte.Screen", new_rows: int, new_cols: int) -> None:
    """Resize a pyte screen for a GUI window resize, without going
    through `pyte.Screen.resize()`.

    pyte 0.8.2's own resize() has two compounding bugs when shrinking the
    row count, both easy to hit just by dragging this window smaller
    after having made it bigger:

      1. `Screen.delete_lines()` (which it uses to drop rows from the
         top) only overwrites a destination row if the correspondingly
         shifted *source* row happens to already hold data -- if the
         source is blank (very common: most of a screen usually is),
         the destination row is left holding its old, stale content
         instead of being cleared or shifted. Called repeatedly across
         many small resize steps (exactly what a live window-drag
         produces -- X11 sends a stream of intermediate `<Configure>`
         events, not just one final one), this visibly corrupts the
         buffer.
      2. `Screen.restore_cursor()` (which resize() uses to preserve the
         cursor across that same shift) restores the cursor to its
         pre-shift *absolute* row number, never adjusted for the shift
         that just happened -- so the cursor visibly drifts away from
         the text it was next to.

    This reimplements the same documented contract (grow: blank rows/
    columns appended; shrink: rows dropped from the top, columns from
    the right, only as far as needed to keep the cursor on-screen)
    directly against `screen.buffer`/`screen.cursor`, unconditionally
    (no "only if the source has data" gap) and cursor-aware (only drops
    the exact number of rows needed to bring the cursor back onto the
    last visible row -- a screen that already comfortably fits its
    content, the common case, is left completely untouched).
    """
    old_cols = screen.columns
    if new_rows < screen.lines:
        dropped = max(0, screen.cursor.y - (new_rows - 1))
        if dropped > 0:
            shifted = {y: screen.buffer[y + dropped]
                       for y in range(new_rows) if (y + dropped) in screen.buffer}
            screen.buffer.clear()
            screen.buffer.update(shifted)
            screen.cursor.y -= dropped
    if new_cols < old_cols:
        for line in screen.buffer.values():
            for x in range(new_cols, old_cols):
                line.pop(x, None)
        screen.cursor.x = min(screen.cursor.x, new_cols - 1)
    screen.lines, screen.columns = new_rows, new_cols
    screen.set_margins()
    screen.dirty.update(range(new_rows))


class PillowMenu:
    """A right-click context menu with no Tk-drawn text anywhere in it.

    A native `tkinter.Menu` renders its own item labels internally, using
    Tk's font engine -- there is no way to hand it a pre-rasterized label
    instead, so as long as one is used, this program would still have one
    spot where the bundled Tk's non-antialiased core bitmap font shows
    through. This class replaces it: a borderless (`overrideredirect`)
    `Toplevel` holding a single `Canvas`, which displays one composited
    Pillow image of the *entire* menu (border, items, separators, hover
    highlight) -- redrawn on mouse motion for the highlight, exactly the
    same technique `HudTerminal._redraw_frame` uses for the terminal grid
    itself. There is no per-item Tk widget, and Tk never draws a glyph.
    """

    PADX = 14
    PADY = 6
    SEP_HEIGHT = 7
    MIN_WIDTH = 160

    def __init__(self, root: tk.Tk, font: ImageFont.FreeTypeFont,
                 fg_rgb: tuple[int, int, int], fg_bright_rgb: tuple[int, int, int],
                 bg_rgb: tuple[int, int, int], border_rgb: tuple[int, int, int]):
        self.root = root
        self.font = font
        self.fg_rgb = fg_rgb
        self.fg_bright_rgb = fg_bright_rgb
        self.bg_rgb = bg_rgb
        self.border_rgb = border_rgb
        # Each entry is either (label, command) or None for a separator.
        self._items: list[tuple[str, object] | None] = []
        self._row_boxes: list[tuple[int, int] | None] = []
        self._hover: int | None = None
        self._toplevel: tk.Toplevel | None = None
        self._canvas: tk.Canvas | None = None
        self._photo: ImageTk.PhotoImage | None = None

        ascent, descent = font.getmetrics()
        self._row_h = ascent + descent + 2 * self.PADY

    def add_command(self, label: str, command) -> None:
        self._items.append((label, command))

    def add_separator(self) -> None:
        self._items.append(None)

    # -- building -------------------------------------------------------
    def _measure_width(self, draw: ImageDraw.ImageDraw) -> int:
        width = self.MIN_WIDTH
        for item in self._items:
            if item is None:
                continue
            label, _ = item
            bbox = draw.textbbox((0, 0), label, font=self.font)
            width = max(width, (bbox[2] - bbox[0]) + 2 * self.PADX)
        return width

    def _build_image(self) -> Image.Image:
        probe = Image.new("RGB", (1, 1))
        width = self._measure_width(ImageDraw.Draw(probe))

        row_tops = []
        height = 2  # 1px border top/bottom
        for item in self._items:
            row_tops.append(height)
            height += self.SEP_HEIGHT if item is None else self._row_h
        height += 2

        img = Image.new("RGB", (width, height), self.border_rgb)
        draw = ImageDraw.Draw(img)
        draw.rectangle([1, 1, width - 2, height - 2], fill=self.bg_rgb)

        self._row_boxes = []
        for idx, (item, y0) in enumerate(zip(self._items, row_tops)):
            if item is None:
                y = y0 + self.SEP_HEIGHT // 2
                draw.line([(6, y), (width - 6, y)], fill=self.border_rgb, width=1)
                self._row_boxes.append(None)
                continue
            label, _ = item
            y1 = y0 + self._row_h
            hovered = (idx == self._hover)
            if hovered:
                draw.rectangle([2, y0, width - 3, y1 - 1], fill=self.fg_rgb)
                fg = self.bg_rgb
            else:
                fg = self.fg_rgb
            bbox = draw.textbbox((0, 0), label, font=self.font)
            draw.text((self.PADX - bbox[0], y0 + self.PADY - bbox[1]), label,
                       font=self.font, fill=fg)
            self._row_boxes.append((y0, y1))
        return img

    def _redraw(self) -> None:
        if self._canvas is None:
            return
        img = self._build_image()
        self._photo = ImageTk.PhotoImage(img)
        self._canvas.configure(width=img.width, height=img.height)
        self._canvas.delete("all")
        self._canvas.create_image(0, 0, anchor="nw", image=self._photo)

    # -- showing / interaction -------------------------------------------
    def show(self, x_root: int, y_root: int) -> None:
        self.close()
        self._hover = None
        top = tk.Toplevel(self.root)
        top.overrideredirect(True)
        try:
            top.attributes("-topmost", True)
        except tk.TclError:
            pass
        top.configure(bg=_rgb_to_hex(self.border_rgb))
        canvas = tk.Canvas(top, bd=0, highlightthickness=0,
                            bg=_rgb_to_hex(self.bg_rgb))
        canvas.pack()
        self._toplevel = top
        self._canvas = canvas
        self._redraw()
        top.geometry(f"+{x_root}+{y_root}")
        canvas.bind("<Motion>", self._on_motion)
        canvas.bind("<ButtonRelease-1>", self._on_release)
        top.bind("<Escape>", lambda e: self.close())
        top.bind("<FocusOut>", lambda e: self.close())
        top.update_idletasks()
        top.focus_force()

    def close(self) -> None:
        if self._toplevel is not None:
            self._toplevel.destroy()
        self._toplevel = None
        self._canvas = None
        self._photo = None

    def _row_at(self, y: int) -> int | None:
        for idx, box in enumerate(self._row_boxes):
            if box is not None and box[0] <= y < box[1]:
                return idx
        return None

    def _on_motion(self, event: tk.Event) -> None:
        idx = self._row_at(event.y)
        if idx != self._hover:
            self._hover = idx
            self._redraw()

    def _on_release(self, event: tk.Event) -> None:
        idx = self._row_at(event.y)
        item = self._items[idx] if idx is not None else None
        self.close()
        if item is not None:
            _, command = item
            command()


class HudTerminal:
    """Owns the pty, the child process, the pyte screen model, and the
    tkinter/Pillow widgets that render it."""

    def __init__(self, root: tk.Tk, child_argv: list[str]):
        if not child_argv:
            raise ValueError("HudTerminal requires a non-empty child argv")
        self.root = root
        self.child_argv = child_argv
        self.cols = DEFAULT_COLS
        self.rows = DEFAULT_ROWS
        self._cursor_on = True
        self._closing = False

        # Mouse-drag text selection, in (row, col) grid coordinates.
        # _sel_anchor is where the current drag started; _sel_start/_sel_end
        # are only set once the drag has actually moved, so a plain click
        # (no drag) clears any old selection instead of "selecting" one cell.
        self._sel_anchor: tuple[int, int] | None = None
        self._sel_start: tuple[int, int] | None = None
        self._sel_end: tuple[int, int] | None = None

        self._glyph_cache: dict[tuple[str, bool], Image.Image] = {}
        self._tile_cache: dict[tuple[int, int, int], Image.Image] = {}
        self._frame_photo: ImageTk.PhotoImage | None = None
        self._frame_img_id: int | None = None

        # Scrollback view state. pyte.HistoryScreen tracks the actual
        # scroll position itself (screen.history.position); these two
        # flags only track *our own* bookkeeping around it: whether the
        # footer is currently showing the "scrolled back" hint (so we only
        # touch it on an actual state change, not on every redraw -- see
        # _set_scrolled_footer), and whether the child has already exited
        # (so that bookkeeping never fights with the "process terminated"
        # footer -- see _on_child_eof).
        self._footer_scrolled = False
        self._child_exited = False

        self._build_window()
        self._spawn_child()

        # Default HistoryScreen ratio (0.5) moves half a screen per
        # prev_page()/next_page() call -- fine for a keyboard Page Up/Down,
        # but far too coarse for a single mouse-wheel notch (15 lines at
        # this HUD's default 30 rows). Passing a much smaller ratio here
        # makes each page call move roughly _SCROLLBACK_LINES_PER_NOTCH
        # lines instead (scaling with the *current* row count, so it stays
        # proportionate across window resizes too); see _on_mousewheel /
        # _scroll_pages below, which call prev_page()/next_page() directly
        # -- there is no separate "scroll by N lines" primitive in pyte.
        self.screen = pyte.HistoryScreen(
            self.cols, self.rows, history=4000,
            ratio=_SCROLLBACK_LINES_PER_NOTCH / max(1, self.rows),
        )
        self.stream = pyte.ByteStream(self.screen)

        self._outq: "queue.Queue[bytes]" = queue.Queue()
        self._reader_thread = threading.Thread(target=self._read_loop, daemon=True)
        self._reader_thread.start()

        self.canvas.focus_set()
        self._redraw_frame()
        self.root.after(30, self._poll_output)
        self.root.after(500, self._blink_cursor)

    # -- fonts -------------------------------------------------------------
    def _load_fonts(self) -> None:
        self.font_size = _resolve_font_size()  # points
        font_px = _pt_to_px(self.font_size)
        header_px = _pt_to_px(max(10, self.font_size - 4))
        footer_px = _pt_to_px(max(9, self.font_size - 5))

        reg_path, bold_path = _resolve_font_paths()
        if reg_path and bold_path:
            self.font = ImageFont.truetype(reg_path, font_px)
            self.bold_font = ImageFont.truetype(bold_path, font_px)
            self.header_font = ImageFont.truetype(bold_path, header_px)
            self.footer_font = ImageFont.truetype(reg_path, footer_px)
        else:
            # Should not happen in a correctly built AppImage (the fonts
            # are bundled right next to this script) -- but fail soft
            # rather than crash the whole terminal over a packaging slip.
            sys.stderr.write(
                "melband-roformer HUD terminal: bundled font not found "
                f"(looked for {BUNDLED_FONT_REGULAR} next to this script); "
                "falling back to Pillow's built-in bitmap font, which will "
                "look worse than intended.\n"
            )
            self.font = ImageFont.load_default()
            self.bold_font = self.font
            self.header_font = self.font
            self.footer_font = self.font

        self.char_w, self.char_h = _measure_font(self.font)

    # -- window / widget setup -------------------------------------------
    def _build_window(self) -> None:
        self.root.title("melband-roformer -- audio separation")
        self.root.configure(bg=HUD_FRAME, padx=2, pady=2)
        self.root.protocol("WM_DELETE_WINDOW", self.close)

        icon_path = os.environ.get("MELBAND_ROFORMER_HUD_ICON")
        if icon_path and os.path.isfile(icon_path):
            try:
                self._icon_img = tk.PhotoImage(file=icon_path)
                self.root.iconphoto(True, self._icon_img)
            except tk.TclError:
                pass  # non-fatal: an unreadable icon shouldn't block startup

        self._bg_rgb = _hex_to_rgb(HUD_BG)
        self._fg_rgb = _hex_to_rgb(HUD_FG)
        self._fg_bright_rgb = _hex_to_rgb(HUD_FG_BRIGHT)
        self._frame_rgb = _hex_to_rgb(HUD_FRAME)

        self._load_fonts()

        header_photo = _render_label_image(
            ">>> MELBAND-ROFORMER // AUDIO ANALYSIS UNIT <<<",
            self.header_font, self._fg_rgb, self._bg_rgb,
        )
        self._header_photo = header_photo  # keep alive
        header = tk.Label(self.root, image=header_photo, bg=HUD_BG, anchor="w")
        header.pack(fill="x")

        self.canvas = tk.Canvas(
            self.root, bg=HUD_BG, bd=0, highlightthickness=0, cursor="xterm",
            width=self.cols * self.char_w, height=self.rows * self.char_h,
        )
        self.canvas.pack(fill="both", expand=True)

        self._footer_photo_normal = _render_label_image(
            "[ COPY: Ctrl+Shift+C ]  [ PASTE: Ctrl+Shift+V ]  "
            "[ SCROLL: wheel, Shift+Home/Shift+End ]  "
            "[ RIGHT-CLICK: MENU ]  [ EXIT: type 'exit' ]",
            self.footer_font, self._fg_bright_rgb, self._bg_rgb, pady=3,
        )
        self._footer_photo_scrolled = _render_label_image(
            "[ SCROLLED BACK -- wheel down or Shift+End for live output ]",
            self.footer_font, self._fg_bright_rgb, self._bg_rgb, pady=3,
        )
        self._footer_photo_closing = _render_label_image(
            "[ PROCESS TERMINATED -- CLOSING ]",
            self.footer_font, self._fg_bright_rgb, self._bg_rgb, pady=3,
        )
        self.footer = tk.Label(self.root, image=self._footer_photo_normal,
                                bg=HUD_BG, anchor="w")
        self.footer.pack(fill="x")

        self._build_context_menu()
        self.canvas.bind("<Key>", self._on_key)
        self.canvas.bind("<Button-1>", self._on_button1)
        self.canvas.bind("<B1-Motion>", self._on_b1_motion)
        self.canvas.bind("<ButtonRelease-1>", self._on_b1_release)
        self.canvas.bind("<Button-3>", self._show_context_menu)
        self.canvas.bind("<Button-2>", self._paste_primary)   # X11 middle-click paste
        # Scrollback: classic X11 delivers the wheel as button clicks
        # (Button-4 = up, Button-5 = down, always one notch each); some
        # newer Linux input stacks (libinput over XWayland, in particular)
        # instead deliver a delta-based <MouseWheel> like Windows/macOS do
        # -- both are bound so the wheel works either way, whichever this
        # host's X server actually sends.
        self.canvas.bind("<Button-4>", lambda e: self._scroll_pages(1))
        self.canvas.bind("<Button-5>", lambda e: self._scroll_pages(-1))
        self.canvas.bind("<MouseWheel>", self._on_mousewheel)
        self.canvas.bind("<Configure>", self._on_resize)
        self.canvas.selection_handle(self._provide_selection)

    def _build_context_menu(self) -> None:
        # Reuses the footer's font/size: a Pillow-rendered menu, not a
        # native tk.Menu -- see PillowMenu's own docstring for why.
        self.menu = PillowMenu(
            self.root, self.footer_font,
            fg_rgb=self._fg_rgb, fg_bright_rgb=self._fg_bright_rgb,
            bg_rgb=self._bg_rgb, border_rgb=self._frame_rgb,
        )
        self.menu.add_command("Copy   (Ctrl+Shift+C)", self._copy_selection)
        self.menu.add_command("Paste  (Ctrl+Shift+V)", self._paste_clipboard)
        self.menu.add_separator()
        self.menu.add_command("Select all", self._select_all)
        self.menu.add_separator()
        self.menu.add_command("Scroll to top    (Shift+Home)", self._scroll_to_top)
        self.menu.add_command("Scroll to bottom (Shift+End)", self._scroll_to_bottom)

    # -- child process -----------------------------------------------------
    def _spawn_child(self) -> None:
        import pty
        self.master_fd, slave_fd = pty.openpty()
        self._apply_winsize()
        pid = os.fork()
        if pid == 0:
            # Child: become the pty's controlling process, wire its stdio to
            # the slave end, then exec the exact argv we were given -- never
            # a shell, never anything derived from user input at this point.
            try:
                os.setsid()
                fcntl.ioctl(slave_fd, termios.TIOCSCTTY, 0)
                os.dup2(slave_fd, 0)
                os.dup2(slave_fd, 1)
                os.dup2(slave_fd, 2)
                os.close(self.master_fd)
                os.close(slave_fd)
                os.environ["TERM"] = "xterm-256color"
                os.execvp(self.child_argv[0], self.child_argv)
            except Exception as exc:  # pragma: no cover - child-side only
                os.write(2, f"melband-roformer HUD terminal: exec failed: {exc}\n"
                             .encode("utf-8", "replace"))
            os._exit(1)
        os.close(slave_fd)
        self.child_pid = pid

    def _apply_winsize(self) -> None:
        packed = struct.pack("HHHH", self.rows, self.cols, 0, 0)
        try:
            fcntl.ioctl(self.master_fd, termios.TIOCSWINSZ, packed)
        except OSError:
            pass

    # -- reading child output (background thread) ---------------------------
    def _read_loop(self) -> None:
        while True:
            try:
                ready, _, _ = select.select([self.master_fd], [], [], 0.5)
            except (OSError, ValueError):
                return
            if self.master_fd not in ready:
                continue
            try:
                data = os.read(self.master_fd, 8192)
            except OSError:
                data = b""
            self._outq.put(data)
            if not data:
                return

    def _poll_output(self) -> None:
        if self._closing:
            return
        got_data = False
        try:
            while True:
                chunk = self._outq.get_nowait()
                if chunk == b"":
                    self._on_child_eof()
                    return
                self.stream.feed(chunk)
                got_data = True
        except queue.Empty:
            pass
        if got_data:
            self.screen.dirty.clear()
            self._redraw_frame()
            # New output already snapped the view back to the bottom of
            # history (pyte.HistoryScreen.before_event() does that
            # automatically for every event except prev_page/next_page
            # itself -- see the class docstring on the HistoryScreen
            # construction above), so the "scrolled back" footer hint, if
            # it was showing, is stale now and needs clearing to match.
            self._set_scrolled_footer(False)
        self.root.after(30, self._poll_output)

    def _on_child_eof(self) -> None:
        self._child_exited = True
        try:
            os.waitpid(self.child_pid, os.WNOHANG)
        except ChildProcessError:
            pass
        self.footer.configure(image=self._footer_photo_closing)
        self.root.after(400, self.close)

    # -- rendering -----------------------------------------------------------
    def _glyph_mask(self, ch: str, bold: bool) -> Image.Image:
        key = (ch, bold)
        mask = self._glyph_cache.get(key)
        if mask is None:
            font = self.bold_font if bold else self.font
            mask = Image.new("L", (self.char_w, self.char_h), 0)
            draw = ImageDraw.Draw(mask)
            try:
                draw.text((0, 0), ch, font=font, fill=255, anchor="la")
            except (ValueError, TypeError):
                # Fonts without anchor support (e.g. the load_default()
                # bitmap fallback) -- draw at the origin instead.
                draw.text((0, 0), ch, font=font, fill=255)
            self._glyph_cache[key] = mask
        return mask

    def _color_tile(self, rgb: tuple[int, int, int]) -> Image.Image:
        tile = self._tile_cache.get(rgb)
        if tile is None:
            tile = Image.new("RGB", (self.char_w, self.char_h), rgb)
            self._tile_cache[rgb] = tile
        return tile

    def _normalized_selection(self) -> tuple[tuple[int, int], tuple[int, int]] | None:
        if self._sel_start is None or self._sel_end is None:
            return None
        return tuple(sorted([self._sel_start, self._sel_end]))  # type: ignore[return-value]

    @staticmethod
    def _cell_in_selection(row: int, col: int,
                            sel_range: tuple[tuple[int, int], tuple[int, int]]) -> bool:
        (row_a, col_a), (row_b, col_b) = sel_range
        if row < row_a or row > row_b:
            return False
        if row == row_a and col < col_a:
            return False
        if row == row_b and col > col_b:
            return False
        return True

    def _redraw_frame(self) -> None:
        if self._closing:
            return
        width = max(1, self.cols * self.char_w)
        height = max(1, self.rows * self.char_h)
        frame = Image.new("RGB", (width, height), self._bg_rgb)
        draw = ImageDraw.Draw(frame)

        sel_range = self._normalized_selection()
        cursor_visible = (not self.screen.cursor.hidden) and self._cursor_on
        cursor_pos = (self.screen.cursor.y, self.screen.cursor.x)

        for row in range(self.rows):
            buf_row = self.screen.buffer[row]
            y = row * self.char_h
            for col in range(self.cols):
                cell = buf_row[col]
                ch = cell.data or " "
                bold = bool(cell.bold)
                reverse = bool(cell.reverse)
                is_cursor = cursor_visible and cursor_pos == (row, col)
                is_selected = sel_range is not None and self._cell_in_selection(row, col, sel_range)

                if is_cursor:
                    bg_rgb, fg_rgb = self._fg_rgb, self._bg_rgb
                elif is_selected:
                    bg_rgb, fg_rgb = self._fg_bright_rgb, self._bg_rgb
                elif reverse:
                    bg_rgb = self._fg_bright_rgb if bold else self._fg_rgb
                    fg_rgb = self._bg_rgb
                else:
                    bg_rgb = None
                    fg_rgb = self._fg_bright_rgb if bold else self._fg_rgb

                x = col * self.char_w
                if bg_rgb is not None:
                    draw.rectangle([x, y, x + self.char_w, y + self.char_h], fill=bg_rgb)
                if ch != " ":
                    mask = self._glyph_mask(ch, bold)
                    tile = self._color_tile(fg_rgb)
                    frame.paste(tile, (x, y), mask)

        self._frame_photo = ImageTk.PhotoImage(frame)
        if self._frame_img_id is None:
            self._frame_img_id = self.canvas.create_image(0, 0, anchor="nw", image=self._frame_photo)
        else:
            self.canvas.itemconfig(self._frame_img_id, image=self._frame_photo)
        self.canvas.configure(scrollregion=(0, 0, width, height))

    def _blink_cursor(self) -> None:
        if self._closing:
            return
        self._cursor_on = not self._cursor_on
        self._redraw_frame()
        self.root.after(500, self._blink_cursor)

    # -- resize --------------------------------------------------------------
    def _on_resize(self, event: tk.Event) -> None:
        if event.widget is not self.canvas or self.char_w <= 0 or self.char_h <= 0:
            return
        new_cols = max(20, event.width // self.char_w)
        new_rows = max(5, event.height // self.char_h)
        if (new_cols, new_rows) == (self.cols, self.rows):
            return
        self.cols, self.rows = new_cols, new_rows
        _resize_screen(self.screen, self.rows, self.cols)
        self._apply_winsize()
        try:
            os.kill(self.child_pid, signal.SIGWINCH)
        except OSError:
            pass
        self._redraw_frame()

    # -- scrollback ------------------------------------------------------------
    # pyte.HistoryScreen keeps its own scroll position (screen.history.
    # position / .top / .bottom); everything here just drives its
    # prev_page()/next_page() (there is no "scroll by N lines" primitive --
    # see the ratio comment on the HistoryScreen construction above for how
    # one page call is kept close to _SCROLLBACK_LINES_PER_NOTCH lines) and
    # keeps the footer hint in sync. Scrolling itself is a pure view
    # operation -- it never touches the pty or the child process, and
    # typing (or any new child output) snaps straight back to live output,
    # the same as in any other terminal emulator.
    def _set_scrolled_footer(self, scrolled: bool) -> None:
        if self._closing or self._child_exited:
            return  # don't fight the "process terminated" footer
        if scrolled == self._footer_scrolled:
            return  # only touch the widget on an actual state change
        self._footer_scrolled = scrolled
        self.footer.configure(
            image=self._footer_photo_scrolled if scrolled else self._footer_photo_normal
        )

    def _scroll_pages(self, notches: int) -> None:
        """Scroll by `notches` wheel-notch-equivalents: positive = up/back
        into history, negative = down/forward toward live output."""
        if notches == 0 or self._closing:
            return
        page = self.screen.prev_page if notches > 0 else self.screen.next_page
        for _ in range(abs(notches)):
            page()
        self._redraw_frame()
        self._set_scrolled_footer(self.screen.history.position < self.screen.history.size)

    def _on_mousewheel(self, event: tk.Event) -> None:
        # Delta-based wheel event (Windows/macOS, and some Linux input
        # stacks over XWayland): +/-120 is the traditional one-notch unit,
        # but be defensive about hosts that report smaller/fractional
        # deltas (some XWayland/libinput setups do) -- always scroll at
        # least one notch rather than silently doing nothing.
        notches = max(1, abs(event.delta) // 120)
        self._scroll_pages(notches if event.delta > 0 else -notches)

    def _scroll_to_top(self) -> None:
        if self._closing:
            return
        while self.screen.history.position > self.screen.lines and self.screen.history.top:
            self.screen.prev_page()
        self._redraw_frame()
        self._set_scrolled_footer(self.screen.history.position < self.screen.history.size)

    def _scroll_to_bottom(self) -> None:
        if self._closing:
            return
        while self.screen.history.position < self.screen.history.size:
            self.screen.next_page()
        self._redraw_frame()
        self._set_scrolled_footer(False)

    # -- keyboard input --------------------------------------------------------
    _SPECIAL_KEYS = {
        "Return": b"\r", "KP_Enter": b"\r", "BackSpace": b"\x7f",
        "Tab": b"\t", "Escape": b"\x1b",
        "Up": b"\x1b[A", "Down": b"\x1b[B", "Right": b"\x1b[C", "Left": b"\x1b[D",
        "Home": b"\x1b[H", "End": b"\x1b[F", "Delete": b"\x1b[3~",
        "Prior": b"\x1b[5~", "Next": b"\x1b[6~",
    }

    # Ctrl+Shift+C / Ctrl+Shift+V must not be matched by X11 *keysym* alone
    # (event.keysym) -- keysym is the symbol the active keyboard *layout*
    # produces, not the physical key. On a Cyrillic (ЙЦУКЕН) layout, the
    # physical "C"/"V" keys produce the keysyms "Cyrillic_es"/"Cyrillic_em"
    # (they type С/М), not "c"/"v" -- so `keysym.lower() == "c"` silently
    # never matches for a Cyrillic-layout user, and Ctrl+Shift+C/V then fall
    # through to the "any other keystroke dismisses the selection" branch
    # below instead of copying/pasting (reported: selection is cleared on
    # every keypress, and only the right-click menu's copy/paste -- which
    # calls _copy_selection/_paste_clipboard directly, bypassing this
    # keysym check entirely -- still works).
    #
    # Fix: match on event.keycode first -- the X11 *hardware* keycode is
    # tied to the key's physical position, not to whatever the active XKB
    # layout/group maps it to, so it stays the same whether the layout is
    # Latin, Cyrillic, or anything else. 54/55 are the physical "C"/"V" key
    # positions under the evdev driver (keycode = evdev scancode + 8; C is
    # KEY_C=46, V is KEY_V=47) -- effectively universal on modern Linux/X11.
    # The keysym check is kept alongside as a fallback for the rare setups
    # where keycodes don't follow that mapping (some VNC/remote-X/Xvfb
    # servers), extended with the two most common Cyrillic keysym names so
    # it degrades gracefully there too, instead of only covering Latin
    # layouts as before.
    _COPY_KEYCODES = {54}
    _PASTE_KEYCODES = {55}
    _COPY_KEYSYMS = {"c", "cyrillic_es"}
    _PASTE_KEYSYMS = {"v", "cyrillic_em"}

    def _is_copy_chord(self, event: tk.Event) -> bool:
        return event.keycode in self._COPY_KEYCODES or event.keysym.lower() in self._COPY_KEYSYMS

    def _is_paste_chord(self, event: tk.Event) -> bool:
        return event.keycode in self._PASTE_KEYCODES or event.keysym.lower() in self._PASTE_KEYSYMS

    # A modifier key pressed on its own (Ctrl, Shift, Alt, ...) fires its
    # own KeyPress event before the letter it's held for. Crucially,
    # X11's event.state on THAT event reflects the modifier state *before*
    # this key, so e.g. the Control_L keypress itself still has ctrl=False
    # in event.state -- it doesn't match any binding below, including the
    # copy/paste chord checks above, and previously fell all the way
    # through to the "any other keystroke dismisses the selection" branch.
    # That means the sequence "press Ctrl, press Shift, press C" cleared
    # the selection on the very first keydown (Control_L), before Ctrl+
    # Shift+C could ever be recognized as a chord -- reproducing "the
    # selection resets the instant I touch Ctrl or Shift", independent of
    # keyboard layout (unlike the Cyrillic keysym issue above, this one
    # also happens on a plain Latin layout). Fix: a bare modifier keypress
    # must be a no-op here -- it neither clears the selection nor sends
    # anything to the child -- so the selection survives until the actual
    # letter key of the chord arrives.
    _MODIFIER_KEYSYMS = {
        "Control_L", "Control_R", "Shift_L", "Shift_R",
        "Alt_L", "Alt_R", "Meta_L", "Meta_R", "Super_L", "Super_R",
        "Hyper_L", "Hyper_R", "ISO_Level3_Shift", "ISO_Level5_Shift",
        "Mode_switch", "Caps_Lock", "Shift_Lock", "Num_Lock", "Scroll_Lock",
    }

    def _on_key(self, event: tk.Event) -> str:
        if event.keysym in self._MODIFIER_KEYSYMS:
            return "break"

        ctrl = bool(event.state & 0x0004)
        shift = bool(event.state & 0x0001)
        keysym = event.keysym

        if ctrl and shift and self._is_copy_chord(event):
            self._copy_selection()
            return "break"
        if ctrl and shift and self._is_paste_chord(event):
            self._paste_clipboard()
            return "break"
        if keysym == "Insert" and shift:
            self._paste_clipboard()
            return "break"
        # Shift+Home / Shift+End: jump to the top/bottom of scrollback --
        # the xterm/gnome-terminal/konsole convention. Plain (unshifted)
        # Home/End are deliberately left going to the child as normal
        # terminal input further down (via _SPECIAL_KEYS) instead of being
        # repurposed here, same reasoning as Ctrl+Shift+C/V above: the
        # child's own line editor (readline, in this shell) already uses
        # bare Home/End for cursor-to-line-start/end, and overriding that
        # would break normal typing.
        if keysym == "Home" and shift:
            self._scroll_to_top()
            return "break"
        if keysym == "End" and shift:
            self._scroll_to_bottom()
            return "break"

        # Any other keystroke dismisses the current visual selection, same
        # as typing into a normal terminal.
        if self._sel_start is not None or self._sel_end is not None or self._sel_anchor is not None:
            self._sel_anchor = self._sel_start = self._sel_end = None
            self._redraw_frame()

        if keysym in self._SPECIAL_KEYS:
            self._send(self._SPECIAL_KEYS[keysym])
            return "break"
        if ctrl and len(keysym) == 1 and keysym.isalpha():
            self._send(bytes([ord(keysym.lower()) - ord("a") + 1]))
            return "break"
        if event.char and event.char.isprintable():
            self._send(event.char.encode("utf-8", "ignore"))
            return "break"
        if keysym == "space":
            self._send(b" ")
            return "break"
        return "break"  # never fall through to tkinter's default text edits

    def _send(self, data: bytes) -> None:
        try:
            os.write(self.master_fd, data)
        except OSError:
            pass

    # -- mouse selection -------------------------------------------------------
    def _event_to_cell(self, event: tk.Event) -> tuple[int, int]:
        col = max(0, min(self.cols - 1, event.x // self.char_w))
        row = max(0, min(self.rows - 1, event.y // self.char_h))
        return row, col

    def _on_button1(self, event: tk.Event) -> None:
        self.canvas.focus_set()
        self._sel_anchor = self._event_to_cell(event)
        self._sel_start = self._sel_end = None
        self._redraw_frame()

    def _on_b1_motion(self, event: tk.Event) -> None:
        if self._sel_anchor is None:
            return
        cell = self._event_to_cell(event)
        if cell == self._sel_anchor and self._sel_start is None:
            return  # not enough movement yet to call it a selection
        self._sel_start, self._sel_end = self._sel_anchor, cell
        self._redraw_frame()

    def _on_b1_release(self, event: tk.Event) -> None:
        if self._sel_start is not None:
            try:
                self.canvas.selection_own()
            except tk.TclError:
                pass

    def _get_selected_text(self) -> str:
        sel_range = self._normalized_selection()
        if sel_range is None:
            return ""
        (row_a, col_a), (row_b, col_b) = sel_range
        lines = []
        for row in range(row_a, row_b + 1):
            col_from = col_a if row == row_a else 0
            col_to = col_b if row == row_b else self.cols - 1
            buf_row = self.screen.buffer[row]
            chars = [buf_row[c].data or " " for c in range(col_from, col_to + 1)]
            lines.append("".join(chars).rstrip())
        return "\n".join(lines)

    def _provide_selection(self, offset: str, length: str) -> str:
        # Signature required by tkinter's selection_handle: both args
        # arrive as strings that look like integers.
        text = self._get_selected_text()
        start = int(offset)
        count = int(length)
        return text[start:start + count]

    # -- clipboard -------------------------------------------------------------
    # tkinter's clipboard_get/clipboard_append talk to the host's X11
    # CLIPBOARD selection directly -- no external clipboard tool (xclip,
    # xsel, ...) needs to be bundled or shelled out to.
    def _copy_selection(self) -> None:
        text = self._get_selected_text()
        if not text:
            return
        self.root.clipboard_clear()
        self.root.clipboard_append(text)

    def _paste_clipboard(self) -> None:
        try:
            data = self.root.clipboard_get()
        except tk.TclError:
            return  # clipboard empty or holds non-text content
        self._send(data.replace("\n", "\r").encode("utf-8", "ignore"))

    def _paste_primary(self, event: tk.Event | None = None) -> None:
        try:
            data = self.canvas.selection_get(selection="PRIMARY")
        except tk.TclError:
            return
        self._send(data.replace("\n", "\r").encode("utf-8", "ignore"))

    def _select_all(self) -> None:
        self._sel_anchor = (0, 0)
        self._sel_start = (0, 0)
        self._sel_end = (self.rows - 1, self.cols - 1)
        try:
            self.canvas.selection_own()
        except tk.TclError:
            pass
        self._redraw_frame()

    def _show_context_menu(self, event: tk.Event) -> None:
        self.menu.show(event.x_root, event.y_root)

    # -- shutdown ----------------------------------------------------------
    def close(self) -> None:
        if self._closing:
            return
        self._closing = True
        try:
            os.kill(self.child_pid, signal.SIGHUP)
        except OSError:
            pass
        self.root.after(200, self.root.destroy)


def main() -> int:
    child_argv = sys.argv[1:]
    if not child_argv:
        sys.stderr.write(
            "melband-roformer HUD terminal: no command given.\n"
            "This program is only ever meant to be launched by AppRun with "
            "a fixed command to run inside it.\n"
        )
        return 2

    root = tk.Tk()
    try:
        HudTerminal(root, child_argv)
    except tk.TclError as exc:
        sys.stderr.write(
            f"melband-roformer HUD terminal: could not open a display: {exc}\n"
            "Set DISPLAY to a running X server, or run the AppImage from an "
            "existing terminal instead (it will skip this window entirely).\n"
        )
        return 1
    root.mainloop()
    return 0


if __name__ == "__main__":
    sys.exit(main())
