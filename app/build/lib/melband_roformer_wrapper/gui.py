"""Minimal GTK4 GUI front-end for melband-roformer.

Deliberately thin: it shells out to the same `melband-roformer` CLI used
from the terminal (via a background thread) rather than duplicating the
separation/device/model logic, so the GUI and CLI can never disagree about
behaviour.

NOTE (packaging honesty): this module has not been visually exercised in
the environment this package was prepared in (no X11/Wayland display
server was available there). It is included because the task asked for a
GTK4 GUI and the code is a straightforward, standard PyGObject layout with
no exotic API usage, but treat first real runs of `melband-roformer-gui`
as the first real test of this file.
"""
from __future__ import annotations

import subprocess
import sys
import threading
from pathlib import Path

import gi

gi.require_version("Gtk", "4.0")
from gi.repository import Gtk, Gio, GLib  # noqa: E402


class MainWindow(Gtk.ApplicationWindow):
    def __init__(self, app: Gtk.Application):
        super().__init__(application=app, title="Mel-Band RoFormer")
        self.set_default_size(560, 380)

        self.input_path: Path | None = None
        self.output_dir: Path = Path.home() / "melband-roformer-output"

        root = Gtk.Box(orientation=Gtk.Orientation.VERTICAL, spacing=12)
        root.set_margin_top(16)
        root.set_margin_bottom(16)
        root.set_margin_start(16)
        root.set_margin_end(16)
        self.set_child(root)

        # --- Input file ---
        row = Gtk.Box(orientation=Gtk.Orientation.HORIZONTAL, spacing=8)
        self.input_label = Gtk.Label(label="No file selected", xalign=0)
        self.input_label.set_hexpand(True)
        pick_btn = Gtk.Button(label="Choose Audio File…")
        pick_btn.connect("clicked", self.on_pick_input)
        row.append(self.input_label)
        row.append(pick_btn)
        root.append(row)

        # --- Model ---
        model_row = Gtk.Box(orientation=Gtk.Orientation.HORIZONTAL, spacing=8)
        model_row.append(Gtk.Label(label="Model:"))
        self.model_combo = Gtk.DropDown.new_from_strings([
            "melband-roformer-kim-vocals (recommended)",
        ])
        self.model_combo.set_hexpand(True)
        model_row.append(self.model_combo)
        root.append(model_row)

        # --- Device ---
        device_row = Gtk.Box(orientation=Gtk.Orientation.HORIZONTAL, spacing=8)
        device_row.append(Gtk.Label(label="Device:"))
        self.device_combo = Gtk.DropDown.new_from_strings(["auto", "cuda", "cpu"])
        device_row.append(self.device_combo)
        root.append(device_row)

        # --- Output dir ---
        out_row = Gtk.Box(orientation=Gtk.Orientation.HORIZONTAL, spacing=8)
        self.output_label = Gtk.Label(label=str(self.output_dir), xalign=0)
        self.output_label.set_hexpand(True)
        out_btn = Gtk.Button(label="Choose Output Folder…")
        out_btn.connect("clicked", self.on_pick_output)
        out_row.append(self.output_label)
        out_row.append(out_btn)
        root.append(out_row)

        # --- Run ---
        self.run_btn = Gtk.Button(label="Separate")
        self.run_btn.connect("clicked", self.on_run)
        root.append(self.run_btn)

        self.progress = Gtk.ProgressBar()
        self.progress.set_show_text(True)
        root.append(self.progress)

        status_scroller = Gtk.ScrolledWindow()
        status_scroller.set_vexpand(True)
        self.status_view = Gtk.TextView()
        self.status_view.set_editable(False)
        self.status_view.set_monospace(True)
        status_scroller.set_child(self.status_view)
        root.append(status_scroller)

        self.open_output_btn = Gtk.Button(label="Open Output Folder")
        self.open_output_btn.set_sensitive(False)
        self.open_output_btn.connect("clicked", self.on_open_output)
        root.append(self.open_output_btn)

    def _append_status(self, text: str) -> None:
        buf = self.status_view.get_buffer()
        buf.insert(buf.get_end_iter(), text + "\n")

    def on_pick_input(self, _btn) -> None:
        dialog = Gtk.FileChooserNative.new(
            "Choose an audio file", self, Gtk.FileChooserAction.OPEN,
            "_Open", "_Cancel",
        )

        def on_response(dlg, response):
            if response == Gtk.ResponseType.ACCEPT:
                gfile = dlg.get_file()
                if gfile:
                    self.input_path = Path(gfile.get_path())
                    self.input_label.set_text(self.input_path.name)
            dlg.destroy()

        dialog.connect("response", on_response)
        dialog.show()

    def on_pick_output(self, _btn) -> None:
        dialog = Gtk.FileChooserNative.new(
            "Choose output folder", self, Gtk.FileChooserAction.SELECT_FOLDER,
            "_Select", "_Cancel",
        )

        def on_response(dlg, response):
            if response == Gtk.ResponseType.ACCEPT:
                gfile = dlg.get_file()
                if gfile:
                    self.output_dir = Path(gfile.get_path())
                    self.output_label.set_text(str(self.output_dir))
            dlg.destroy()

        dialog.connect("response", on_response)
        dialog.show()

    def on_open_output(self, _btn) -> None:
        Gio.AppInfo.launch_default_for_uri(f"file://{self.output_dir}", None)

    def on_run(self, _btn) -> None:
        if self.input_path is None:
            self._append_status("Choose an input file first.")
            return

        from .modeldl import _model_is_cached
        from mel_band_roformer import DEFAULT_MODEL
        if not _model_is_cached(DEFAULT_MODEL):
            dialog = Gtk.AlertDialog()
            dialog.set_message("Download model?")
            dialog.set_detail(
                f"'{DEFAULT_MODEL}' (~913 MB) is not downloaded yet. It "
                "will be fetched from its upstream host and its checksum "
                "verified before use."
            )
            dialog.set_buttons(["Cancel", "Download and continue"])
            dialog.set_cancel_button(0)

            def on_dialog_response(dlg, result):
                try:
                    choice = dlg.choose_finish(result)
                except GLib.Error:
                    return
                if choice == 1:
                    self._start_run(consented_download=True)

            dialog.choose(self, None, on_dialog_response)
            return

        self._start_run(consented_download=False)

    def _start_run(self, consented_download: bool) -> None:
        device = self.device_combo.get_selected_item().get_string()
        self.run_btn.set_sensitive(False)
        self.progress.set_fraction(0.0)
        self.progress.set_text("Running…")
        self._append_status(f"Starting: {self.input_path} -> {self.output_dir} "
                             f"(device={device})")

        def worker():
            cmd = [
                sys.executable, "-m", "melband_roformer_wrapper.cli",
                str(self.input_path),
                "--output-dir", str(self.output_dir),
                "--device", device,
            ]
            if consented_download:
                # User just explicitly confirmed the download dialog above;
                # --yes here only skips the *redundant* terminal prompt,
                # it does not bypass consent.
                cmd.append("--yes")
            proc = subprocess.Popen(cmd, stdout=subprocess.PIPE,
                                     stderr=subprocess.STDOUT, text=True)
            for line in proc.stdout:
                GLib.idle_add(self._append_status, line.rstrip())
            proc.wait()
            GLib.idle_add(self._on_finished, proc.returncode)

        threading.Thread(target=worker, daemon=True).start()

    def _on_finished(self, returncode: int) -> None:
        self.run_btn.set_sensitive(True)
        self.progress.set_fraction(1.0)
        if returncode == 0:
            self.progress.set_text("Done")
            self.open_output_btn.set_sensitive(True)
        else:
            self.progress.set_text(f"Failed (exit {returncode})")


class Application(Gtk.Application):
    def __init__(self):
        super().__init__(application_id="org.debian.melband_roformer")

    def do_activate(self) -> None:
        win = MainWindow(self)
        win.present()


def main(argv: list[str] | None = None) -> int:
    app = Application()
    return app.run(argv or sys.argv)


if __name__ == "__main__":
    sys.exit(main())
