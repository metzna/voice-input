#!/usr/bin/env python3
"""Voice input tool — real-time speech-to-text typed at the cursor position."""

import sys
import argparse
import json
import os
import queue
import subprocess
import threading
import time

VERSION = "0.1.0"

# ---------------------------------------------------------------------------
# Dependency checks
# ---------------------------------------------------------------------------
# Imported lazily / guarded so that --version and --help work on machines
# without a display or GUI toolkit installed.

try:
    import tkinter as tk
    _TK = True
except ImportError:
    tk = None
    _TK = False

try:
    import pyaudio
    _PYAUDIO = True
except ImportError:
    _PYAUDIO = False

try:
    from vosk import Model, KaldiRecognizer
    _VOSK = True
except ImportError:
    _VOSK = False

try:
    from Xlib import X, XK, display as xdisplay
    _XLIB = True
except ImportError:
    _XLIB = False


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

_MODEL_SEARCH_PATHS = [
    os.path.expanduser("~/.local/share/vosk"),
    os.path.expanduser("~/vosk-model"),
    "/usr/share/vosk",
    "/usr/local/share/vosk",
    os.path.join(os.path.dirname(os.path.abspath(__file__)), "model"),
]


def find_vosk_model() -> str | None:
    for base in _MODEL_SEARCH_PATHS:
        if not os.path.isdir(base):
            continue
        # base itself might be a model
        if os.path.exists(os.path.join(base, "am", "final.mdl")):
            return base
        # or a directory containing one or more models
        try:
            for entry in sorted(os.listdir(base)):
                candidate = os.path.join(base, entry)
                if os.path.isdir(candidate) and os.path.exists(
                    os.path.join(candidate, "am", "final.mdl")
                ):
                    return candidate
        except PermissionError:
            continue
    return None


def _run(*cmd, **kwargs) -> subprocess.CompletedProcess:
    return subprocess.run(list(cmd), capture_output=True, **kwargs)


def get_active_window_id() -> str | None:
    """Return the X11 window ID that currently holds input focus."""
    try:
        r = _run("xdotool", "getactivewindow")
        if r.returncode == 0:
            return r.stdout.decode().strip()
    except FileNotFoundError:
        pass
    return None


def type_text_to_window(text: str, window_id: str | None) -> None:
    """Type *text* into *window_id* (X11) or the active window as a fallback."""
    if not text.strip():
        return

    # X11 path — xdotool
    try:
        if window_id:
            # Focus target, type, done. We purposely do NOT re-focus our own
            # window afterwards; the tiny refocus flash is less annoying than
            # fighting the WM on every utterance.
            _run(
                "xdotool",
                "windowfocus", "--sync", window_id,
                "type", "--clearmodifiers", "--delay", "12", "--", text,
            )
        else:
            _run(
                "xdotool",
                "type", "--clearmodifiers", "--delay", "12", "--", text,
            )
        return
    except FileNotFoundError:
        pass

    # Wayland path — wtype
    try:
        _run("wtype", "--", text)
        return
    except FileNotFoundError:
        pass

    # Wayland path — ydotool
    try:
        _run("ydotool", "type", "--", text)
        return
    except FileNotFoundError:
        pass

    print(
        "warning: could not type text — install xdotool (X11) or wtype / ydotool (Wayland)",
        file=sys.stderr,
    )


# ---------------------------------------------------------------------------
# Overlay window
# ---------------------------------------------------------------------------

_DOT_COLORS = [
    "#f38ba8", "#fab387", "#f9e2af",
    "#a6e3a1", "#89dceb", "#89b4fa", "#b4befe",
]

_BG      = "#1e1e2e"
_FG      = "#cdd6f4"
_FG_DIM  = "#585b70"
_FG_HINT = "#a6adc8"


class VoiceOverlay:
    """Small always-on-top floating window that shows recognition state."""

    SAMPLE_RATE = 16000
    CHUNK       = 8000

    def __init__(self, model_path: str | None = None) -> None:
        self._model_path   = model_path
        self._audio_q: queue.Queue[bytes] = queue.Queue()
        self._running      = False
        self._target_wid: str | None = None
        self._root: tk.Tk | None = None

        # Labels (set during _build_ui)
        self._lbl_status: tk.Label
        self._lbl_partial: tk.Label
        self._lbl_dot: tk.Label

    # ------------------------------------------------------------------
    # Window
    # ------------------------------------------------------------------

    def _build_ui(self) -> None:
        root = tk.Tk()
        self._root = root

        root.title("Voice Input")
        root.resizable(False, False)
        root.configure(bg=_BG)

        # Stay a *managed* window so the WM can move / rule / focus it, but
        # hint it as a utility window. Tiling WMs (i3, bspwm, Hyprland, …)
        # honour _NET_WM_WINDOW_TYPE and float utility windows instead of
        # tiling them, while still keeping them under WM control.
        try:
            root.attributes("-type", "utility")
        except tk.TclError:
            pass  # non-X11 / unsupported — degrade gracefully
        root.attributes("-topmost", True)

        # Position: horizontally centred, near the top of the screen.
        W, H = 320, 72
        sw = root.winfo_screenwidth()
        root.geometry(f"{W}x{H}+{(sw - W) // 2}+48")

        # Rounded-ish border via a 1-px coloured frame
        border = tk.Frame(root, bg="#313244", padx=1, pady=1)
        border.pack(fill=tk.BOTH, expand=True)

        inner = tk.Frame(border, bg=_BG, padx=12, pady=8)
        inner.pack(fill=tk.BOTH, expand=True)

        # Top row
        top = tk.Frame(inner, bg=_BG)
        top.pack(fill=tk.X)

        self._lbl_dot = tk.Label(top, text="●", bg=_BG, fg=_DOT_COLORS[0],
                                 font=("monospace", 10))
        self._lbl_dot.pack(side=tk.LEFT)

        self._lbl_status = tk.Label(top, text="Loading…", bg=_BG, fg=_FG,
                                    font=("sans-serif", 10, "bold"))
        self._lbl_status.pack(side=tk.LEFT, padx=(6, 0))

        tk.Label(top, text="ESC to close", bg=_BG, fg=_FG_DIM,
                 font=("monospace", 8)).pack(side=tk.RIGHT)

        # Partial text preview
        self._lbl_partial = tk.Label(inner, text="", bg=_BG, fg=_FG_HINT,
                                     font=("sans-serif", 9), anchor="w",
                                     wraplength=296, justify=tk.LEFT)
        self._lbl_partial.pack(fill=tk.X, pady=(4, 0))

        # Local ESC binding — works when the overlay itself happens to hold
        # focus. The global Super+Esc grab (see _global_hotkey_loop) is what
        # closes it during dictation, when focus lives in the target window.
        root.bind("<Escape>", lambda _e: self.stop())

        # Pulse the dot
        self._dot_idx = 0
        self._pulse()

    def _pulse(self) -> None:
        if not self._running or self._root is None:
            return
        self._dot_idx = (self._dot_idx + 1) % len(_DOT_COLORS)
        self._lbl_dot.configure(fg=_DOT_COLORS[self._dot_idx])
        self._root.after(380, self._pulse)

    # ------------------------------------------------------------------
    # Thread: global hotkey (Super+Esc)
    # ------------------------------------------------------------------

    def _global_hotkey_loop(self) -> None:
        """Grab Super+Esc on the root window so the overlay closes from
        anywhere, without swallowing the bare Escape key system-wide."""
        if not _XLIB:
            print(
                "warning: python-xlib not installed — global Super+Esc disabled "
                "(click the window and press Esc to close). pip install python-xlib",
                file=sys.stderr,
            )
            return

        try:
            disp = xdisplay.Display()
            root = disp.screen().root
            keycode = disp.keysym_to_keycode(XK.XK_Escape)
            modifier = X.Mod4Mask  # Super / Meta

            # Re-grab under each lock-key combination so NumLock / CapsLock
            # being on doesn't break the binding.
            lock_variants = [
                0,
                X.LockMask,                 # CapsLock
                X.Mod2Mask,                 # NumLock
                X.LockMask | X.Mod2Mask,
            ]
            root.change_attributes(event_mask=X.KeyPressMask)
            for lv in lock_variants:
                root.grab_key(keycode, modifier | lv, True,
                              X.GrabModeAsync, X.GrabModeAsync)
            disp.sync()
        except Exception as exc:
            print(f"warning: could not register global hotkey: {exc}",
                  file=sys.stderr)
            return

        try:
            while self._running:
                if disp.pending_events() == 0:
                    time.sleep(0.05)
                    continue
                event = disp.next_event()
                if event.type == X.KeyPress and event.detail == keycode:
                    self._ui(self.stop)
                    break
        finally:
            try:
                for lv in lock_variants:
                    root.ungrab_key(keycode, modifier | lv)
                disp.sync()
                disp.close()
            except Exception:
                pass

    # ------------------------------------------------------------------
    # Thread: audio capture
    # ------------------------------------------------------------------

    def _audio_loop(self) -> None:
        p = pyaudio.PyAudio()
        stream = p.open(
            format=pyaudio.paInt16,
            channels=1,
            rate=self.SAMPLE_RATE,
            input=True,
            frames_per_buffer=self.CHUNK,
        )
        try:
            while self._running:
                data = stream.read(self.CHUNK, exception_on_overflow=False)
                self._audio_q.put(data)
        finally:
            stream.stop_stream()
            stream.close()
            p.terminate()

    # ------------------------------------------------------------------
    # Thread: recognition
    # ------------------------------------------------------------------

    def _recognition_loop(self, model_path: str) -> None:
        try:
            model = Model(model_path)
            rec   = KaldiRecognizer(model, self.SAMPLE_RATE)
            rec.SetWords(True)
        except Exception as exc:
            self._ui(self._on_error, f"Model load failed: {exc}")
            return

        self._ui(self._on_ready)

        while self._running:
            try:
                data = self._audio_q.get(timeout=0.1)
            except queue.Empty:
                continue

            if rec.AcceptWaveform(data):
                text = json.loads(rec.Result()).get("text", "").strip()
                if text:
                    self._ui(self._on_final, text)
            else:
                partial = json.loads(rec.PartialResult()).get("partial", "").strip()
                self._ui(self._on_partial, partial)

    # ------------------------------------------------------------------
    # UI callbacks (always called via root.after → main thread)
    # ------------------------------------------------------------------

    def _ui(self, fn, *args) -> None:
        if self._root and self._running:
            self._root.after(0, fn, *args)

    def _on_ready(self) -> None:
        self._lbl_status.configure(text="Listening…", fg=_FG)

    def _on_partial(self, text: str) -> None:
        display = text if len(text) <= 45 else "…" + text[-44:]
        self._lbl_partial.configure(text=display)

    def _on_final(self, text: str) -> None:
        self._lbl_partial.configure(text="")
        # Type in background so the UI stays responsive
        threading.Thread(
            target=type_text_to_window,
            args=(text + " ", self._target_wid),
            daemon=True,
        ).start()

    def _on_error(self, msg: str) -> None:
        self._lbl_status.configure(text="Error", fg="#f38ba8")
        self._lbl_partial.configure(text=msg[:60])

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def run(self) -> None:
        if not _TK:
            sys.exit(
                "error: tkinter not available — install it via your package "
                "manager (e.g. apt install python3-tk)."
            )
        if not _PYAUDIO:
            sys.exit("error: pyaudio not installed — run: pip install pyaudio")
        if not _VOSK:
            sys.exit("error: vosk not installed — run: pip install vosk")

        model_path = self._model_path or find_vosk_model()
        if not model_path:
            sys.exit(
                "error: no Vosk model found.\n"
                "Download a model from https://alphacephei.com/vosk/models\n"
                "and extract it to ~/.local/share/vosk/<model-name>/\n"
                "or pass --model <path>."
            )

        # Record the window that currently has focus BEFORE we create ours.
        self._target_wid = get_active_window_id()

        self._running = True
        self._build_ui()

        threading.Thread(target=self._audio_loop, daemon=True).start()
        threading.Thread(target=self._recognition_loop, args=(model_path,),
                         daemon=True).start()
        threading.Thread(target=self._global_hotkey_loop, daemon=True).start()

        # Some WMs focus a freshly-mapped window. Hand focus back to the
        # target *once* so dictation lands there immediately; from then on the
        # overlay never grabs focus, so the WM can still control the window
        # (move it, focus it on demand) without disturbing the target.
        if self._target_wid:
            self._root.after(120, lambda: _run("xdotool", "windowfocus",
                                               "--sync", self._target_wid))

        assert self._root is not None
        self._root.mainloop()

    def stop(self) -> None:
        self._running = False
        if self._root:
            self._root.destroy()


# ---------------------------------------------------------------------------
# Entry point
# ---------------------------------------------------------------------------

def build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(
        prog="voice-input",
        description="Real-time voice input — types recognised speech at the cursor position.",
    )
    p.add_argument("--version", action="version", version=f"%(prog)s {VERSION}")
    p.add_argument(
        "--model", metavar="PATH",
        help="Path to a Vosk model directory (auto-detected if omitted).",
    )
    return p


def main() -> None:
    args = build_parser().parse_args()
    VoiceOverlay(model_path=args.model).run()


if __name__ == "__main__":
    main()
