"""
Elevator Data Collection app  (MIE 1050 - "Which Elevator Am I In?", Part 1)

Pairs with the ElevatorLogger_v2 firmware, which streams CSV sensor data over USB serial.
This one desktop app does two things at once:

  1. Capture: reads the board's serial stream and writes every line to a timestamped raw.csv.
  2. Label:   you set the direction and the destination buttons pressed inside the cabin; then
              each time the doors open you tap DOORS OPEN and the app infers which floor you
              arrived at (the next requested stop in the current direction) and advances. A
              correction nudge is always available for unexpected (hall-call) stops.

Alignment: every raw row is written as  host_epoch,<original serial line>  and every event row
stores both host_epoch and the latest t_us seen from the stream. host_epoch is the common clock
between raw.csv and events.csv; t_us is the precise sensor-timeline anchor.

Runs on a Windows laptop or a Surface tablet (tkinter; touch taps register as clicks).
Only external dependency: pyserial.
"""

import json
import os
import threading
import time
from datetime import datetime

import tkinter as tk
from tkinter import ttk, messagebox

try:
    import serial
    from serial.tools import list_ports
except ImportError:  # pragma: no cover - surfaced to the user at startup
    serial = None
    list_ports = None


BASE_DIR = os.path.dirname(os.path.abspath(__file__))
CONFIG_PATH = os.path.join(BASE_DIR, "config.json")
SESSIONS_DIR = os.path.join(BASE_DIR, "sessions")

DEFAULT_CONFIG = {
    "baud": 1000000,
    "buildings": {
        "Hart House":      ["B", "G", "1", "2", "3"],
        "Mechanical (MC)": ["B", "G", "1", "2", "3", "4", "5"],
        "Myhal":           ["B", "1", "2", "3", "4", "5", "6", "7"],
        "Rosebrugh":       ["B", "G", "1", "2", "3", "4", "5"],
    },
    # Generic placeholders - correct the floor lists per building on-site by editing config.json.
    "custom_events": ["Bike", "Mobility device", "Other heavy object"],
}

# ---- palette (modern dark dashboard) ----
BG      = "#0f1222"   # app background
CARD    = "#1a1f36"   # panels
INPUT   = "#252b45"   # entry/combobox fields
TEXT    = "#e8ebf5"
MUTED   = "#9aa3c4"
GREEN   = "#22c55e"; GREEN_H = "#16a34a"   # doors open
BLUE    = "#3b82f6"; BLUE_H  = "#2563eb"   # doors close / direction active
INDIGO  = "#6366f1"; INDIGO_H = "#4f46e5"  # pressed destination
SLATE   = "#3a4166"; SLATE_H = "#4a5280"   # idle buttons
AMBER   = "#f59e0b"; AMBER_H = "#d97706"   # other / undo
RED     = "#ef4444"; RED_H   = "#dc2626"   # stop

TITLE_FONT = ("Segoe UI Semibold", 20)
H_FONT     = ("Segoe UI Semibold", 12)
BTN_FONT   = ("Segoe UI", 13)
BIG_FONT   = ("Segoe UI Semibold", 15)
FLOOR_FONT = ("Segoe UI", 14, "bold")
HUGE_FONT  = ("Segoe UI", 64, "bold")
LOG_FONT   = ("Consolas", 10)

COMMAND_KEYS = {"o", "c", "u", "d", "i", "k", "z"}


def load_config():
    """Load config.json, writing the defaults on first run or if the file is unreadable."""
    if not os.path.exists(CONFIG_PATH):
        with open(CONFIG_PATH, "w", encoding="utf-8") as f:
            json.dump(DEFAULT_CONFIG, f, indent=2)
        return json.loads(json.dumps(DEFAULT_CONFIG))
    try:
        with open(CONFIG_PATH, "r", encoding="utf-8") as f:
            cfg = json.load(f)
    except (json.JSONDecodeError, OSError):
        return json.loads(json.dumps(DEFAULT_CONFIG))
    for key, val in DEFAULT_CONFIG.items():
        cfg.setdefault(key, val)
    return cfg


def _safe_bg(btn, color):
    """Recolour a button, ignoring the TclError raised if it is mid-teardown (e.g. a <Leave>
    event fires while the window is closing)."""
    try:
        btn.config(bg=color)
    except tk.TclError:
        pass


def make_btn(parent, text, command, bg, hover, fg="white", font=BTN_FONT, **grid):
    """A flat, hover-highlighted button with a stored base colour so it can be recoloured later."""
    b = tk.Button(parent, text=text, command=command, bg=bg, fg=fg,
                  activebackground=hover, activeforeground=fg, relief="flat", bd=0,
                  font=font, cursor="hand2", highlightthickness=0,
                  padx=grid.pop("padx", 10), pady=grid.pop("pady", 8))
    b._bg, b._hover = bg, hover
    b.bind("<Enter>", lambda e: _safe_bg(b, b._hover))
    b.bind("<Leave>", lambda e: _safe_bg(b, b._bg))
    return b


def recolor(btn, bg, hover):
    btn._bg, btn._hover = bg, hover
    btn.config(bg=bg, activebackground=hover)


class SharedState:
    """Thread-safe snapshot of capture status, written by the reader, read by the GUI."""

    def __init__(self):
        self._lock = threading.Lock()
        self.connected = False
        self.last_t_us = None
        self.fast_rows = 0
        self.slow_rows = 0
        self.missed_ticks = 0
        self.bad_lines = 0
        self.first_data_seen = False
        self.connect_time = None
        self.error = None

    def snapshot(self):
        with self._lock:
            return dict(self.__dict__)

    def get_last_t_us(self):
        with self._lock:
            return self.last_t_us


class SerialReader(threading.Thread):
    """Reads serial lines, writes raw.csv, parses header comments, updates SharedState."""

    def __init__(self, port, baud, raw_path, state, header_lines):
        super().__init__(daemon=True)
        self.port = port
        self.baud = baud
        self.raw_path = raw_path
        self.state = state
        self.header_lines = header_lines
        # NB: not "_stop" — threading.Thread._stop is an internal method that join() calls, and
        # shadowing it with an Event makes join() raise "'Event' object is not callable".
        self._stop_evt = threading.Event()

    def stop(self):
        self._stop_evt.set()

    def run(self):
        try:
            ser = serial.Serial(self.port, self.baud, timeout=0.5)
        except Exception as exc:  # pragma: no cover - hardware dependent
            with self.state._lock:
                self.state.error = f"Could not open {self.port}: {exc}"
                self.state.connected = False
            return

        with self.state._lock:
            self.state.connected = True
            self.state.connect_time = time.time()
            self.state.error = None

        last_flush = time.time()
        try:
            with open(self.raw_path, "w", encoding="utf-8", buffering=1 << 16) as raw:
                while not self._stop_evt.is_set():
                    try:
                        raw_bytes = ser.readline()
                    except Exception as exc:
                        with self.state._lock:
                            self.state.connected = False
                            self.state.error = f"Serial read failed: {exc}"
                        break
                    now = time.time()
                    if raw_bytes:
                        line = raw_bytes.decode("utf-8", errors="replace").strip()
                        if line:
                            self._handle_line(line, now, raw)
                    if now - last_flush >= 1.0:
                        raw.flush()
                        last_flush = now
                raw.flush()
        finally:
            try:
                ser.close()
            except Exception:
                pass
            with self.state._lock:
                self.state.connected = False

    # expected comma-separated field count per record type (F=8, S=5, M=3)
    FIELD_COUNT = {"F": 8, "S": 5, "M": 3}

    def _handle_line(self, line, host_epoch, raw):
        if line.startswith("#"):
            # Keep every comment line in the session header; flag hard errors.
            raw.write(line + "\n")
            self.header_lines.append(line.lstrip("# ").rstrip())
            if "ERROR" in line.upper():
                with self.state._lock:
                    self.state.error = line.strip()
            return

        rec = line[0]
        expected = self.FIELD_COUNT.get(rec)
        parts = line.split(",")
        if expected is None or len(parts) != expected:
            # Boot-ROM garbage (opening the port resets the ESP32) or a truncated/partial row.
            # Ignore anything that is not a well-formed F/S/M row; just count it.
            with self.state._lock:
                self.state.bad_lines += 1
            return

        raw.write(f"{host_epoch:.6f},{line}\n")

        try:
            t_us = int(parts[1])
        except ValueError:
            t_us = None

        with self.state._lock:
            if t_us is not None:
                self.state.last_t_us = t_us
            self.state.first_data_seen = True
            if rec == "F":
                self.state.fast_rows += 1
            elif rec == "S":
                self.state.slow_rows += 1
            elif rec == "M":
                try:
                    self.state.missed_ticks += int(parts[2])
                except ValueError:
                    pass


class EventLogger:
    """Writes events.csv (host_epoch,iso_time,t_us_at_event,event_type,value,note). Supports undo/amend."""

    HEADER = "host_epoch,iso_time,t_us_at_event,event_type,value,note"

    def __init__(self, path, get_t_us):
        self.path = path
        self.get_t_us = get_t_us
        self.rows = []  # list of lists matching HEADER columns
        self._write_all()

    @staticmethod
    def _csv_escape(value):
        s = "" if value is None else str(value)
        if any(ch in s for ch in [",", '"', "\n"]):
            s = '"' + s.replace('"', '""') + '"'
        return s

    def _write_all(self):
        with open(self.path, "w", encoding="utf-8", newline="") as f:
            f.write(self.HEADER + "\n")
            for row in self.rows:
                f.write(",".join(self._csv_escape(c) for c in row) + "\n")
            f.flush()
            os.fsync(f.fileno())

    def log(self, event_type, value="", note=""):
        host = time.time()
        iso = datetime.fromtimestamp(host).isoformat(timespec="milliseconds")
        t_us = self.get_t_us()
        row = [f"{host:.6f}", iso, "" if t_us is None else t_us, event_type, value, note]
        self.rows.append(row)
        with open(self.path, "a", encoding="utf-8", newline="") as f:
            f.write(",".join(self._csv_escape(c) for c in row) + "\n")
            f.flush()
            os.fsync(f.fileno())
        return row

    def amend_last(self, event_type, new_value):
        for row in reversed(self.rows):
            if row[3] == event_type:
                row[4] = new_value
                self._write_all()
                return True
        return False

    def set_last_note(self, note):
        if self.rows:
            self.rows[-1][5] = note
            self._write_all()

    def undo(self):
        if not self.rows:
            return None
        removed = self.rows.pop()
        self._write_all()
        return removed


class App(tk.Tk):
    def __init__(self, config):
        super().__init__()
        self.config_data = config
        self.title("Elevator Data Collector")
        self.geometry("1060x760")
        self.minsize(900, 680)
        self.configure(bg=BG)

        self._setup_ttk_style()

        self.state_obj = SharedState()
        self.reader = None
        self.event_logger = None
        self.session_dir = None
        self.header_lines = []
        self.capturing = False

        # ride state
        self.current_floor = ""
        self.direction = ""          # "up" / "down"
        self.pressed = set()         # destination floor labels currently requested
        self.dest_buttons = {}       # floor label -> button widget
        self.dir_buttons = {}        # "up"/"down" -> button widget
        self.other_buttons = {}      # custom-event name -> button widget
        self.other_on = set()        # custom-event items currently on board
        self.occupancy = 0           # running count of people currently on board

        self.cur_floor_var = tk.StringVar(value="--")
        self.dir_var = tk.StringVar(value="direction: -")
        self.occ_var = tk.StringVar(value="0")

        self._closing = False
        self._poll_id = None

        self._build_header()
        self._build_status()
        self._build_body()
        self._bind_keys()

        self.protocol("WM_DELETE_WINDOW", self._on_close)
        self._poll_id = self.after(200, self._poll_status)

    # ---------- styling ----------
    def _setup_ttk_style(self):
        style = ttk.Style(self)
        try:
            style.theme_use("clam")
        except tk.TclError:
            pass
        style.configure(".", background=CARD, foreground=TEXT, font=("Segoe UI", 11))
        style.configure("TFrame", background=CARD)
        style.configure("Bg.TFrame", background=BG)
        style.configure("TLabel", background=CARD, foreground=MUTED)
        style.configure("Card.TLabel", background=CARD, foreground=MUTED)
        style.configure("TEntry", fieldbackground=INPUT, foreground=TEXT, bordercolor=INPUT,
                        insertcolor=TEXT, padding=4)
        style.configure("TCombobox", fieldbackground=INPUT, background=INPUT, foreground=TEXT,
                        arrowcolor=TEXT, bordercolor=INPUT, padding=4)
        style.map("TCombobox", fieldbackground=[("readonly", INPUT)],
                  foreground=[("readonly", TEXT)])
        self.option_add("*TCombobox*Listbox.background", INPUT)
        self.option_add("*TCombobox*Listbox.foreground", TEXT)
        self.option_add("*TCombobox*Listbox.selectBackground", INDIGO)

    def _card(self, parent, **kw):
        return tk.Frame(parent, bg=CARD, bd=0, highlightthickness=1,
                        highlightbackground="#2b3254", **kw)

    # ---------- header / setup ----------
    def _build_header(self):
        bar = tk.Frame(self, bg=BG)
        bar.pack(fill="x", padx=16, pady=(14, 8))
        tk.Label(bar, text="🛗  Elevator Data Collector", bg=BG, fg=TEXT,
                 font=TITLE_FONT).pack(side="left")
        self.start_btn = make_btn(bar, "START  (Space)", self._toggle_capture, GREEN, GREEN_H,
                                  font=BIG_FONT, padx=22, pady=10)
        self.start_btn.pack(side="right")

        setup = self._card(self)
        setup.pack(fill="x", padx=16, pady=4)
        inner = tk.Frame(setup, bg=CARD)
        inner.pack(fill="x", padx=14, pady=12)

        def label(txt, c, r=0):
            tk.Label(inner, text=txt, bg=CARD, fg=MUTED, font=("Segoe UI", 10)).grid(
                row=r, column=c, sticky="w", padx=(0, 6), pady=(0, 2))

        label("Port", 0); label("Baud", 1); label("Building", 2); label("Car", 3)
        label("Board orientation", 4); label("Operator / notes", 5)

        # Editable (state="normal") so a port that Windows doesn't enumerate can be typed in
        # directly, e.g. COM6. postcommand re-scans every time the dropdown is opened.
        self.port_cb = ttk.Combobox(inner, width=14, values=self._list_ports(),
                                    postcommand=self._refresh_ports)
        self.port_cb.grid(row=1, column=0, sticky="w", padx=(0, 8))
        if self.port_cb["values"]:
            self.port_cb.current(0)
        self.baud_cb = ttk.Combobox(inner, width=9, state="normal",
                                    values=["9600", "115200", "230400", "460800", "921600", "1000000"])
        self.baud_cb.set(str(self.config_data.get("baud", 921600)))
        self.baud_cb.grid(row=1, column=1, sticky="w", padx=(0, 8))
        buildings = list(self.config_data.get("buildings", {}).keys())
        self.building_cb = ttk.Combobox(inner, width=16, state="readonly", values=buildings)
        if buildings:
            self.building_cb.current(0)
        self.building_cb.grid(row=1, column=2, sticky="w", padx=(0, 8))
        self.building_cb.bind("<<ComboboxSelected>>", lambda e: self._rebuild_destinations())
        self.car_entry = ttk.Entry(inner, width=10)
        self.car_entry.grid(row=1, column=3, sticky="w", padx=(0, 8))
        self.orient_entry = ttk.Entry(inner, width=18)
        self.orient_entry.grid(row=1, column=4, sticky="w", padx=(0, 8))
        self.operator_entry = ttk.Entry(inner, width=22)
        self.operator_entry.grid(row=1, column=5, sticky="w", padx=(0, 8))
        self.refresh_btn = make_btn(inner, "⟳", self._refresh_ports, SLATE, SLATE_H, padx=8, pady=2)
        self.refresh_btn.grid(row=1, column=6, sticky="w")

    def _build_status(self):
        self.status_var = tk.StringVar(value="Idle — not capturing.")
        self.status_lbl = tk.Label(self, textvariable=self.status_var, bg=BG, fg=MUTED,
                                   font=("Segoe UI", 10), anchor="w")
        self.status_lbl.pack(fill="x", padx=18, pady=(2, 0))
        self.banner_var = tk.StringVar(value="")
        self.banner = tk.Label(self, textvariable=self.banner_var, bg=RED, fg="white",
                               font=("Segoe UI Semibold", 11), anchor="w", padx=10, pady=4)

    # ---------- body ----------
    def _build_body(self):
        body = tk.Frame(self, bg=BG)
        body.pack(fill="both", expand=True, padx=16, pady=8)
        body.columnconfigure(0, weight=3)
        body.columnconfigure(1, weight=4)
        body.rowconfigure(0, weight=1)

        self._build_ride_panel(body)
        self._build_input_panel(body)
        self._build_log_panel()

    def _build_ride_panel(self, parent):
        card = self._card(parent)
        card.grid(row=0, column=0, sticky="nsew", padx=(0, 8))
        tk.Label(card, text="CURRENT FLOOR", bg=CARD, fg=MUTED,
                 font=("Segoe UI", 10)).pack(pady=(16, 0))

        row = tk.Frame(card, bg=CARD)
        row.pack(pady=4)
        make_btn(row, "▼", lambda: self._nudge_floor(-1), SLATE, SLATE_H,
                 font=("Segoe UI", 20), padx=16, pady=4).pack(side="left", padx=8)
        tk.Label(row, textvariable=self.cur_floor_var, bg=CARD, fg=TEXT,
                 font=HUGE_FONT, width=3).pack(side="left")
        make_btn(row, "▲", lambda: self._nudge_floor(1), SLATE, SLATE_H,
                 font=("Segoe UI", 20), padx=16, pady=4).pack(side="left", padx=8)

        tk.Label(card, textvariable=self.dir_var, bg=CARD, fg=MUTED,
                 font=("Segoe UI", 11, "italic")).pack()

        dirrow = tk.Frame(card, bg=CARD)
        dirrow.pack(pady=(14, 6))
        self.dir_buttons["up"] = make_btn(dirrow, "▲ UP  (U)", lambda: self._set_direction("up"),
                                          SLATE, SLATE_H, font=BIG_FONT, padx=16, pady=10)
        self.dir_buttons["up"].pack(side="left", padx=6)
        self.dir_buttons["down"] = make_btn(dirrow, "▼ DOWN  (D)", lambda: self._set_direction("down"),
                                            SLATE, SLATE_H, font=BIG_FONT, padx=16, pady=10)
        self.dir_buttons["down"].pack(side="left", padx=6)

        make_btn(card, "DOORS  OPEN  (O)", self._doors_open, GREEN, GREEN_H,
                 font=("Segoe UI Semibold", 18), padx=20, pady=18).pack(fill="x", padx=20, pady=(16, 6))
        make_btn(card, "DOORS  CLOSE  (C)", self._doors_close, BLUE, BLUE_H,
                 font=("Segoe UI Semibold", 18), padx=20, pady=16).pack(fill="x", padx=20, pady=(0, 18))

    def _build_input_panel(self, parent):
        col = tk.Frame(parent, bg=BG)
        col.grid(row=0, column=1, sticky="nsew")
        col.rowconfigure(0, weight=1)
        col.columnconfigure(0, weight=1)

        dest = self._card(col)
        dest.grid(row=0, column=0, sticky="nsew", pady=(0, 8))
        tk.Label(dest, text="BUTTONS PRESSED  (tap to request a floor)", bg=CARD, fg=MUTED,
                 font=("Segoe UI", 10)).pack(anchor="w", padx=14, pady=(12, 6))
        self.dest_grid = tk.Frame(dest, bg=CARD)
        self.dest_grid.pack(padx=12, pady=(0, 12))

        ppl = self._card(col)
        ppl.grid(row=1, column=0, sticky="ew", pady=(0, 8))
        phead = tk.Frame(ppl, bg=CARD)
        phead.pack(fill="x", padx=14, pady=(12, 6))
        tk.Label(phead, text="PEOPLE  (I = in +1,  K = out +1)", bg=CARD, fg=MUTED,
                 font=("Segoe UI", 10)).pack(side="left")
        tk.Label(phead, text="on board", bg=CARD, fg=MUTED,
                 font=("Segoe UI", 10)).pack(side="right", padx=(6, 0))
        tk.Label(phead, textvariable=self.occ_var, bg=CARD, fg=GREEN,
                 font=("Segoe UI", 22, "bold")).pack(side="right")
        pr = tk.Frame(ppl, bg=CARD)
        pr.pack(padx=12, pady=(0, 12), anchor="w")
        tk.Label(pr, text="IN", bg=CARD, fg=TEXT, font=H_FONT, width=4).grid(row=0, column=0)
        tk.Label(pr, text="OUT", bg=CARD, fg=TEXT, font=H_FONT, width=4).grid(row=1, column=0)
        for i, n in enumerate([1, 2, 3, 4, 5], start=1):
            make_btn(pr, f"+{n}", lambda n=n: self._log_people("people_in", n), SLATE, SLATE_H,
                     padx=12, pady=6).grid(row=0, column=i, padx=3, pady=3)
            make_btn(pr, f"+{n}", lambda n=n: self._log_people("people_out", n), SLATE, SLATE_H,
                     padx=12, pady=6).grid(row=1, column=i, padx=3, pady=3)

        other = self._card(col)
        other.grid(row=2, column=0, sticky="ew")
        tk.Label(other, text="OTHER EVENT / CONDITION  (tap = entered, tap again = left)",
                 bg=CARD, fg=MUTED, font=("Segoe UI", 10)).pack(anchor="w", padx=14, pady=(12, 6))
        orow = tk.Frame(other, bg=CARD)
        orow.pack(padx=12, pady=(0, 8), anchor="w")
        for i, name in enumerate(self.config_data.get("custom_events", [])):
            b = make_btn(orow, name, lambda name=name: self._toggle_other(name), SLATE, SLATE_H,
                         padx=14, pady=8)
            b.grid(row=0, column=i, padx=4)
            self.other_buttons[name] = b
        trow = tk.Frame(other, bg=CARD)
        trow.pack(fill="x", padx=12, pady=(0, 12))
        self.other_entry = ttk.Entry(trow)
        self.other_entry.pack(side="left", fill="x", expand=True, padx=(0, 6))
        make_btn(trow, "Log text", self._log_other_text, SLATE, SLATE_H, padx=12, pady=6).pack(side="left")

        self._rebuild_destinations()

    def _build_log_panel(self):
        card = self._card(self)
        card.pack(fill="both", expand=False, padx=16, pady=(0, 14))
        head = tk.Frame(card, bg=CARD)
        head.pack(fill="x", padx=14, pady=(10, 4))
        tk.Label(head, text="EVENT LOG", bg=CARD, fg=MUTED, font=("Segoe UI", 10)).pack(side="left")
        make_btn(head, "⟲ UNDO last  (Z)", self._undo, AMBER, AMBER_H, padx=12, pady=4).pack(side="right")
        body = tk.Frame(card, bg=CARD)
        body.pack(fill="both", expand=True, padx=14, pady=(0, 12))
        self.log_text = tk.Text(body, height=8, font=LOG_FONT, state="disabled", wrap="none",
                                bg=INPUT, fg=TEXT, relief="flat", bd=0, highlightthickness=0,
                                insertbackground=TEXT)
        self.log_text.pack(side="left", fill="both", expand=True)
        sb = ttk.Scrollbar(body, command=self.log_text.yview)
        sb.pack(side="left", fill="y")
        self.log_text.config(yscrollcommand=sb.set)

    def _bind_keys(self):
        self.bind("<space>", lambda e: self._toggle_capture())
        self.bind("<Up>", lambda e: self._nudge_floor(1))
        self.bind("<Down>", lambda e: self._nudge_floor(-1))
        self.bind("<Key>", self._on_key)

    # ---------- destinations keypad ----------
    def _current_floors(self):
        return self.config_data.get("buildings", {}).get(self.building_cb.get(), [])

    def _rebuild_destinations(self):
        self.pressed = set()
        self.dest_buttons = {}
        for w in self.dest_grid.winfo_children():
            w.destroy()
        for i, fl in enumerate(self._current_floors()):
            b = make_btn(self.dest_grid, fl, lambda fl=fl: self._toggle_dest(fl),
                         SLATE, SLATE_H, font=FLOOR_FONT, padx=4, pady=12)
            b.config(width=3)
            b.grid(row=i // 6, column=i % 6, padx=4, pady=4)
            self.dest_buttons[fl] = b

    def _toggle_dest(self, fl):
        # Destination buttons only drive the on-screen floor prediction; they are intentionally
        # NOT written to events.csv (the floors actually visited are captured by door_open/close).
        if not self._ensure_capturing():
            return
        if fl in self.pressed:
            self.pressed.discard(fl)
            recolor(self.dest_buttons[fl], SLATE, SLATE_H)
        else:
            self.pressed.add(fl)
            recolor(self.dest_buttons[fl], INDIGO, INDIGO_H)

    # ---------- floor inference ----------
    def _predict_next_floor(self):
        floors = self._current_floors()
        if not floors:
            return self.current_floor
        if self.current_floor not in floors:
            return floors[0]
        i = floors.index(self.current_floor)
        pressed_idx = sorted(floors.index(f) for f in self.pressed if f in floors)
        if self.direction == "up":
            above = [j for j in pressed_idx if j > i]
            if above:
                return floors[above[0]]
            if i + 1 < len(floors):
                return floors[i + 1]
        elif self.direction == "down":
            below = [j for j in pressed_idx if j < i]
            if below:
                return floors[below[-1]]
            if i - 1 >= 0:
                return floors[i - 1]
        if pressed_idx:
            return floors[min(pressed_idx, key=lambda j: abs(j - i))]
        return self.current_floor

    def _set_floor(self, fl):
        self.current_floor = fl
        self.cur_floor_var.set(fl if fl else "--")

    def _nudge_floor(self, step):
        """Correct the current floor up/down by one physical level."""
        if not self.capturing:
            return
        floors = self._current_floors()
        if not floors:
            return
        i = floors.index(self.current_floor) if self.current_floor in floors else 0
        i = max(0, min(len(floors) - 1, i + step))
        self._set_floor(floors[i])
        # keep the most recent door_open label in sync with the correction
        if self.event_logger and self.event_logger.amend_last("door_open", floors[i]):
            self._append_log("corrected →", floors[i])

    # ---------- ride events ----------
    def _set_direction(self, direction):
        if not self._ensure_capturing():
            return
        self.direction = direction
        self.dir_var.set(f"direction: {direction}")
        for key, btn in self.dir_buttons.items():
            if key == direction:
                recolor(btn, BLUE, BLUE_H)
            else:
                recolor(btn, SLATE, SLATE_H)
        self.event_logger.log("direction", direction)
        self._append_log("direction", direction)

    def _doors_open(self):
        if not self._ensure_capturing():
            return
        predicted = self._predict_next_floor()
        self._set_floor(predicted)
        self.event_logger.log("door_open", predicted)
        self._append_log("door_open", predicted, "(▲/▼ to correct)")

    def _doors_close(self):
        if not self._ensure_capturing():
            return
        self.event_logger.log("door_close", self.current_floor)
        self._append_log("door_close", self.current_floor)
        # the floor we just visited has been served — clear its request if it was pressed
        if self.current_floor in self.pressed:
            self.pressed.discard(self.current_floor)
            recolor(self.dest_buttons[self.current_floor], SLATE, SLATE_H)

    def _log_people(self, which, count):
        if not self._ensure_capturing():
            return
        self.event_logger.log(which, count)
        self._recompute_occupancy()
        self.event_logger.set_last_note(f"on_board={self.occupancy}")
        self._append_log(which, count, f"(on board: {self.occupancy})")

    def _recompute_occupancy(self):
        """Single source of truth: derive on-board count from the logged people events."""
        total = 0
        if self.event_logger:
            for row in self.event_logger.rows:
                try:
                    n = int(row[4])
                except (ValueError, TypeError):
                    continue
                if row[3] == "people_in":
                    total += n
                elif row[3] == "people_out":
                    total -= n
        self.occupancy = max(0, total)
        self.occ_var.set(str(self.occupancy))

    def _toggle_other(self, name):
        """Custom-event items (bike, mobility device, heavy object) toggle entered/left so you can
        see what is still on board and bracket how long it was present."""
        if not self._ensure_capturing():
            return
        if name in self.other_on:
            self.other_on.discard(name)
            recolor(self.other_buttons[name], SLATE, SLATE_H)
            self.event_logger.log("other", name, "left")
            self._append_log("other", name, "(left)")
        else:
            self.other_on.add(name)
            recolor(self.other_buttons[name], AMBER, AMBER_H)
            self.event_logger.log("other", name, "entered")
            self._append_log("other", name, "(entered)")

    def _log_other_text(self):
        if not self._ensure_capturing():
            return
        text = self.other_entry.get().strip()
        if not text:
            return
        self.event_logger.log("other", text)
        self._append_log("other", text)
        self.other_entry.delete(0, "end")

    def _undo(self):
        if not self.event_logger:
            return
        removed = self.event_logger.undo()
        if removed:
            self._recompute_occupancy()
            self._append_log("UNDID", removed[3], str(removed[4]))

    def _append_log(self, event_type, value, note=""):
        ts = datetime.now().strftime("%H:%M:%S")
        line = f"{ts}   {event_type:<16} {value}  {note}".rstrip() + "\n"
        self.log_text.config(state="normal")
        self.log_text.insert("end", line)
        self.log_text.see("end")
        self.log_text.config(state="disabled")

    # ---------- capture lifecycle ----------
    def _list_ports(self):
        if list_ports is None:
            return []
        return [p.device for p in list_ports.comports()]

    def _refresh_ports(self):
        ports = self._list_ports()
        self.port_cb["values"] = ports
        if ports and not self.port_cb.get():
            self.port_cb.current(0)

    def _ensure_capturing(self):
        if not self.capturing:
            self.status_var.set("Not capturing — press START first.")
            return False
        return True

    def _toggle_capture(self):
        if self.capturing:
            self._stop_capture()
        else:
            self._start_capture()

    def _start_capture(self):
        if serial is None:
            messagebox.showerror("pyserial missing",
                                 "pyserial is not installed.\nRun: pip install -r requirements.txt")
            return
        port = self.port_cb.get().strip()
        if not port:
            messagebox.showerror("No port", "Select a serial port first.")
            return
        try:
            baud = int(self.baud_cb.get().strip())
        except ValueError:
            messagebox.showerror("Bad baud", "Baud rate must be a whole number.")
            return
        building = self.building_cb.get().strip() or "Unknown"
        car = self.car_entry.get().strip() or "NA"

        os.makedirs(SESSIONS_DIR, exist_ok=True)
        stamp = datetime.now().strftime("%Y-%m-%d_%H%M%S")
        safe = lambda s: "".join(c if c.isalnum() else "_" for c in s)
        self.session_dir = os.path.join(SESSIONS_DIR, f"{stamp}_{safe(building)}_{safe(car)}")
        os.makedirs(self.session_dir, exist_ok=True)

        self.state_obj = SharedState()
        self.header_lines = []
        raw_path = os.path.join(self.session_dir, "raw.csv")
        self.reader = SerialReader(port, baud, raw_path, self.state_obj, self.header_lines)
        self.reader.start()
        self.event_logger = EventLogger(os.path.join(self.session_dir, "events.csv"),
                                        self.state_obj.get_last_t_us)

        # reset ride state
        floors = self._current_floors()
        start = "G" if "G" in floors else (floors[0] if floors else "")
        self._rebuild_destinations()
        self._set_floor(start)
        self._set_direction_initial("up")
        self.occupancy = 0
        self.occ_var.set("0")

        self._write_session_json(building, car, port, baud, closing=False)
        self.event_logger.log("session_start", building, f"car={car}")
        self._append_log("session_start", building, f"car={car}")

        self.capturing = True
        recolor(self.start_btn, RED, RED_H)
        self.start_btn.config(text="STOP  (Space)")
        self._set_setup_enabled(False)

    def _set_direction_initial(self, direction):
        self.direction = direction
        self.dir_var.set(f"direction: {direction}")
        for key, btn in self.dir_buttons.items():
            recolor(btn, BLUE if key == direction else SLATE, BLUE_H if key == direction else SLATE_H)

    def _stop_capture(self):
        if self.event_logger:
            self.event_logger.log("session_end")
            self._append_log("session_end", "")
        if self.reader:
            self.reader.stop()
            self.reader.join(timeout=2.0)
        self._write_session_json(self.building_cb.get().strip(), self.car_entry.get().strip(),
                                 self.port_cb.get().strip(), self.baud_cb.get().strip(), closing=True)
        self.capturing = False
        recolor(self.start_btn, GREEN, GREEN_H)
        self.start_btn.config(text="START  (Space)")
        self._set_setup_enabled(True)
        self.status_var.set(f"Stopped. Session saved to: {self.session_dir}")

    def _write_session_json(self, building, car, port, baud, closing):
        path = os.path.join(self.session_dir, "session.json")
        data = {
            "building": building, "car": car,
            "board_orientation": self.orient_entry.get().strip(),
            "operator_notes": self.operator_entry.get().strip(),
            "port": port, "baud": baud,
            "floors": self._current_floors(),
            "firmware_header": list(self.header_lines),
            "stopped": closing,
            "stop_time": datetime.now().isoformat(timespec="seconds") if closing else None,
        }
        if not closing:
            data["start_time"] = datetime.now().isoformat(timespec="seconds")
        else:
            try:
                with open(path, "r", encoding="utf-8") as f:
                    data["start_time"] = json.load(f).get("start_time")
            except (OSError, json.JSONDecodeError):
                data["start_time"] = None
        with open(path, "w", encoding="utf-8") as f:
            json.dump(data, f, indent=2)

    def _set_setup_enabled(self, enabled):
        state = "normal" if enabled else "disabled"
        ro = "readonly" if enabled else "disabled"
        for w in (self.car_entry, self.orient_entry, self.operator_entry):
            w.config(state=state)
        self.port_cb.config(state=state)  # keep the port field typeable, not dropdown-only
        self.baud_cb.config(state=state)
        self.building_cb.config(state=ro)
        self.refresh_btn.config(state=state)

    # ---------- keyboard ----------
    def _on_key(self, event):
        if isinstance(event.widget, (tk.Entry, ttk.Entry, ttk.Combobox)):
            return
        ch = event.char.lower()
        if not ch:
            return
        if ch == "o":
            self._doors_open()
        elif ch == "c":
            self._doors_close()
        elif ch == "u":
            self._set_direction("up")
        elif ch == "d":
            self._set_direction("down")
        elif ch == "i":
            self._log_people("people_in", 1)
        elif ch == "k":
            self._log_people("people_out", 1)
        elif ch == "z":
            self._undo()
        else:
            for fl in self._current_floors():
                if len(fl) == 1 and fl.lower() == ch and ch not in COMMAND_KEYS:
                    self._toggle_dest(fl)
                    break

    # ---------- status ----------
    def _poll_status(self):
        if self._closing or not self.winfo_exists():
            return
        snap = self.state_obj.snapshot()
        if self.capturing:
            conn = "● CONNECTED" if snap["connected"] else "○ DISCONNECTED"
            t_us = snap["last_t_us"]
            t_s = f"{t_us / 1e6:.2f}s" if t_us is not None else "-"
            self.status_var.set(
                f"{conn}   F:{snap['fast_rows']}  S:{snap['slow_rows']}   "
                f"last t_us: {t_us if t_us is not None else '-'} ({t_s})   "
                f"missed ticks: {snap['missed_ticks']}   skipped: {snap['bad_lines']}"
            )
            self._update_banner(snap)
        self._poll_id = self.after(200, self._poll_status)

    def _update_banner(self, snap):
        msg = ""
        if snap["error"]:
            msg = snap["error"]
        elif snap["connected"] and not snap["first_data_seen"] and snap["connect_time"]:
            if time.time() - snap["connect_time"] > 5.0:
                msg = "No sensor rows yet — is the board in timing-test mode (RUN_TIMING_TEST = 1)?"
        elif snap["missed_ticks"] > 0:
            msg = f"⚠ {snap['missed_ticks']} missed ticks — this ride dropped samples."
        if msg:
            self.banner_var.set(msg)
            if not self.banner.winfo_ismapped():
                self.banner.pack(fill="x", padx=16, pady=(2, 0), after=self.status_lbl)
        else:
            self.banner_var.set("")
            if self.banner.winfo_ismapped():
                self.banner.pack_forget()

    def _on_close(self, event=None):
        # Accept an optional event: some window managers invoke the handler with one, and binding
        # _on_close elsewhere would otherwise raise "takes 1 positional argument but 2 were given".
        self._closing = True
        if self._poll_id is not None:
            try:
                self.after_cancel(self._poll_id)
            except Exception:
                pass
        try:
            if self.capturing:
                self._stop_capture()
        except Exception:
            pass
        self.destroy()


def main():
    config = load_config()
    app = App(config)
    if serial is None:
        messagebox.showwarning(
            "pyserial not installed",
            "pyserial is missing, so capture will not work.\n"
            "Install it with:  pip install -r requirements.txt\n\n"
            "You can still explore the GUI.",
        )
    app.mainloop()


if __name__ == "__main__":
    main()
