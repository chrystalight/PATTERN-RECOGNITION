"""
Elevator ride post-processing  (MIE 1050 - "Which Elevator Am I In?", Part 1)

ONE script, ONE output. It walks every capture session, auto-segments the continuous log into
individual rides from the sensors (no reliance on the hand-logged events), extracts per-ride
features, and writes a single consolidated table:

    Data_Collector/ride_features.csv      <- one row per ride, every session, the single source of truth

It also prints a short console report (sample rate, pressure coverage, detected-vs-labeled ride
counts) to surface data-quality issues, but writes no other files.

Segmentation: the cabin's barometric altitude changes while moving and is flat when stopped, so a
smoothed pressure-derived vertical speed cleanly marks each ride even through the constant-velocity
cruise (where vertical acceleration is ~0 and looks just like a dwell). The accelerometer then
supplies the dynamic features within each ride. The hand-logged door events are used only to label
each ride (building/floors/direction/occupancy) and to sanity-check the segmenter.

Run from the Data_Collector folder:  python postprocess.py
"""

import os
import glob
import json
import numpy as np
import pandas as pd
from scipy import ndimage
import matplotlib
matplotlib.use("Agg")                     # save a PNG, never block on a window
import matplotlib.pyplot as plt

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
SESSIONS_DIR = os.path.join(BASE_DIR, "sessions")
CONFIG_PATH = os.path.join(BASE_DIR, "config.json")
OUT_CSV = os.path.join(BASE_DIR, "ride_features.csv")
OUT_PNG = os.path.join(BASE_DIR, "ride_features.png")
OUT_SEG_PNG = os.path.join(BASE_DIR, "segmentation_check.png")

FAST_HZ = 200.0
PA_PER_M = 12.0            # ~0.12 hPa per metre (spec)
SPEED_ON = 0.30           # m/s; cabin counts as "moving" above this smoothed vertical speed
SMOOTH_S = 1.5            # pressure smoothing window (s) before differentiating
GRID_DT = 0.1            # altitude resample step (s)
MERGE_GAP_S = 2.0        # join moving spans separated by less than this
MIN_RIDE_S = 2.0         # drop blips shorter than this
MIN_RIDE_M = 1.0         # drop moves smaller than this
CRUISE_ACCEL_THR = 0.03  # g; |vertical accel| below this = constant-velocity (cruise)
MIN_CRUISE_S = 1.5       # need at least this much clean cruise for the vibration/FFT features


# ----------------------------- loading -----------------------------
def load_config_floors():
    """building name -> ordered floor list, for turning floor labels into a floor count."""
    try:
        with open(CONFIG_PATH, encoding="utf-8") as f:
            return json.load(f).get("buildings", {})
    except (OSError, json.JSONDecodeError):
        return {}


def load_raw(path):
    """Parse raw.csv into fast (F) and slow (S) DataFrames. Columns are host_epoch + t_us + sensors."""
    f_rows, s_rows, missed = [], [], 0
    with open(path, encoding="utf-8") as fh:
        for ln in fh:
            if not ln or ln[0] == "#":
                continue
            a = ln.rstrip("\n").split(",")
            rec = a[1] if len(a) > 1 else ""
            if rec == "F" and len(a) == 9:
                f_rows.append(a)
            elif rec == "S" and len(a) == 6:
                s_rows.append(a)
            elif rec == "M":
                try:
                    missed += int(a[3])
                except (IndexError, ValueError):
                    pass
    F = pd.DataFrame(f_rows, columns=["host", "rec", "t_us", "ax", "ay", "az", "light", "emf", "pres"])
    for c in ["host", "t_us", "ax", "ay", "az", "light", "emf"]:
        F[c] = pd.to_numeric(F[c], errors="coerce")
    F["pres"] = pd.to_numeric(F["pres"].replace("", np.nan), errors="coerce")
    S = pd.DataFrame(s_rows, columns=["host", "rec", "t_us", "temp", "rh", "gas"])
    for c in ["host", "t_us", "temp", "rh"]:
        S[c] = pd.to_numeric(S[c], errors="coerce")
    return F, S, missed


def load_events(path):
    if not os.path.exists(path):
        return pd.DataFrame(columns=["host_epoch", "event_type", "value", "note"])
    return pd.read_csv(path)


# ----------------------------- segmentation -----------------------------
def pick_vertical_axis(F):
    """The axis reading ~1 g at rest is the gravity/vertical axis (board mounting may vary)."""
    means = {ax: F[ax].mean() for ax in ("ax", "ay", "az")}
    axis = max(means, key=lambda k: abs(means[k]))
    return axis, means[axis]


def altitude_series(F):
    """Uniform-grid altitude (m, up positive) from the sparse pressure, plus its smoothed speed."""
    t = (F["t_us"].values - F["t_us"].values[0]) / 1e6
    P = F["pres"].values
    ok = ~np.isnan(P)
    if ok.sum() < 5:
        return None, None, None
    grid = np.arange(t[0], t[-1], GRID_DT)
    alt = -np.interp(grid, t[ok], P[ok]) / PA_PER_M           # metres, up positive
    k = max(1, int(SMOOTH_S / GRID_DT))
    alt_s = np.convolve(alt, np.ones(k) / k, mode="same")
    speed = np.gradient(alt_s, grid)                          # m/s
    return grid, alt_s, speed


def segment_rides(F):
    """Return a list of (i0, i1) row-index ranges into F, one per detected ride."""
    grid, alt_s, speed = altitude_series(F)
    if grid is None:
        return []
    t = (F["t_us"].values - F["t_us"].values[0]) / 1e6
    moving = np.abs(speed) > SPEED_ON
    idx = np.where(moving)[0]
    if len(idx) == 0:
        return []
    spans, s, p = [], idx[0], idx[0]
    for i in idx[1:]:
        if grid[i] - grid[p] > MERGE_GAP_S:
            spans.append((s, p))
            s = i
        p = i
    spans.append((s, p))

    rides = []
    for a, b in spans:
        if grid[b] - grid[a] < MIN_RIDE_S or abs(alt_s[b] - alt_s[a]) < MIN_RIDE_M:
            continue
        # map grid times back to F row indices
        i0 = int(np.searchsorted(t, grid[a]))
        i1 = int(np.searchsorted(t, grid[b]))
        rides.append((max(0, i0), min(len(F) - 1, i1)))
    return rides


# ----------------------------- features -----------------------------
def ride_features(F, i0, i1, vaxis, gmean):
    seg = F.iloc[i0:i1 + 1]
    t = (seg["t_us"].values - seg["t_us"].values[0]) / 1e6
    dur = float(t[-1] - t[0]) if len(t) > 1 else 0.0

    a_v = (seg[vaxis].values - gmean)                     # vertical specific accel, g (sign of +axis)
    if gmean < 0:
        a_v = -a_v                                        # make +a_v = upward regardless of mounting
    jerk = np.gradient(a_v, t) if len(t) > 2 else np.array([0.0])

    # pressure-derived altitude over the ride
    P = seg["pres"].dropna()
    height = float(-(P.iloc[-1] - P.iloc[0]) / PA_PER_M) if len(P) > 1 else np.nan
    direction = "up" if (height or 0) > 0 else "down"
    cruise_speed = float(abs(height) / dur) if dur > 0 and not np.isnan(height) else np.nan

    # Cruise phase = the longest interior stretch of near-zero vertical accel (constant velocity).
    # Short "triangular" rides (accel straight into decel, no plateau) have no real cruise, so the
    # vibration/FFT features are left NaN rather than measuring leftover start/stop motion.
    cruise_s = 0.0
    rms_vib, dom_freq, band_energy = np.nan, np.nan, np.nan
    quiet = np.convolve(np.abs(a_v), np.ones(20) / 20, mode="same") < CRUISE_ACCEL_THR
    quiet = ndimage.binary_closing(quiet, structure=np.ones(41))  # bridge <0.2 s flicker
    if quiet.any():
        lab, nlab = ndimage.label(quiet)
        sizes = ndimage.sum(np.ones_like(lab), lab, index=range(1, nlab + 1))
        run = np.where(lab == int(np.argmax(sizes)) + 1)[0]
        cruise_s = len(run) / FAST_HZ
        if cruise_s >= MIN_CRUISE_S and len(run) >= 64:
            ac = a_v[run] - a_v[run].mean()
            rms_vib = float(np.sqrt(np.mean(ac ** 2)))
            freqs = np.fft.rfftfreq(len(ac), d=1.0 / FAST_HZ)
            psd = np.abs(np.fft.rfft(ac)) ** 2
            m = freqs >= 2.0                              # ignore sub-2 Hz bulk motion
            dom_freq = float(freqs[m][np.argmax(psd[m])])
            band = (freqs >= 5.0) & (freqs <= 50.0)
            band_energy = float(psd[band].sum())

    emf = seg["emf"].values.astype(float)
    light = seg["light"].values.astype(float)
    return {
        "direction": direction,
        "duration_s": round(dur, 2),
        "height_m": round(height, 2) if not np.isnan(height) else np.nan,
        "cruise_speed_mps": round(cruise_speed, 3) if not np.isnan(cruise_speed) else np.nan,
        "peak_up_accel_g": round(float(a_v.max()), 4),
        "peak_down_accel_g": round(float(-a_v.min()), 4),
        "peak_jerk_g_s": round(float(np.abs(jerk).max()), 3),
        "cruise_s": round(cruise_s, 2),
        "rms_vib_g": round(rms_vib, 5) if not np.isnan(rms_vib) else np.nan,
        "dom_vib_freq_hz": round(dom_freq, 2) if not np.isnan(dom_freq) else np.nan,
        "band_energy_5_50hz": round(band_energy, 1) if not np.isnan(band_energy) else np.nan,
        "emf_mean": round(float(emf.mean()), 2),
        "emf_rms_ac": round(float(np.sqrt(np.mean((emf - emf.mean()) ** 2))), 3),
        "light_mean": round(float(light.mean()), 1),
        "light_std": round(float(light.std()), 2),
    }


# ----------------------------- labels from events -----------------------------
def travel_windows(ev):
    """Pair each door_close (departure) with the next door_open (arrival) -> labeled ride windows."""
    out = []
    pending = None
    for _, r in ev.iterrows():
        et = str(r["event_type"]).strip()
        if et == "door_close":
            pending = (float(r["host_epoch"]), str(r["value"]))
        elif et == "door_open" and pending is not None:
            out.append((pending[0], float(r["host_epoch"]), pending[1], str(r["value"])))
            pending = None
    return out


def occupancy_before(ev, host_t):
    n = 0
    for _, r in ev.iterrows():
        if float(r["host_epoch"]) > host_t:
            break
        try:
            v = int(r["value"])
        except (ValueError, TypeError):
            continue
        if r["event_type"] == "people_in":
            n += v
        elif r["event_type"] == "people_out":
            n -= v
    return max(0, n)


def label_ride(F, i0, i1, windows, ev, floors):
    """Attach building-floor labels to a detected ride by overlap with a door travel window."""
    h0, h1 = float(F["host"].iloc[i0]), float(F["host"].iloc[i1])
    best, best_ov = None, 0.0
    for (c, o, f_from, f_to) in windows:
        ov = max(0.0, min(h1, o) - max(h0, c))
        if ov > best_ov:
            best_ov, best = ov, (f_from, f_to)
    if best is None:
        return {"start_floor": "", "end_floor": "", "n_floors_label": "",
                "direction_label": "", "people_on_board": "", "matched_label": False}
    f_from, f_to = best
    n_floors, dir_lab = "", ""
    if f_from in floors and f_to in floors:
        n_floors = abs(floors.index(f_to) - floors.index(f_from))
        dir_lab = "up" if floors.index(f_to) > floors.index(f_from) else "down"
    return {"start_floor": f_from, "end_floor": f_to, "n_floors_label": n_floors,
            "direction_label": dir_lab, "people_on_board": occupancy_before(ev, h0),
            "matched_label": True}


# ----------------------------- visualization -----------------------------
# Each feature as a strip of per-ride dots grouped by building, so you can see at a glance which
# features separate the classes. Last panel is a 2-D scatter of the two strongest.
STRIP_FEATURES = [
    ("cruise_speed_mps", "Cruise speed (m/s)"),
    ("peak_up_accel_g", "Peak up accel (g)"),
    ("peak_down_accel_g", "Peak down accel (g)"),
    ("peak_jerk_g_s", "Peak jerk (g/s)"),
    ("rms_vib_g", "Cruise vibration RMS (g)"),
    ("dom_vib_freq_hz", "Dominant vib freq (Hz)"),
    ("light_mean", "Cabin light (ADC)"),
    ("floor_height_m", "Floor height (m)"),
]


def plot_features(df, out_png):
    df = df.copy()
    buildings = sorted(df["building"].dropna().unique())
    palette = plt.cm.tab10.colors
    cmap = {b: palette[i % 10] for i, b in enumerate(buildings)}
    rng = np.random.default_rng(0)

    fig, axes = plt.subplots(3, 3, figsize=(15, 11))
    for ax, (col, label) in zip(axes.flat, STRIP_FEATURES):
        for i, b in enumerate(buildings):
            vals = df.loc[df["building"] == b, col].dropna().values
            if len(vals) == 0:
                continue
            ax.scatter(i + rng.normal(0, 0.07, len(vals)), vals, s=20, alpha=0.7,
                       color=cmap[b], edgecolor="none")
            ax.hlines(vals.mean(), i - 0.22, i + 0.22, color="black", lw=2, zorder=3)
        ax.set_xticks(range(len(buildings)))
        ax.set_xticklabels(buildings, rotation=12, fontsize=9)
        ax.set_title(label, fontsize=11)
        ax.grid(True, axis="y", ls=":", alpha=0.5)

    # last panel: cruise speed vs jerk scatter (the two physically strongest separators)
    ax = axes.flat[8]
    for b in buildings:
        sub = df[df["building"] == b]
        ax.scatter(sub["cruise_speed_mps"], sub["peak_jerk_g_s"], s=28, alpha=0.75,
                   color=cmap[b], edgecolor="none", label=b)
    ax.set_xlabel("Cruise speed (m/s)")
    ax.set_ylabel("Peak jerk (g/s)")
    ax.set_title("Speed vs jerk", fontsize=11)
    ax.grid(True, ls=":", alpha=0.5)
    ax.legend(fontsize=9)

    fig.suptitle("Per-ride features by building  (dots = rides, black bar = mean)", fontsize=14)
    plt.tight_layout(rect=(0, 0, 1, 0.98))
    fig.savefig(out_png, dpi=110)
    plt.close(fig)


def plot_segmentation_check(panels, out_png):
    """One panel per session: altitude + |vertical speed| with detected ride spans (green) and the
    door close/open events (dotted), so the auto-segmentation can be eyeballed against the truth."""
    if not panels:
        return
    n = len(panels)
    fig, axes = plt.subplots(n, 1, figsize=(14, 2.3 * n))
    if n == 1:
        axes = [axes]
    for ax, p in zip(axes, panels):
        # raw pressure line; axis auto-zooms to the data range so ~40-50 Pa/floor steps are visible
        ax.plot(p["pt"], p["pp"], color="#1f77b4", lw=0.9)
        ax.set_ylabel("pressure (Pa)", color="#1f77b4", fontsize=9)
        ax.ticklabel_format(axis="y", style="plain", useOffset=False)
        ax2 = ax.twinx()
        ax2.plot(p["grid"], np.abs(p["speed"]), color="#ff7f0e", lw=0.9, alpha=0.7)
        ax2.axhline(SPEED_ON, ls="--", color="#ff7f0e", lw=0.8)
        ax2.set_ylabel("|v| m/s", color="#ff7f0e", fontsize=9)
        for a, b in p["spans"]:
            ax.axvspan(a, b, color="green", alpha=0.15)
        for c, o in p["doors"]:
            ax.axvline(c, color="red", ls=":", lw=1.0)
            ax.axvline(o, color="purple", ls=":", lw=1.0)
        ax.set_title(f"{p['name']}    detected {p['nseg']}  vs labeled {p['nlab']}", fontsize=10)
        ax.grid(True, axis="x", ls=":", alpha=0.4)
    axes[-1].set_xlabel("time since session start (s)")
    fig.suptitle("Auto-segmentation check  —  blue = pressure, green span = detected ride, "
                 "orange = |vertical speed| (dashed = 0.30 m/s), red/purple dotted = door close/open",
                 fontsize=12)
    plt.tight_layout(rect=(0, 0, 1, 0.98))
    fig.savefig(out_png, dpi=110)
    plt.close(fig)


# ----------------------------- main -----------------------------
def main():
    building_floors = load_config_floors()
    all_rows = []
    seg_panels = []
    print(f"{'session':44} {'Hz':>5} {'pres%':>6} {'rides':>6} {'labeled':>8}  note")
    print("-" * 92)

    for sess in sorted(glob.glob(os.path.join(SESSIONS_DIR, "*"))):
        name = os.path.basename(sess)
        raw_path = os.path.join(sess, "raw.csv")
        if not os.path.exists(raw_path):
            print(f"{name:44} {'-':>5} {'-':>6} {'-':>6} {'-':>8}  no raw.csv")
            continue

        F, S, missed = load_raw(raw_path)
        if len(F) < 50:
            print(f"{name:44} {'-':>5} {'-':>6} {'-':>6} {'-':>8}  too few F rows ({len(F)})")
            continue

        sj = {}
        try:
            with open(os.path.join(sess, "session.json"), encoding="utf-8") as f:
                sj = json.load(f)
        except (OSError, json.JSONDecodeError):
            pass
        building = sj.get("building", "")
        car = sj.get("car", "")
        # prefer the config floor order; fall back to the session's own list (e.g. Burton)
        floors = building_floors.get(building) or sj.get("floors", [])

        dt = np.diff(F["t_us"].values)
        hz = FAST_HZ if len(dt) == 0 else 1e6 / np.median(dt)
        pres_pct = 100.0 * F["pres"].notna().mean()

        ev = load_events(os.path.join(sess, "events.csv"))
        windows = travel_windows(ev)
        vaxis, gmean = pick_vertical_axis(F)
        rides = segment_rides(F)

        # collect what the segmentation-check figure needs (times on the t_us base, in seconds)
        grid, alt_s, speed = altitude_series(F)
        if grid is not None:
            t0 = F["t_us"].values[0]
            Ft = (F["t_us"].values - t0) / 1e6
            Fh = F["host"].values
            okp = F["pres"].notna().values
            seg_panels.append({
                "name": name.replace("2026-10-28_", ""),
                "grid": grid, "pres_s": -np.asarray(alt_s) * PA_PER_M, "speed": speed,
                "pt": Ft[okp], "pp": F["pres"].values[okp],      # raw pressure samples
                "spans": [((F["t_us"].iloc[i0] - t0) / 1e6, (F["t_us"].iloc[i1] - t0) / 1e6)
                          for i0, i1 in rides],
                "doors": [(float(np.interp(c, Fh, Ft)), float(np.interp(o, Fh, Ft)))
                          for (c, o, _, _) in windows],
                "nseg": len(rides), "nlab": len(windows),
            })

        for k, (i0, i1) in enumerate(rides, start=1):
            feat = ride_features(F, i0, i1, vaxis, gmean)
            lab = label_ride(F, i0, i1, windows, ev, floors)
            row = {"session": name, "building": building, "car": car, "ride": k,
                   "t_start_s": round((F["t_us"].iloc[i0] - F["t_us"].iloc[0]) / 1e6, 1),
                   "t_end_s": round((F["t_us"].iloc[i1] - F["t_us"].iloc[0]) / 1e6, 1)}
            row.update(lab)
            row.update(feat)
            # Per-floor pitch = metres travelled / floors travelled (building architecture).
            # Needs the labeled floor count, so it is blank on rides with no door label.
            nf = lab.get("n_floors_label", "")
            h = feat.get("height_m")
            valid_h = h is not None and not (isinstance(h, float) and np.isnan(h))
            row["floor_height_m"] = (round(abs(h) / nf, 2)
                                     if isinstance(nf, int) and nf > 0 and valid_h else np.nan)
            all_rows.append(row)

        notes = []
        if len(rides) != len(windows):
            notes.append(f"seg {len(rides)} vs {len(windows)} labeled")
        if missed:
            notes.append(f"{missed} missed ticks")
        if pres_pct < 10:
            notes.append("low pressure coverage")
        if len(rides) == 0:
            notes.append("NO rides detected")
        print(f"{name:44} {hz:5.0f} {pres_pct:5.1f}% {len(rides):6d} {len(windows):8d}  {'; '.join(notes)}")

    if not all_rows:
        print("\nNo rides extracted.")
        return

    cols = ["session", "building", "car", "ride", "direction", "start_floor", "end_floor",
            "n_floors_label", "direction_label", "people_on_board",
            "duration_s", "height_m", "floor_height_m", "cruise_speed_mps",
            "peak_up_accel_g", "peak_down_accel_g", "peak_jerk_g_s",
            "cruise_s", "rms_vib_g", "dom_vib_freq_hz", "band_energy_5_50hz",
            "emf_mean", "emf_rms_ac", "light_mean", "light_std",
            "t_start_s", "t_end_s", "matched_label"]
    df = pd.DataFrame(all_rows)[cols]
    try:
        df.to_csv(OUT_CSV, index=False)
        csv_msg = os.path.relpath(OUT_CSV, BASE_DIR)
    except PermissionError:
        csv_msg = f"NOT WRITTEN ({os.path.basename(OUT_CSV)} is open elsewhere — close it and re-run)"
    def save(fn, path):
        try:
            fn()
            return os.path.relpath(path, BASE_DIR)
        except OSError:
            return f"NOT WRITTEN ({os.path.basename(path)} is open elsewhere — close it and re-run)"

    png_msg = save(lambda: plot_features(df, OUT_PNG), OUT_PNG)
    seg_msg = save(lambda: plot_segmentation_check(seg_panels, OUT_SEG_PNG), OUT_SEG_PNG)
    print("-" * 92)
    print(f"{len(df)} rides from {df['session'].nunique()} sessions -> {csv_msg}")
    print("Buildings:", ", ".join(f"{b}:{n}" for b, n in df["building"].value_counts().items()))
    print(f"Feature plot        -> {png_msg}")
    print(f"Segmentation check  -> {seg_msg}")


if __name__ == "__main__":
    main()
