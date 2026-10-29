"""
Elevator ride post-processing  (MIE 1050 - "Which Elevator Am I In?", Part 1)

ONE script, ONE output. It walks every capture session, auto-segments the continuous log into
individual rides from the sensors (no reliance on the hand-logged events), extracts per-ride
features, and writes a single consolidated table:

    Data_Collector/ride_features.csv      <- one row per ride, every session, the single source of truth

plus two diagnostic figures: ride_features.png (every feature, per ride, grouped by building with
each car in its own shade) and segmentation_check.png (per-session pressure / speed / detected
spans / door events).

Segmentation: the cabin's barometric altitude changes while moving and is flat when stopped, so a
smoothed pressure-derived vertical speed cleanly marks each ride even through the constant-velocity
cruise (where vertical acceleration is ~0 and looks just like a dwell). The accelerometer then
supplies the dynamic features within each ride. The hand-logged door events are used to label
each ride (building/floors/direction/occupancy), to sanity-check the segmenter, and to time the two
door-open features (pressure pop, light step).

Filtering (the raw 200 Hz accelerometer is too noisy to differentiate directly):
  * motion profile  a_lp  = 4th-order zero-phase Butterworth low-pass at 1.5 Hz. Drives peak
    accel/decel, rise time, jerk, phase timing and the cruise detector. 1.5 Hz keeps the S-curve
    (jerk phases last ~0.5-1 s) but removes the ~3 Hz car bounce and sensor noise; it gave the most
    repeatable per-car jerk of the cutoffs tried (1 / 1.5 / 2 / 3 Hz).
  * velocity  v  = integral of the raw vertical accel from motion start to motion end, with the
    linear drift removed so v = 0 at both ends. Its distance matches the barometer to ~5%.
  * vibration = 1 Hz high-pass of the raw vertical accel over the cruise; spectra are Welch PSDs
    (1.28 s Hann segments, 50% overlap) instead of a single raw FFT.
Aliasing: at 200 Hz sampling the Nyquist is 100 Hz, so 120 Hz lamp flicker shows at 80 Hz and the
60 Hz mains harmonics fold to 60 / 80 / 20 / 40 Hz (60, 120, 180, 240+360 Hz).

Run from the Data_Collector folder:  python postprocess.py
"""

import os
import glob
import json
import numpy as np
import pandas as pd
from scipy import ndimage, signal
import matplotlib
matplotlib.use("Agg")                     # save a PNG, never block on a window
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
SESSIONS_DIR = os.path.join(BASE_DIR, "sessions")
CONFIG_PATH = os.path.join(BASE_DIR, "config.json")
OUT_CSV = os.path.join(BASE_DIR, "ride_features.csv")
OUT_PNG = os.path.join(BASE_DIR, "ride_features.png")
OUT_SEG_PNG = os.path.join(BASE_DIR, "segmentation_check.png")

FAST_HZ = 200.0
G = 9.81
PA_PER_M = 12.0            # ~0.12 hPa per metre (spec)

# segmentation (barometer)
SPEED_ON = 0.30           # m/s; cabin counts as "moving" above this smoothed vertical speed
SMOOTH_S = 1.5            # pressure smoothing window (s) before differentiating
GRID_DT = 0.1            # altitude resample step (s)
MERGE_GAP_S = 2.0        # join moving spans separated by less than this
MIN_RIDE_S = 2.0         # drop blips shorter than this
MIN_RIDE_M = 1.0         # drop moves smaller than this

# motion profile (accelerometer)
MOTION_LP_HZ = 1.5       # low-pass for the motion profile a_lp
PHASE_SEARCH_S = 3.0     # look this far either side of the baro span for the launch / brake peaks
MOTION_FRAC = 0.10       # motion starts/ends where |a_lp| crosses 10% of the launch/brake peak
MIN_PHASE_ACCEL = 0.01   # g; launch and brake peaks must both reach this (gentlest car ~0.04 g)
RISE_FRAC = 0.90         # rise time = motion start -> 90% of the launch peak
CRUISE_ACCEL_THR = 0.01  # g; |a_lp| below this between launch and brake = constant-velocity cruise
MIN_CRUISE_S = 1.5       # need at least this much cruise for the cruise-only features
STILL_S = 1.0            # pressure averaged over this long just before start / after stop

# vibration / spectra
VIB_HP_HZ = 1.0          # high-pass before cruise vibration RMS / PSD
VIB_PEAK_BAND = (2.0, 95.0)    # dominant-frequency search band (below 2 Hz = residual bulk motion)
VIB_ENERGY_BAND = (5.0, 50.0)
SWAY_BAND = (0.5, 20.0)  # lateral sway band-pass
WOBBLE_BAND = (0.5, 10.0)      # post-stop wobble band-pass (car bounce on ropes / re-levelling)
WOBBLE_S = 2.0           # window after motion end
WELCH_NPERSEG = 256      # 1.28 s segments -> 0.78 Hz resolution
FLICKER_BAND = (50.0, 99.0)    # 120 Hz flicker aliases to 80 Hz
MAINS_ALIAS_HZ = (60.0, 80.0, 20.0, 40.0)   # 60, 120, 180, 240/360 Hz folded about 100 Hz
TONE_HALF_BW = 0.8       # Hz either side of a tone when integrating its power
EMF_ENV_S = 0.5          # EMF envelope smoothing for the EMF-accel correlation

# door-open features: windows (s) around the logged door_open, clear of the ~1 s logging latency
DOOR_PRE = (-4.0, -1.5)
DOOR_POST = (1.5, 4.0)

# slow sensors
SLOW_PAD_S = 5.0         # widen the ride window for temp / RH (1 Hz samples)
MIN_SLOW_N = 5
MIN_SLOPE_M = 3.0        # need this much altitude change to fit a temp / RH vs altitude slope


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
    F = F.dropna(subset=["t_us"]).reset_index(drop=True)
    for c in ["ax", "ay", "az", "light", "emf"]:          # filters can't take NaN; gaps are rare
        F[c] = F[c].interpolate(limit_direction="both")
    S = pd.DataFrame(s_rows, columns=["host", "rec", "t_us", "temp", "rh", "gas"])
    for c in ["host", "t_us", "temp", "rh"]:
        S[c] = pd.to_numeric(S[c], errors="coerce")
    return F, S, missed


def load_events(path):
    if not os.path.exists(path):
        return pd.DataFrame(columns=["host_epoch", "event_type", "value", "note"])
    return pd.read_csv(path)


# ----------------------------- segmentation -----------------------------
def gravity_frame(F):
    """Gravity-aligned frame from the session's resting accel vector, so any board orientation or
    tilt works. The median is dominated by the dwells (rides are a minority and their up/down accel
    cancels). Returns |g| (g), the unit 'up' vector and two orthonormal horizontal unit vectors."""
    g = np.median(F[["ax", "ay", "az"]].values, axis=0)
    gmag = float(np.linalg.norm(g))
    up = g / gmag                            # at rest the accelerometer reads +1 g pointing up
    ref = np.eye(3)[int(np.argmin(np.abs(up)))]          # axis least aligned with up
    h1 = np.cross(up, ref)
    h1 /= np.linalg.norm(h1)
    return gmag, up, h1, np.cross(up, h1)


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
    pad = np.pad(alt, (k // 2, k - 1 - k // 2), mode="edge")  # edge pad: no fake drop to 0 at the ends
    alt_s = np.convolve(pad, np.ones(k) / k, mode="valid")
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


# ----------------------------- signal helpers -----------------------------
def butter(x, fc, btype):
    sos = signal.butter(4, fc, btype=btype, fs=FAST_HZ, output="sos")
    return signal.sosfiltfilt(sos, x)


def session_signals(F):
    """Filtered whole-session signals, computed once so ride windows have no filter edge effects."""
    gmag, up, e1, e2 = gravity_frame(F)
    A = F[["ax", "ay", "az"]].values
    a_v = A @ up - gmag                                   # vertical specific accel, g, + = upward
    h = [A @ e1, A @ e2]                                  # horizontal components (band-passed below)
    light = F["light"].values.astype(float)
    emf = F["emf"].values.astype(float)
    emf_env = np.abs(emf - pd.Series(emf).rolling(int(4 * FAST_HZ), center=True, min_periods=1)
                     .mean().values)
    k = int(EMF_ENV_S * FAST_HZ)
    return {
        "t": (F["t_us"].values - F["t_us"].values[0]) / 1e6,
        "host": F["host"].values,
        "pres": F["pres"].values,
        "a_v": a_v,
        "a_lp": butter(a_v, MOTION_LP_HZ, "lowpass"),
        "a_vib": butter(a_v, VIB_HP_HZ, "highpass"),
        "a_wob": butter(a_v, WOBBLE_BAND, "bandpass"),
        "h1": butter(h[0], SWAY_BAND, "bandpass"),
        "h2": butter(h[1], SWAY_BAND, "bandpass"),
        "light": light,
        "emf": emf,
        "emf_env": np.convolve(emf_env, np.ones(k) / k, mode="same"),
    }


def welch(x):
    n = min(WELCH_NPERSEG, len(x))
    return signal.welch(x, fs=FAST_HZ, nperseg=n, noverlap=n // 2, detrend="constant")


def band_power(f, P, lo, hi):
    m = (f >= lo) & (f <= hi)
    return float(np.trapezoid(P[m], f[m])) if m.sum() > 1 else float(P[m].sum() * (f[1] - f[0]))


def longest_run(mask):
    """(start, stop) of the longest True run in mask, or None."""
    if not mask.any():
        return None
    lab, n = ndimage.label(mask)
    sizes = ndimage.sum(np.ones_like(lab), lab, index=range(1, n + 1))
    run = np.where(lab == int(np.argmax(sizes)) + 1)[0]
    return run[0], run[-1] + 1


def pres_median(sig, lo, hi):
    P = sig["pres"][lo:hi]
    P = P[~np.isnan(P)]
    return float(np.median(P)) if len(P) else np.nan


def host_window(sig, t_host, rel):
    """Row slice for host-time window t_host + rel (s)."""
    h = sig["host"]
    return slice(int(np.searchsorted(h, t_host + rel[0])), int(np.searchsorted(h, t_host + rel[1])))


def r(x, n):
    return round(float(x), n) if x is not None and np.isfinite(x) else np.nan


# ----------------------------- features -----------------------------
def motion_phases(sig, i0, i1, up):
    """Refine the baro span to true motion start / end from the motion profile.
    Launch peak = strongest accel in the travel direction near the span start, brake peak = strongest
    accel against it near the span end; motion start/end = where |a_lp| falls to 10% of those peaks."""
    a = sig["a_lp"]
    n = len(a)
    w = int(PHASE_SEARCH_S * FAST_HZ)
    d = 1.0 if up else -1.0
    lo, hi = max(0, i0 - w), min(n, i0 + w)
    k_launch = lo + int(np.argmax(d * a[lo:hi]))
    lo, hi = max(0, i1 - w), min(n, i1 + w)
    k_brake = lo + int(np.argmax(-d * a[lo:hi]))
    if k_brake <= k_launch:
        return None
    pk_l, pk_b = abs(a[k_launch]), abs(a[k_brake])
    if min(pk_l, pk_b) < MIN_PHASE_ACCEL:
        return None                                   # no real launch or brake (e.g. log ends mid-ride)
    below = np.where(np.abs(a[:k_launch]) < MOTION_FRAC * pk_l)[0]
    s = int(below[-1]) if len(below) else 0
    below = np.where(np.abs(a[k_brake:]) < MOTION_FRAC * pk_b)[0]
    e = k_brake + int(below[0]) if len(below) else n - 1
    if s < STILL_S * FAST_HZ or e > n - 1 - STILL_S * FAST_HZ:
        return None                                   # ride cut off by the session start / stop
    return s, k_launch, k_brake, e


def ride_features(sig, S, i0, i1, door_open_host):
    t = sig["t"]
    still = int(STILL_S * FAST_HZ)

    # direction + height from the barometer (median of the still second either side of the span)
    p0 = pres_median(sig, max(0, i0 - still), i0)
    p1 = pres_median(sig, i1, i1 + still)
    up = (p0 - p1) > 0
    ph = motion_phases(sig, i0, i1, up)
    if ph is None:
        return None
    s, k_l, k_b, e = ph
    p0 = pres_median(sig, max(0, s - still), s)
    p1 = pres_median(sig, e, e + still)
    height = -(p1 - p0) / PA_PER_M
    dur = t[e] - t[s]

    a_lp = sig["a_lp"][s:e + 1]
    tt = t[s:e + 1]
    dt = np.gradient(tt)
    jerk = np.gradient(a_lp, tt)
    launch, brake = abs(sig["a_lp"][k_l]), abs(sig["a_lp"][k_b])
    rise = np.where(np.abs(a_lp[:k_l - s + 1]) >= RISE_FRAC * launch)[0]
    rise_s = tt[rise[0]] - tt[0] if len(rise) else np.nan

    # velocity by integrating raw accel, drift removed so it is 0 at rest on both ends
    v = np.cumsum(sig["a_v"][s:e + 1] * dt) * G
    v -= np.linspace(0.0, v[-1], len(v))
    dist = np.abs(v) * dt

    # cruise = longest constant-velocity stretch between the launch and brake peaks
    quiet = np.zeros(len(a_lp), bool)
    quiet[k_l - s:k_b - s] = np.abs(a_lp[k_l - s:k_b - s]) < CRUISE_ACCEL_THR
    run = longest_run(quiet)
    cruise_s = (run[1] - run[0]) / FAST_HZ if run else 0.0
    has_cruise = cruise_s >= MIN_CRUISE_S
    accel_end = run[0] if has_cruise else int(np.argmax(np.abs(v)))
    f = {
        "direction": "up" if up else "down",
        "duration_s": r(dur, 2),
        "height_m": r(height, 2),
        "cruise_speed_mps": r(np.median(np.abs(v[run[0]:run[1]])) if has_cruise
                              else np.abs(v).max(), 3),
        "speed_ripple": np.nan,
        "accel_dist_frac": r(dist[:accel_end].sum() / dist.sum(), 3) if dist.sum() > 0 else np.nan,
        "peak_up_accel_g": r(a_lp.max(), 4),
        "peak_down_accel_g": r(-a_lp.min(), 4),
        "accel_decel_ratio": r(launch / brake, 3) if brake > 0 else np.nan,
        "accel_rise_s": r(rise_s, 2),
        "peak_jerk_g_s": r(np.abs(jerk).max(), 3),
        "cruise_s": r(cruise_s, 2),
        "rms_vib_g": np.nan, "dom_vib_freq_hz": np.nan, "band_energy_5_50hz": np.nan,
        "lateral_sway_g": np.nan,
    }

    if has_cruise:
        c0, c1 = s + run[0], s + run[1]
        vc = np.abs(v[run[0]:run[1]])
        f["speed_ripple"] = r(vc.std() / vc.mean(), 4)
        vib = sig["a_vib"][c0:c1]
        fr, P = welch(vib)
        m = (fr >= VIB_PEAK_BAND[0]) & (fr <= VIB_PEAK_BAND[1])
        f["rms_vib_g"] = r(np.sqrt(np.mean(vib ** 2)), 5)
        f["dom_vib_freq_hz"] = r(fr[m][np.argmax(P[m])], 2)
        f["band_energy_5_50hz"] = r(band_power(fr, P, *VIB_ENERGY_BAND) * 1e6, 3)   # mg^2
        f["lateral_sway_g"] = r(np.sqrt(np.mean(sig["h1"][c0:c1] ** 2 + sig["h2"][c0:c1] ** 2)), 5)

    # post-stop wobble: residual bounce once the brake phase has died out
    wob = sig["a_wob"][e:e + int(WOBBLE_S * FAST_HZ)]
    f["post_stop_wobble_g"] = r(np.abs(wob).max(), 4) if len(wob) else np.nan
    f["post_stop_wobble_rms_g"] = r(np.sqrt(np.mean(wob ** 2)), 5) if len(wob) else np.nan

    # light: level, variance, and the 120 Hz flicker (aliased to 80 Hz)
    light = sig["light"][s:e + 1]
    fr, P = welch(light)
    m = (fr >= FLICKER_BAND[0]) & (fr <= FLICKER_BAND[1])
    fk = fr[m][np.argmax(P[m])]
    amp = np.sqrt(2 * band_power(fr, P, fk - TONE_HALF_BW, fk + TONE_HALF_BW))
    f["light_mean"] = r(light.mean(), 1)
    f["light_std"] = r(light.std(), 2)
    f["light_flicker_hz"] = r(fk, 1)
    f["light_flicker_index"] = r(amp / light.mean(), 4) if light.mean() > 0 else np.nan

    # EMF: level, AC RMS, mains-harmonic content, and coupling to drive torque
    emf = sig["emf"][s:e + 1]
    fr, P = welch(emf)
    tones = [band_power(fr, P, h - TONE_HALF_BW, h + TONE_HALF_BW) for h in MAINS_ALIAS_HZ]
    total = band_power(fr, P, 1.0, 99.0)
    env = sig["emf_env"][s:e + 1]
    f["emf_mean"] = r(emf.mean(), 2)
    f["emf_rms_ac"] = r(np.sqrt(np.mean((emf - emf.mean()) ** 2)), 3)
    f["emf_mains_frac"] = r(sum(tones) / total, 3) if total > 0 else np.nan
    f["emf_mains_peak_hz"] = MAINS_ALIAS_HZ[int(np.argmax(tones))] if total > 0 else np.nan
    f["emf_accel_corr"] = (r(np.corrcoef(np.abs(a_lp), env)[0, 1], 3)
                           if env.std() > 0 and a_lp.std() > 0 else np.nan)

    # door-open pressure pop and light step, from the logged door_open of the matched label
    f["door_pressure_pop_pa"], f["door_light_step"] = np.nan, np.nan
    if door_open_host is not None:
        pre, post = host_window(sig, door_open_host, DOOR_PRE), host_window(sig, door_open_host, DOOR_POST)
        if post.stop - post.start > 0.5 * FAST_HZ and pre.stop - pre.start > 0.5 * FAST_HZ:
            f["door_pressure_pop_pa"] = r(pres_median(sig, post.start, post.stop)
                                          - pres_median(sig, pre.start, pre.stop), 2)
            f["door_light_step"] = r(sig["light"][post].mean() - sig["light"][pre].mean(), 1)

    # slow sensors: cabin climate and its gradient with altitude
    ts = (S["t_us"].values - sig["t0_us"]) / 1e6
    m = ((ts >= t[s] - SLOW_PAD_S) & (ts <= t[e] + SLOW_PAD_S)
         & np.isfinite(S["temp"].values) & np.isfinite(S["rh"].values))
    temp, rh = S["temp"].values[m], S["rh"].values[m]
    f["temp_mean_c"] = r(np.nanmean(temp), 2) if m.any() else np.nan
    f["rh_mean_pct"] = r(np.nanmean(rh), 2) if m.any() else np.nan
    f["temp_alt_slope_c_per_m"], f["rh_alt_slope_pct_per_m"] = np.nan, np.nan
    okp = ~np.isnan(sig["pres"])
    if m.sum() >= MIN_SLOW_N and okp.sum() > 5:
        alt = -np.interp(ts[m], t[okp], sig["pres"][okp]) / PA_PER_M
        if alt.max() - alt.min() >= MIN_SLOPE_M:
            f["temp_alt_slope_c_per_m"] = r(np.polyfit(alt, temp, 1)[0], 4)
            f["rh_alt_slope_pct_per_m"] = r(np.polyfit(alt, rh, 1)[0], 4)
    f["_span"] = (s, e)
    return f


# ----------------------------- labels from events -----------------------------
def travel_windows(ev):
    """Pair each door_close (departure) with the next door_open (arrival) -> labeled ride windows."""
    out = []
    pending = None
    for _, r_ in ev.iterrows():
        et = str(r_["event_type"]).strip()
        if et == "door_close":
            pending = (float(r_["host_epoch"]), str(r_["value"]))
        elif et == "door_open" and pending is not None:
            out.append((pending[0], float(r_["host_epoch"]), pending[1], str(r_["value"])))
            pending = None
    return out


def occupancy_before(ev, host_t):
    n = 0
    for _, r_ in ev.iterrows():
        if float(r_["host_epoch"]) > host_t:
            break
        try:
            v = int(r_["value"])
        except (ValueError, TypeError):
            continue
        if r_["event_type"] == "people_in":
            n += v
        elif r_["event_type"] == "people_out":
            n -= v
    return max(0, n)


def label_ride(F, i0, i1, windows, ev, floors):
    """Attach building-floor labels to a detected ride by overlap with a door travel window.
    Also returns the matched arrival door_open host time (for the door-open features)."""
    h0, h1 = float(F["host"].iloc[i0]), float(F["host"].iloc[i1])
    best, best_ov = None, 0.0
    for (c, o, f_from, f_to) in windows:
        ov = max(0.0, min(h1, o) - max(h0, c))
        if ov > best_ov:
            best_ov, best = ov, (f_from, f_to, o)
    if best is None:
        return {"start_floor": "", "end_floor": "", "n_floors_label": "",
                "direction_label": "", "people_on_board": "", "matched_label": False}, None
    f_from, f_to, t_open = best
    n_floors, dir_lab = "", ""
    if f_from in floors and f_to in floors:
        n_floors = abs(floors.index(f_to) - floors.index(f_from))
        dir_lab = "up" if floors.index(f_to) > floors.index(f_from) else "down"
    return {"start_floor": f_from, "end_floor": f_to, "n_floors_label": n_floors,
            "direction_label": dir_lab, "people_on_board": occupancy_before(ev, h0),
            "matched_label": True}, t_open


# ----------------------------- visualization -----------------------------
# Every feature as a strip of per-ride dots, one column per building. Cars of the same building
# share the column and the building's hue, told apart by shade + marker. Black bar = building mean.
FEATURES = [
    ("cruise_speed_mps", "Cruise speed (m/s)"),
    ("peak_jerk_g_s", "Peak jerk (g/s)"),
    ("peak_up_accel_g", "Peak up accel (g)"),
    ("peak_down_accel_g", "Peak down accel (g)"),
    ("accel_decel_ratio", "Launch / brake accel ratio"),
    ("accel_rise_s", "Accel rise time (s)"),
    ("accel_dist_frac", "Accel-distance fraction"),
    ("speed_ripple", "Cruise speed ripple (std/mean)"),
    ("floor_height_m", "Floor height (m)"),
    ("duration_s", "Ride duration (s)"),
    ("post_stop_wobble_g", "Post-stop wobble peak (g)"),
    ("post_stop_wobble_rms_g", "Post-stop wobble RMS (g)"),
    ("rms_vib_g", "Cruise vibration RMS (g)"),
    ("dom_vib_freq_hz", "Dominant vib freq (Hz)"),
    ("band_energy_5_50hz", "Vib power 5-50 Hz (mg²)"),
    ("lateral_sway_g", "Lateral sway RMS (g)"),
    ("door_pressure_pop_pa", "Door-open pressure pop (Pa)"),
    ("light_mean", "Cabin light (ADC)"),
    ("light_std", "Light std (ADC)"),
    ("light_flicker_index", "Light flicker index"),
    ("light_flicker_hz", "Flicker freq, aliased (Hz)"),
    ("door_light_step", "Door-open light step (ADC)"),
    ("emf_mean", "EMF mean (ADC)"),
    ("emf_rms_ac", "EMF AC RMS (ADC)"),
    ("emf_mains_frac", "EMF mains-harmonic fraction"),
    ("emf_mains_peak_hz", "EMF strongest harmonic, aliased (Hz)"),
    ("emf_accel_corr", "EMF-accel correlation"),
    ("temp_mean_c", "Cabin temp (°C)"),
    ("rh_mean_pct", "Cabin RH (%)"),
    ("temp_alt_slope_c_per_m", "Temp vs altitude (°C/m)"),
    ("rh_alt_slope_pct_per_m", "RH vs altitude (%/m)"),
]

# fixed hue per building (never by rank); [dark, light, mid] shades for up to three cars
BUILDING_COLORS = {
    "Hart House": ["#1f5fae", "#8fb8ec", "#2a78d6"],
    "Mechanical (MC)": ["#c04a1c", "#f5a882", "#eb6834"],
    "Myhal": ["#0f7a52", "#7fd8b4", "#1baf7a"],
    "Rosebrugh": ["#4a3ab8", "#aaa0f0", "#6250d6"],
}
OTHER_COLORS = ["#6b6a66", "#bdbcb6", "#8f8e89"]
CAR_MARKERS = ["o", "^", "s"]
SHORT_NAME = {"Mechanical (MC)": "MC"}


def car_styles(df, buildings):
    """(building, car) -> (color, marker, legend label)."""
    styles = {}
    for b in buildings:
        cars = sorted(df.loc[df["building"] == b, "car"].fillna("").astype(str).unique())
        shades = BUILDING_COLORS.get(b, OTHER_COLORS)
        for j, c in enumerate(cars):
            name = SHORT_NAME.get(b, b)
            label = f"{name} · {c}" if c and len(cars) > 1 else name
            styles[(b, c)] = (shades[j % len(shades)], CAR_MARKERS[j % len(CAR_MARKERS)], label)
    return styles


def plot_features(df, out_png, config_order):
    df = df.copy()
    df["car"] = df["car"].fillna("").astype(str)
    present = list(df["building"].dropna().unique())
    buildings = [b for b in config_order if b in present] + sorted(b for b in present
                                                                   if b not in config_order)
    styles = car_styles(df, buildings)
    rng = np.random.default_rng(0)

    ncol = 6
    nrow = int(np.ceil(len(FEATURES) / ncol))
    fig, axes = plt.subplots(nrow, ncol, figsize=(4.0 * ncol, 3.1 * nrow))
    for ax, (col, label) in zip(axes.flat, FEATURES):
        for i, b in enumerate(buildings):
            sub = df[df["building"] == b]
            for (bb, c), (color, marker, _) in styles.items():
                if bb != b:
                    continue
                vals = sub.loc[sub["car"] == c, col].dropna().values
                ax.scatter(i + rng.uniform(-0.2, 0.2, len(vals)), vals, s=22, alpha=0.85,
                           color=color, marker=marker, edgecolor="white", linewidth=0.5, zorder=2)
            allv = sub[col].dropna().values
            if len(allv):
                ax.hlines(allv.mean(), i - 0.28, i + 0.28, color="#222222", lw=1.8, zorder=3)
        ax.set_xticks(range(len(buildings)))
        ax.set_xticklabels([SHORT_NAME.get(b, b) for b in buildings], fontsize=9)
        ax.set_xlim(-0.6, len(buildings) - 0.4)
        ax.set_title(label, fontsize=10.5)
        ax.tick_params(axis="y", labelsize=8.5, colors="#52514e")
        ax.grid(True, axis="y", color="#e4e3df", lw=0.8)
        ax.set_axisbelow(True)
        for sp in ("top", "right"):
            ax.spines[sp].set_visible(False)
        for sp in ("left", "bottom"):
            ax.spines[sp].set_color("#b5b4ae")
    for ax in axes.flat[len(FEATURES):]:
        ax.set_visible(False)

    handles = [Line2D([], [], ls="", marker=m, color=c, markersize=8, markeredgecolor="white",
                      label=lab) for (c, m, lab) in styles.values()]
    handles.append(Line2D([], [], color="#222222", lw=1.8, label="building mean"))
    fig.legend(handles=handles, loc="upper center", ncol=len(handles), frameon=False, fontsize=10,
               bbox_to_anchor=(0.5, 0.985))
    fig.suptitle(f"Per-ride features by building  ({len(df)} rides; dots = rides)", fontsize=15,
                 y=0.998)
    plt.tight_layout(rect=(0, 0, 1, 0.965))
    fig.savefig(out_png, dpi=100)
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
    fig.suptitle("Auto-segmentation check  —  blue = pressure, green = detected ride (motion start→stop),\n"
                 "orange = |vertical speed| (dashed = 0.30 m/s), red / purple dotted = door close / open",
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
        if building_floors and building not in building_floors:
            print(f"{name:44} {'-':>5} {'-':>6} {'-':>6} {'-':>8}  skipped (not a target building)")
            continue
        car = str(sj.get("car", "") or "").strip()
        if car.upper() in ("NA", "N/A", "NONE"):          # single-car buildings: blank or "NA"
            car = ""
        # prefer the config floor order; fall back to the session's own list
        floors = building_floors.get(building) or sj.get("floors", [])

        dt = np.diff(F["t_us"].values)
        hz = FAST_HZ if len(dt) == 0 else 1e6 / np.median(dt)
        pres_pct = 100.0 * F["pres"].notna().mean()

        ev = load_events(os.path.join(sess, "events.csv"))
        windows = travel_windows(ev)
        sig = session_signals(F)
        sig["t0_us"] = F["t_us"].values[0]
        rides = segment_rides(F)

        spans, dropped = [], 0
        for i0, i1 in rides:
            lab, t_open = label_ride(F, i0, i1, windows, ev, floors)
            feat = ride_features(sig, S, i0, i1, t_open)
            if feat is None:
                dropped += 1
                continue
            s, e = feat.pop("_span")
            spans.append((sig["t"][s], sig["t"][e]))
            row = {"session": name, "building": building, "car": car, "ride": len(spans),
                   "t_start_s": round(sig["t"][s], 1), "t_end_s": round(sig["t"][e], 1)}
            row.update(lab)
            row.update(feat)
            # Per-floor pitch = metres travelled / floors travelled (building architecture).
            # Needs the labeled floor count, so it is blank on rides with no door label.
            nf = lab.get("n_floors_label", "")
            h = feat.get("height_m")
            row["floor_height_m"] = (round(abs(h) / nf, 2)
                                     if isinstance(nf, int) and nf > 0 and np.isfinite(h) else np.nan)
            all_rows.append(row)

        # collect what the segmentation-check figure needs (times on the t_us base, in seconds)
        grid, alt_s, speed = altitude_series(F)
        if grid is not None:
            okp = F["pres"].notna().values
            seg_panels.append({
                "name": name.replace("2026-10-28_", ""),
                "grid": grid, "speed": speed,
                "pt": sig["t"][okp], "pp": F["pres"].values[okp],      # raw pressure samples
                "spans": spans,
                "doors": [(float(np.interp(c, sig["host"], sig["t"])),
                           float(np.interp(o, sig["host"], sig["t"]))) for (c, o, _, _) in windows],
                "nseg": len(spans), "nlab": len(windows),
            })

        notes = []
        if len(spans) != len(windows):
            notes.append(f"seg {len(spans)} vs {len(windows)} labeled")
        if dropped:
            notes.append(f"{dropped} dropped (truncated / no clear launch+brake)")
        if missed:
            notes.append(f"{missed} missed ticks")
        if pres_pct < 10:
            notes.append("low pressure coverage")
        if len(spans) == 0:
            notes.append("NO rides detected")
        print(f"{name:44} {hz:5.0f} {pres_pct:5.1f}% {len(spans):6d} {len(windows):8d}  {'; '.join(notes)}")

    if not all_rows:
        print("\nNo rides extracted.")
        return

    label_cols = ["session", "building", "car", "ride", "direction", "start_floor", "end_floor",
                  "n_floors_label", "direction_label", "people_on_board"]
    feat_cols = [c for c, _ in FEATURES]
    cols = label_cols + feat_cols + ["height_m", "cruise_s", "t_start_s", "t_end_s", "matched_label"]
    df = pd.DataFrame(all_rows)[cols]

    def save(fn, path):
        try:
            fn()
            return os.path.relpath(path, BASE_DIR)
        except OSError as exc:
            return f"NOT WRITTEN ({os.path.basename(path)}: {exc.strerror or exc} — close it if open, re-run)"

    csv_msg = save(lambda: df.to_csv(OUT_CSV, index=False), OUT_CSV)
    png_msg = save(lambda: plot_features(df, OUT_PNG, list(building_floors)), OUT_PNG)
    seg_msg = save(lambda: plot_segmentation_check(seg_panels, OUT_SEG_PNG), OUT_SEG_PNG)
    print("-" * 92)
    print(f"{len(df)} rides from {df['session'].nunique()} sessions, {len(feat_cols)} features -> {csv_msg}")
    print("Buildings:", ", ".join(f"{b}:{n}" for b, n in df["building"].value_counts().items()))
    print(f"Feature plot        -> {png_msg}")
    print(f"Segmentation check  -> {seg_msg}")


if __name__ == "__main__":
    main()
