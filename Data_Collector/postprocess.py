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
supplies the dynamic features within each ride. The hand-logged door events (where logged) only
label each ride (building/floors/direction/occupancy) and sanity-check the segmenter; no feature
depends on them, so every feature is available on the board at run time.

Filtering (the raw 200 Hz accelerometer is too noisy to differentiate directly):
  * vertical axis = the session's resting gravity vector (any board orientation / tilt works).
  * motion profile  a_lp  = 4th-order zero-phase Butterworth low-pass at 1.5 Hz. Drives launch /
    brake accel, rise / release time, plateaus, jerk, phase timing and the cruise detector. 1.5 Hz keeps the S-curve
    (jerk phases last ~0.5-1 s) but removes the ~3 Hz car bounce and sensor noise; it gave the most
    repeatable per-car jerk of the cutoffs tried (1 / 1.5 / 2 / 3 Hz).
  * velocity  v  = integral of the raw vertical accel from motion start to motion end, with the
    linear drift removed so v = 0 at both ends. Its distance matches the barometer to ~5%.
  * jerk = least-squares slope of the 20-80% part of the launch / brake ramps (many samples),
    plus the max derivative of a_lp for reference.
  * lateral sway = 0.5-20 Hz band-pass of the two horizontal axes, divided by the same quantity in
    the dwell just before the ride (board placement changed raw vibration ~2x between days).
  * spectra (light flicker, EMF harmonics) are Welch PSDs (1.28 s Hann segments, 50% overlap).
Dropped after the cross-day check: vertical vibration (unstable even normalized), speed ripple, RH
(weather), EMF strongest harmonic, and the features that depended on hand-logged doors or showed
nothing (EMF-accel correlation, flicker frequency, temp & RH vs altitude).
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
REST_PRE_S = (4.0, 1.0)  # rest baseline = from 4 s to 1 s before motion start
SWAY_BAND = (0.5, 20.0)  # lateral sway band-pass
WOBBLE_BAND = (0.5, 10.0)      # post-stop wobble band-pass (car bounce on ropes / re-levelling)
WOBBLE_S = 2.0           # window after motion end
BOUNCE_S = 3.0           # bounce-frequency window after motion end
BOUNCE_BAND = (1.0, 8.0)
BOUNCE_NFFT = 4096       # zero-padded FFT -> ~0.05 Hz bins
WELCH_NPERSEG = 256      # 1.28 s segments -> 0.78 Hz resolution
FLICKER_BAND = (50.0, 99.0)    # 120 Hz flicker aliases to 80 Hz
MAINS_ALIAS_HZ = (60.0, 80.0, 20.0, 40.0)   # 60, 120, 180, 240/360 Hz folded about 100 Hz
TONE_HALF_BW = 0.8       # Hz either side of a tone when integrating its power

# slow sensors
SLOW_PAD_S = 5.0         # widen the ride window for temperature (1 Hz samples)


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
    return {
        "t": (F["t_us"].values - F["t_us"].values[0]) / 1e6,
        "host": F["host"].values,
        "pres": F["pres"].values,
        "a_v": a_v,
        "a_lp": butter(a_v, MOTION_LP_HZ, "lowpass"),
        "a_wob": butter(a_v, WOBBLE_BAND, "bandpass"),
        "h1": butter(h[0], SWAY_BAND, "bandpass"),
        "h2": butter(h[1], SWAY_BAND, "bandpass"),
        "light": light,
        "emf": emf,
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


def ramp_slope(t, a, lo_frac, hi_frac, peak):
    """|slope| (g/s) of a least-squares line through the part of a ramp between lo_frac and hi_frac of
    peak: a jerk estimate from many samples instead of one derivative maximum."""
    m = (np.abs(a) >= lo_frac * peak) & (np.abs(a) <= hi_frac * peak)
    return abs(np.polyfit(t[m], np.abs(a[m]), 1)[0]) if m.sum() >= 5 else np.nan


def plateau_s(a, k, peak):
    """Length (s) of the contiguous stretch around index k where |a| stays >= 90% of peak."""
    hi = np.abs(a) >= RISE_FRAC * peak
    lo = k
    while lo > 0 and hi[lo - 1]:
        lo -= 1
    up = k
    while up < len(a) - 1 and hi[up + 1]:
        up += 1
    return (up - lo + 1) / FAST_HZ


def ride_features(sig, S, i0, i1):
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
    kl, kb = k_l - s, k_b - s                             # launch / brake peak, ride-relative
    launch, brake = abs(a_lp[kl]), abs(a_lp[kb])
    rise = np.where(np.abs(a_lp[:kl + 1]) >= RISE_FRAC * launch)[0]
    rise_s = tt[rise[0]] - tt[0] if len(rise) else np.nan
    release = np.where(np.abs(a_lp[kb:]) >= RISE_FRAC * brake)[0]
    release_s = tt[-1] - tt[kb + release[-1]] if len(release) else np.nan

    # velocity by integrating raw accel, drift removed so it is 0 at rest on both ends
    v = np.cumsum(sig["a_v"][s:e + 1] * dt) * G
    v -= np.linspace(0.0, v[-1], len(v))
    dist = np.abs(v) * dt

    # cruise = longest constant-velocity stretch between the launch and brake peaks
    quiet = np.zeros(len(a_lp), bool)
    quiet[kl:kb] = np.abs(a_lp[kl:kb]) < CRUISE_ACCEL_THR
    run = longest_run(quiet)
    cruise_s = (run[1] - run[0]) / FAST_HZ if run else 0.0
    has_cruise = cruise_s >= MIN_CRUISE_S
    accel_end = run[0] if has_cruise else int(np.argmax(np.abs(v)))

    # Launch = the accel phase that starts the ride, brake = the one that ends it, whatever the
    # direction (up/down peaks would mix the two and split each car into two clusters).
    f = {
        "direction": "up" if up else "down",
        "duration_s": r(dur, 2),
        "height_m": r(height, 2),
        "cruise_speed_mps": r(np.median(np.abs(v[run[0]:run[1]])) if has_cruise
                              else np.abs(v).max(), 3),
        "launch_accel_g": r(launch, 4),
        "brake_accel_g": r(brake, 4),
        "accel_decel_ratio": r(launch / brake, 3) if brake > 0 else np.nan,
        "accel_rise_s": r(rise_s, 2),
        "brake_release_s": r(release_s, 2),
        "launch_jerk_g_s": r(ramp_slope(tt[:kl + 1], a_lp[:kl + 1], 0.2, 0.8, launch), 3),
        "brake_jerk_g_s": r(ramp_slope(tt[kb:], a_lp[kb:], 0.2, 0.8, brake), 3),
        "peak_jerk_g_s": r(np.abs(jerk).max(), 3),
        "launch_plateau_s": r(plateau_s(a_lp, kl, launch), 2),
        "brake_plateau_s": r(plateau_s(a_lp, kb, brake), 2),
        "launch_phase_s": r(tt[accel_end] - tt[0], 2),
        "accel_dist_frac": r(dist[:accel_end].sum() / dist.sum(), 3) if dist.sum() > 0 else np.nan,
        "cruise_s": r(cruise_s, 2),
        "sway_rest_ratio": np.nan,
    }

    # Cruise lateral sway divided by the sway in the dwell just before the ride: the board's
    # placement sets how much motion couples in, and the ratio cancels it (stable across days).
    # Vertical vibration (RMS, centroid, band power) stayed day/placement-dependent even
    # normalized this way, so it is not a feature.
    pre = slice(max(0, s - int(REST_PRE_S[0] * FAST_HZ)), max(0, s - int(REST_PRE_S[1] * FAST_HZ)))
    if has_cruise and pre.stop - pre.start > FAST_HZ:
        c0, c1 = s + run[0], s + run[1]
        rest_sway = np.sqrt(np.mean(sig["h1"][pre] ** 2 + sig["h2"][pre] ** 2))
        sway = np.sqrt(np.mean(sig["h1"][c0:c1] ** 2 + sig["h2"][c0:c1] ** 2))
        f["sway_rest_ratio"] = r(sway / rest_sway, 3) if rest_sway > 0 else np.nan

    # post-stop wobble: residual bounce once the brake phase has died out, and its frequency
    # (rope bounce: set by rope stiffness and car mass)
    wob = sig["a_wob"][e:e + int(WOBBLE_S * FAST_HZ)]
    f["post_stop_wobble_g"] = r(np.abs(wob).max(), 4) if len(wob) else np.nan
    f["post_stop_wobble_rms_g"] = r(np.sqrt(np.mean(wob ** 2)), 5) if len(wob) else np.nan
    w = sig["a_v"][e:e + int(BOUNCE_S * FAST_HZ)]
    f["bounce_freq_hz"] = np.nan
    if len(w) == int(BOUNCE_S * FAST_HZ):
        w = signal.detrend(w) * np.hanning(len(w))
        fq = np.fft.rfftfreq(BOUNCE_NFFT, 1 / FAST_HZ)
        A = np.abs(np.fft.rfft(w, BOUNCE_NFFT))
        m = (fq >= BOUNCE_BAND[0]) & (fq <= BOUNCE_BAND[1])
        f["bounce_freq_hz"] = r(fq[m][np.argmax(A[m])], 2)

    # light: level, variance, and the 120 Hz flicker depth (aliased to 80 Hz)
    light = sig["light"][s:e + 1]
    fr, P = welch(light)
    m = (fr >= FLICKER_BAND[0]) & (fr <= FLICKER_BAND[1])
    fk = fr[m][np.argmax(P[m])]
    amp = np.sqrt(2 * band_power(fr, P, fk - TONE_HALF_BW, fk + TONE_HALF_BW))
    f["light_mean"] = r(light.mean(), 1)
    f["light_std"] = r(light.std(), 2)
    f["light_flicker_index"] = r(amp / light.mean(), 4) if light.mean() > 0 else np.nan

    # EMF: level, AC RMS, mains-harmonic content
    emf = sig["emf"][s:e + 1]
    fr, P = welch(emf)
    tones = [band_power(fr, P, h - TONE_HALF_BW, h + TONE_HALF_BW) for h in MAINS_ALIAS_HZ]
    total = band_power(fr, P, 1.0, 99.0)
    f["emf_mean"] = r(emf.mean(), 2)
    f["emf_rms_ac"] = r(np.sqrt(np.mean((emf - emf.mean()) ** 2)), 3)
    f["emf_mains_frac"] = r(sum(tones) / total, 3) if total > 0 else np.nan

    # cabin temperature (RH dropped: it follows the weather, not the building)
    ts = (S["t_us"].values - sig["t0_us"]) / 1e6
    m = (ts >= t[s] - SLOW_PAD_S) & (ts <= t[e] + SLOW_PAD_S) & np.isfinite(S["temp"].values)
    f["temp_mean_c"] = r(np.mean(S["temp"].values[m]), 2) if m.any() else np.nan
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
# Every feature as a strip of per-ride dots, one column per building. Cars of the same building
# share the column and the building's hue, told apart by shade + marker. Black bar = building mean.
FEATURES = [
    # kinematics (motion profile)
    ("cruise_speed_mps", "Cruise speed (m/s)"),
    ("launch_accel_g", "Launch accel (g)"),
    ("brake_accel_g", "Brake accel (g)"),
    ("accel_decel_ratio", "Launch / brake accel ratio"),
    ("accel_rise_s", "Launch rise time (s)"),
    ("brake_release_s", "Brake release time (s)"),
    ("launch_jerk_g_s", "Launch ramp jerk (g/s)"),
    ("brake_jerk_g_s", "Brake ramp jerk (g/s)"),
    ("peak_jerk_g_s", "Peak jerk (g/s)"),
    ("launch_plateau_s", "Launch accel plateau (s)"),
    ("brake_plateau_s", "Brake accel plateau (s)"),
    ("launch_phase_s", "Launch phase, start to cruise (s)"),
    ("accel_dist_frac", "Accel-distance fraction"),
    ("duration_s", "Ride duration (s)"),
    # stop
    ("post_stop_wobble_g", "Post-stop wobble peak (g)"),
    ("post_stop_wobble_rms_g", "Post-stop wobble RMS (g)"),
    ("bounce_freq_hz", "Post-stop bounce freq (Hz)"),
    # lateral sway (placement-normalized)
    ("sway_rest_ratio", "Lateral sway / rest"),
    # light, EMF, climate
    ("light_mean", "Cabin light (ADC)"),
    ("light_std", "Light std (ADC)"),
    ("light_flicker_index", "Light flicker index"),
    ("emf_mean", "EMF mean (ADC)"),
    ("emf_rms_ac", "EMF AC RMS (ADC)"),
    ("emf_mains_frac", "EMF mains-harmonic fraction"),
    ("temp_mean_c", "Cabin temp (°C)"),
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
            lab = label_ride(F, i0, i1, windows, ev, floors)
            feat = ride_features(sig, S, i0, i1)
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
    # height_m feeds the floor-lattice feature (fit per CV fold in the classifier stage);
    # floor_height_m needs logged floors, so it is for analysis only, not a classifier input
    cols = label_cols + feat_cols + ["height_m", "floor_height_m", "cruise_s", "t_start_s", "t_end_s",
                                     "matched_label"]
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
