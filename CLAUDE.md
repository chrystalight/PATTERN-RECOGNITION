# MIE 1050 — "Which Elevator Am I In?" (Pattern Recognition)

Group project: classify which of four UofT elevators a ride came from, from one ride's onboard
sensor data.

**Target classes (4):** Hart House · Mechanical (MC) · Myhal · Rosebrugh

## Repo layout
- `Data_Collector/elevator_collector.py` — tkinter capture app. Reads the board's serial stream to
  `sessions/<timestamp>_<Building>_<Car>/raw.csv` and logs ride events (door open/close, direction,
  people in/out, other) to `events.csv`, with per-session `session.json`. Building floor lists and
  custom events live in `config.json`.
- `Data_Collector/postprocess.py` — **the one analysis script.** Auto-segments each session into
  rides and writes the single source of truth `ride_features.csv` (one row per ride), plus
  `ride_features.png` (features by building) and `segmentation_check.png` (per-session pressure /
  speed / detected spans / door events). Run: `python postprocess.py`.
- `Data_Collector/plot_elevator_rides.py` — per-session sensor time series with event overlays.
- `Data_Collector/Elevator_Data_Collection_Summary.md` — assignment brief + firmware/data spec.
- `ElevatorLogger_v2/ElevatorLogger_v2.ino` — board firmware.

## Firmware / data facts
- Serial CSV over USB @ **1,000,000 baud**. Opening the port resets the ESP32, so boot-ROM garbage
  (115200 baud) arrives first — skip any line that isn't a valid `F`/`S`/`M` record.
- `F` fast rows @200 Hz: `F,t_us,ax_g,ay_g,az_g,light_raw,emf_raw,pres_Pa` (pressure present on
  ~20% of rows / ~40 Hz, empty otherwise → NaN, do not forward-fill).
- `S` slow rows @1 Hz: `temp_C, rh_pct` (gas = nan). `M` = missed ticks. `#` = header/config.
- Use `t_us` (board clock) for timing, not host time. Vertical axis = **Z** (az ≈ 0.98 g at rest).
  Pressure ≈ **12 Pa per metre**.

## Pipeline approach
- Rides are segmented from **smoothed barometric vertical speed** (robust through the
  constant-velocity cruise, where vertical accel looks identical to a dwell). The accelerometer
  supplies the dynamic features. Hand-logged door events (where logged) are labels only; **no
  feature depends on them**, so every feature is available on the board at run time.
- The baro span is refined to true motion start/stop from the accel launch/brake peaks (10% of
  peak). Rides cut off by the session start/end, or with no real launch/brake, are dropped.
- Vertical axis = the session's resting gravity vector (orientation/tilt-proof). Motion profile =
  1.5 Hz zero-phase Butterworth low-pass. Velocity = integrated raw accel, drift removed (matches
  the barometer to ~5%). Jerk = least-squares slope of the 20–80% launch/brake ramps. Lateral sway
  is divided by the sway in the dwell just before the ride. 200 Hz sampling aliases 120 Hz flicker
  to 80 Hz and mains harmonics to 60/80/20/40 Hz.
- **25 features**, phase-based (launch = start of ride, brake = end, regardless of direction):
  cruise speed; launch/brake accel, ratio, rise/release time, ramp jerk, plateau; peak jerk; launch
  phase; accel-distance fraction; duration; post-stop wobble peak/RMS + bounce frequency; sway/rest;
  light mean/std/flicker index; EMF mean/AC RMS/mains fraction; temp. Extra columns: `direction`
  (classifier input), `height_m` (for the floor-lattice feature), `floor_height_m` (analysis only).
- Dropped after the cross-day check: vertical vibration (RMS, centroid, band power — unstable even
  normalized), speed ripple, RH, EMF strongest harmonic, door pop/step, EMF-accel correlation,
  flicker frequency, temp/RH-vs-altitude slopes. Levelling creep was tested and is below the
  barometer's noise, so it isn't a feature.
- Keep it to ONE script, ONE consolidated output (`ride_features.csv`). `ride_features.png` puts
  one column per building, with cars in the same building shown as shades of that building's hue.

## Current state (2026-10-29)
- Collected: **Myhal** (cars right + left, 10-28, 26 rides), **Rosebrugh** (1 car; 10-28 4
  sessions + 10-29 3 sessions, 48 rides), **MC** (10-29, 2 sessions, 37 rides). Only Myhal has
  more than one car; MC, Hart House and Rosebrugh each have a single car (blank `car` field).
  **Burton** was a pure test run (non-target). Its session stays in the repo, but `postprocess.py`
  skips any building not listed in `config.json`, so it is in no output, plot or analysis. Most 10-29 sessions
  have no door events, so their rides have no floor labels or floor height (not needed for classifying).
- **Hart House** (10-29, 6 sessions, 33 rides; collection complete). Very smooth car: jerk
  ~0.044 g/s (lowest), peak accel ~0.04 g, rise 0.85 s, cruise 0.74 m/s (≈ MC's 0.75). No lamp
  flicker (like MC) but light ~705 vs MC ~900. Uneven floors: one-floor rides ≈ 3.0–3.2, 3.8–3.9
  and 4.2–4.4 m (MC: 2.76 and 3.9 m).
- Empty <1 s sessions (`2026-10-29_163840_Rosebrugh_NA`, `2026-10-29_173058_Hart_House_NA`) can be
  deleted.
- Findings: most kinematic features form one tight cluster per building (within-building
  spread 1–5%). Cruise speed Myhal 0.98, MC 0.75, Hart House 0.74, Rosebrugh 0.66 up / 0.45 down.
  Brake release time: Rosebrugh 0.4, MC 0.55, Myhal 1.6, Hart House 1.7 s. Launch plateau: Rosebrugh
  0.3, Myhal 0.46, MC 0.92, Hart House 1.0 s. Brake ramp jerk: Hart House 0.023, Myhal 0.029, MC
  0.076, Rosebrugh 0.14 g/s. Bounce freq: Hart House ~5.5 Hz vs MC/Myhal ~2.2 Hz (noisy). Sway/rest
  stable across Rosebrugh days (2.40 vs 2.35). Rosebrugh brakes harder after going up (0.078 vs
  0.051 g), so direction matters there. Light flicker: none in MC or Hart House.
- Myhal_left rides 3→1 and 4→1 measure as one floor (4.58 m), which is probably a labeling error.

## Next steps
- Part 2/3: classifiers (KNN, naive Bayes, 1D CNN, tree/RF) vs a threshold baseline, with PCA /
  feature selection; then run on the board.

## Working-notes doc (live project log)
https://claude.ai/artifact/FSLMuYPCVkdbEFoYDCJco7 — goals, data status, features (with physical
justification + exact calculation), flaws found, and the candidate feature catalog (teammates edit
it, so re-read it before assuming which features are wanted).

## Working style (what the collaborator prefers)
- One script / one output; avoid proliferating scripts. No separability or pipeline smoke tests
  unless asked. Keep `events.csv` to real ride events only. On-board testing is done by the user.
