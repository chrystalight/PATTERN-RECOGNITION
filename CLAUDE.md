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
  supplies the dynamic features. Hand-logged door events are ground-truth **labels only**.
- Keep it to ONE script, ONE consolidated output (`ride_features.csv`).

## Current state (2026-10-29)
- Collected: **Myhal** (cars right + left, 28 rides), **Rosebrugh** (1 car, 4 sessions, 25 rides).
  **Burton** = a non-target test elevator (5 rides), kept as its own label.
- **Still to collect: Hart House and Mechanical (MC)** — then re-run `postprocess.py` (folds in
  automatically).
- Findings: clean separators so far = **cruise speed** (Rosebrugh 0.55, Myhal 0.86, Burton
  1.55 m/s), **jerk**, **cabin light**; the two Myhal cars agree (within-building consistency
  holds). Weak: vibration RMS / dominant frequency (noisy), EMF (near-constant / effectively dead).
  `floor_height_m` ≈ 4.2–4.6 m, not strongly separating yet.
- **No filtering on acceleration** (raw samples) — jerk and the FFT features are the noisiest; a
  low-pass / Savitzky-Golay derivative and a windowed Welch PSD would stabilize them.

## Next steps
- Collect Hart House + MC; re-run `postprocess.py`.
- Implement spectral-shape features (spectral centroid / flatness / band-ratio) and up/down speed
  asymmetry — see the full candidate catalog in the working-notes doc.
- Part 2/3: classifiers (KNN, naive Bayes, 1D CNN, tree/RF) vs a threshold baseline, with PCA /
  feature selection; then run on the board.

## Working-notes doc (live project log)
https://claude.ai/artifact/FSLMuYPCVkdbEFoYDCJco7 — goals, data status, features (with physical
justification + exact calculation), flaws found, and the 40-feature candidate catalog.

## Working style (what the collaborator prefers)
- One script / one output; avoid proliferating scripts. No separability or pipeline smoke tests
  unless asked. Keep `events.csv` to real ride events only. On-board testing is done by the user.
