# Elevator Data Collector

Capture + event-labeling app for Part 1 of the "Which Elevator Am I In?" assignment. It reads the
ElevatorLogger_v2 serial stream to a timestamped `raw.csv` **and** lets you log ride events
(door open/close, direction, people in/out, odd conditions) that are time-aligned to the sensor
data. You set the direction and the buttons pressed inside the cabin, and the app infers which
floor you arrive at on each stop.

## Setup (once)

```
pip install -r requirements.txt
```

Needs Python 3 with tkinter (bundled with the standard Windows python.org installer).

## Run

```
python elevator_collector.py
```

1. Pick the board's **serial port** (Refresh to re-scan) and confirm the **baud** (default
   1,000,000, editable in case the firmware changes).
2. Choose the **Building**, type the **Car**, **board orientation**, and any operator notes.
3. Press **START** (or Space). The status line shows connection, F/S row counts, last `t_us`, and a
   missed-tick warning. A red banner appears if no data arrives (e.g. the firmware is still built
   with `RUN_TIMING_TEST 1`).
4. Set up the ride, then log stops:
   - **Direction** Up/Down (`U`/`D`).
   - **Buttons pressed**: tap each destination button selected inside the cabin — these light up and
     drive the floor prediction (they are *not* written to `events.csv`).
   - **DOORS OPEN** (`O`): the big centre display shows the inferred floor (next requested stop in
     your direction). If it stopped somewhere unexpected, tap **▲/▼** (or the arrow keys) to correct
     it — the correction updates that stop in place.
   - **DOORS CLOSE** (`C`): logs the departure; a served floor's button un-lights.
   - **People**: `+N` buttons for in/out (`I` = in +1, `K` = out +1). The **on board** counter shows
     the running total so you can check it against the cabin.
   - **Other**: custom-event buttons (Bike / Mobility device / Other heavy object) — tap once for
     *entered*, tap again for *left* (they stay lit while on board) — or the free-text field for a
     one-off note.
   - **UNDO last** (`Z`) fixes a mis-tap.
5. Press **STOP** (or Space / close the window) to end the session.

## Output

Each session is a folder under `sessions/<date_time>_<Building>_<Car>/`:

- `raw.csv` — sensor stream; well-formed data rows are `host_epoch,<original F/S/M line>` and
  firmware `#` header lines are kept verbatim. Boot-ROM garbage (the ESP32 resets when the port
  opens) and any malformed rows are skipped and counted (shown as "skipped" in the status line),
  not written as data.
- `events.csv` — `host_epoch,iso_time,t_us_at_event,event_type,value,note`, one real ride event per
  row (`session_start`/`session_end`, `door_open`/`door_close` with the floor as `value`,
  `direction`, `people_in`/`people_out` with the `+N` delta as `value` and `on_board=N` in `note`,
  and `other`). Button presses and floor corrections are not separate rows — corrections edit the
  relevant `door_open` row in place.
- `session.json` — building, car, orientation, operator notes, port/baud, floor list, firmware
  config header, start/stop times.

`host_epoch` is the shared clock between the two CSVs; `t_us_at_event` anchors each event to the
sensor timeline for the later segmentation step.

## Configuring floors

Floor buttons per building live in `config.json` (created on first run with generic placeholders).
Edit the `buildings` lists on-site to match each building's real floors, and `custom_events` for
the quick "other condition" buttons. No code change needed.
