"""
Checks that the firmware's classifier (ElevatorLogger_v2/classifier.h) matches the Python pipeline before
anything goes on the board.

It compiles board_sim.cpp (the same C++ code the board runs) with g++, replays every recorded session
through it sample by sample, and compares each ride with ride_features.csv:
  - are the same rides found?
  - do launch jerk and light flicker agree with postprocess.py?
  - which building does the board name, and does its cross-check pass?

Run from this folder:  python check_board.py      (needs g++ on the PATH)
"""
import os
import subprocess
import pandas as pd

HERE = os.path.dirname(os.path.abspath(__file__))
REPO = os.path.dirname(os.path.dirname(HERE))
SESSIONS = os.path.join(REPO, "Data_Collector", "sessions")
EXE = os.path.join(HERE, "board_sim.exe" if os.name == "nt" else "board_sim")
SHORT = {"Mechanical (MC)": "MC"}
MATCH_S = 1.0          # a board ride and a Python ride are the same ride if their starts are within 1 s

# 1. compile the board code for the PC
subprocess.run(["g++", "-O2", "-std=c++17", "-o", EXE, os.path.join(HERE, "board_sim.cpp")], check=True)

# 2. replay every session the Python pipeline used
python_rides = pd.read_csv(os.path.join(REPO, "Data_Collector", "ride_features.csv"))
rows = []
for session, py in python_rides.groupby("session", sort=False):
    out = subprocess.run([EXE, os.path.join(SESSIONS, session, "raw.csv")],
                         capture_output=True, text=True, check=True).stdout
    board = pd.read_csv(pd.io.common.StringIO(out))
    board = board[board["valid"] == 1]
    for _, ride in py.iterrows():
        gap = (board["t_start_s"] - ride["t_start_s"]).abs()
        b = board.loc[gap.idxmin()] if len(board) and gap.min() <= MATCH_S else None
        rows.append({
            "session": session, "ride": ride["ride"],
            "building": SHORT.get(ride["building"], ride["building"]),
            "py_jerk": ride["launch_jerk_g_s"], "py_flicker": ride["light_flicker_index"],
            "found": b is not None,
            "board_jerk": b["launch_jerk_g_s"] if b is not None else None,
            "board_flicker": b["light_flicker_index"] if b is not None else None,
            "board_building": b["building"] if b is not None else None,
            "agree": bool(b["agree"]) if b is not None else None,
        })
    print(f"{session:44s} python {len(py):3d} rides, board {len(board):3d} rides")

# 3. compare
r = pd.DataFrame(rows)
found = r[r["found"]]
jerk_err = (found["board_jerk"] - found["py_jerk"]).abs()
flicker_err = (found["board_flicker"] - found["py_flicker"]).abs()
print(f"\nRides found by the board: {len(found)} of {len(r)}")
print(f"Launch jerk  |board - python|: median {jerk_err.median():.4f}, max {jerk_err.max():.4f} g/s "
      f"(median relative {(jerk_err / found['py_jerk']).median():.1%})")
print(f"Flicker      |board - python|: median {flicker_err.median():.5f}, max {flicker_err.max():.5f}")
print(f"Board names the right building: {(found['board_building'] == found['building']).mean():.1%} "
      f"of found rides")
print(f"Cross-check passes: {found['agree'].mean():.1%}")
wrong = found[(found["board_building"] != found["building"]) | ~found["agree"].astype(bool)]
if len(wrong):
    print("\nRides the board got wrong or flagged:")
    print(wrong[["session", "ride", "building", "board_building", "py_jerk", "board_jerk", "agree"]].to_string())
missing = r[~r["found"]]
if len(missing):
    print("\nRides the board did not find:")
    print(missing[["session", "ride", "building"]].to_string())
