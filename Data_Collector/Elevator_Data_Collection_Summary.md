# Which Elevator Am I In? Data Collection Summary

MIE 1050 group assignment. This document covers the assignment at a high level, the data collection requirements in detail, and what the Elevator Logger firmware (`ElevatorLogger_v2.ino`) outputs.

## 1. Assignment in brief

**Goal:** build a classifier that identifies which elevator a ride was taken in, from one ride's sensor data.

**Classes (4):**
- Hart House
- Mechanical Building (MC)
- Myhal Centre
- Rosebrugh Building

| Part | What it asks for |
|---|---|
| 1. Data collection (groups) | Record rides, log metadata, write a script that detects each ride's start/stop from the accelerometer and cuts the recording into one segment per ride |
| 2. Features | At least 8 features, each justified physically (peak accel/decel, jerk, cruise speed, cruise-phase vibration frequencies from an FFT, etc.) |
| 3. Classifiers | KNN, naive Bayes, 1D CNN, and a decision tree or random forest; compare against a simple heuristic baseline (e.g. a threshold on peak acceleration); apply PCA or wrapper feature selection and report which features survive |
| 4. Implementation | Run the algorithm on the board |

**Deliverables:**
1. A MATLAB or Python classification algorithm. *The brief says "classify the occupant status based on the 4 input parameters", which does not match the rest of the assignment. Confirm with the instructor or TA what this means.*
2. Board firmware, uploaded to the board.
3. A 2 to 4 page summary of the method and why it was chosen, including remaining issues in classification accuracy.

**Current focus: Part 1, data collection only.**

## 2. Data collection requirements (Part 1)

### Quantity
- At least **20 rides per elevator**, so at least **80 rides per group**.

### Conditions to vary on purpose
- Direction: up and down
- Distance: 1 floor and several floors
- Load: busy and empty cabins
- Time of day

### Cars
- Where a building has several cars, record from **more than one**, so the classifier learns the building rather than one car.

### Metadata to log for every ride
| Field | Notes |
|---|---|
| Building | Hart House, MC, Myhal, or Rosebrugh |
| Car | Which elevator car within the building |
| Direction | Up or down |
| Start floor | |
| End floor | |
| Approx. number of people | |
| Board orientation | Which way the board is facing/mounted in the cabin |
| Date | |

### Segmentation script
- Detect each ride's start and stop **from the accelerometer** and cut the recording into one segment per ride.
- The logger records continuously, so rides are cut offline from the log.

### Why the logger is dual-rate
The features in Part 2 need different sample rates from different sensors:
- **Accelerometer:** peak accel/decel and jerk need a fast rate, and the FFT of cruise vibration needs a high rate.
- **Pressure:** cruise speed is estimated from the pressure rate (about 0.12 hPa per metre), which needs steady pressure readings at 10 Hz or more with the gas heater off.
- **Temperature and humidity:** slow-changing, so a slow stream is enough.

## 3. Elevator Logger output

### Connection
- Serial over USB at **921600 baud**. The capture tool must use the same baud rate.
- Comma-separated text, one record per line. The **first column is the record type**.
- Timestamps `t_us` are microseconds since boot (64-bit, so they don't wrap).

### Record types

| Type | Columns | Rate | Notes |
|---|---|---|---|
| `F` (fast) | `F,t_us,ax_g,ay_g,az_g,light_raw,emf_raw,pres_Pa` | `FAST_PERIOD_US` (placeholder 5000 = 200 Hz) | `pres_Pa` is **empty** unless a new pressure reading arrived on that tick |
| `S` (slow) | `S,t_us,temp_C,rh_pct,gas_kOhm` | `SLOW_PERIOD_US` (placeholder 1,000,000 = 1 Hz) | `gas_kOhm` is `nan` unless `LOG_GAS` is set to 1 |
| `M` (missed) | `M,t_us,missed_ticks` | Only when it happens | Means the loop fell behind and skipped fast ticks. Should normally never appear |
| `#` | Free text | At start | Header and config lines. Ignore as data |

### Units
| Column | Unit |
|---|---|
| `ax_g`, `ay_g`, `az_g` | g (about 1 g on the vertical axis when stationary) |
| `light_raw`, `emf_raw` | ADC counts. Volts per count is printed in the header (`adc_volts_per_count`) |
| `pres_Pa` | Pascals |
| `temp_C` | Degrees C (from the AHT20; falls back to the BME680 if no AHT20) |
| `rh_pct` | Percent relative humidity (`nan` if no AHT20) |
| `gas_kOhm` | Kilo-ohms (only if `LOG_GAS` is 1) |

### Example (illustrative values only)
```
# Elevator Logger v2
# F,t_us,ax_g,ay_g,az_g,light_raw,emf_raw,pres_Pa
# S,t_us,temp_C,rh_pct,gas_kOhm
# M,t_us,missed_ticks
# config fast_period_us=5000 slow_period_us=1000000 baud=921600 log_gas=0 aht=1
# adc_volts_per_count=0.003000
F,5012345,0.0021,-0.0103,1.0004,812,-14,101325.40
F,5017346,0.0019,-0.0098,1.0011,811,-12,
F,5022344,0.0025,-0.0101,0.9998,813,-15,
S,5102210,22.31,38.42,nan
F,5027350,0.0301,-0.0099,1.0420,812,-13,101325.31
```

### Things to know when processing the data
- **Pressure is sparse.** Rows without a new reading have an empty pressure field, which pandas reads as NaN. Drop or interpolate as needed.
- **Fast rows are not perfectly uniform in time.** Use `t_us`, not the row index, and resample if a uniform grid is needed.
- **The accelerometer output rate** should match the fast rate. Slight clock differences mean an occasional repeated or skipped sample is possible.
- **Temperature and humidity (`S` rows)** arrive about 90 ms after the AHT20 is triggered, and their timestamp is the readback time.
- **Boot time resets** the timestamp, so each capture session starts a new time base.

## 4. Status and open items

- [x] Dual-rate logger firmware written (`ElevatorLogger_v2.ino`); compile fix applied for the Arduino prototype error.
- [ ] Run the timing test on the board and pick `FAST_PERIOD_US` and the matching `BMA_ODR`.
- [x] Python capture script (serial to timestamped CSV, plus per-ride metadata/event entry): `elevator_collector.py` (see `README.md`). Needs on-board validation.
- [ ] Ride segmentation script (accelerometer-based start/stop detection).
- [ ] Confirm the wording of deliverable 1 ("occupant status", "4 input parameters").
- [ ] Decide board mounting and orientation, and keep it consistent or record it per ride.
