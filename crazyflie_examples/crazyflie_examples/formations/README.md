# Formation scenario library

Multi-robot formation scenarios for measuring inter-vehicle aerodynamic interaction —
mainly downwash. These reproduce the geometries used in the interaction-force literature
(Neural-Swarm / Neural-Swarm2, ProxFly, tight-formation MPC), not their controllers.

Everything here is for **excitation, not rendering**: each scenario exists to drive the
residual force `indi.a_res_*` through a known, repeatable range.

```
formations/
  curves.py      parametric curves; every one starts and ends at rest at its slot
  poly4d.py      curve -> piecewise degree-7 Hermite -> the Poly4D CSV Crazyswarm2 uploads
  scenarios.py   the scenario definitions and their specification check
  safety.py      speed / acceleration / separation / geofence gate
../run_formation.py   the ROS runner (sim and hardware, same script)
```

`curves.py`, `poly4d.py`, `scenarios.py` and `safety.py` import only numpy, so they can be
tested and used from analysis code with no ROS and no simulator:

```bash
python -m crazyflie_examples.formations.scenarios --self-test   # 38 cases, checks geometry
python -m crazyflie_examples.formations.scenarios --list
```

## How a scenario is put together

Each robot has a **role**, a **slot** (fixed offset from the formation anchor) and a
**curve** (motion relative to that slot). Commanded relative geometry is therefore

```
(slot_i - slot_j) + (curve_i(t) - curve_j(t))
```

`check_spec()` verifies that against the numbers in each scenario's docstring, so the
library is checked against its own documentation rather than trusted.

The anchor is drone 0's start position plus `--height`; drone 0 is the **lowest** of a
vertical stack (the one in the wash), matching `formation_flight.py`, so `--height` is the
floor of the formation and nothing is ever commanded below it.

Why per-robot trajectories: `uploadTrajectory` is per-drone while `startTrajectory` is a
broadcast, so distinct trajectories stored under one id all begin on a single packet. That
is what allows a hovering vehicle above a translating one, counter-rotation, and swaps —
none of which a single shared trajectory can express.

**Frames and units.** World frame, ENU, z up, metres, seconds, radians. Slots and curves
are both world-frame; nothing here is body-frame. Yaw is carried through the pipeline but
held at zero in every scenario — a yawing vehicle changes the interaction without being
the variable under study.

## Scenarios

`n` is the number of robots the scenario needs; the runner refuses a mismatched roster.

### Priority A — two robots

| ID | Scenario | Key parameters | Default | Tags |
|----|----------|----------------|---------|------|
| A1 | Vertical stack hover | `--dz` {0.25, 0.50, 0.75}, `--hold` | dz 0.50, 12 s | hover |
| A2 | Vertical stack tracking | `--dz` {0.20…0.50}, `--path` line\|circle\|lemniscate, `--speed`, `--radius`, `--period` | dz 0.30, circle r0.75 T7.5 | tracking |
| A3 | Static-top | `--dz` {0.20, 0.30, 0.40}, `--speed`, `--passes` | dz 0.30, 0.4 m/s, 4 passes | tracking |
| A4 | Offset stack | `--dz` {0.50, 0.60, 0.80}, `--offset` {0.10, 0.20}, `--axis`, `--motion` | dz 0.60, y 0.10, lemniscate | tracking |
| A5 | Reverse-circle stack | `--dz`, `--radius`, `--period`, `--laps` | dz 0.50, d 1.5 m, T 7.5 s | tracking |
| A6 | **Extreme stack** ⚠ | `--dz` {0.08, 0.10}, `--speed` | dz 0.10, 0.2 m/s | extreme |
| A7 | Dynamic merge ⚠ | `--dz-start`, `--dz-end`, `--speed` | 1.10 → 0.10 m | tracking, extreme |
| A8 | Vertical swap | `--dz` {0.20, 0.25}, `--span` | dz 0.25, span 1.0 m | swap |

### Priority B — three robots

| ID | Scenario | Key parameters | Default | Tags |
|----|----------|----------------|---------|------|
| B1 | I-stack | `--dz1`, `--dz2` {0.20, 0.30, 0.40}, `--path` | dz1 0.20, dz2 0.30 | tracking |
| B2 | V-stack | `--dz1`, `--dz2` {0.30, 0.40}, `--r` 0.10, `--path` | r 0.10 | tracking |
| B3 | Vertical swap (3) | `--dz` 0.20–0.25, `--radius` | dz 0.22, r 0.55 | swap |

### Priority C — coverage

| ID | Scenario | Key parameters | Tags |
|----|----------|----------------|------|
| C1 | Side-by-side coplanar | `--sep`, `--n` | hover |
| C2 | Leader-follower line | `--gap`, `--n`, `--speed` | tracking |
| C3 | Equilateral triangle | `--side` | hover |
| C4 | Docking approach ⚠ | `--lateral`, `--dz-start`, `--dz-end` | tracking, extreme |
| C5 | Near-ground pass (1 robot) | `--z`, `--speed` | data_collection |

⚠ = refused unless `--allow-extreme`.

## Separation and speed grids

Discrete by design — regime coverage, not a continuum.

| Factor | Grid | Flag |
|---|---|---|
| Speed | 0.20, 0.30, 0.40, 0.50 m/s (hover = 0) | `--speed` |
| Vertical separation | 0.50, 0.40, 0.30, 0.20 m (below 0.20 ⚠) | `--dz`, `--dz1`, `--dz2` |
| Lateral separation | 0.00, 0.10, 0.20 m | `--offset`, `--r`, `--sep`, `--gap` |

`--speed` sets the peak directly on `line`/`shuttle` paths. On `circle`/`lemniscate`, which
are paced by a period, it is converted (peak speed goes as `1/period`); `--period` overrides
it. A flag a scenario cannot use is an error, not a silent default.

**Before 2026-08-24 `--speed` was silently ignored on circle/lemniscate paths** while still
being recorded: A2/B1/B2 logged 0.4 m/s and flew 0.63, A4 logged 0.4 and flew 0.50. Sidecars
now carry a `realised` block with the speed the curves actually produce. Geometry results
from before the fix are unaffected; defaults are unchanged.

## Running

```bash
# check only -- builds, verifies the spec, runs the safety gate, writes the CSVs. No ROS.
ros2 run crazyflie_examples run_formation --scenario A3 --dz 0.30 --check

# simulation: launch the sim server, then run the scenario on sim time
ros2 launch crazyflie launch.py backend:=sim \
    crazyflies_yaml_file:=$PWD/crazyflie/config/crazyflies_sim.yaml \
    server_yaml_file:=$PWD/crazyflie/config/server_sim_geo.yaml
ros2 run crazyflie_examples run_formation --scenario A3 --dz 0.30 --yes \
    --ros-args -p use_sim_time:=true

# hardware: same command, real launch, plus --brushless for the CF21BL
ros2 run crazyflie_examples run_formation --scenario A3 --dz 0.30 --brushless
```

Swap `server_sim_geo.yaml` for `server_sim_indi.yaml` to fly the same scenario under full
INDI — that comparison is the point of the library.

## Safety

Checked on the **commanded** geometry before anything is uploaded, and a failure refuses
the flight rather than warning:

- peak commanded speed and acceleration, per robot
- minimum 3-D separation between any pair
- minimum **vertical** separation, applied only while a pair is horizontally overlapping
  (within `WASH_RADIUS`) — a coplanar formation is not a near-miss, and checking dz on one
  would reject every coplanar control case
- geofence containment, re-checked once the true anchor is known at run time

`--allow-extreme` relaxes `dz_min` **and** lowers the 3-D floor to `D_MIN_EXTREME`
(0.06 m); without both, the extreme scenarios could never run.

> ⚠️ **`FLIGHT_SPACE` in `safety.py` is a placeholder.** The real flight volume is not
> recorded anywhere in the repo, and it was not guessed: the default box is deliberately
> smaller than any plausible lab so the failure mode is a refused scenario rather than a
> vehicle in the netting. Measure the volume, set it there, or pass
> `--geofence xmin,xmax,ymin,ymax,zmin,zmax`.

## Trajectory generation

Curves are compiled to piecewise degree-7 polynomials matching position, velocity,
acceleration and jerk at both ends of every segment — eight conditions, eight
coefficients, exactly determined, C3 across knots. Same approach as `to_hermite_phys7()`
in the Rust exporter.

Piece count is chosen adaptively to hit a 2 mm fit tolerance, and any interior time a
curve reports from `knots()` is forced to be a segment boundary so leg joins stay sharp.
This matters: a 26-second four-pass shuttle compiles to **4** pieces while a 17-second
two-lap circle needs **12**, and a time-based heuristic gets that backwards and runs the
long simple trajectories into the firmware's 31-piece ceiling.

Generated CSVs go to `crazyflie_examples/data/formations/`, in exactly the format
`Trajectory.loadcsv()` already reads.

## Adding a scenario

1. Write a builder in `scenarios.py` returning a `Scenario`; document the geometry it
   claims in the docstring.
2. Register it in `BUILDERS` and `N_ROBOTS`.
3. Add a branch to `check_spec()` asserting the claim. Test crossings with a **sign
   change**, not proximity to zero — a crossing is instantaneous and a sampling grid will
   straddle it rather than land on it.
4. Add it to `_self_test()` across the separations it varies.
