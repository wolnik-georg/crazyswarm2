#!/usr/bin/env python3
"""
Generic CS2 flight script — runs any Rust-exported Poly4D trajectory.

Usage:
  ros2 run crazyflie_examples flight -- --trajectory figure8 --mode 0
  ros2 run crazyflie_examples flight -- --trajectory figure8 --mode 1 --kt 0.008
  ros2 run crazyflie_examples flight -- --trajectory circle  --mode 1 --kt 0.1

Arguments:
  --trajectory : figure8 | circle  (must match an exported CSV in data/)
  --mode       : 0=Spline 1=Richter 2=Se3 3=Paper (default: 0)
  --kt         : aggressiveness for modes 1/2/3 (default: 0.008 for figure8, 0.1 for circle)
  --speed      : time scale for playback — 1.0 = normal, <1 = slower (default: 1.0)
  --reps       : number of trajectory repetitions (default: 1)
  --height     : hover height in metres (default: 1.0)

How logging works:
  CS2 server reads firmware log variables over radio and publishes two ROS2 topics:
    /{cf_name}/state    — 20 Hz: pos, vel, attitude, thrust, vbat  (LogDataGeneric)
    /{cf_name}/gyro_acc — 20 Hz: gyro and accelerometer           (LogDataGeneric)

  flight.py subscribes to both topics on the same node as Crazyswarm
  (swarm.allcfs IS the ROS2 node — CrazyflieServer inherits from rclpy.node.Node).
  Callbacks fire automatically inside timeHelper.sleep() which calls
  rclpy.spin_once() in a loop — no extra threads needed.

  Rows are collected in _log_rows during the flight and written to
  RUNS_DIR/{trajectory}_mode{N}_kt{val}_{timestamp}.csv after landing.
  Column names match the Rust onboard scripts exactly → analytics unchanged.

CSV file naming (must match export_poly4d output):
  Mode 0 (speed=1.0):  {trajectory}_mode0.csv
  Mode 1/2/3:          {trajectory}_mode{N}_kt{val}.csv

Values in LogDataGeneric are in the same order as the 'vars' list in crazyflies.yaml.
The YAML vars list is defined by us, so the mapping below is authoritative.
"""

import argparse
import csv
import sys
import time
from datetime import datetime
from pathlib import Path

import numpy as np
from crazyflie_interfaces.msg import LogDataGeneric
from crazyflie_py import Crazyswarm
from crazyflie_py.uav_trajectory import Trajectory

# ── Paths ──────────────────────────────────────────────────────────────────

DATA_DIR = Path(__file__).parent / "data"

LOGS_DIR = Path(__file__).resolve().parents[3] / "flying_robot_course" / "Controls" / "logs"

# ── Logging state (single-threaded — callbacks fire inside timeHelper.sleep) ──

_log_rows       = []   # accumulated rows during flight
_log_t0         = None # wall-clock at trajectory start (None = not logging yet)
_logging_active = False
_latest_att     = {}   # most recent attitude/thrust/vbat values
_latest_gyro    = {}   # most recent gyro_acc values
_latest_rpm     = {}   # most recent per-motor RPM values

# Variable order must match crazyflies.yaml custom_topics vars lists exactly.
# Split into 3 blocks to stay within the 26-byte CRTP log packet limit (6 floats max).
_STATE_VARS    = ["stateEstimate.x", "stateEstimate.y", "stateEstimate.z",
                  "stateEstimate.vx", "stateEstimate.vy", "stateEstimate.vz"]
_ATTITUDE_VARS = ["stabilizer.roll", "stabilizer.pitch", "stabilizer.yaw",
                  "stabilizer.thrust", "pm.vbat"]
_GYRO_VARS     = ["gyro.x", "gyro.y", "gyro.z", "acc.x", "acc.y", "acc.z"]
_RPM_VARS      = ["rpm.m1", "rpm.m2", "rpm.m3", "rpm.m4"]


def _state_cb(msg: LogDataGeneric):
    """Fires at 20 Hz. Appends one log row using latest cached attitude + gyro."""
    if not _logging_active:
        return
    if len(msg.values) != len(_STATE_VARS):
        return

    s = dict(zip(_STATE_VARS, msg.values))
    a = _latest_att
    g = _latest_gyro

    _log_rows.append({
        "time_s":    round(time.monotonic() - _log_t0, 4),
        "x":         s["stateEstimate.x"],
        "y":         s["stateEstimate.y"],
        "z":         s["stateEstimate.z"],
        "vx":        s["stateEstimate.vx"],
        "vy":        s["stateEstimate.vy"],
        "vz":        s["stateEstimate.vz"],
        "roll_deg":  a.get("stabilizer.roll",   float("nan")),
        "pitch_deg": a.get("stabilizer.pitch",  float("nan")),
        "yaw_deg":   a.get("stabilizer.yaw",    float("nan")),
        "thrust":    a.get("stabilizer.thrust", float("nan")),
        "vbat":      a.get("pm.vbat",           float("nan")),
        "gyro_x":    g.get("gyro.x",  float("nan")),
        "gyro_y":    g.get("gyro.y",  float("nan")),
        "gyro_z":    g.get("gyro.z",  float("nan")),
        "acc_x":     g.get("acc.x",   float("nan")),
        "acc_y":     g.get("acc.y",   float("nan")),
        "acc_z":     g.get("acc.z",   float("nan")),
        "rpm_m1":    _latest_rpm.get("rpm.m1", float("nan")),
        "rpm_m2":    _latest_rpm.get("rpm.m2", float("nan")),
        "rpm_m3":    _latest_rpm.get("rpm.m3", float("nan")),
        "rpm_m4":    _latest_rpm.get("rpm.m4", float("nan")),
    })


def _attitude_cb(msg: LogDataGeneric):
    """Fires at 20 Hz. Caches latest attitude/thrust/vbat for the next state row."""
    if len(msg.values) != len(_ATTITUDE_VARS):
        return
    for name, val in zip(_ATTITUDE_VARS, msg.values):
        _latest_att[name] = val


def _gyro_cb(msg: LogDataGeneric):
    """Fires at 20 Hz. Caches latest gyro/acc values for the next state row."""
    if len(msg.values) != len(_GYRO_VARS):
        return
    for name, val in zip(_GYRO_VARS, msg.values):
        _latest_gyro[name] = val


def _rpm_cb(msg: LogDataGeneric):
    """Fires at 20 Hz. Caches latest per-motor RPM for the next state row."""
    if len(msg.values) != len(_RPM_VARS):
        return
    for name, val in zip(_RPM_VARS, msg.values):
        _latest_rpm[name] = val


# ── Filename helpers ────────────────────────────────────────────────────────

def _csv_label(trajectory: str, mode: int, kt: float, speed: float) -> str:
    """Consistent label matching export_poly4d naming convention."""
    if mode == 0:
        if abs(speed - 1.0) < 1e-3:
            return f"{trajectory}_mode0"
        return f"{trajectory}_mode0_speed{speed:.2f}"
    kt_str = f"{kt:.6f}".rstrip("0").rstrip(".")
    return f"{trajectory}_mode{mode}_kt{kt_str}"


def _find_csv(trajectory: str, mode: int, kt: float, speed: float) -> Path:
    label = _csv_label(trajectory, mode, kt, speed)
    path = DATA_DIR / f"{label}.csv"
    if not path.exists():
        print(f"[error] CSV not found: {path}")
        print(f"  Generate it with (from flying_drone_stack/):")
        print(f"    cargo run --release --bin export_poly4d -- "
              f"--trajectory {trajectory} --mode {mode} --kt {kt}")
        sys.exit(1)
    return path


def _save_log(trajectory: str, mode: int, kt: float, speed: float,
               reps: int, lap_time_s: float):
    LOGS_DIR.mkdir(parents=True, exist_ok=True)
    ts = datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
    label = _csv_label(trajectory, mode, kt, speed)
    path = LOGS_DIR / f"{label}_{ts}.csv"

    fieldnames = [
        "time_s", "x", "y", "z", "vx", "vy", "vz",
        "roll_deg", "pitch_deg", "yaw_deg", "thrust", "vbat",
        "gyro_x", "gyro_y", "gyro_z", "acc_x", "acc_y", "acc_z",
        "rpm_m1", "rpm_m2", "rpm_m3", "rpm_m4",
    ]
    kt_str = f"{kt:.6f}".rstrip("0").rstrip(".")
    with open(path, "w", newline="") as f:
        # meta comments — read by analyze_flight.py load_csv_with_meta()
        f.write(f"# meta:run_trajectory={trajectory}\n")
        f.write(f"# meta:run_mode={mode}\n")
        f.write(f"# meta:run_kt={kt_str}\n")
        f.write(f"# meta:run_speed={speed}\n")
        f.write(f"# meta:run_reps={reps}\n")
        f.write(f"# meta:run_lap_time_s={lap_time_s:.4f}\n")
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(_log_rows)

    print(f"[log] {len(_log_rows)} rows → {path}")


# ── Main ────────────────────────────────────────────────────────────────────

def main():
    global _log_t0, _logging_active

    parser = argparse.ArgumentParser(description="Generic CS2 flight script")
    parser.add_argument("--trajectory", default="figure8",
                        choices=["figure8", "circle", "hover"])
    parser.add_argument("--mode", type=int, default=0, choices=[0, 1, 2, 3])
    parser.add_argument("--kt",    type=float, default=None)
    parser.add_argument("--speed", type=float, default=1.0)
    parser.add_argument("--reps",  type=int,   default=1)
    parser.add_argument("--height", type=float, default=1.0)
    parser.add_argument("--duration", type=float, default=20.0,
                        help="Hover duration in seconds (hover trajectory only)")
    args, _ = parser.parse_known_args()

    if args.kt is None:
        args.kt = 0.1 if args.trajectory == "circle" else 0.008

    hover_mode = (args.trajectory == "hover")
    traj = None
    if not hover_mode:
        csv_path = _find_csv(args.trajectory, args.mode, args.kt, args.speed)
        print(f"[flight] {csv_path.name}  mode={args.mode} kt={args.kt} "
              f"speed={args.speed} reps={args.reps}")
        traj = Trajectory()
        traj.loadcsv(csv_path)
        print(f"[flight] Duration per rep: {traj.duration:.2f}s")
    else:
        print(f"[flight] hover mode — {args.duration:.0f}s at {args.height:.2f}m")

    # ── Init CS2 ────────────────────────────────────────────────────────────
    # swarm.allcfs IS the ROS2 node (CrazyflieServer inherits rclpy.node.Node).
    # Subscriptions created here are processed by timeHelper.sleep(), which
    # calls rclpy.spin_once() in a loop — no extra threads required.
    swarm   = Crazyswarm()
    th      = swarm.timeHelper
    allcfs  = swarm.allcfs          # this IS the node
    cf      = allcfs.crazyflies[0]
    cf_name = cf.prefix.lstrip("/") # e.g. "cf231"

    allcfs.create_subscription(
        LogDataGeneric, f"{cf_name}/state",    _state_cb,    10)
    allcfs.create_subscription(
        LogDataGeneric, f"{cf_name}/attitude", _attitude_cb, 10)
    allcfs.create_subscription(
        LogDataGeneric, f"{cf_name}/gyro_acc", _gyro_cb,     10)
    allcfs.create_subscription(
        LogDataGeneric, f"{cf_name}/rpm",      _rpm_cb,      10)
    print(f"[log] Subscribed to {cf_name}/state, {cf_name}/attitude, {cf_name}/gyro_acc, {cf_name}/rpm")

    # ── Wait for EKF to converge on mocap poses ──────────────────────────────
    print("[flight] Waiting for EKF to converge on mocap poses...")
    th.sleep(3.0)
    print("[flight] EKF ready. Taking off...")

    # ── Takeoff and position ─────────────────────────────────────────────────
    allcfs.takeoff(targetHeight=args.height, duration=2.0)
    th.sleep(2.5)

    for c in allcfs.crazyflies:
        pos = np.array(c.initialPosition) + np.array([0, 0, args.height])
        c.goTo(pos, 0, 2.0)
    th.sleep(2.5)

    _logging_active = True
    _log_t0 = time.monotonic()

    try:
        if hover_mode:
            # ── Hover: no trajectory upload, just hold position ───────────────
            print(f"[flight] Hovering for {args.duration:.0f}s...")
            th.sleep(args.duration)
            print("[flight] Done. Landing...")
        else:
            # ── Upload and fly trajectory ─────────────────────────────────────
            for c in allcfs.crazyflies:
                c.uploadTrajectory(0, 0, traj)

            print("[flight] Starting trajectory...")
            for rep in range(args.reps):
                if rep > 0:
                    th.sleep(1.0)
                _log_t0 = time.monotonic()
                allcfs.startTrajectory(0, timescale=args.speed)
                th.sleep(traj.duration * args.speed + 1.0)

            print("[flight] Done. Landing...")

        _logging_active = False
        allcfs.land(targetHeight=0.06, duration=2.0)
        th.sleep(3.0)

    finally:
        # ── Save log (always — even on crash/Ctrl+C) ──────────────────────────
        _logging_active = False
        if _log_rows:
            dur = args.duration if hover_mode else traj.duration
            _save_log(args.trajectory, args.mode, args.kt, args.speed,
                      args.reps, dur)
        else:
            print("[log] No rows collected — log not saved.")


if __name__ == "__main__":
    main()
