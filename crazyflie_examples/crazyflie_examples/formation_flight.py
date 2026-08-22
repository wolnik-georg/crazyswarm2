#!/usr/bin/env python3
"""
Multi-robot formation flight via the official Crazyswarm2 trajectory path (Mode E).

Every drone flies the SAME exported Poly4D trajectory, held in a fixed formation.
This works because `startTrajectory(relative=True)` shifts the trajectory to begin at
each drone's current position setpoint -- so if the drones are already sitting at their
formation offsets when the trajectory starts, each one flies the same shape from its own
position and the formation is preserved throughout.

Companion to `flight.py`:
  * flight.py         -- single drone, supports Mode D (--onboard) and Mode E. Unchanged.
  * formation_flight.py -- N drones, Mode E only (Mode D cannot do multi-robot: the
                         traj.* parameter namespace is shared and each upload takes ~5 s).

Trajectories valid on Mode E: hover, figure8, circle, oval, tilted_oval.
Do NOT use slalom/helix/corner/corkscrew/roller_coaster/loop_train (degree-8 content) or
loop/teardrop/teardrop_wide/flip (zero-thrust). See
flying_drone_stack/docs/TRAJECTORY_UPLOAD_PATHS.md.

Usage
-----
  # Dry run first -- prints the formation plan, commands nothing
  ros2 run crazyflie_examples formation_flight -- --trajectory figure8 --mode 1 --kt 0.05 \
      --formation vertical --separation 0.4 --dry-run

  # Real flight
  ros2 run crazyflie_examples formation_flight -- --trajectory figure8 --mode 1 --kt 0.05 \
      --formation vertical --separation 0.4 --brushless

  # Continuous multi-lap (export the laps first, then fly one uninterrupted flight)
  cargo run --release --bin export_poly4d -- --trajectory circle --mode 1 --kt 0.1 --laps 3
  ros2 run crazyflie_examples formation_flight -- --trajectory circle --mode 1 --kt 0.1 \
      --laps 3 --formation vertical --separation 0.4 --brushless

CSV per drone -> experiments/logs/{traj}_{formation}{sep}_{cfname}_{timestamp}.csv

Not yet covered (deliberately, pending first flights): per-drone distinct trajectories,
dynamic formation changes mid-flight, and collision checking of the stage-2 XY transit.
"""

import argparse
from collections import defaultdict
from pathlib import Path
import sys
import time

from ament_index_python.packages import get_package_share_directory
from crazyflie_interfaces.msg import LogDataGeneric
from crazyflie_py import Crazyswarm
from crazyflie_py.uav_trajectory import Trajectory
import numpy as np
import yaml

DATA_DIR = Path(__file__).parent / "data"
# Thesis experiment logs live in experiments/, separate from the archived course-phase
# Controls/logs that flight.py still writes to.
LOG_DIR = Path("/home/georg/Desktop/flying_robot_course/experiments/logs")

# Takeoff/landing always run on the geometric controller, matching flight.py: the INDI
# gains are tuned for trajectory tracking and the ramp is a different operating point.
_RAMP_CONTROLLER = 6
_RAMP_CTRL_MODE = 0
_CTRL_SETTLE_S = 0.3

# Trajectories verified equivalent between Mode D and Mode E (offline reference diff).
_MODE_E_SAFE = {"hover", "figure8", "circle", "oval", "tilted_oval"}


# ── Config ──────────────────────────────────────────────────────────────────

def load_controller_config():
    """Read stabilizer.controller + indi_gains/pos_gains from crazyflies.yaml (all: block)."""
    path = Path(get_package_share_directory("crazyflie")) / "config" / "crazyflies.yaml"
    with open(path) as f:
        cfg = yaml.safe_load(f)
    fp = cfg.get("all", {}).get("firmware_params", {})
    stab, indi = fp.get("stabilizer", {}), fp.get("indi_gains", {})
    if "controller" not in stab or "ctrl_mode" not in indi:
        print("[formation] ERROR: crazyflies.yaml is missing all.firmware_params."
              "stabilizer.controller or .indi_gains.ctrl_mode")
        sys.exit(1)
    gains = {k: indi[k] for k in
             ("kr", "kw", "kr_z", "kw_z", "fc_bw", "mass",
              "kt1", "kt2", "kt3", "kt4", "j_scale", "notch_f0", "notch_bw") if k in indi}
    pos = fp.get("pos_gains", {})
    pos_gains = {k: pos[k] for k in ("kp_xy", "kp_z", "kv_xy", "kv_z") if k in pos}
    return int(stab["controller"]), int(indi["ctrl_mode"]), gains, pos_gains


def csv_label(trajectory, mode, kt, speed, laps=1):
    """Must match export_poly4d's naming."""
    if mode == 0:
        base = f"{trajectory}_mode0" if abs(speed - 1.0) < 1e-3 \
            else f"{trajectory}_mode0_speed{speed:.2f}"
    else:
        base = f"{trajectory}_mode{mode}_kt{('%.6f' % kt).rstrip('0').rstrip('.')}"
    return f"{base}_laps{laps}" if laps > 1 else base


def find_csv(trajectory, mode, kt, speed, laps=1):
    label = csv_label(trajectory, mode, kt, speed, laps)
    path = DATA_DIR / f"{label}.csv"
    if not path.exists():
        lap_arg = f" --laps {laps}" if laps > 1 else ""
        print(f"[formation] ERROR: {path} not found.\n"
              f"  Generate it with:\n"
              f"    cargo run --release --bin export_poly4d -- "
              f"--trajectory {trajectory} --mode {mode} --kt {kt}{lap_arg}")
        sys.exit(1)
    return path


# ── Formation geometry ──────────────────────────────────────────────────────

def formation_offsets(kind: str, n: int, sep: float) -> np.ndarray:
    """Offsets (n,3) added to each drone's initial_position, in formation order.

    Drone 0 is always the reference at zero offset; subsequent drones stack away
    from it, so `--height` is the floor of the formation and nothing ever flies
    below it. For `vertical`, drone 0 is the LOWEST -- it receives the downwash.
    """
    i = np.arange(n, dtype=float)
    z = np.zeros(n)
    if kind == "vertical":       # stacked in Z -- the downwash case
        return np.stack([z, z, i * sep], axis=1)
    if kind == "horizontal":     # offset along X
        return np.stack([i * sep, z, z], axis=1)
    if kind == "side_by_side":   # offset along Y
        return np.stack([z, i * sep, z], axis=1)
    print(f"[formation] ERROR: unknown formation '{kind}'")
    sys.exit(1)


# ── Logging ─────────────────────────────────────────────────────────────────

CSV_COLS = ["time_s", "pos_x", "pos_y", "pos_z", "vel_x", "vel_y", "vel_z",
            "roll", "pitch", "yaw", "thrust", "vbat",
            "gyro_x", "gyro_y", "gyro_z", "acc_x", "acc_y", "acc_z",
            "tau_x", "tau_y", "tau_z", "alp_x", "alp_y", "alp_z",
            "a_res_x", "a_res_y", "a_res_z"]


class DroneLogger:
    """Caches the latest value of each log topic; emits a CSV row per `state` message."""

    def __init__(self, node, name, t0):
        self.name = name
        self.latest = defaultdict(float)
        self.rows = []
        self.active = False
        # ONE clock shared by every drone in the run. Each logger having its own t0 would offset
        # the drones by however long their construction took, and the whole point of these logs is
        # to correlate one drone's residual against another's relative position -- that needs a
        # common time base.
        self.t0 = t0
        node.create_subscription(LogDataGeneric, f"{name}/state", self._state, 10)
        node.create_subscription(LogDataGeneric, f"{name}/attitude", self._att, 10)
        node.create_subscription(LogDataGeneric, f"{name}/gyro_acc", self._imu, 10)
        node.create_subscription(LogDataGeneric, f"{name}/indi_state", self._indi, 10)
        node.create_subscription(LogDataGeneric, f"{name}/indi_a_res", self._a_res, 10)

    def _state(self, msg):
        v = msg.values
        self.latest.update(zip(("pos_x", "pos_y", "pos_z", "vel_x", "vel_y", "vel_z"), v))
        if self.active:
            self.rows.append([time.monotonic() - self.t0]
                             + [self.latest[c] for c in CSV_COLS[1:]])

    def _att(self, msg):
        self.latest.update(zip(("roll", "pitch", "yaw", "thrust", "vbat"), msg.values))

    def _imu(self, msg):
        self.latest.update(
            zip(("gyro_x", "gyro_y", "gyro_z", "acc_x", "acc_y", "acc_z"), msg.values))

    def _indi(self, msg):
        self.latest.update(
            zip(("tau_x", "tau_y", "tau_z", "alp_x", "alp_y", "alp_z"), msg.values))

    def _a_res(self, msg):
        """Residual acceleration [m/s^2], world frame. f_res = mass * a_res."""
        self.latest.update(zip(("a_res_x", "a_res_y", "a_res_z"), msg.values))

    def position(self):
        return np.array([self.latest["pos_x"], self.latest["pos_y"], self.latest["pos_z"]])

    def save(self, path: Path, meta: dict):
        path.parent.mkdir(parents=True, exist_ok=True)
        with open(path, "w") as f:
            for k, v in meta.items():
                f.write(f"# meta:{k}={v}\n")
            f.write(",".join(CSV_COLS) + "\n")
            for r in self.rows:
                f.write(",".join(f"{x:.6f}" for x in r) + "\n")
        print(f"[log] {self.name}: {len(self.rows)} rows -> {path}")


def residual_report(loggers, mass):
    """Residual force per drone -- the quantity the thesis is measuring."""
    print("\n[formation] residual force |f_res| = mass * |a_res| (downwash signature):")
    for lg in loggers:
        if not lg.rows:
            continue
        a = np.array([r[24:27] for r in lg.rows])
        if not np.any(a):
            print(f"  {lg.name}: all zero -- no RPM source? a_res needs rpm/motor telemetry")
            continue
        mag = np.linalg.norm(a, axis=1)
        print(f"  {lg.name}: |a_res| mean {mag.mean():.3f} max {mag.max():.3f} m/s^2   "
              f"|f_res| mean {1000*mass*mag.mean():.1f} max {1000*mass*mag.max():.1f} mN   "
              f"(a_res_z mean {a[:, 2].mean():+.3f})")


def separation_report(loggers):
    """Pairwise separation statistics over the logged flight -- the headline metric."""
    if len(loggers) < 2:
        return
    n = min(len(lg.rows) for lg in loggers)
    if n == 0:
        return
    print("\n[formation] pairwise separation over the flight:")
    for a in range(len(loggers)):
        for b in range(a + 1, len(loggers)):
            pa = np.array([r[1:4] for r in loggers[a].rows[:n]])
            pb = np.array([r[1:4] for r in loggers[b].rows[:n]])
            d = np.linalg.norm(pa - pb, axis=1)
            print(f"  {loggers[a].name} <-> {loggers[b].name}: "
                  f"min {d.min():.3f}  mean {d.mean():.3f}  max {d.max():.3f} m")


# ── Main ────────────────────────────────────────────────────────────────────

def main():
    p = argparse.ArgumentParser(description="Multi-robot formation flight (Mode E / HLC).")
    p.add_argument("--trajectory", default="figure8")
    p.add_argument("--mode", type=int, default=1, choices=[0, 1, 2, 3])
    p.add_argument("--kt", type=float, default=0.05)
    p.add_argument("--speed", type=float, default=1.0, help="HLC timescale")
    p.add_argument("--laps", type=int, default=1,
                   help="continuous core laps baked into one upload (needs a matching "
                        "export_poly4d --laps N); flown as ramp-up -> N laps -> ramp-down")
    p.add_argument("--reps", type=int, default=1,
                   help="repeat the WHOLE flight this many times (separate runs, with a "
                        "pause and ramps each time). For continuous laps use --laps.")
    p.add_argument("--height", type=float, default=1.0,
                   help="height of the LOWEST drone in the formation [m]")
    p.add_argument("--formation", default="vertical",
                   choices=["vertical", "horizontal", "side_by_side"])
    p.add_argument("--separation", type=float, default=0.4,
                   help="spacing between neighbouring drones [m]")
    p.add_argument("--brushless", action="store_true", help="arm ESCs (required for CF21BL)")
    p.add_argument("--dry-run", action="store_true", help="print the plan and exit")
    p.add_argument("--yes", action="store_true", help="skip the pre-flight confirmation")
    args = p.parse_args()

    if args.trajectory not in _MODE_E_SAFE:
        print(f"[formation] ERROR: '{args.trajectory}' is not verified for Mode E.\n"
              f"  Verified: {sorted(_MODE_E_SAFE)}\n"
              f"  Others differ from the Mode D reference (degree-8 content or zero-thrust);\n"
              f"  fly them with flight.py --onboard instead. See TRAJECTORY_UPLOAD_PATHS.md.")
        sys.exit(1)

    csv_path = find_csv(args.trajectory, args.mode, args.kt, args.speed, args.laps)
    traj = Trajectory()
    traj.loadcsv(csv_path)
    controller, traj_ctrl_mode, indi_gains, pos_gains = load_controller_config()

    swarm = Crazyswarm()
    th = swarm.timeHelper
    allcfs = swarm.allcfs
    cfs = allcfs.crazyflies
    n = len(cfs)
    if n == 0:
        print("[formation] ERROR: no drones enabled in crazyflies.yaml")
        sys.exit(1)

    # The formation is anchored on ONE reference point (drone 0's start + height), not on
    # each drone's own initial_position -- otherwise a "vertical" stack would inherit the
    # drones' differing takeoff XY and never actually be vertical, which is exactly the
    # geometry a downwash experiment depends on.
    offsets = formation_offsets(args.formation, n, args.separation)
    anchor = np.array(cfs[0].initialPosition) + np.array([0.0, 0.0, args.height])
    targets = [anchor + o for o in offsets]
    starts = [np.array(c.initialPosition) for c in cfs]
    transits = [float(np.linalg.norm(t[:2] - s[:2])) for s, t in zip(starts, targets)]

    print(f"\n[formation] {csv_path.name}  dur={traj.duration:.2f}s  "
          f"laps={args.laps} (continuous)  reps={args.reps} (separate runs)")
    print(f"[formation] {args.formation} formation, {args.separation:.2f} m separation, "
          f"{n} drone(s)")
    print(f"[formation] anchored on {cfs[0].prefix.lstrip('/')} at {anchor.round(2)}")
    print(f"[formation] controller={controller} ctrl_mode={traj_ctrl_mode} "
          f"(ramp: {_RAMP_CONTROLLER}/{_RAMP_CTRL_MODE})")
    for c, s, t, tr in zip(cfs, starts, targets, transits):
        print(f"    {c.prefix.lstrip('/'):12s} start {s.round(2)}  ->  formation {t.round(2)}"
              f"   (XY transit {tr:.2f} m)")
    if n > 1:
        print(f"[formation] closest pair at rest: {args.separation:.2f} m")
        if args.formation == "vertical":
            print("[formation] approach: climb to individual heights first, then converge in XY"
                  " (drones are vertically separated before any XY motion)")
        else:
            print("[formation] NOTE: all drones share one height in this formation -- check the"
                  " XY transits above do not cross before flying.")
    if max(transits, default=0.0) > 1.0:
        print(f"[formation] WARNING: largest XY transit is {max(transits):.2f} m -- confirm the"
              " path is clear.")

    if args.dry_run:
        print("\n[formation] --dry-run: nothing commanded.")
        return
    if not args.yes:
        try:
            if input("\nProceed? [y/N] ").strip().lower() not in ("y", "yes"):
                print("[formation] aborted.")
                return
        except (EOFError, KeyboardInterrupt):
            print("\n[formation] aborted.")
            return

    log_t0 = time.monotonic()
    loggers = [DroneLogger(allcfs, c.prefix.lstrip("/"), log_t0) for c in cfs]
    print(f"[log] subscribed to {n} drone(s)")
    th.sleep(2.0)  # let the EKF settle on mocap and the log streams start

    stamp = time.strftime("%Y-%m-%d_%H-%M-%S")
    meta = {
        "run_eval_mode": "hlc_e",
        "trajectory": args.trajectory, "mode": args.mode, "kt": args.kt,
        "speed": args.speed, "reps": args.reps, "laps": args.laps,
        "formation": args.formation, "separation": args.separation,
        "height": args.height, "n_drones": n,
        "controller": controller, "ctrl_mode": traj_ctrl_mode,
        "csv": csv_path.name,
    }
    meta.update({f"indi_{k}": v for k, v in indi_gains.items()})
    meta.update({f"pos_{k}": v for k, v in pos_gains.items()})

    def apply(phase, ctrl, mode_, gains=None, pgains=None):
        for c in cfs:
            c.setParam("stabilizer.controller", ctrl)
            c.setParam("indi_gains.ctrl_mode", mode_)
            for k, v in (gains or {}).items():
                c.setParam(f"indi_gains.{k}", float(v))
            for k, v in (pgains or {}).items():
                c.setParam(f"pos_gains.{k}", float(v))
        th.sleep(_CTRL_SETTLE_S)
        print(f"[formation] {phase}: controller={ctrl} ctrl_mode={mode_}")

    try:
        apply("takeoff", _RAMP_CONTROLLER, _RAMP_CTRL_MODE)
        if args.brushless:
            for c in cfs:
                c.arm(True)
            th.sleep(0.5)

        for lg in loggers:
            lg.active = True

        # Stage 1 -- each drone climbs straight up to its OWN formation height, no XY motion.
        # For a vertical stack this establishes the Z separation before anything moves
        # laterally, so the stage-2 XY convergence cannot bring two drones together.
        print("[formation] stage 1: climbing to individual formation heights...")
        for c, t in zip(cfs, targets):
            c.takeoff(targetHeight=float(t[2]), duration=3.0)
        th.sleep(3.5)

        # Stage 2 -- converge in XY onto the anchored formation slots.
        print("[formation] stage 2: converging to formation slots...")
        for c, t in zip(cfs, targets):
            c.goTo(t, 0, 3.0)
        th.sleep(3.5)

        if n > 1:
            actual = [lg.position() for lg in loggers]
            for a in range(n):
                for b in range(a + 1, n):
                    d = float(np.linalg.norm(actual[a] - actual[b]))
                    print(f"[formation] achieved {loggers[a].name}<->{loggers[b].name}: "
                          f"{d:.3f} m (target {np.linalg.norm(offsets[a]-offsets[b]):.3f})")

        # Each drone gets the same trajectory; relative=True anchors it to that drone's
        # own current setpoint, so the formation offsets carry through unchanged.
        print("[formation] uploading trajectory to all drones...")
        for c in cfs:
            c.uploadTrajectory(0, 0, traj)
        th.sleep(0.5)

        apply("trajectory", controller, traj_ctrl_mode, indi_gains, pos_gains)
        for c in cfs:
            try:
                c.setParam("usd.logging", 1)
            except Exception:
                pass  # no uSD deck

        for rep in range(args.reps):
            if rep > 0:
                th.sleep(1.0)
            print(f"[formation] starting trajectory (rep {rep + 1}/{args.reps})...")
            allcfs.startTrajectory(0, timescale=args.speed)
            th.sleep(traj.duration * args.speed + 0.5)

        print("[formation] done, landing...")
        apply("landing", _RAMP_CONTROLLER, _RAMP_CTRL_MODE)
        for c in cfs:
            try:
                c.setParam("usd.logging", 0)
            except Exception:
                pass
        allcfs.land(targetHeight=0.06, duration=3.0)
        th.sleep(4.0)
        if args.brushless:
            for c in cfs:
                c.arm(False)

    finally:
        for lg in loggers:
            lg.active = False
        label = f"{args.trajectory}_{args.formation}{args.separation:.2f}"
        for lg in loggers:
            lg.save(LOG_DIR / f"{label}_{lg.name}_{stamp}.csv", meta)
        separation_report(loggers)
        residual_report(loggers, float(indi_gains.get("mass", 0.041)))


if __name__ == "__main__":
    main()
