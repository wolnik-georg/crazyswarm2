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
import rclpy
import yaml
from ament_index_python.packages import get_package_share_directory
from crazyflie_interfaces.msg import LogDataGeneric
from crazyflie_interfaces.srv import NotifySetpointsStop
from crazyflie_py import Crazyswarm
from crazyflie_py.uav_trajectory import Trajectory
from rcl_interfaces.msg import Parameter, ParameterType, ParameterValue
from rcl_interfaces.srv import SetParameters

# ── Paths ──────────────────────────────────────────────────────────────────

DATA_DIR = Path(__file__).parent / "data"

LOGS_DIR = (
    Path(__file__).resolve().parents[3] / "flying_robot_course" / "Controls" / "logs"
)

# ── Logging state (single-threaded — callbacks fire inside timeHelper.sleep) ──

_log_rows = []  # accumulated rows during flight
_log_t0 = None  # wall-clock at trajectory start (None = not logging yet)
_logging_active = False
_latest_att = {}  # most recent attitude/thrust/vbat values
_latest_gyro = {}  # most recent gyro_acc values
_latest_rpm = {}  # most recent per-motor RPM values
_latest_indi = {}  # most recent INDI values (from indi_state OR indi_filter_char)
_latest_state = {}  # most recent EKF state (always cached — used for onboard origin)
_controller_meta = {}  # phase -> (stabilizer.controller, indi_gains.ctrl_mode)
_yaml_indi_gains = {}  # kr, kw, kr_z, kw_z, fc_bw, mass, kt1..4 from crazyflies.yaml at startup
_yaml_pos_gains = {}   # kp_xy, kp_z, kv_xy, kv_z from crazyflies.yaml at startup
_onboard_mode = False  # True when --onboard (Mode D); written in main, read in _save_log

# Variable order must match crazyflies.yaml custom_topics vars lists exactly.
# Split into 3 blocks to stay within the 26-byte CRTP log packet limit (6 floats max).
_STATE_VARS = [
    "stateEstimate.x",
    "stateEstimate.y",
    "stateEstimate.z",
    "stateEstimate.vx",
    "stateEstimate.vy",
    "stateEstimate.vz",
]
_ATTITUDE_VARS = [
    "stabilizer.roll",
    "stabilizer.pitch",
    "stabilizer.yaw",
    "stabilizer.thrust",
    "pm.vbat",
]
_GYRO_VARS = ["gyro.x", "gyro.y", "gyro.z", "acc.x", "acc.y", "acc.z"]
_RPM_VARS = ["rpm.m1", "rpm.m2", "rpm.m3", "rpm.m4"]
_INDI_STATE_VARS = [
    "indi.tau_x",
    "indi.tau_y",
    "indi.tau_z",
    "indi.alp_x",
    "indi.alp_y",
    "indi.alp_z",
]
_INDI_ALPraw_VARS = [
    "indi.alp_raw_x",
    "indi.alp_raw_y",
    "indi.alp_raw_z",
]
_INDI_FILTER_VARS = [
    "indi.alp_raw_x",
    "indi.alp_raw_y",
    "indi.alp_raw_z",
    "indi.alp_x",
    "indi.alp_y",
    "indi.alp_z",
]


def _state_cb(msg: LogDataGeneric):
    """Fires at state topic rate. Always caches EKF; logs rows when active."""
    if len(msg.values) != len(_STATE_VARS):
        return

    s = dict(zip(_STATE_VARS, msg.values))
    _latest_state.update(s)
    if not _logging_active:
        return

    a = _latest_att
    g = _latest_gyro

    _log_rows.append(
        {
            "time_s": round(time.monotonic() - _log_t0, 4),
            "x": s["stateEstimate.x"],
            "y": s["stateEstimate.y"],
            "z": s["stateEstimate.z"],
            "vx": s["stateEstimate.vx"],
            "vy": s["stateEstimate.vy"],
            "vz": s["stateEstimate.vz"],
            "roll_deg": a.get("stabilizer.roll", float("nan")),
            "pitch_deg": a.get("stabilizer.pitch", float("nan")),
            "yaw_deg": a.get("stabilizer.yaw", float("nan")),
            "thrust": a.get("stabilizer.thrust", float("nan")),
            "vbat": a.get("pm.vbat", float("nan")),
            "gyro_x": g.get("gyro.x", float("nan")),
            "gyro_y": g.get("gyro.y", float("nan")),
            "gyro_z": g.get("gyro.z", float("nan")),
            "acc_x": g.get("acc.x", float("nan")),
            "acc_y": g.get("acc.y", float("nan")),
            "acc_z": g.get("acc.z", float("nan")),
            "rpm_m1": _latest_rpm.get("rpm.m1", float("nan")),
            "rpm_m2": _latest_rpm.get("rpm.m2", float("nan")),
            "rpm_m3": _latest_rpm.get("rpm.m3", float("nan")),
            "rpm_m4": _latest_rpm.get("rpm.m4", float("nan")),
            "tau_x": _latest_indi.get("indi.tau_x", float("nan")),
            "tau_y": _latest_indi.get("indi.tau_y", float("nan")),
            "tau_z": _latest_indi.get("indi.tau_z", float("nan")),
            "alp_x": _latest_indi.get("indi.alp_x", float("nan")),
            "alp_y": _latest_indi.get("indi.alp_y", float("nan")),
            "alp_z": _latest_indi.get("indi.alp_z", float("nan")),
            "alp_raw_x": _latest_indi.get("indi.alp_raw_x", float("nan")),
            "alp_raw_y": _latest_indi.get("indi.alp_raw_y", float("nan")),
            "alp_raw_z": _latest_indi.get("indi.alp_raw_z", float("nan")),
        }
    )


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


def _indi_state_cb(msg: LogDataGeneric):
    """Caches tau_x/y/z + alp_x/y/z from indi_state topic (normal operation, 20 Hz)."""
    if len(msg.values) != len(_INDI_STATE_VARS):
        return
    for name, val in zip(_INDI_STATE_VARS, msg.values):
        _latest_indi[name] = val


def _indi_alp_raw_cb(msg: LogDataGeneric):
    """Caches alp_raw_x/y/z from indi_alp_raw topic (100 Hz, normal operation)."""
    if len(msg.values) != len(_INDI_ALPraw_VARS):
        return
    for name, val in zip(_INDI_ALPraw_VARS, msg.values):
        _latest_indi[name] = val


def _indi_filter_cb(msg: LogDataGeneric):
    """Caches alp_raw_x/y/z + alp_x/y/z from indi_filter_char topic (100 Hz, filter char step)."""
    if len(msg.values) != len(_INDI_FILTER_VARS):
        return
    for name, val in zip(_INDI_FILTER_VARS, msg.values):
        _latest_indi[name] = val


# ── Controller / ctrl_mode helpers ─────────────────────────────────────────

_CTRL_SETTLE_S = 1.0  # wait after each controller/ctrl_mode change before proceeding
_TRAJ_ARM_DELAY_S = 0.05  # Rust: traj.mode=1 then brief pause before traj.start=1
# Firmware wraps onboard t at total_dur (periodic reps). Stop slightly early so reps=1
# never hits the wrap — position is C0 at the seam but velocity is not, which spirals.
_TRAJ_STOP_MARGIN_S = 0.50
_COEF_UPLOAD_DELAY_S = 0.008  # Rust upload_coef: 8 ms after each ci/cv/cw commit
# Lee/Mellinger (controller 2) for takeoff/landing — always available, no OOT required.
_RAMP_CONTROLLER = 2
_RAMP_CTRL_MODE = 0  # geometric — takeoff and landing (hardcoded)


def _load_firmware_controller_config() -> tuple[int, int, dict, dict]:
    """Read stabilizer.controller, indi_gains.ctrl_mode, and indi gains from crazyflies.yaml."""
    path = Path(get_package_share_directory("crazyflie")) / "config" / "crazyflies.yaml"
    try:
        with open(path) as f:
            cfg = yaml.safe_load(f)
    except Exception as exc:
        print(f"[flight] ERROR: could not read {path}: {exc}")
        sys.exit(1)

    fp = cfg.get("all", {}).get("firmware_params", {})
    stabilizer = fp.get("stabilizer", {})
    indi = fp.get("indi_gains", {})
    if "controller" not in stabilizer:
        print(
            "[flight] ERROR: missing all.firmware_params.stabilizer.controller in crazyflies.yaml"
        )
        sys.exit(1)
    if "ctrl_mode" not in indi:
        print(
            "[flight] ERROR: missing all.firmware_params.indi_gains.ctrl_mode in crazyflies.yaml"
        )
        sys.exit(1)
    indi_gains = {
        k: indi[k]
        for k in ("kr", "kw", "kr_z", "kw_z", "fc_bw", "mass", "kt1", "kt2", "kt3", "kt4")
        if k in indi
    }
    pos = fp.get("pos_gains", {})
    pos_gains = {
        k: pos[k]
        for k in ("kp_xy", "kp_z", "kv_xy", "kv_z")
        if k in pos
    }
    return int(stabilizer["controller"]), int(indi["ctrl_mode"]), indi_gains, pos_gains


def _log_phase(phase: str, controller: int, ctrl_mode: int):
    """Record and print the controller settings applied for this flight phase."""
    _controller_meta[phase] = (controller, ctrl_mode)
    print(
        f"[flight] {phase}: stabilizer.controller={controller}  "
        f"indi_gains.ctrl_mode={ctrl_mode}"
    )


def _apply_flight_settings(
    allcfs, th, phase: str, controller: int, ctrl_mode: int, indi_gains: dict | None = None
):
    """Set stabilizer.controller, indi_gains.* on all drones, settle, then log."""
    for c in allcfs.crazyflies:
        c.setParam("stabilizer.controller", controller)
        c.setParam("indi_gains.ctrl_mode", ctrl_mode)
        if indi_gains:
            for k, v in indi_gains.items():
                c.setParam(f"indi_gains.{k}", float(v))
    th.sleep(_CTRL_SETTLE_S)
    _log_phase(phase, controller, ctrl_mode)


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
        print(
            f"    cargo run --release --bin export_poly4d -- "
            f"--trajectory {trajectory} --mode {mode} --kt {kt}"
        )
        sys.exit(1)
    return path


def _save_log(
    trajectory: str, mode: int, kt: float, speed: float, reps: int, lap_time_s: float
):
    LOGS_DIR.mkdir(parents=True, exist_ok=True)
    ts = datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
    label = _csv_label(trajectory, mode, kt, speed)
    path = LOGS_DIR / f"{label}_{ts}.csv"

    fieldnames = [
        "time_s",
        "x",
        "y",
        "z",
        "vx",
        "vy",
        "vz",
        "roll_deg",
        "pitch_deg",
        "yaw_deg",
        "thrust",
        "vbat",
        "gyro_x",
        "gyro_y",
        "gyro_z",
        "acc_x",
        "acc_y",
        "acc_z",
        "rpm_m1",
        "rpm_m2",
        "rpm_m3",
        "rpm_m4",
        "tau_x",
        "tau_y",
        "tau_z",
        "alp_x",
        "alp_y",
        "alp_z",
        "alp_raw_x",
        "alp_raw_y",
        "alp_raw_z",
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
        f.write(f"# meta:run_eval_mode={'onboard_d' if _onboard_mode else 'hlc_e'}\n")
        for k in ("kr", "kw", "kr_z", "kw_z", "fc_bw", "mass", "kt1", "kt2", "kt3", "kt4"):
            if k in _yaml_indi_gains:
                f.write(f"# meta:indi_{k}={_yaml_indi_gains[k]}\n")
        for k in ("kp_xy", "kp_z", "kv_xy", "kv_z"):
            if k in _yaml_pos_gains:
                f.write(f"# meta:pos_{k}={_yaml_pos_gains[k]}\n")
        if "yaml" in _controller_meta:
            c, m = _controller_meta["yaml"]
            f.write(f"# meta:yaml_stabilizer_controller={c}\n")
            f.write(f"# meta:yaml_ctrl_mode={m}\n")
        for phase in ("takeoff", "trajectory", "landing"):
            if phase in _controller_meta:
                c, m = _controller_meta[phase]
                f.write(f"# meta:{phase}_stabilizer_controller={c}\n")
                f.write(f"# meta:{phase}_ctrl_mode={m}\n")
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(_log_rows)

    print(f"[log] {len(_log_rows)} rows → {path}")


# ── Onboard (Mode D) trajectory helpers ─────────────────────────────────────


def _find_onboard_csv(trajectory: str, mode: int, kt: float, speed: float) -> Path:
    label = _csv_label(trajectory, mode, kt, speed)
    path = DATA_DIR / f"{label}_onboard.csv"
    if not path.exists():
        print(f"[error] Onboard CSV not found: {path}")
        print(f"  Generate it with (from flying_drone_stack/):")
        print(
            f"    cargo run --release --bin export_poly4d -- "
            f"--trajectory {trajectory} --mode {mode} --kt {kt} --onboard"
        )
        sys.exit(1)
    return path


def _load_onboard_csv(path: Path) -> tuple:
    """Return (segments, total_duration). segments: list of 19-float lists [dur,cx0..cx8,cy0..cy8]."""
    segs = []
    total_dur = 0.0
    with open(path) as f:
        reader = csv.DictReader(f)
        for row in reader:
            vals = [float(row["duration"])]
            for i in range(9):
                vals.append(float(row[f"cx{i}"]))
            for i in range(9):
                vals.append(float(row[f"cy{i}"]))
            segs.append(vals)
            total_dur += vals[0]
    return segs, total_dur


def _set_param_sync(cf, th, name: str, value):
    """Blocking setParam — required for traj coefficient upload ordering.

    crazyflie_py.setParam uses call_async; firing ci/cv/cw back-to-back without
    waiting can reorder or drop writes over the ROS→CRTP path and corrupt g_traj_coefs.
    Rust upload_coef awaits each write and sleeps 8 ms per coefficient.
    """
    param_name = cf.prefix[1:] + ".params." + name
    param_type = cf.paramTypeDict[name]
    if param_type == ParameterType.PARAMETER_INTEGER:
        param_value = ParameterValue(type=param_type, integer_value=int(value))
    elif param_type == ParameterType.PARAMETER_DOUBLE:
        param_value = ParameterValue(type=param_type, double_value=float(value))
    else:
        raise ValueError(f"unsupported param type for {name}")
    req = SetParameters.Request()
    req.parameters = [Parameter(name=param_name, value=param_value)]
    future = cf.setParamsService.call_async(req)
    while rclpy.ok() and not future.done():
        th.sleep(0.001)


def _notify_setpoints_stop_sync(cf, th, remain_ms: int = 100):
    """Blocking notify_setpoints_stop — required before HLC takeoff after cmdFullState."""
    req = NotifySetpointsStop.Request()
    req.remain_valid_millisecs = remain_ms
    req.group_mask = 0
    future = cf.notifySetpointsStopService.call_async(req)
    while rclpy.ok() and not future.done():
        th.sleep(0.001)


def _kalman_reset_pulse(cf, th):
    """Rust kalman_reset(): pulse resetEstimation so back-to-back flights start clean."""
    if "kalman.resetEstimation" not in cf.paramTypeDict:
        print("[flight] WARN: kalman.resetEstimation not in TOC — skipping EKF reset.")
        return
    _set_param_sync(cf, th, "kalman.resetEstimation", 1)
    th.sleep(0.2)
    _set_param_sync(cf, th, "kalman.resetEstimation", 0)
    print("[flight] Kalman reset pulse sent.")


def _firmware_idle_reset(cf, th):
    """Return firmware to idle so the next ros2 run can use HLC takeoff/land.

    After onboard flights the drone stays in low-level cmdFullState mode until
    notify_setpoints_stop is called. traj.start must be cleared so the next
    Mode D arm gets a fresh T0 latch.
    """
    _set_param_sync(cf, th, "traj.mode", 0)
    _set_param_sync(cf, th, "traj.start", 0)
    _set_param_sync(cf, th, "indi_gains.ctrl_mode", _RAMP_CTRL_MODE)
    _notify_setpoints_stop_sync(cf, th)
    th.sleep(0.2)


def _upload_traj_to_oot(cf, th, segs: list, hz: float, ox: float, oy: float):
    """Upload trajectory segments to OOT traj params (Mode D).

    Protocol: for each float, set traj.ci=index, traj.cv=value, traj.cw=1.
    Each triplet is written synchronously with 8 ms spacing (Rust upload_coef).
    """
    n_segs = len(segs)
    n_coefs = n_segs * 19
    print(f"[traj] Uploading {n_segs} segs ({n_coefs} coefs) to OOT params...")

    _set_param_sync(cf, th, "traj.mode", 0)   # stay in passthrough during upload
    _set_param_sync(cf, th, "traj.nseg", n_segs)
    _set_param_sync(cf, th, "traj.hz", hz)
    _set_param_sync(cf, th, "traj.ox", ox)
    _set_param_sync(cf, th, "traj.oy", oy)
    _set_param_sync(cf, th, "traj.dz", 0.0)
    _set_param_sync(cf, th, "traj.z_mode", 0)
    _set_param_sync(cf, th, "traj.att_ctrl_mode", 1)

    t0 = time.monotonic()
    coef_idx = 0
    for seg_i, seg in enumerate(segs):
        for val in seg:
            _set_param_sync(cf, th, "traj.ci", coef_idx)
            _set_param_sync(cf, th, "traj.cv", float(val))
            _set_param_sync(cf, th, "traj.cw", 1)
            th.sleep(_COEF_UPLOAD_DELAY_S)
            coef_idx += 1
        if (seg_i + 1) % 3 == 0 or seg_i == n_segs - 1:
            print(f"[traj]   [{seg_i + 1}/{n_segs}] segs uploaded")

    print(f"[traj] Upload done in {time.monotonic() - t0:.1f}s")


def _wait_for_state(th, timeout_s: float = 3.0) -> bool:
    """Spin until at least one EKF state sample is cached."""
    t0 = th.time()
    while th.time() - t0 < timeout_s:
        if _latest_state:
            return True
        th.sleep(0.05)
    return bool(_latest_state)


def _sample_origin(th, fallback_xy) -> tuple[float, float]:
    """Sample traj origin from live EKF position (Rust sample_origin pattern)."""
    _wait_for_state(th)
    if _latest_state:
        ox = float(_latest_state["stateEstimate.x"])
        oy = float(_latest_state["stateEstimate.y"])
        print(f"[traj] Sampled origin ox={ox:.4f} oy={oy:.4f} (EKF)")
        return ox, oy
    ox, oy = float(fallback_xy[0]), float(fallback_xy[1])
    print(
        f"[traj] WARN: no EKF state — using fallback origin ox={ox:.4f} oy={oy:.4f}"
    )
    return ox, oy


# Firmware OOT arms only while sp.position.z > 0.05 (controllerOutOfTree). Landing must
# stay kt-/trajectory-independent: use live EKF z and fixed physics limits, not lap time.
_FIRMWARE_Z_ARM_MAX = 0.05
_Z_CMD_MIN_AIRBORNE = _FIRMWARE_Z_ARM_MAX + 0.02  # stay armed while airborne
_Z_ON_GROUND = 0.12  # EKF z below this → on the ground, safe to disarm
_LAND_DESCENT_RATE = 0.20  # m/s commanded descent (kt-independent)
_LAND_Z_CHASE = 0.04  # command this far below measured z to pull down
_LAND_STREAM_HZ = 20.0


def _stop_onboard_traj_and_hold(cf, th, fallback_pos):
    """Stop Mode D while streaming hover at measured pose (no wrap, no radio gap)."""
    zero3 = np.zeros(3)
    # Read measured pose first so hold_pos reflects where the drone actually is.
    _wait_for_state(th)
    z_meas = float(_latest_state.get("stateEstimate.z", fallback_pos[2]))
    x_meas = float(_latest_state.get("stateEstimate.x", fallback_pos[0]))
    y_meas = float(_latest_state.get("stateEstimate.y", fallback_pos[1]))
    hold_pos = np.array([x_meas, y_meas, max(z_meas, _Z_CMD_MIN_AIRBORNE)])
    # Stop trajectory FIRST (prevent wrap), then switch ctrl_mode.
    # Each _set_param_sync takes ~50-100 ms; wrong order lets the trajectory wrap before
    # traj.mode=0 lands on the firmware, causing a discontinuous velocity reference.
    t_end = th.time() + 0.50
    traj_stopped = False
    ctrl_switched = False
    while not th.isShutdown() and th.time() < t_end:
        cf.cmdFullState(hold_pos, zero3, zero3, 0.0, zero3)
        if not traj_stopped:
            _set_param_sync(cf, th, "traj.start", 0)
            _set_param_sync(cf, th, "traj.mode", 0)
            traj_stopped = True
        elif not ctrl_switched:
            cf.setParam("stabilizer.controller", _RAMP_CONTROLLER)
            _set_param_sync(cf, th, "indi_gains.ctrl_mode", _RAMP_CTRL_MODE)
            _log_phase("landing", _RAMP_CONTROLLER, _RAMP_CTRL_MODE)
            ctrl_switched = True
        th.sleepForRate(_LAND_STREAM_HZ)
    return hold_pos


def _stream_hover_hold(cf, th, hover_pos, duration_s: float, rate_hz: float = _LAND_STREAM_HZ):
    """Stream cmdFullState hover keepalive — firmware armed while sp.position.z > 0.05."""
    zero3 = np.zeros(3)
    t_end = th.time() + duration_s
    while not th.isShutdown() and th.time() < t_end:
        cf.cmdFullState(hover_pos, zero3, zero3, 0.0, zero3)
        th.sleepForRate(rate_hz)


def _onboard_stream_land(cf, th, hover_pos, hold_s: float = 1.0):
    """Closed-loop streamed landing after onboard traj (Rust hold_and_land pattern).

    Why landing is fragile:
    - cmdFullState forces low-level mode; HLC land()/goTo() do not work afterward.
    - OOT disarms when sp.position.z <= 0.05 even if EKF z is still ~0.8 m.
    - Open-loop z ramps can hit the disarm threshold before the drone physically descends.
    - A gap in streaming (>~0.5 s) triggers the setpoint watchdog → motor cut.

    This routine keeps 20 Hz cmdFullState throughout, switches to geometric ctrl_mode,
    chases measured height down at a fixed rate, and only disarms once EKF z confirms
    ground contact. No kt or lap-time constants.
    """
    zero3 = np.zeros(3)
    xy = np.array(hover_pos[:2])

    _wait_for_state(th)
    z_meas = float(_latest_state.get("stateEstimate.z", hover_pos[2]))
    hold_pos = np.array([xy[0], xy[1], z_meas])
    print(f"[flight] Post-traj hold z={z_meas:.2f}m for {hold_s:.1f}s...")
    _stream_hover_hold(cf, th, hold_pos, hold_s)

    _wait_for_state(th)
    z_meas = float(_latest_state.get("stateEstimate.z", hold_pos[2]))
    est_s = max(z_meas - _Z_ON_GROUND, 0.0) / _LAND_DESCENT_RATE
    max_s = est_s + 8.0  # safety cap from measured height, not trajectory/kt
    print(
        f"[flight] Landing: chase descent from z={z_meas:.2f}m at "
        f"{_LAND_DESCENT_RATE:.2f}m/s (est ~{est_s:.1f}s)..."
    )

    t_land = th.time()
    while not th.isShutdown() and th.time() - t_land < max_s:
        z_meas = float(_latest_state.get("stateEstimate.z", hold_pos[2]))
        if z_meas <= _Z_ON_GROUND:
            print(f"[flight] On ground (z={z_meas:.2f}m)")
            break

        # Chase measured height — never command z at/under arm threshold while airborne.
        z_cmd = max(z_meas - _LAND_Z_CHASE, _Z_CMD_MIN_AIRBORNE)
        pos = np.array([xy[0], xy[1], z_cmd])
        vel = np.array([0.0, 0.0, -_LAND_DESCENT_RATE])
        cf.cmdFullState(pos, vel, zero3, 0.0, zero3)
        th.sleepForRate(_LAND_STREAM_HZ)
    else:
        print(
            f"[flight] WARN: landing timeout at z="
            f"{_latest_state.get('stateEstimate.z', float('nan')):.2f}m — disarming"
        )

    # Rust: setpoint_rpyt(0,0,0,0). CS2 has no raw RPYT after cmdFullState — disarm via z.
    disarm_pos = np.array([xy[0], xy[1], 0.02])
    t0 = th.time()
    while not th.isShutdown() and th.time() - t0 < 0.5:
        cf.cmdFullState(disarm_pos, zero3, zero3, 0.0, zero3)
        th.sleepForRate(_LAND_STREAM_HZ)
    th.sleep(0.2)
    # Exit low-level streaming so the next takeoff() / finally cleanup can use HLC.
    _notify_setpoints_stop_sync(cf, th)


def _apply_onboard_traj_origin(cf, th, ox: float, oy: float, height: float):
    """Set traj frame + attitude policy immediately before arming onboard eval."""
    _set_param_sync(cf, th, "traj.ox", ox)
    _set_param_sync(cf, th, "traj.oy", oy)
    _set_param_sync(cf, th, "traj.hz", float(height))
    # Hybrid: polynomial omega_d feedforward + flatness-derived attitude (Rust default).
    _set_param_sync(cf, th, "traj.att_ctrl_mode", 1)


# ── Main ────────────────────────────────────────────────────────────────────


def main():
    global _log_t0, _logging_active, _onboard_mode

    parser = argparse.ArgumentParser(description="Generic CS2 flight script")
    parser.add_argument(
        "--trajectory", default="figure8", choices=["figure8", "circle", "hover"]
    )
    parser.add_argument("--mode", type=int, default=0, choices=[0, 1, 2, 3])
    parser.add_argument("--kt", type=float, default=None)
    parser.add_argument("--speed", type=float, default=1.0)
    parser.add_argument("--reps", type=int, default=1)
    parser.add_argument("--height", type=float, default=1.0)
    parser.add_argument(
        "--duration",
        type=float,
        default=20.0,
        help="Hover duration in seconds (hover trajectory only)",
    )
    parser.add_argument(
        "--onboard",
        action="store_true",
        help="Mode D: upload to OOT traj params (500 Hz onboard eval, full jerk/snap feedforward) "
             "instead of HLC Poly4D (Mode E). Requires {label}_onboard.csv from export_poly4d --onboard.",
    )
    args, _ = parser.parse_known_args()

    if args.kt is None:
        args.kt = 0.1 if args.trajectory == "circle" else 0.008

    hover_mode = args.trajectory == "hover"
    onboard_mode = args.onboard and not hover_mode
    _onboard_mode = onboard_mode
    traj = None
    traj_dur = 0.0
    onboard_segs = None

    if hover_mode:
        traj_dur = args.duration
        print(f"[flight] hover mode — {args.duration:.0f}s at {args.height:.2f}m")
    elif onboard_mode:
        onboard_csv_path = _find_onboard_csv(args.trajectory, args.mode, args.kt, args.speed)
        onboard_segs, traj_dur = _load_onboard_csv(onboard_csv_path)
        print(
            f"[flight] onboard: {onboard_csv_path.name}  mode={args.mode} kt={args.kt} "
            f"reps={args.reps}  ({len(onboard_segs)} segs, {traj_dur:.2f}s/rep)"
        )
    else:
        csv_path = _find_csv(args.trajectory, args.mode, args.kt, args.speed)
        print(
            f"[flight] {csv_path.name}  mode={args.mode} kt={args.kt} "
            f"speed={args.speed} reps={args.reps}"
        )
        traj = Trajectory()
        traj.loadcsv(csv_path)
        traj_dur = traj.duration
        print(f"[flight] Duration per rep: {traj_dur:.2f}s")

    # ── Init CS2 ────────────────────────────────────────────────────────────
    # swarm.allcfs IS the ROS2 node (CrazyflieServer inherits rclpy.node.Node).
    # Subscriptions created here are processed by timeHelper.sleep(), which
    # calls rclpy.spin_once() in a loop — no extra threads required.
    swarm = Crazyswarm()
    th = swarm.timeHelper
    allcfs = swarm.allcfs  # this IS the node
    cf = allcfs.crazyflies[0]
    cf_name = cf.prefix.lstrip("/")  # e.g. "cf231"

    allcfs.create_subscription(LogDataGeneric, f"{cf_name}/state", _state_cb, 10)
    allcfs.create_subscription(LogDataGeneric, f"{cf_name}/attitude", _attitude_cb, 10)
    allcfs.create_subscription(LogDataGeneric, f"{cf_name}/gyro_acc", _gyro_cb, 10)
    allcfs.create_subscription(LogDataGeneric, f"{cf_name}/rpm", _rpm_cb, 10)
    allcfs.create_subscription(
        LogDataGeneric, f"{cf_name}/indi_state", _indi_state_cb, 10
    )
    allcfs.create_subscription(
        LogDataGeneric, f"{cf_name}/indi_alp_raw", _indi_alp_raw_cb, 10
    )
    allcfs.create_subscription(
        LogDataGeneric, f"{cf_name}/indi_filter_char", _indi_filter_cb, 10
    )
    print(
        f"[log] Subscribed to {cf_name}/state, attitude, gyro_acc, rpm, indi_state, indi_alp_raw, indi_filter_char"
    )

    # ── Wait for EKF + clear stale state from any previous run ───────────────
    print("[flight] Preflight: idle reset + Kalman pulse (place drone on pad)...")
    _firmware_idle_reset(cf, th)
    _kalman_reset_pulse(cf, th)
    print("[flight] Waiting for EKF to converge on mocap poses...")
    th.sleep(3.0)

    # ── Controller config from crazyflies.yaml firmware_params ─────────────
    # Trajectory uses yaml stabilizer.controller + indi_gains.ctrl_mode.
    # Takeoff/landing keep OOT geometric (controller=6, ctrl_mode=0); only ctrl_mode changes.
    yaml_controller, traj_ctrl_mode, indi_gains_from_yaml, pos_gains_from_yaml = _load_firmware_controller_config()
    _yaml_indi_gains.update(indi_gains_from_yaml)
    _yaml_pos_gains.update(pos_gains_from_yaml)
    _controller_meta["yaml"] = (yaml_controller, traj_ctrl_mode)
    print(
        f"[flight] crazyflies.yaml (trajectory): stabilizer.controller={yaml_controller}  "
        f"indi_gains.ctrl_mode={traj_ctrl_mode}"
    )
    print(
        f"[flight] ramp (takeoff/landing): stabilizer.controller={_RAMP_CONTROLLER}  "
        f"indi_gains.ctrl_mode={_RAMP_CTRL_MODE}"
    )

    # ── Mode D: upload trajectory BEFORE takeoff (Rust pattern) ──────────────
    # Must happen while OOT controller (stabilizer.controller=6) is active.
    # Never switch to a non-OOT controller (e.g. mellinger=2) or traj.cw commits
    # inside controllerOutOfTree never run → zero segment durations → NaN on start.
    # ox/oy here are placeholders; real origin is sampled from EKF at hover (below).
    if onboard_mode:
        ox = float(cf.initialPosition[0])
        oy = float(cf.initialPosition[1])
        print("[flight] Uploading onboard trajectory (OOT active, drone on ground)...")
        _upload_traj_to_oot(cf, th, onboard_segs, args.height, ox, oy)

    _apply_flight_settings(allcfs, th, "takeoff", _RAMP_CONTROLLER, _RAMP_CTRL_MODE)
    for c in allcfs.crazyflies:
        try:
            c.setParam("usd.logging", 1)
        except Exception:
            pass  # uSD deck not present — skip silently
    print("[flight] Taking off...")

    # ── Takeoff and position ─────────────────────────────────────────────────
    _logging_active = True
    _log_t0 = time.monotonic()

    allcfs.takeoff(targetHeight=args.height, duration=2.0)
    th.sleep(2.5)

    for c in allcfs.crazyflies:
        pos = np.array(c.initialPosition) + np.array([0, 0, args.height])
        c.goTo(pos, 0, 2.0)
    th.sleep(2.5)

    onboard_landed = False
    try:
        if hover_mode:
            # ── Hover: no trajectory upload, just hold position ───────────────
            # Restore configured ctrl_mode for the hover itself
            if (yaml_controller, traj_ctrl_mode) != (_RAMP_CONTROLLER, _RAMP_CTRL_MODE):
                _apply_flight_settings(
                    allcfs, th, "trajectory", yaml_controller, traj_ctrl_mode, indi_gains_from_yaml
                )
            else:
                _log_phase("trajectory", yaml_controller, traj_ctrl_mode)
            print(f"[flight] Hovering for {args.duration:.0f}s...")
            th.sleep(args.duration)
            print("[flight] Done. Landing...")
        elif onboard_mode:
            # ── Mode D: OOT traj params, 500 Hz onboard eval ─────────────────
            # Trajectory was already uploaded before takeoff (see above) so that
            # traj.cw commits ran inside controllerOutOfTree while OOT was active.
            #
            # Keepalive: cmdFullState at 20 Hz with actual EKF z (arming reads setpoint z).
            _zero3 = np.zeros(3)
            fallback_xy = cf.initialPosition[:2]
            keepalive_pos = None

            if (yaml_controller, traj_ctrl_mode) != (_RAMP_CONTROLLER, _RAMP_CTRL_MODE):
                _apply_flight_settings(
                    allcfs, th, "trajectory", yaml_controller, traj_ctrl_mode, indi_gains_from_yaml
                )
            else:
                _log_phase("trajectory", yaml_controller, traj_ctrl_mode)
            print("[flight] Starting onboard trajectory (Mode D)...")

            for rep in range(args.reps):
                if rep > 0:
                    _stream_hover_hold(cf, th, keepalive_pos, 1.0)
                    _set_param_sync(cf, th, "traj.start", 0)
                    _set_param_sync(cf, th, "traj.mode", 0)
                    th.sleep(0.1)
                ox, oy = _sample_origin(th, fallback_xy)
                _apply_onboard_traj_origin(cf, th, ox, oy, args.height)
                _wait_for_state(th)
                z_hover = float(_latest_state.get("stateEstimate.z", args.height))
                keepalive_pos = np.array([ox, oy, max(z_hover, _Z_CMD_MIN_AIRBORNE)])

                _log_t0 = time.monotonic()
                _set_param_sync(cf, th, "traj.start", 0)
                _set_param_sync(cf, th, "traj.mode", 1)
                th.sleep(_TRAJ_ARM_DELAY_S)
                _set_param_sync(cf, th, "traj.start", 1)
                # Stop margin before total_dur on the last rep — firmware wraps t at lap end
                # and restarts the path (C0 position, discontinuous velocity).
                if rep < args.reps - 1:
                    t_run = traj_dur
                else:
                    t_run = max(traj_dur - _TRAJ_STOP_MARGIN_S, traj_dur * 0.95)
                    print(
                        f"[flight] Onboard lap {t_run:.2f}s "
                        f"(margin {_TRAJ_STOP_MARGIN_S:.2f}s before {traj_dur:.2f}s wrap)"
                    )
                t_end = th.time() + t_run
                while not th.isShutdown() and th.time() < t_end:
                    cf.cmdFullState(keepalive_pos, _zero3, _zero3, 0.0, _zero3)
                    th.sleepForRate(20)

            print("[flight] Done. Stopping trajectory and landing...")
            hold_pos = _stop_onboard_traj_and_hold(cf, th, keepalive_pos)
            # Settle at hold_pos for 1 s, then hand back to HLC for smooth descent.
            _stream_hover_hold(cf, th, hold_pos, 1.0)
            _notify_setpoints_stop_sync(cf, th, remain_ms=200)
            print("[flight] Landing...")
            allcfs.land(targetHeight=0.06, duration=3.0)
            th.sleep(4.0)
            _logging_active = False
            onboard_landed = True  # skip duplicate land in common block below

        else:
            # ── Mode E: HLC Poly4D upload (existing behaviour) ────────────────
            for c in allcfs.crazyflies:
                c.uploadTrajectory(0, 0, traj)

            # Switch to configured ctrl_mode for the trajectory
            if (yaml_controller, traj_ctrl_mode) != (_RAMP_CONTROLLER, _RAMP_CTRL_MODE):
                _apply_flight_settings(
                    allcfs, th, "trajectory", yaml_controller, traj_ctrl_mode, indi_gains_from_yaml
                )
            else:
                _log_phase("trajectory", yaml_controller, traj_ctrl_mode)
            print("[flight] Starting trajectory...")
            for c in allcfs.crazyflies:
                try:
                    c.setParam("usd.logging", 1)
                except Exception:
                    pass  # uSD deck not present — skip silently

            for rep in range(args.reps):
                if rep > 0:
                    th.sleep(1.0)
                _log_t0 = time.monotonic()
                allcfs.startTrajectory(0, timescale=args.speed)
                th.sleep(traj_dur * args.speed)
                print(
                    f"[flight] HLC lap {traj_dur * args.speed:.2f}s done — overriding HLC"
                )

            print("[flight] Done. Stopping HLC trajectory and landing...")
            # Override HLC by streaming position setpoints, then hand back to HLC for land.
            hlc_hold_pos = np.array([
                float(_latest_state.get("stateEstimate.x", 0.0)),
                float(_latest_state.get("stateEstimate.y", 0.0)),
                float(_latest_state.get("stateEstimate.z", args.height)),
            ])
            _stream_hover_hold(cf, th, hlc_hold_pos, 2.5)
            # Switch controller while low-level setpoints still hold the drone,
            # so there is no gap between notify_setpoints_stop and allcfs.land().
            for c in allcfs.crazyflies:
                c.setParam("stabilizer.controller", _RAMP_CONTROLLER)
                c.setParam("indi_gains.ctrl_mode", _RAMP_CTRL_MODE)
            th.sleep(0.2)
            _notify_setpoints_stop_sync(cf, th, remain_ms=200)

        # Land immediately — ctrl_mode already switched above
        if not onboard_landed:
            _logging_active = False
            _log_phase("landing", _RAMP_CONTROLLER, _RAMP_CTRL_MODE)
            print("[flight] Landing...")
            allcfs.land(targetHeight=0.06, duration=2.0)
            th.sleep(3.0)
            for c in allcfs.crazyflies:
                try:
                    c.setParam("usd.logging", 0)
                except Exception:
                    pass  # uSD deck not present — skip silently

    finally:
        # ── Save log + always return firmware to HLC-idle (even on Ctrl+C) ─
        _logging_active = False
        try:
            _firmware_idle_reset(cf, th)
            print("[flight] Cleanup done — ready for another run.")
        except Exception as exc:
            print(f"[flight] WARN: cleanup failed: {exc}")
        if _log_rows:
            _save_log(args.trajectory, args.mode, args.kt, args.speed, args.reps, traj_dur)
        else:
            print("[log] No rows collected — log not saved.")


if __name__ == "__main__":
    main()
