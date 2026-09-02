#!/usr/bin/env python3
"""Simple multi-drone CS2 flight script — the figure8.py pattern, generalized.

Stock uploadTrajectory/startTrajectory only (Mode E) -- no Mode D onboard eval, no
--onboard/--mode flags. Every drone in the roster flies the SAME trajectory shape
relative to its own initialPosition -- this is not a formation with per-drone offsets,
use run_formation for that.

Controller-mode/gain switching (from crazyflies.yaml, same convention as always: yaml
sets the trajectory controller, the ramp/landing controller is the fixed OOT-geometric
baseline), radio logging, and the takeoff/landing state machine are the SAME functions
flight.py uses -- imported from there, not copy-pasted, so the two scripts cannot drift
apart silently. That logging is single-drone (the first drone in the roster), exactly
matching flight.py's own current scope -- not a new limitation introduced here.

Brushless-only: ESCs are armed/disarmed unconditionally around takeoff/land. No
--brushless flag -- this project now only flies CF21BL.

Coupling note: this reaches into flight.py's underscore-prefixed internals (module
globals and helper functions) rather than duplicating them. If flight.py's internal
names or signatures change, this script breaks with them -- that is the price of reuse
over copy-paste, and is why both live in the same package.

Usage:
  ros2 run crazyflie_examples simple_flight -- --trajectory figure8 --kt 0.008
  ros2 run crazyflie_examples simple_flight -- --trajectory circle  --kt 0.1
  ros2 run crazyflie_examples simple_flight -- --trajectory oval    --kt 0.2
  ros2 run crazyflie_examples simple_flight -- --trajectory hover   --duration 15

Arguments:
  --trajectory : figure8 | circle | oval | hover
  --kt         : aggressiveness/speed for figure8/circle/oval (default: 0.1 for circle,
                 0.008 otherwise). Must match an exported CSV in data/.
  --speed      : playback timescale, 1.0 = normal, <1 = slower (default: 1.0)
  --height     : flight height in metres (default: 1.0)
  --duration   : hover duration in seconds, hover trajectory only (default: 15.0)
  --reps       : number of trajectory repetitions (default: 1)
"""

import argparse
import sys
import time
from pathlib import Path

import numpy as np
from crazyflie_interfaces.msg import LogDataGeneric
from crazyflie_py import Crazyswarm
from crazyflie_py.uav_trajectory import Trajectory

from crazyflie_examples import flight as _f

DATA_DIR = Path(__file__).parent / 'data'


def _csv_path(trajectory: str, kt: float) -> Path:
    kt_str = f'{kt:.6f}'.rstrip('0').rstrip('.')
    path = DATA_DIR / f'{trajectory}_mode1_kt{kt_str}.csv'
    if not path.exists():
        print(f'[error] CSV not found: {path}')
        print('  Generate it with (from flying_drone_stack/):')
        print(
            f'    cargo run --release --bin export_poly4d -- '
            f'--trajectory {trajectory} --mode 1 --kt {kt}'
        )
        sys.exit(1)
    return path


def main():
    parser = argparse.ArgumentParser(description='Simple multi-drone CS2 flight script')
    parser.add_argument(
        '--trajectory', default='figure8', choices=['figure8', 'circle', 'oval', 'hover']
    )
    parser.add_argument('--kt', type=float, default=None)
    parser.add_argument('--speed', type=float, default=1.0)
    parser.add_argument('--height', type=float, default=1.0)
    parser.add_argument('--duration', type=float, default=15.0)
    parser.add_argument('--reps', type=int, default=1)
    args, _ = parser.parse_known_args()

    if args.kt is None:
        args.kt = 0.1 if args.trajectory == 'circle' else 0.008

    hover_mode = args.trajectory == 'hover'
    _f._onboard_mode = False  # Mode E only, ever -- feeds flight.py's _save_log meta field

    traj = None
    traj_dur = args.duration
    if not hover_mode:
        csv_path = _csv_path(args.trajectory, args.kt)
        traj = Trajectory()
        traj.loadcsv(csv_path)
        traj_dur = traj.duration

    swarm = Crazyswarm()
    th = swarm.timeHelper
    allcfs = swarm.allcfs
    cf = allcfs.crazyflies[0]
    cf_name = cf.prefix.lstrip('/')
    n_drones = len(allcfs.crazyflies)

    if not hover_mode:
        print(
            f'[simple_flight] {csv_path.name}  duration={traj_dur:.2f}s  n_drones={n_drones}'
        )

    # Same radio-log subscriptions as flight.py, reused not copy-pasted (first drone only).
    allcfs.create_subscription(LogDataGeneric, f'{cf_name}/state', _f._state_cb, 10)
    allcfs.create_subscription(LogDataGeneric, f'{cf_name}/attitude', _f._attitude_cb, 10)
    allcfs.create_subscription(LogDataGeneric, f'{cf_name}/gyro_acc', _f._gyro_cb, 10)
    allcfs.create_subscription(LogDataGeneric, f'{cf_name}/rpm', _f._rpm_cb, 10)
    allcfs.create_subscription(LogDataGeneric, f'{cf_name}/indi_state', _f._indi_state_cb, 10)
    allcfs.create_subscription(
        LogDataGeneric, f'{cf_name}/indi_alp_raw', _f._indi_alp_raw_cb, 10
    )
    allcfs.create_subscription(
        LogDataGeneric, f'{cf_name}/indi_filter_char', _f._indi_filter_cb, 10
    )
    allcfs.create_subscription(
        LogDataGeneric, f'{cf_name}/indi_alp_notch', _f._indi_alp_notch_cb, 10
    )

    print('[simple_flight] Preflight: idle reset + Kalman pulse (place drone on pad)...')
    _f._firmware_idle_reset(cf, th)
    _f._kalman_reset_pulse(cf, th)
    print('[simple_flight] Waiting for EKF to converge on mocap poses...')
    th.sleep(3.0)

    (
        yaml_controller,
        traj_ctrl_mode,
        indi_gains_from_yaml,
        pos_gains_from_yaml,
        diag_gains_from_yaml,
    ) = _f._load_firmware_controller_config()
    _f._yaml_indi_gains.update(indi_gains_from_yaml)
    _f._yaml_pos_gains.update(pos_gains_from_yaml)
    _f._yaml_diag_gains.update(diag_gains_from_yaml)
    _f._controller_meta['yaml'] = (yaml_controller, traj_ctrl_mode)
    print(
        f'[simple_flight] crazyflies.yaml (trajectory): stabilizer.controller={yaml_controller} '
        f'indi_gains.ctrl_mode={traj_ctrl_mode}'
    )
    print(
        f'[simple_flight] ramp (takeoff/landing): stabilizer.controller={_f._RAMP_CONTROLLER} '
        f'indi_gains.ctrl_mode={_f._RAMP_CTRL_MODE}'
    )

    if not hover_mode:
        for c in allcfs.crazyflies:
            c.uploadTrajectory(0, 0, traj)

    _f._apply_flight_settings(
        allcfs, th, 'takeoff', _f._RAMP_CONTROLLER, _f._RAMP_CTRL_MODE,
        pos_gains=_f._RAMP_POS_GAINS,
    )

    for c in allcfs.crazyflies:
        c.arm(True)
    th.sleep(0.5)

    _f._logging_active = True
    _f._log_t0 = time.monotonic()

    print('[simple_flight] Taking off...')
    allcfs.takeoff(targetHeight=args.height, duration=3.0)
    th.sleep(3.5)
    for c in allcfs.crazyflies:
        pos = np.array(c.initialPosition) + np.array([0.0, 0.0, args.height])
        c.goTo(pos, 0, 2.0)
    th.sleep(2.5)

    try:
        if (yaml_controller, traj_ctrl_mode, pos_gains_from_yaml) != (
            _f._RAMP_CONTROLLER, _f._RAMP_CTRL_MODE, _f._RAMP_POS_GAINS,
        ):
            _f._apply_flight_settings(
                allcfs, th, 'trajectory', yaml_controller, traj_ctrl_mode,
                indi_gains_from_yaml, pos_gains_from_yaml,
            )
        else:
            _f._log_phase('trajectory', yaml_controller, traj_ctrl_mode)

        for c in allcfs.crazyflies:
            try:
                c.setParam('usd.logging', 1)
            except Exception:
                pass  # uSD deck not present -- skip silently

        if hover_mode:
            print(f'[simple_flight] Hovering {args.duration:.0f}s ({n_drones} drones)...')
            th.sleep(args.duration)
        else:
            print('[simple_flight] Starting trajectory...')
            for rep in range(args.reps):
                if rep > 0:
                    th.sleep(1.0)
                _f._log_t0 = time.monotonic()
                allcfs.startTrajectory(0, timescale=args.speed)
                th.sleep(traj_dur * args.speed)

        print('[simple_flight] Done. Landing...')
        th.sleep(0.5)
        hold_pos = np.array(
            [
                float(_f._latest_state.get('stateEstimate.x', 0.0)),
                float(_f._latest_state.get('stateEstimate.y', 0.0)),
                float(_f._latest_state.get('stateEstimate.z', args.height)),
            ]
        )
        _f._stream_hover_hold(cf, th, hold_pos, 2.5)
        for c in allcfs.crazyflies:
            c.setParam('stabilizer.controller', _f._RAMP_CONTROLLER)
            c.setParam('indi_gains.ctrl_mode', _f._RAMP_CTRL_MODE)
        th.sleep(0.2)
        _f._notify_setpoints_stop_sync(cf, th, remain_ms=200)

        _f._logging_active = False
        _f._apply_flight_settings(
            allcfs, th, 'landing', _f._RAMP_CONTROLLER, _f._RAMP_CTRL_MODE,
            pos_gains=_f._RAMP_POS_GAINS,
        )
        print('[simple_flight] Landing...')
        allcfs.land(targetHeight=0.06, duration=2.0)
        th.sleep(3.0)
        for c in allcfs.crazyflies:
            c.arm(False)

    finally:
        _f._logging_active = False
        try:
            _f._firmware_idle_reset(cf, th)
            print('[simple_flight] Cleanup done — ready for another run.')
        except Exception as exc:
            print(f'[simple_flight] WARN: cleanup failed: {exc}')
        if _f._log_rows:
            _f._save_log(args.trajectory, 1, args.kt, args.speed, args.reps, traj_dur)
        else:
            print('[log] No rows collected — log not saved.')


if __name__ == '__main__':
    main()
