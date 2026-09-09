#!/usr/bin/env python3
"""Simple multi-drone CS2 flight script — the figure8.py pattern, generalized.

Stock uploadTrajectory/startTrajectory only (Mode E) -- no Mode D onboard eval, no
--onboard/--mode flags. Every drone in the roster flies the SAME trajectory shape
relative to its own initialPosition -- this is not a formation with per-drone offsets,
use run_formation for that.

Multi-drone takeoff history: confirmed in sim 2026-09-03 that a broadcast
`allcfs.takeoff(targetHeight=args.height)` sends every drone to the SAME height first,
regardless of its own configured initialPosition offset -- two drones sharing similar
(x,y) came within ~0-1mm of each other for several seconds before the later per-drone
`goTo` ever separated them. Fixed same day: takeoff is now per-drone, straight to each
drone's own target height (`initialPosition.z + args.height`), same pattern
`formation_flight.py` already uses for exactly this reason. Re-validated same day, 8/8
matrix cases pass (`run_simple_flight_matrix.sh`): min in-flight inter-drone separation
0.197-0.494 m across figure8/circle/oval/hover, all real positive numbers, no more
near-zero climb-together window.

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


# Position gains that belong to the GEOMETRIC law (ctrl_mode==0). crazyflies.yaml's
# pos_gains block is INDI's: kp_xy=64/kv_xy=5, locked 2026-07-19 against kr=2400/kw=170
# ("KV FLOOR FOUND: kv=4 CRASHED 2 out of 2 flights ... LOCKED at kv=5"). Geometric's
# attitude loop is far slower (kr_geo=0.010, the 27 g platform's value on a 41 g airframe),
# so it does not satisfy the cascade separation those gains assume -- at kp_xy=64/kv_xy=5
# (zeta=0.31) it rang up exponentially on 2026-09-09, ~1.45x/s. These are the values the
# yaml itself names as the alternative ("To revert to fully original tuned config:
# kp_xy=40/kv_xy=8/kp_z=30/kv_z=10", zeta=0.63) and are the same gains that flew the ramp
# phase of all five flights that day at |roll| 0.3-1.8 deg. Not a guess -- a previously
# tuned config, re-selected for the controller it was tuned for.
GEOMETRIC_POS_GAINS = {'kp_xy': 40.0, 'kp_z': 30.0, 'kv_xy': 8.0, 'kv_z': 10.0}


def _select_pos_gains(controller, ctrl_mode, yaml_pos_gains):
    """Pick the position gains tuned for the attitude law actually being flown.

    pos_gains.* are OUR params (traj_iface.c) and are read only by the out-of-tree
    controller, stabilizer.controller==6. Every stock controller (1 PID, 2 Mellinger,
    3 INDI, 4 Brescianini, 5 Lee) runs its own internal position loop and ignores them
    entirely, so for those the yaml values pass through untouched -- substituting there
    would be a no-op dressed up as a decision.

    Within the OOT controller, ctrl_mode 0 is geometric; 1/2/3 run INDI in at least one
    loop and keep the yaml's INDI-tuned block. Returns (gains, source_label) so the caller
    can say out loud which set it used -- a silent gain substitution is exactly the class
    of bug that cost the 2026-09-09 session.
    """
    if controller != 6:
        return dict(yaml_pos_gains), f'crazyflies.yaml (inert: stock controller {controller})'
    if ctrl_mode == 0:
        return dict(GEOMETRIC_POS_GAINS), 'GEOMETRIC_POS_GAINS (simple_flight.py)'
    return dict(yaml_pos_gains), 'crazyflies.yaml pos_gains (INDI-tuned)'


def _pin_ramp_to_target(controller, ctrl_mode, pos_gains, indi_gains):
    """Take off, fly and land in ONE controller configuration -- no mid-air switch.

    Root cause of the 2026-09-09 crash session. flight.py ramps takeoff/landing on a fixed
    OOT-geometric baseline (_RAMP_CONTROLLER=6, _RAMP_CTRL_MODE=0, _RAMP_POS_GAINS
    kp_xy=40/kv_xy=8) and switches to the yaml controller once at altitude. Every flight
    that day diverged at that switch, ~6.06 s in, and nowhere else:

      * INDI -- OURS (ctrl_mode=3) AND STOCK (stabilizer.controller=3, pure bitcraze code
        that never touches this firmware): |roll| 0.6-0.7 deg at t=5 s, 22-32 deg one second
        later, full tumble ~1.4 s after the switch. INDI is incremental
        (tau = tau_current + J*(alpha_ref - alpha_meas)); handed control at altitude its
        tau_prev/filter state is cold while the motors are already producing hover thrust,
        so the first increment is computed off a baseline that does not match reality. On
        the ground that same cold state is CONSISTENT (tau_prev = 0, actual torque = 0),
        which is why taking off in INDI is the fix rather than a bigger risk. Matches the
        previously logged "INSTANT CRASH within 1.4s of switch" exactly.
      * GEOMETRIC (ctrl_mode=0): the pos-gain swap alone destabilises it --
        kp_xy 40->64 (+60%) with kv_xy 8->5 (-37%) halves the damping ratio (0.63 -> 0.31).
        Rings up exponentially, ~1.45x/s: 2.2 -> 3.1 -> 4.9 -> 7.2 -> 13.3 -> 20.1 -> 29.4 deg.
        Those gains were tuned against INDI at kr=2400; geometric runs kr_geo=0.010 (the 27 g
        platform's value on a 41 g airframe), a much slower attitude loop, so the cascade
        separation the gains assume does not hold.

    Stock Lee (stabilizer.controller=5), switched at the same instant, hovered 25 s at roll
    std 0.74 deg -- the vehicle, motors, mocap, EKF and RPM deck are all healthy, and the OOT
    geometric controller itself flew the first 6 s of all five flights at |roll| 0.3-1.8 deg.
    Nothing was broken except the transition.

    Pinning the ramp constants to the trajectory values makes flight.py's own
    "did anything change?" guard compare equal, so it logs the phase and never re-pushes
    params in flight. Done by assigning flight.py's module globals rather than editing it --
    flight.py is frozen legacy (Mode D) and only borrowed here for its internals.
    """
    _f._RAMP_CONTROLLER = controller
    _f._RAMP_CTRL_MODE = ctrl_mode
    _f._RAMP_POS_GAINS = dict(pos_gains)
    _f._RAMP_INDI_GAINS = dict(indi_gains)


def _append_full_gains_meta():
    """Append the full firmware_params block to the CSV flight.py just saved, as more
    `# meta:` lines -- same prefix its own meta block uses, just appended after the data
    rows instead of before the header.

    flight.py's own meta block records a fixed key list that predates kr_geo/kw_geo and
    the clamp settings, so a geometric flight log states only `indi_kr=2400` -- the INDI
    gain, which ctrl_mode=0 never uses. On 2026-09-09 that gap meant two crashed hover
    flights could not be told apart from their logs at all: nothing recorded which
    attitude gains actually flew, and a THIRD blind spot -- clamp_en reading 0 instead of
    11 from a stale installed crazyflies.yaml -- took a direct torque-telemetry read to
    find, specifically because it wasn't in any log.

    Appending directly into the CSV (not a separate sidecar, which is what this used to be
    -- see git history) means one file always has everything: every consumer here already
    treats '#' as a comment marker independent of position (np.loadtxt's default
    comments='#' skips it anywhere; analyze_flight.py's load_csv_with_meta scans the whole
    file for "# meta:" regardless of where it falls; compute_kt_motor.py/indi_tune.py
    guard on column-count mismatch). Verified before writing this way -- see
    experiments/analysis/README.md's log-format notes.

    This lives here rather than in flight.py deliberately -- flight.py is frozen legacy
    (Mode D) and is only borrowed for its logging internals, so it does not get edited.
    Appending is additive: it cannot change the CSV schema anything else already parses.

    Reads crazyflies.yaml DIRECTLY rather than reusing _f._yaml_indi_gains: that dict is
    already filtered down to a hardcoded 14-key subset in flight.py's
    _load_firmware_controller_config, which predates kr_geo/kw_geo/clamp_en/tau_xy_max and
    so drops exactly the values this exists to capture. Dumping the whole
    all.firmware_params block instead means a gain added to the yaml in future is recorded
    without anyone remembering to extend a list here.

    Reads the INSTALLED copy via get_package_share_directory, the same path flight.py
    itself reads -- not the source tree file, which can silently disagree with it (that
    mismatch was the actual root cause found tonight). Still the yaml's stated INTENT at
    connect, not a value read back from the drone -- a param that failed to apply still
    shows here as whatever the yaml asked for. Confirm on the vehicle when it matters.
    """
    import yaml as _yaml
    from ament_index_python.packages import get_package_share_directory
    try:
        csvs = sorted(_f.LOGS_DIR.glob('*.csv'), key=lambda p: p.stat().st_mtime)
        if not csvs:
            return
        cfg_path = (
            Path(get_package_share_directory('crazyflie')) / 'config' / 'crazyflies.yaml'
        )
        with open(cfg_path) as fh:
            cfg = _yaml.safe_load(fh)
        fp = (cfg.get('all') or {}).get('firmware_params', {})
        path = csvs[-1]
        with open(path, 'a') as out:
            out.write(f'# meta:full_gains_source={cfg_path} (installed copy, intent not readback)\n')
            for group, values in fp.items():
                if not isinstance(values, dict):
                    continue
                for k, v in values.items():
                    out.write(f'# meta:full_{group}_{k}={v}\n')
            for phase, (controller, mode) in _f._controller_meta.items():
                out.write(f'# meta:full_{phase}_stabilizer_controller={controller}\n')
                out.write(f'# meta:full_{phase}_ctrl_mode={mode}\n')
        print(f'[log] full gains appended to {path}')
    except Exception as exc:
        print(f'[log] WARN: full gains not appended: {exc}')


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
    parser.add_argument(
        '--pin-controller', action='store_true',
        help='Take off, fly and land in ONE controller config -- no mid-air switch. NOT the '
             'frozen behaviour: finalized-version-for-INDI-project ramped on OOT-geometric '
             'and switched at altitude, and that handover flew the whole July campaign, so '
             'it is not the fault. Opt-in experiment only -- taking off directly in INDI has '
             'never been flown. See _pin_ramp_to_target().',
    )
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
    try:
        _f._firmware_idle_reset(cf, th)
    except Exception as exc:
        # Mode-D-only params (traj.mode etc.) -- this script never uses Mode D, and the sim
        # backend doesn't declare them at all, so there's nothing to actually reset here.
        # On hardware this clears residual state from a PREVIOUS Mode D flight; skip cleanly
        # when it's not there rather than crash before ever reaching a Mode-E-only script.
        print(f'[simple_flight] WARN: idle reset skipped ({exc}) -- Mode-D params not present')
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
    # Position gains are selected for the law actually being flown REGARDLESS of --pin-controller:
    # this is not part of the handover question. crazyflies.yaml's pos_gains are INDI's
    # (kp_xy=64/kv_xy=5, locked 2026-07-19 against kr=2400/kw=170) and geometric was never
    # flown on them until 2026-09-07, when it oscillated -- with a_indi identically zero in
    # geometric, the residual sign cannot explain that, the gains can. See _select_pos_gains.
    pos_gains_from_yaml, pg_source = _select_pos_gains(
        yaml_controller, traj_ctrl_mode, pos_gains_from_yaml
    )
    print(f'[simple_flight] position gains: {pos_gains_from_yaml}  <- {pg_source}')
    _f._yaml_pos_gains.clear()
    _f._yaml_pos_gains.update(pos_gains_from_yaml)

    if args.pin_controller:
        print(
            '[simple_flight] --pin-controller: one config from takeoff to landing, no mid-air '
            'switch. NOT the frozen behaviour -- taking off directly in INDI has never flown.'
        )
        _pin_ramp_to_target(
            yaml_controller, traj_ctrl_mode, pos_gains_from_yaml, indi_gains_from_yaml
        )
    print(
        f'[simple_flight] ramp (takeoff/landing): stabilizer.controller={_f._RAMP_CONTROLLER} '
        f'indi_gains.ctrl_mode={_f._RAMP_CTRL_MODE} pos_gains={_f._RAMP_POS_GAINS}'
    )

    if not hover_mode:
        for c in allcfs.crazyflies:
            c.uploadTrajectory(0, 0, traj)

    # Push indi_gains at takeoff too, not just pos_gains: with the ramp pinned to the
    # trajectory config the vehicle must already be in its FINAL configuration before it
    # leaves the ground, or the mid-air param push this fix exists to remove comes back.
    _f._apply_flight_settings(
        allcfs, th, 'takeoff', _f._RAMP_CONTROLLER, _f._RAMP_CTRL_MODE,
        indi_gains=getattr(_f, '_RAMP_INDI_GAINS', None),
        pos_gains=_f._RAMP_POS_GAINS,
    )

    for c in allcfs.crazyflies:
        c.arm(True)
    th.sleep(0.5)

    _f._logging_active = True
    _f._log_t0 = time.monotonic()

    # Staged takeoff (2026-09-03 fix): per-drone takeoff straight to each drone's own
    # target height, not one broadcast to a shared height first. The broadcast form sends
    # every drone to the SAME z regardless of its own initialPosition offset -- confirmed
    # in sim to bring two drones sharing similar (x,y) within 0-1mm of each other for
    # several seconds before the later goTo ever separated them. Same fix formation_flight.py
    # already uses for exactly this reason (its "stage 1: climb to individual heights").
    print('[simple_flight] Taking off (per-drone target heights)...')
    for c in allcfs.crazyflies:
        target_z = float(c.initialPosition[2]) + args.height
        c.takeoff(targetHeight=target_z, duration=3.0)
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

        # Pure high-level landing, exactly as figure8.py does it: startTrajectory ... land().
        #
        # This used to call _stream_hover_hold() (2.5 s of cmdFullState) and then
        # _notify_setpoints_stop_sync() before land(). That is Mode B / Mode D machinery and
        # it BREAKS the landing, per flight.py's own docstring on _onboard_stream_land:
        # "cmdFullState forces low-level mode; HLC land()/goTo() do not work afterward."
        # So land() was being issued into a firmware still in low-level mode, the vehicle
        # never descended, and arm(False) three seconds later cut the motors from ~0.7 m --
        # the "does not land, just shuts the motors off" seen on 2026-09-09.
        #
        # simple_flight is Mode E only (stock uploadTrajectory/startTrajectory, never Mode D),
        # so nothing here ever put the firmware into low-level mode in the first place and
        # there is no low-level state to hand back. Dropping both calls makes takeoff, upload,
        # trajectory and landing the same high-level sequence figure8.py uses.

        # Logging deliberately stays ON through the descent. It used to be stopped here,
        # before land() was even called, so every log ended mid-air (2026-09-09: the one
        # clean flight's log ends at z=0.743 m) and the descent -- the phase the operator
        # reported as "does not land, just shuts the motors off" -- was never recorded at
        # all. It cannot be diagnosed from a log that stops before it starts.
        _f._apply_flight_settings(
            allcfs, th, 'landing', _f._RAMP_CONTROLLER, _f._RAMP_CTRL_MODE,
            indi_gains=getattr(_f, '_RAMP_INDI_GAINS', None),
            pos_gains=_f._RAMP_POS_GAINS,
        )
        print('[simple_flight] Landing...')
        allcfs.land(targetHeight=0.06, duration=2.0)
        th.sleep(3.0)
        z_end = float(_f._latest_state.get('stateEstimate.z', float('nan')))
        # Disarming at altitude drops the vehicle. If land() did not take (the HLC/low-level
        # handover after notify_setpoints_stop is the usual reason), say so instead of
        # silently cutting the motors and calling it a landing.
        if z_end == z_end and z_end > 0.20:
            print(
                f'[simple_flight] WARN: still at z={z_end:.2f} m after land() + 3.0 s -- '
                'landing did not complete. Holding 2 s more before disarm.'
            )
            th.sleep(2.0)
            z_end = float(_f._latest_state.get('stateEstimate.z', float('nan')))
            if z_end == z_end and z_end > 0.20:
                print(
                    f'[simple_flight] WARN: z={z_end:.2f} m -- disarming anyway, the vehicle '
                    'WILL drop. Check the land()/notify_setpoints_stop handover.'
                )
        _f._logging_active = False
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
            _append_full_gains_meta()
        else:
            print('[log] No rows collected — log not saved.')


if __name__ == '__main__':
    main()
