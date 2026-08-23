"""Run a named formation scenario, in simulation or on hardware.

    ros2 run crazyflie_examples run_formation -- --scenario A3 --dz 0.30
    ros2 run crazyflie_examples run_formation -- --scenario A2 --dz 0.3 --path circle
    ros2 run crazyflie_examples run_formation -- --scenario B1 --dz2 0.4

Sibling to `formation_flight.py`, which flies ONE trajectory shared by every drone in a
fixed formation. This runner gives each drone its OWN trajectory, which is what the
literature's interaction scenarios need -- a hovering vehicle above a translating one, two
counter-rotating vehicles, a swap. The mechanism is already in Crazyswarm2:
`uploadTrajectory` is per-drone while `startTrajectory` is a broadcast, so distinct
trajectories stored under the same id all begin on one packet.

Scenario definitions live in `formations/scenarios.py` and are ROS-free; run
`python -m crazyflie_examples.formations.scenarios --self-test` to check them without
flying anything. `--check` here does the same plus safety and trajectory compilation.

Simulation and hardware use this identical script; only the launch differs. See
docs/10_Formation_Library.md in the thesis repo.
"""

from __future__ import annotations

import argparse
from pathlib import Path
import sys
import time

import numpy as np

import json

from .formations import poly4d, safety, scenarios

DATA_DIR = Path(__file__).parent / 'data' / 'formations'
LOG_DIR = Path('/home/georg/Desktop/flying_robot_course/experiments/logs')

_RAMP_CONTROLLER = 6      # geometric SE(3) for takeoff/landing, as formation_flight does
_RAMP_CTRL_MODE = 0
_CTRL_SETTLE_S = 0.3


def build_parser():
    p = argparse.ArgumentParser(
        description='Run a formation scenario', formatter_class=argparse.RawTextHelpFormatter)
    p.add_argument('--scenario', help='scenario id, e.g. A3 (see --list)')
    p.add_argument('--list', action='store_true', help='list scenarios and exit')

    g = p.add_argument_group('scenario parameters (only the relevant ones are used)')
    g.add_argument('--dz', type=float, help='vertical separation [m]')
    g.add_argument('--dz1', type=float, help='top-to-centre gap, 3-robot [m]')
    g.add_argument('--dz2', type=float, help='centre-to-bottom gap, 3-robot [m]')
    g.add_argument('--dz-start', type=float, dest='dz_start', help='initial dz for merge/dock')
    g.add_argument('--dz-end', type=float, dest='dz_end', help='final dz for merge/dock')
    g.add_argument('--path', help='hover | line | shuttle | circle | lemniscate')
    g.add_argument('--motion', help='A4 bottom motion: lemniscate | line')
    g.add_argument('--offset', type=float, help='lateral offset [m]')
    g.add_argument('--axis', help='lateral offset axis: x | y')
    g.add_argument('--speed', type=float, help='peak translation speed [m/s]')
    g.add_argument('--length', type=float, help='translation length [m]')
    g.add_argument('--radius', type=float, help='circle / lemniscate radius [m]')
    g.add_argument('--period', type=float, help='circle period [s]')
    g.add_argument('--laps', type=float, help='circle / lemniscate laps')
    g.add_argument('--passes', type=int, help='shuttle passes')
    g.add_argument('--span', type=float, help='swap span [m]')
    g.add_argument('--sep', type=float, help='lateral separation, C1 [m]')
    g.add_argument('--gap', type=float, help='along-track gap, C2 [m]')
    g.add_argument('--r', type=float, help='V-stack lateral offset [m]')
    g.add_argument('--hold', type=float, help='hover duration [s]')
    g.add_argument('--n', type=int, help='robot count where the scenario allows it')
    g.add_argument('--rotate', type=float, default=None,
                   help='turn the whole formation about the vertical axis [deg]. Rigid, so '
                        'no inter-robot distance changes -- use it to lay a translating '
                        'scenario along the long axis of the room')

    s = p.add_argument_group('execution and safety')
    s.add_argument('--height', type=float, default=1.0,
                   help='base height of the LOWEST robot [m] (default 1.0)')
    s.add_argument('--allow-extreme', action='store_true',
                   help='permit separations below the safe minimum')
    s.add_argument('--v-max', type=float, default=None, help='override commanded speed limit')
    s.add_argument('--a-max', type=float, default=None, help='override commanded accel limit')
    s.add_argument('--dz-min', type=float, default=None, help='override minimum dz')
    s.add_argument('--geofence', type=str, default=None,
                   help='xmin,xmax,ymin,ymax,zmin,zmax [m] -- overrides the placeholder box')
    s.add_argument('--check', action='store_true',
                   help='build, verify and compile the scenario, then exit (no ROS)')
    s.add_argument('--dry-run', action='store_true', help='print the plan and exit')
    s.add_argument('--auto-center', action='store_true',
                   help='place the formation centred in the flight volume instead of on '
                        'drone 0 (recommended in a tight lab; relative geometry is '
                        'unchanged, only where it flies)')
    s.add_argument('--yes', action='store_true', help='skip the confirmation prompt')
    s.add_argument('--brushless', action='store_true', help='arm ESCs (required for CF21BL)')
    s.add_argument('--timescale', type=float, default=1.0,
                   help='HLC timescale; >1 slows the trajectory down')
    return p


def scenario_params(args) -> dict:
    keys = ('rotate_deg', 'dz', 'dz1', 'dz2', 'dz_start', 'dz_end', 'path', 'motion', 'offset', 'axis',
            'speed', 'length', 'radius', 'period', 'laps', 'passes', 'span', 'sep', 'gap',
            'r', 'hold', 'n')
    out = {k: getattr(args, k) for k in keys if getattr(args, k, None) is not None}
    if args.rotate is not None:
        out['rotate_deg'] = args.rotate
    return out


def make_limits(args) -> safety.Limits:
    lim = safety.Limits(allow_extreme=args.allow_extreme)
    if args.v_max is not None:
        lim.v_max = args.v_max
    if args.a_max is not None:
        lim.a_max = args.a_max
    if args.dz_min is not None:
        lim.dz_min = args.dz_min
    if args.geofence:
        v = [float(x) for x in args.geofence.split(',')]
        if len(v) != 6:
            sys.exit('[formation] --geofence needs xmin,xmax,ymin,ymax,zmin,zmax')
        lim.geofence = dict(x=(v[0], v[1]), y=(v[2], v[3]), z=(v[4], v[5]))
    return lim


def compile_scenario(sc, base_height: float):
    """Compile every robot's curve and write it as a Poly4D CSV. Returns (paths, tables)."""
    paths, tables = [], []
    for i, r in enumerate(sc.robots):
        table = poly4d.compile_curve(r.curve)
        tag = '_'.join(f'{k}{v}' for k, v in sc.params.items()
                       if isinstance(v, (int, float)))
        path = DATA_DIR / f'{sc.sid}_{tag}_r{i}_{r.role}.csv'
        poly4d.write_csv(path, table)
        # Verify what was WRITTEN, not what was computed. The file is what flies.
        poly4d.verify_csv(path, r.curve)
        paths.append(path)
        tables.append(table)
    return paths, tables


def centred_anchor(sc, lim) -> np.ndarray:
    """Anchor that puts the scenario's commanded bounding box in the middle of the volume.

    Only moves WHERE the formation flies. Every relative distance -- what the scenario
    specifies and what verification checks -- is unchanged.
    """
    _, box = safety.sample_positions(sc, base=np.zeros(3))
    lo, hi = box.reshape(-1, 3).min(axis=0), box.reshape(-1, 3).max(axis=0)
    gf = lim.geofence
    mid = np.array([(gf['x'][0] + gf['x'][1]) / 2,
                    (gf['y'][0] + gf['y'][1]) / 2,
                    (gf['z'][0] + gf['z'][1]) / 2])
    return mid - (lo + hi) / 2


def geometry_table(sc, n: int = 400) -> str:
    """Commanded relative geometry over the flight -- what the scenario promises."""
    ts = np.linspace(0.0, sc.duration, n)
    out = []
    for i in range(sc.n_robots):
        for j in range(i + 1, sc.n_robots):
            d = np.array([sc.relative(i, j, t) for t in ts])
            dist = np.linalg.norm(d, axis=1)
            out.append(
                f'    {sc.robots[i].role:>7} - {sc.robots[j].role:<7} '
                f'dx {d[:, 0].min():+6.2f}..{d[:, 0].max():+6.2f}  '
                f'dy {d[:, 1].min():+6.2f}..{d[:, 1].max():+6.2f}  '
                f'dz {d[:, 2].min():+6.2f}..{d[:, 2].max():+6.2f}  '
                f'|d| {dist.min():.3f}..{dist.max():.3f} m')
    return '\n'.join(out)


def print_plan(sc, lim, base_height, tables, problems, spec_problems):
    print(f'\n[formation] {sc.sid}  {sc.title}')
    print(f'[formation] {sc.notes}')
    print(f'[formation] {sc.n_robots} robot(s), {sc.duration:.1f} s, '
          f'tags: {", ".join(sc.tags)}')
    print(f'[formation] anchor {np.round(base_height, 2) if hasattr(base_height, "shape") else base_height}')
    print(safety.describe(sc, lim, base=base_height if hasattr(base_height, 'shape')
                          else np.array([0.0, 0.0, base_height])))
    for i, t in enumerate(tables):
        print(f'    robot {i} trajectory: {len(t):2d} pieces, '
              f'{t[:, 0].sum():.2f} s')
    print('[formation] commanded relative geometry:')
    print(geometry_table(sc))
    if spec_problems:
        print('[formation] SPEC MISMATCH:')
        for m in spec_problems:
            print(f'    - {m}')
    if problems:
        print('[formation] SAFETY:')
        for m in problems:
            print(f'    - {m}')


def relative_report(loggers, sc, poses=None):
    """Realised vs commanded geometry, after the flight.

    `poses` is the fallback: the radio log topics DroneLogger subscribes to do not exist in
    simulation, so in sim it records nothing and this report would silently print zeros --
    which reads as a total tracking failure rather than as missing telemetry. /pose is
    published in both environments.
    """
    print('\n[formation] realised relative geometry (last settled sample):')
    pos = [lg.position() for lg in loggers]
    if poses is not None and all(np.allclose(p, 0.0) for p in pos):
        print('    (radio log topics empty -- using /pose; expected in simulation)')
        pos = poses
    for i in range(len(loggers)):
        for j in range(i + 1, len(loggers)):
            got = pos[i] - pos[j]
            want = sc.relative(i, j, sc.duration)
            err = got - want
            print(f'    {loggers[i].name} - {loggers[j].name}: '
                  f'got {np.round(got, 3)}  want {np.round(want, 3)}  '
                  f'err {np.round(err, 3)} m  |ez| {abs(err[2]) * 1000:.0f} mm')


def main():
    # parse_known_args, not parse_args: ROS options (--ros-args -p use_sim_time:=true) are
    # appended to the same command line and are consumed later by rclpy, not by us.
    args, _ros = build_parser().parse_known_args()

    if args.list:
        for k in sorted(scenarios.BUILDERS):
            sc = scenarios.build(k)
            print(f'  {k}  n={scenarios.N_ROBOTS[k]:<2} {sc.title}')
            print(f'        tags={",".join(sc.tags)}')
            print(f'        {sc.notes}')
        return

    if not args.scenario:
        sys.exit('[formation] --scenario is required (use --list to see them)')

    sc = scenarios.build(args.scenario, **scenario_params(args))
    lim = make_limits(args)
    spec_problems = scenarios.check_spec(sc)
    # Offline the anchor XY is unknown (it comes from drone 0's start position), so the
    # fence is checked about the origin at the requested base height. The runner
    # re-checks against the real anchor once ROS reports it.
    base = np.array([0.0, 0.0, args.height])
    if args.auto_center:
        base = centred_anchor(sc, lim)
    problems = safety.check(sc, lim, base=base)
    paths, tables = compile_scenario(sc, args.height)

    print_plan(sc, lim, base, tables, problems, spec_problems)
    for p in paths:
        print(f'[formation] wrote {p.relative_to(DATA_DIR.parent.parent)}')

    if spec_problems:
        sys.exit('\n[formation] refusing to fly: the scenario does not match its own '
                 'specification.')
    if problems:
        sys.exit('\n[formation] refusing to fly: safety check failed (see above). '
                 'Use --allow-extreme / --geofence if these limits are wrong.')
    print('\n[formation] checks passed.')

    if args.check:
        return
    if args.dry_run:
        print('[formation] --dry-run: nothing commanded.')
        return

    # ── everything below needs ROS ──────────────────────────────────────────
    from crazyflie_py import Crazyswarm
    from crazyflie_py.uav_trajectory import Trajectory

    from .formation_flight import DroneLogger, load_controller_config

    controller, traj_ctrl_mode, indi_gains, pos_gains = load_controller_config()

    swarm = Crazyswarm()
    th = swarm.timeHelper
    allcfs = swarm.allcfs
    cfs = allcfs.crazyflies
    if len(cfs) != sc.n_robots:
        sys.exit(f'[formation] {sc.sid} needs {sc.n_robots} robot(s) but '
                 f'{len(cfs)} are enabled in crazyflies.yaml')

    # The formation is anchored on ONE point -- drone 0's start plus the base height --
    # not on each drone's own initial_position, or a stack would inherit the vehicles'
    # differing takeoff XY and never actually be vertical.
    anchor = np.array(cfs[0].initialPosition) + np.array([0.0, 0.0, args.height])

    if args.auto_center:
        # Anchoring blindly on drone 0 wastes flight volume: the formation lands wherever
        # that drone happens to sit, and a scenario that would fit comfortably in the
        # middle of the room gets refused for leaving the far wall. Centring the
        # scenario's own commanded bounding box inside the geofence changes only WHERE it
        # flies -- every relative distance, which is what the scenario specifies and what
        # verification checks, is untouched.
        anchor = centred_anchor(sc, lim)
        print(f'[formation] --auto-center: anchor moved to {anchor.round(2)} '
              f'(was {(np.array(cfs[0].initialPosition) + [0, 0, args.height]).round(2)})')

    slots = [anchor + r.slot for r in sc.robots]

    # Re-check the geofence now that the true anchor is known -- offline it was assumed at
    # the origin, and a formation that fits about the origin can still leave the volume
    # once it is anchored on a drone parked near an edge.
    late = safety.check(sc, lim, base=anchor)
    if late:
        for m in late:
            print(f'[formation] SAFETY (anchored at {anchor.round(2)}): {m}')
        sys.exit('[formation] refusing to fly.')

    print(f'\n[formation] anchored on {cfs[0].prefix.lstrip("/")} at {anchor.round(2)}')
    for c, r, s in zip(cfs, sc.robots, slots):
        print(f'    {c.prefix.lstrip("/"):12s} [{r.role:>7}]  start '
              f'{np.array(c.initialPosition).round(2)} -> slot {s.round(2)}')
    if not args.yes:
        try:
            if input('\nProceed? [y/N] ').strip().lower() not in ('y', 'yes'):
                return print('[formation] aborted.')
        except (EOFError, KeyboardInterrupt):
            return print('\n[formation] aborted.')

    trajs = []
    for p in paths:
        t = Trajectory()
        t.loadcsv(p)
        trajs.append(t)

    log_t0 = time.monotonic()
    loggers = [DroneLogger(allcfs, c.prefix.lstrip('/'), log_t0) for c in cfs]

    # /pose exists in simulation and on hardware; the custom log topics only on hardware.
    from geometry_msgs.msg import PoseStamped
    latest = [np.zeros(3) for _ in cfs]

    def _sub(k, name):
        def cb(msg):
            latest[k] = np.array([msg.pose.position.x, msg.pose.position.y,
                                  msg.pose.position.z])
        return allcfs.create_subscription(PoseStamped, f'/{name}/pose', cb, 10)

    _subs = [_sub(k, c.prefix.lstrip('/')) for k, c in enumerate(cfs)]
    th.sleep(2.0)

    stamp = time.strftime('%Y-%m-%d_%H-%M-%S')
    meta = {'run_eval_mode': 'formation_scenario', 'scenario': sc.sid,
            'title': sc.title, 'n_drones': sc.n_robots,
            'duration_s': f'{sc.duration:.3f}', 'height': args.height,
            'timescale': args.timescale, 'controller': controller,
            'ctrl_mode': traj_ctrl_mode, 'tags': '|'.join(sc.tags)}
    meta.update({f'param_{k}': v for k, v in sc.params.items()})
    meta.update({f'indi_{k}': v for k, v in indi_gains.items()})
    meta.update({f'pos_{k}': v for k, v in pos_gains.items()})

    def apply(phase, ctrl, mode_, gains=None, pgains=None):
        for c in cfs:
            c.setParam('stabilizer.controller', ctrl)
            c.setParam('indi_gains.ctrl_mode', mode_)
            for k, v in (gains or {}).items():
                c.setParam(f'indi_gains.{k}', float(v))
            for k, v in (pgains or {}).items():
                c.setParam(f'pos_gains.{k}', float(v))
        th.sleep(_CTRL_SETTLE_S)
        print(f'[formation] {phase}: controller={ctrl} ctrl_mode={mode_}')

    usd_start = None
    try:
        apply('takeoff', _RAMP_CONTROLLER, _RAMP_CTRL_MODE)
        if args.brushless:
            for c in cfs:
                c.arm(True)
            th.sleep(0.5)
        for lg in loggers:
            lg.active = True

        # Stage 1: straight up to each robot's own slot height, no lateral motion, so a
        # stack is vertically separated before anything moves sideways.
        print('[formation] stage 1: climbing to slot heights...')
        for c, s in zip(cfs, slots):
            c.takeoff(targetHeight=float(s[2]), duration=3.0)
        th.sleep(3.5)

        print('[formation] stage 2: converging to slots...')
        for c, s in zip(cfs, slots):
            c.goTo(s, 0, 3.0)
        th.sleep(3.5)

        # Per-drone trajectories under a single id, started by one broadcast: each vehicle
        # flies its own curve and they all begin on the same packet.
        print('[formation] uploading per-robot trajectories...')
        for c, t in zip(cfs, trajs):
            c.uploadTrajectory(0, 0, t)
        th.sleep(0.5)

        apply('scenario', controller, traj_ctrl_mode, indi_gains, pos_gains)

        try:
            allcfs.setParam('usd.logging', 1)
            usd_start = time.monotonic()
            print(f'[formation] uSD logging started (broadcast) at '
                  f't={usd_start - log_t0:.3f}s')
        except Exception as e:
            print(f'[formation] uSD logging not started ({e}) -- radio logs only')

        # Record the exact simulation-clock instant the trajectory starts. Verification
        # afterwards needs the window, and guessing it from the recorded states is
        # impossible for a scenario like A1 where nothing moves.
        t_start = float(th.time())
        meta['t_start_sim'] = f'{t_start:.4f}'
        sidecar = LOG_DIR / f'{sc.sid}_{stamp}.meta.json'
        sidecar.parent.mkdir(parents=True, exist_ok=True)
        with open(sidecar, 'w') as fh:
            json.dump({'scenario': sc.sid, 'params': sc.params,
                       'roles': [r.role for r in sc.robots],
                       'names': [c.prefix.lstrip('/') for c in cfs],
                       'height': args.height, 'anchor': list(map(float, anchor)),
                       't_start_sim': t_start, 'duration': sc.duration,
                       'timescale': args.timescale}, fh, indent=2)
        print(f'[formation] t_start(sim) = {t_start:.3f}s -> {sidecar.name}')

        print(f'[formation] running {sc.sid} ({sc.duration:.1f} s)...')
        allcfs.startTrajectory(0, timescale=args.timescale)
        th.sleep(sc.duration * args.timescale + 1.0)

        print('[formation] done, landing...')
        apply('landing', _RAMP_CONTROLLER, _RAMP_CTRL_MODE)
        try:
            allcfs.setParam('usd.logging', 0)
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
        if usd_start is not None:
            meta['usd_start_s'] = f'{usd_start - log_t0:.6f}'
        for lg in loggers:
            lg.save(LOG_DIR / f'{sc.sid}_{lg.name}_{stamp}.csv', meta)
        relative_report(loggers, sc, poses=latest)


if __name__ == '__main__':
    main()
