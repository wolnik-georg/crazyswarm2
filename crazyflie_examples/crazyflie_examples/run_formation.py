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
# 2026-09-14: was hardcoded to this dev machine's own home directory
# (/home/georg/Desktop/...), which does not exist on the actual lab flight-control PC --
# "folder/directory not found" the moment a live (non-dry-run) flight tried to write its
# metadata sidecar. Resolved lazily in main() below, from formation_flight.py's own
# LOG_DIR (itself flight.py's LOGS_DIR, the one absolute path already confirmed correct
# on the lab PC) rather than guessing a second hardcoded machine-specific path.
LOG_DIR = None

_RAMP_CONTROLLER = 6      # geometric SE(3) for takeoff/landing, as formation_flight does
_RAMP_CTRL_MODE = 0
_CTRL_SETTLE_S = 0.3

# 2026-09-14: TEMPORARY compensation for a ~0.40m height shortfall on cf_second, reproduced
# in a 4-point sweep (0.6->0.20, 0.9->0.50, 1.25->0.85, 1.5->1.09m -- a constant offset
# regardless of target, not thrust/PID saturation). Suspected cause: this vehicle's mocap
# rigid-body Z origin is miscalibrated by ~0.4m (cf231_active shows no such offset). Real fix
# is recalibrating the rigid body in the mocap software; this just adds the missing height to
# the commanded slot until then. REMOVE once fixed.
#
# 2026-09-15: hoisted from a local inside main() to module level so tools/find_flight_window.py
# can import it directly -- reconstructing a commanded trajectory to verify against needs the
# SAME compensation that was actually applied at flight time, not a second guess at its value.
#
# 2026-09-18: REMOVED -- cf_second was swapped to brushless hardware (new rigid-body mount),
# and the A8 retest with the old 0.40 compensation still active showed only a ~0.10m
# undershoot (commanded 1.65m incl. compensation, achieved 1.552m), not the original ~0.40m.
# The physical miscalibration this compensated for is gone; leaving it in now overcorrects
# by ~0.30m. Empty for now -- re-add per-drone here if a similar offset ever reappears.
Z_OFFSET_COMPENSATION = {}


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
    s.add_argument('--z-floor', type=float, default=None,
                   help='lowest altitude any vehicle may be commanded to [m]. Default 0.30 keeps '
                        'formations clear of ground effect; pass 0.10 only when ground effect is '
                        'the thing being measured (C5)')
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
    if args.z_floor is not None:
        lim.z_floor = args.z_floor
    if args.geofence:
        v = [float(x) for x in args.geofence.split(',')]
        if len(v) != 6:
            sys.exit('[formation] --geofence needs xmin,xmax,ymin,ymax,zmin,zmax')
        lim.geofence = dict(x=(v[0], v[1]), y=(v[2], v[3]), z=(v[4], v[5]))
    return lim


def apply_a7_lab_defaults(args) -> None:
    """A7 in this lab: low anchor, extreme dz, motion along y (room is 4 m x x 2 m y x 1.3 m z).

    The scenario is authored as a +x shuttle (default length 1.2 m). With x geofence ±1 m, an
    unrotated run parks on the x walls; rotate 90° uses y. Lab mocap still loses track when the
    top vehicle is high (≈1.25–1.3 m) or when the shuttle is long — shorten length and lower the
    whole formation while keeping dz_start/dz_end (relative separation) unchanged.
    """
    if (args.scenario or '').upper() != 'A7':
        return
    dz_start = float(args.dz_start if args.dz_start is not None else 1.10)
    z_lo, z_hi = safety.FLIGHT_SPACE['z']
    max_base = z_hi - dz_start
    if max_base < z_lo:
        sys.exit(f'[formation] A7: geofence z_max {z_hi} m cannot fit dz_start {dz_start} m')
    # Shorter along-track leg → stay nearer volume centre (less time at mocap-soft edges).
    if args.length is None:
        args.length = 1.0
        print('[formation] A7: --length -> 1.0 m (shorter shuttle; dz sweep unchanged)')
    # Lower absolute altitude: same inter-drone dz, top peak ≈1.20 m not ≈1.28 m.
    if args.height == 1.0:
        args.height = round(max(z_lo, max_base - 0.10), 2)
        print(f'[formation] A7: --height -> {args.height} m '
              f'(top ≈ {args.height + dz_start:.2f} m; dz_start/dz_end unchanged)')
    elif args.height + dz_start > z_hi + 1e-6:
        sys.exit(f'[formation] A7: height {args.height} + dz_start {dz_start} '
                 f'→ {args.height + dz_start:.2f} m > geofence z_max {z_hi} m')
    if args.z_floor is None:
        args.z_floor = z_lo
        print(f'[formation] A7: --z-floor -> {args.z_floor} m (low anchor; not the 0.30 m stack floor)')
    if not args.allow_extreme:
        args.allow_extreme = True
        print('[formation] A7: --allow-extreme enabled (dz_end 0.10 m is extreme by design)')
    if args.rotate is None:
        args.rotate = 90.0
        print('[formation] A7: --rotate -> 90° (translate along y — x span is only ±1 m here)')


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


def centred_anchor(sc, lim, height: float) -> np.ndarray:
    """Centre the scenario HORIZONTALLY in the volume, at the requested altitude.

    Horizontal placement is arbitrary -- the room has no preferred spot, so putting the
    scenario in the middle simply buys the most clearance. Altitude is not arbitrary: it is
    an experimental parameter. C5 flies low because ground effect is the thing it measures,
    and centring it vertically silently lifted it from 0.15 m to 0.85 m, which is a different
    experiment. So z comes from --height and only x and y are centred; if the result does not
    fit, the check refuses and says so rather than quietly moving it.
    """
    _, box = safety.sample_positions(sc, base=np.zeros(3))
    lo, hi = box.reshape(-1, 3).min(axis=0), box.reshape(-1, 3).max(axis=0)
    gf = lim.geofence
    mid_xy = np.array([(gf['x'][0] + gf['x'][1]) / 2, (gf['y'][0] + gf['y'][1]) / 2])
    xy = mid_xy - (lo[:2] + hi[:2]) / 2
    return np.array([xy[0], xy[1], height])


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

    # A parameter that does not apply is a mistake worth stopping for, not a traceback:
    # it means the flight about to run is not the flight that was asked for.
    apply_a7_lab_defaults(args)
    try:
        sc = scenarios.build(args.scenario, **scenario_params(args))
    except (ValueError, KeyError) as e:
        sys.exit(f'[formation] {str(e).strip(chr(39))}')
    lim = make_limits(args)
    spec_problems = scenarios.check_spec(sc)
    # Offline the anchor XY is unknown (it comes from drone 0's start position), so the
    # fence is checked about the origin at the requested base height. The runner
    # re-checks against the real anchor once ROS reports it.
    base = np.array([0.0, 0.0, args.height])
    if args.auto_center:
        base = centred_anchor(sc, lim, args.height)
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

    from .formation_flight import DroneLogger, load_controller_config, GEOMETRIC_POS_GAINS
    from .formation_flight import LOG_DIR as _FF_LOG_DIR

    # Same base directory formation_flight.py/flight.py already use (the one confirmed
    # correct on the lab PC), just this runner's own "experiments/logs" subfolder instead
    # of flight.py's "Controls/logs" -- see the LOG_DIR = None comment above.
    global LOG_DIR
    LOG_DIR = _FF_LOG_DIR.parent.parent / 'experiments' / 'logs'

    controller, traj_ctrl_mode, indi_gains, pos_gains, per_robot = load_controller_config()

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
        anchor = centred_anchor(sc, lim, args.height)
        print(f'[formation] --auto-center: anchor moved to {anchor.round(2)} '
              f'(was {(np.array(cfs[0].initialPosition) + [0, 0, args.height]).round(2)})')

    slots = [anchor + r.slot for r in sc.robots]

    # Confirmed on the A8 runs this compensation was written for: 1.25m target -> 0.86m
    # achieved without it, |ez|=244mm.
    for i, c in enumerate(cfs):
        name = c.prefix.lstrip('/')
        if name in Z_OFFSET_COMPENSATION:
            slots[i] = slots[i] + np.array([0.0, 0.0, Z_OFFSET_COMPENSATION[name]])

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
    # What the curves actually do, next to what was asked for. `param_speed` is the
    # request and can differ from the flight: circular paths are paced by period, so
    # before this was recorded a scenario could log speed=0.4 and fly 0.63 m/s.
    meta.update({f'realised_{k}': v for k, v in sc.realised.items()})
    meta.update({f'indi_{k}': v for k, v in indi_gains.items()})
    meta.update({f'pos_{k}': v for k, v in pos_gains.items()})

    # ── What each drone ACTUALLY flies ──────────────────────────────────────────────
    # 2026-09-15: the meta above is the shared `all:` block, and until now it was written
    # verbatim into EVERY drone's log. Any drone with its own per-robot firmware_params
    # (cf_second's permanent stock-Lee pin, cf231_active's geometric pin) therefore had its
    # log claim a controller and gain set it never ran -- e.g. every A8 since 2026-09-14
    # recorded ctrl_mode=3/full INDI while cf231_active actually flew ctrl_mode=0/geometric
    # with different pos_gains. The flights were correct; the metadata was not, which is
    # worse, because it silently mislabels the dataset for any later analysis that groups
    # by controller.
    #
    # Per-robot config is a first-class, supported thing: each drone may run its own
    # controller and its own gains, or the same as everyone else. So resolve it explicitly
    # per drone and write THAT into that drone's own log.
    def resolve(name, ctrl, mode_, pgains):
        """Resolve (controller, ctrl_mode, pos_gains) for ONE drone in ONE phase.

        Precedence, and why each step exists:
          1. the shared `all:` values passed in for this phase
          2. GEOMETRIC pos_gains if this drone ends up on OUR controller in geometric mode --
             crazyflies.yaml's pos_gains block is the INDI-tuned one (kp_xy=64/kv_xy=5, damping
             zeta=0.31), and flying geometric on it was 2026-09-09 root cause #1. This rule
             existed in formation_flight.py but was keyed on the SHARED ctrl_mode, and did not
             exist here at all -- so a drone pinned to ctrl_mode=0 silently got INDI gains
             unless its override happened to also carry pos_gains. Now decided PER DRONE.
          3. this drone's own robots.<name>.firmware_params, which always win.
        """
        overrides = per_robot.get(name, {})
        eff_ctrl = int(overrides.get('stabilizer.controller', ctrl))
        eff_mode = int(overrides.get('indi_gains.ctrl_mode', mode_))
        pos = dict(pgains) if pgains else None
        if pos is not None and eff_ctrl == _RAMP_CONTROLLER and eff_mode == 0:
            pos = dict(GEOMETRIC_POS_GAINS)
        if pos is not None:
            for key, v in overrides.items():
                if key.startswith('pos_gains.'):
                    pos[key.split('.', 1)[1]] = float(v)
        return eff_ctrl, eff_mode, pos

    def effective_for(name):
        """Resolve the full config this specific drone actually flies, for the SCENARIO phase
        -- the same precedence apply() enforces on the wire, so the log matches the flight."""
        eff_ctrl, eff_mode, pos = resolve(name, controller, traj_ctrl_mode, pos_gains)
        eff = {'controller': eff_ctrl, 'ctrl_mode': eff_mode,
               'indi': dict(indi_gains), 'pos': pos if pos is not None else dict(pos_gains)}
        for key, v in per_robot.get(name, {}).items():
            if key.startswith('indi_gains.') and key != 'indi_gains.ctrl_mode':
                eff['indi'][key.split('.', 1)[1]] = float(v)
        return eff

    def meta_for(name):
        """This drone's own log metadata: scenario info, plus ITS effective config."""
        m = dict(meta)
        eff = effective_for(name)
        m['controller'] = eff['controller']
        m['ctrl_mode'] = eff['ctrl_mode']
        m.update({f'indi_{k}': v for k, v in eff['indi'].items()})
        m.update({f'pos_{k}': v for k, v in eff['pos'].items()})
        # Make the distinction explicit rather than something to infer by diffing files.
        m['config_source'] = ('all+robot_override' if per_robot.get(name) else 'all')
        m['shared_controller'] = controller
        m['shared_ctrl_mode'] = traj_ctrl_mode
        # indi_gains/pos_gains/ctrl_mode are consumed by OUR OOT controller (6) only. A
        # drone pinned to a stock firmware controller (e.g. cf_second on stock Lee, 5)
        # ignores them entirely, so recording them unqualified would imply they shaped that
        # flight. Flag it instead of dropping them -- the values are still the ones that
        # were pushed, they just had no effect on this vehicle.
        m['gains_apply'] = 1 if eff['controller'] == 6 else 0
        return m

    effective = {c.prefix.lstrip('/'): effective_for(c.prefix.lstrip('/')) for c in cfs}
    print('\n[formation] per-drone effective config (what each vehicle actually flies):')
    for n, e in effective.items():
        src = 'own override' if per_robot.get(n) else 'shared all:'
        print(f'    {n:14s} controller={e["controller"]} ctrl_mode={e["ctrl_mode"]}  '
              f'pos_gains kp_xy={e["pos"].get("kp_xy")} kv_xy={e["pos"].get("kv_xy")} '
              f'kp_z={e["pos"].get("kp_z")} kv_z={e["pos"].get("kv_z")}   [{src}]')

    def apply(phase, ctrl, mode_, gains=None, pgains=None):
        # 2026-09-15: the shared broadcast below and the per-robot override re-push used to
        # be two SEPARATE radio round-trips -- broadcast the shared value to everyone first,
        # then correct any per-robot-pinned drone back to its own value a moment later. In
        # between, a pinned drone (e.g. cf_second on stock Lee) genuinely ran the SHARED
        # controller/gains -- our OOT/INDI controller, tuned for a different airframe's
        # mass/kt -- for however long that gap took. That is a real physical bad-torque
        # command, not a logging artifact: it's what caused the mid-flight tumbles that
        # night (both cf_second and cf231_active glitched at the exact same elapsed time in
        # every A8 flight, regardless of scenario length, because apply('scenario', ...)
        # always fires at the same point in the sequence). Fix: a per-robot override key is
        # never sent in the shared broadcast at all -- only the override loop sets it, so
        # each param is written exactly once. See docs/lab_sessions/2026-09-15.md.
        for c in cfs:
            name = c.prefix.lstrip('/')
            overrides = per_robot.get(name, {})
            # pos_gains are resolved PER DRONE (see resolve()): a drone on our controller in
            # geometric mode must not get the yaml's INDI-tuned pos_gains, whether that mode
            # came from the shared block or from its own pin.
            _, _, pg = resolve(name, ctrl, mode_, pgains)
            if 'stabilizer.controller' not in overrides:
                c.setParam('stabilizer.controller', ctrl)
            if 'indi_gains.ctrl_mode' not in overrides:
                c.setParam('indi_gains.ctrl_mode', mode_)
            for k, v in (gains or {}).items():
                key = f'indi_gains.{k}'
                if key not in overrides:
                    c.setParam(key, float(v))
            for k, v in (pg or {}).items():
                key = f'pos_gains.{k}'
                if key not in overrides:
                    c.setParam(key, float(v))
        # 2026-09-14: same fix as formation_flight.py's apply() (crazyswarm2 fdfc640,
        # 2026-09-12) -- any per-robot firmware_params override (e.g. cf_second's own
        # stock-Lee pin, or cf231_active's temporary ctrl_mode) must still be (re-)applied
        # here -- this is the only place these values are set now that the loop above skips
        # them in the shared broadcast.
        for c in cfs:
            name = c.prefix.lstrip('/')
            for key, v in per_robot.get(name, {}).items():
                is_int_param = key in ('stabilizer.controller', 'indi_gains.ctrl_mode')
                c.setParam(key, int(v) if is_int_param else float(v))
        th.sleep(_CTRL_SETTLE_S)
        # 2026-09-18: this used to print the raw (ctrl, mode_) ARGUMENTS this call was
        # invoked with -- the shared `all:` block's values, not what any per-robot-pinned
        # drone actually resolved to and had pushed. Misleadingly showed e.g. "ctrl_mode=3"
        # for the scenario phase even when every pinned drone was correctly running
        # ctrl_mode=0 underneath -- actively confusing to debug against. Print each drone's
        # real per-robot-resolved value instead.
        for c in cfs:
            name = c.prefix.lstrip('/')
            eff_ctrl, eff_mode, _ = resolve(name, ctrl, mode_, pgains)
            print(f'[formation] {phase}: {name} controller={eff_ctrl} ctrl_mode={eff_mode}')

    usd_start = None
    end_of_flight_poses = latest
    try:
        # ── usec.reset MUST happen HERE, ON THE GROUND, BEFORE TAKEOFF ──────────────
        # 2026-09-15: this broadcast was originally placed mid-flight, right before
        # usd.logging=1. That CRASHED EVERY FLIGHT, on every scenario including a pure
        # A1 hover. The high-level commander's entire time base is this same clock
        # (crtp_commander_high_level.c: `float t = usecTimestamp() / 1e6;`), and the
        # planner stores `t_begin` from it when takeoff/goTo starts. Zeroing the timer
        # mid-flight leaves t_begin holding a large value, so piecewise_eval computes
        # `t - t_begin` ~= MINUS several hundred seconds, evaluates a degree-7 polynomial
        # far outside its domain, and hands the controller an astronomically wrong
        # setpoint -- motors cut or the vehicle slams over, within one HLC tick.
        # On the ground the planner is IDLE (no t_begin in use) and takeoff samples the
        # clock AFTER the reset, so the time base stays monotonic for the whole flight.
        # uSD logs still get their shared origin: what matters is that every drone's
        # clock is zeroed together, not that logging starts at exactly t=0.
        try:
            allcfs.setParam('usec.reset', 1)
        except Exception as e:
            print(f'[formation] WARN: usec.reset broadcast failed ({e}) -- uSD timestamps may '
                  f'carry a per-drone offset')

        # 2026-09-18: MUST pass indi_gains/pos_gains here, not just (ctrl, mode_) -- without
        # them, resolve()'s GEOMETRIC_POS_GAINS substitution (used by apply('scenario', ...)
        # below) never runs for the takeoff/climb/goTo phase, so a drone that resolves to
        # geometric via a per-robot ctrl_mode override (with no separate per-robot pos_gains
        # override) flies the ENTIRE ramp on the shared block's raw pos_gains -- INDI-tuned
        # (64/5/48/7), the documented 2026-09-09 root cause #1 for geometric instability.
        # Root-caused after 3 reproducible A8 crashes today, all at the same point in stage 2
        # (goTo), independent of controller/gains -- because the actual bug was upstream of
        # all of that, in the ramp phase, which every test coincidentally shared. Confirmed
        # against the 2026-09-14 clean A8 flight, which had cf231_active's own explicit
        # pos_gains: override (since removed by the 2026-09-15 resolve()/substitution
        # refactor, which only ever wired the fix into the 'scenario' apply() call).
        # ── EKF reset + convergence gate ────────────────────────────────────────────
        # 2026-09-18: this script had NO EKF reset at all -- every flight ran on
        # whatever estimator state the PREVIOUS flight left behind. After a crash the
        # EKF can stay frozen at wrong values (bad gyro bias), and this project's own
        # documented finding is that `kalman.resetEstimation` alone does NOT clear it:
        # the complementary filter (estimator=1) must re-init attitude from raw accel
        # first, then hand back to the Kalman filter (estimator=2).
        #
        # That is exactly the failure seen all session on 2026-09-18: after the first
        # hard crash (18:14), flight after flight reported a COMPLETELY FROZEN position
        # (x/y std 0.57mm across a whole flight) while the vehicle physically climbed
        # into the ceiling -- the controller saw a constant position error and kept
        # commanding climb because its estimate never moved. Independent of controller
        # (=5 and =6 both), independent of script, and cleared only by a battery/power
        # cycle -- all consistent with a stuck EKF, not with mocap (mocap was verified
        # tracking correctly throughout).
        print('[formation] resetting EKF (complementary re-init, then Kalman)...')
        for c in cfs:
            c.setParam('stabilizer.estimator', 1)   # complementary: re-init attitude from accel
        th.sleep(1.0)
        for c in cfs:
            c.setParam('stabilizer.estimator', 2)   # back to Kalman
        th.sleep(0.2)
        for c in cfs:
            c.setParam('kalman.resetEstimation', 1)
        th.sleep(0.2)
        for c in cfs:
            c.setParam('kalman.resetEstimation', 0)
        print('[formation] waiting for EKF to converge on mocap...')
        th.sleep(3.0)

        # Gate: the drone's OWN estimate must agree with mocap before it is allowed to
        # arm. A stuck EKF is invisible until the vehicle is already in the air, and by
        # then the only outcome is a climb to the ceiling. Compare each drone's logged
        # EKF position against its mocap pose; refuse to fly on disagreement.
        bad = []
        for lg, mocap, c in zip(loggers, latest, cfs):
            name = c.prefix.lstrip('/')
            ekf = np.array([lg.latest['pos_x'], lg.latest['pos_y'], lg.latest['pos_z']])
            err = float(np.linalg.norm(ekf - mocap))
            print(f'    {name:14s} EKF {ekf.round(3)}  mocap {np.asarray(mocap).round(3)}  '
                  f'|err| {err*1000:.0f} mm')
            if err > 0.15:
                bad.append((name, err))
        if bad:
            print('\n[formation] *** ABORT: EKF does not agree with mocap ***')
            for name, err in bad:
                print(f'[formation]   {name}: |err| {err*1000:.0f} mm (limit 150 mm)')
            print('[formation] The estimator is not tracking. Flying now would command a')
            print('[formation] runaway climb. Power-cycle the drone(s), place them on their')
            print('[formation] initial_position, and relaunch. NOT taking off.')
            return

        apply('takeoff', _RAMP_CONTROLLER, _RAMP_CTRL_MODE, indi_gains, pos_gains)
        # 2026-09-14: was gated behind --brushless, same bug formation_flight.py already
        # fixed (2026-09-12) -- standard CF2.1 auto-arms by default, so an explicit arm(True)
        # is a harmless no-op there and the one CF21BL actually needs. Always do it, or a
        # brushless drone silently never takes off when --brushless wasn't passed.
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

        # NOTE: usec.reset is NOT sent here. It is broadcast once, on the ground, before
        # takeoff -- see the long comment at the top of this try block. Sending it at this
        # point (mid-flight, which is where it originally sat) destroys the high-level
        # commander's time base and crashes the vehicle within one control tick.
        run_tag = int(time.time())
        try:
            allcfs.setParam('usd.runTag', run_tag)
            th.sleep(0.05)
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
        # 2026-09-14: this used to be able to raise (bad LOG_DIR, disk full, permissions)
        # and, being inside the main try block, skip straight to `finally` -- which does
        # NOT land the drones. A metadata sidecar failing to write is not a reason to leave
        # a drone flying; isolate it the same way usd.logging's set/reset calls already are.
        try:
            sidecar = LOG_DIR / f'{sc.sid}_{stamp}.meta.json'
            sidecar.parent.mkdir(parents=True, exist_ok=True)
            with open(sidecar, 'w') as fh:
                json.dump({'scenario': sc.sid, 'params': sc.params,
                           'realised': sc.realised,
                           'roles': [r.role for r in sc.robots],
                           'names': [c.prefix.lstrip('/') for c in cfs],
                           'height': args.height, 'anchor': list(map(float, anchor)),
                           't_start_sim': t_start, 'duration': sc.duration,
                           'timescale': args.timescale,
                           'usd_run_tag': run_tag,
                           # 2026-09-15: per-drone effective config, so the sidecar records
                           # what each vehicle actually flew rather than only the shared
                           # block. Drones may run the same controller or different ones;
                           # either way this is the authoritative record for the flight.
                           'per_drone': effective}, fh, indent=2)
            print(f'[formation] t_start(sim) = {t_start:.3f}s -> {sidecar.name}')
        except Exception as e:
            print(f'[formation] WARNING: metadata sidecar not written ({e}) -- '
                  f'flight continues, landing is not skipped')

        print(f'[formation] running {sc.sid} ({sc.duration:.1f} s)...')
        allcfs.startTrajectory(0, timescale=args.timescale)
        th.sleep(sc.duration * args.timescale + 1.0)

        # Snapshot poses HERE, at the end of the commanded trajectory -- not after landing.
        # `sc.relative(i, j, sc.duration)` is the separation the scenario promises at the END
        # OF THE HOVER, and every vertical stack (A1/A2/A5/A7/B1/C4) is designed to converge
        # to near-zero separation once both robots are on the ground. Comparing a post-landing
        # pose against a pre-landing target would report a large "error" on every single
        # flight regardless of whether it actually tracked the formation correctly.
        end_of_flight_poses = [p.copy() for p in latest]

        print('[formation] done, landing...')
        apply('landing', _RAMP_CONTROLLER, _RAMP_CTRL_MODE, indi_gains, pos_gains)
        try:
            allcfs.setParam('usd.logging', 0)
        except Exception:
            pass
        # A pure vertical stack (A1/A2/A5/A7/B1/C4) has two or more robots sharing the same
        # (x, y) for the whole flight, differing only in z. Matching their descent RATE
        # (below) keeps their SEPARATION constant only until the lowest robot reaches the
        # ground and stops -- every robot still above it then continues straight down
        # through that EXACT SAME (x, y), i.e. through the parked robot's position. Rate
        # alone cannot fix this; only giving them distinct (x, y) can. So: while they are
        # still at their very different flight heights (safe -- far apart in z), spread any
        # XY-overlapping robots onto a small circle around their shared point, then begin
        # the vertical descent once everyone has their own column of air.
        LAND_XY_MARGIN = 0.30  # m -- robots closer than this in XY are treated as "stacked"
        LAND_XY_RADIUS = 0.35  # m -- how far each is moved from the shared centre
        cur = [np.array(c.get_position()) for c in cfs]
        overlap = any(
            np.linalg.norm(cur[i][:2] - cur[j][:2]) < LAND_XY_MARGIN
            for i in range(len(cfs)) for j in range(i + 1, len(cfs))
        )
        if overlap:
            print('[formation] landing XY overlap detected -- spreading robots sideways first')
            center = np.mean([p[:2] for p in cur], axis=0)
            n = len(cfs)
            for k, c in enumerate(cfs):
                ang = 2 * np.pi * k / n
                xy = center + LAND_XY_RADIUS * np.array([np.cos(ang), np.sin(ang)])
                c.goTo(np.array([xy[0], xy[1], cur[k][2]]), 0, 3.0)
            # Extra margin past the goTo's own duration: starting the vertical descent
            # while lateral velocity from this move hasn't fully damped out couples a
            # sideways disturbance into the landing, which showed up as an attitude
            # wiggle that got worse across back-to-back runs (no disarm between them
            # to reset the position-integral term via the controller's thrust<0.05 gate).
            th.sleep(4.0)

        # Per-robot, not a single broadcast: a broadcast land() gives every robot the SAME
        # duration regardless of its own height, so a higher robot descends FASTER than a
        # lower one, closing their separation LINEARLY TO ZERO by construction. Scaling
        # duration by each robot's own live height instead keeps their descent RATE equal,
        # so commanded separation is preserved all the way to the ground.
        target_h = 0.06
        descent_rate = 0.25  # m/s -- same for every robot, so relative dz stays constant
        land_durations = []
        for c in cfs:
            h = c.get_position()[2]
            dur = max(3.0, (h - target_h) / descent_rate)
            land_durations.append(dur)
            c.land(targetHeight=target_h, duration=dur)
        th.sleep(max(land_durations) + 1.0)
        for c in cfs:
            c.arm(False)
    finally:
        for lg in loggers:
            lg.active = False
        if usd_start is not None:
            meta['usd_start_s'] = f'{usd_start - log_t0:.6f}'
        for lg in loggers:
            try:
                lg.save(LOG_DIR / f'{sc.sid}_{lg.name}_{stamp}.csv', meta_for(lg.name))
            except Exception as e:
                print(f'[formation] WARN: {lg.name} log not saved ({e}) -- '
                      f'{len(lg.rows)} rows lost')

        # 2026-09-15: twice tonight a per-drone radio log came out with the right number of
        # rows but EVERY value zero -- the subscriber was alive, the drone's log topics simply
        # published nothing. Both times only one drone was affected, and both times it was
        # noticed hours later during analysis, not in the lab. A silent all-zero log is
        # indistinguishable from a real file until you open it, so say it out loud NOW, while
        # the drone is still on the bench and the flight can simply be repeated.
        for lg in loggers:
            rows = getattr(lg, 'rows', None)
            if not rows:
                continue
            try:
                # any non-zero value anywhere in the numeric payload means the link carried data
                alive = any(any(abs(float(v)) > 0.0 for v in r[1:]) for r in rows)
            except (TypeError, ValueError):
                alive = True   # unexpected row shape -- don't cry wolf
            if not alive:
                print(f'[formation] ***** WARNING: {lg.name} radio log is ALL ZERO across '
                      f'{len(rows)} rows *****')
                print(f'[formation]       The log topics published nothing for this drone. The '
                      f'uSD log (if the deck is fitted) is unaffected and still valid, but this '
                      f'drone has NO radio telemetry for this flight. Re-fly if you need it.')

        relative_report(loggers, sc, poses=end_of_flight_poses)


if __name__ == '__main__':
    main()
