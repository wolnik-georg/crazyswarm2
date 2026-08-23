"""Formation scenario library.

Each scenario is a set of robots, each with a role, a slot (fixed offset from the
formation anchor) and a curve (motion relative to that slot). The commanded relative
geometry between robots i and j is therefore

    (slot_i - slot_j) + (curve_i(t) - curve_j(t))

which is what `check_spec()` verifies against the numbers in each scenario's docstring.

Conventions
-----------
World frame, ENU, z up, metres, seconds. Robot 0 is the LOWEST of a vertical stack --
the one in the wash -- matching `formation_flight.py`, so the base height is the floor of
the formation and nothing is ever commanded below it.

Separations are parameters, never constants: every scenario takes the separations its
source varies, with the literature's values as defaults.

The scenarios come from the interaction-force / downwash literature (Neural-Swarm and
Neural-Swarm2, ProxFly, and the tight-formation MPC work); the intent is to reproduce the
geometries those papers use to excite inter-vehicle aerodynamic interaction, not to
reproduce their controllers.

Run `python -m crazyflie_examples.formations.scenarios --self-test` to check every
scenario against its specification without ROS or a simulator.
"""

from __future__ import annotations

import argparse
from dataclasses import dataclass, field
import inspect

import numpy as np

from . import curves as C

# ── Data model ──────────────────────────────────────────────────────────────


@dataclass
class RobotPlan:
    role: str                 # bottom | center | top | leader | follower | solo
    slot: np.ndarray          # (3,) offset from the formation anchor
    curve: C.Curve            # motion relative to the slot; starts and ends at rest
    label: str = ''


@dataclass
class Scenario:
    sid: str
    title: str
    robots: list[RobotPlan]
    tags: list[str] = field(default_factory=list)
    notes: str = ''
    extreme: bool = False
    params: dict = field(default_factory=dict)

    @property
    def n_robots(self):
        return len(self.robots)

    @property
    def duration(self):
        return max(r.curve.duration for r in self.robots)

    def relative(self, i: int, j: int, t: float) -> np.ndarray:
        a, b = self.robots[i], self.robots[j]
        return (a.slot + np.asarray(a.curve(t))[:3]) - (b.slot + np.asarray(b.curve(t))[:3])


def _v(x, y, z):
    return np.array([float(x), float(y), float(z)])


# ── Shared horizontal paths ─────────────────────────────────────────────────

def _path(kind: str, *, speed: float, length: float, radius: float, period: float,
          laps: float, hold: float):
    """The horizontal path a stack follows together. Shared by A2, A4, B1, B2."""
    if kind == 'hover':
        return C.Hover(hold)
    if kind == 'line':
        return C.Line([length, 0.0, 0.0], _line_time(length, speed))
    if kind == 'shuttle':
        return C.Shuttle([length, 0.0, 0.0], _line_time(length, speed), passes=2)
    if kind == 'circle':
        return C.Circle(radius, period, laps=laps)
    if kind == 'lemniscate':
        return C.Lemniscate(radius, period, laps=laps)
    raise ValueError(f'unknown path {kind!r} (hover, line, shuttle, circle, lemniscate)')


def _line_time(length: float, speed: float) -> float:
    """Duration for a rest-to-rest translation whose PEAK speed is `speed`.

    The degree-7 smoothstep peaks at 35/16 of the mean, so a naive length/speed would
    overshoot the requested speed by more than a factor of two -- which matters, because
    the specification quotes traverse speeds and the safety gate checks against them.
    """
    return abs(length) * (35.0 / 16.0) / max(speed, 1e-6)


# ── Priority A -- two robots ────────────────────────────────────────────────

def A1(dz: float = 0.50, hold: float = 12.0, **_):
    """A1 vertical stack hover. dz in {0.25, 0.50, 0.75}, dx=dy=0, hover >= 10 s.

    The baseline: no motion at all, so any relative displacement is the interaction
    itself rather than a tracking error. This is the scenario the residual measurement is
    calibrated on.
    """
    return Scenario(
        'A1', f'Vertical stack hover, dz={dz:.2f} m',
        [RobotPlan('bottom', _v(0, 0, 0), C.Hover(hold)),
         RobotPlan('top', _v(0, 0, dz), C.Hover(hold))],
        tags=['hover', 'data_collection'],
        params=dict(dz=dz, hold=hold),
        notes='Both stationary; the lower vehicle sits in the upper one\'s wash.')


def A2(dz: float = 0.30, path: str = 'circle', speed: float = 0.4, length: float = 1.2,
       radius: float = 0.75, period: float = 7.5, laps: float = 2.0, hold: float = 12.0, **_):
    """A2 vertical stack tracking. dz in {0.20,0.30,0.40,0.50}; both follow the SAME path.

    Line at 0.3-0.5 m/s, or a circle of diameter 1.5 m and period 7.5 s. Because both
    vehicles fly identical curves from vertically stacked slots, the commanded separation
    is constant -- the wash is held on the lower vehicle for the whole flight while both
    are moving, which separates interaction from the trivial hover case.
    """
    p = _path(path, speed=speed, length=length, radius=radius, period=period,
              laps=laps, hold=hold)
    return Scenario(
        'A2', f'Vertical stack tracking {path}, dz={dz:.2f} m',
        [RobotPlan('bottom', _v(0, 0, 0), p),
         RobotPlan('top', _v(0, 0, dz), p)],
        tags=['tracking', 'data_collection'],
        params=dict(dz=dz, path=path, speed=speed, radius=radius, period=period, laps=laps),
        notes='Identical curves -> commanded separation is constant throughout.')


def A3(dz: float = 0.30, speed: float = 0.4, length: float = 1.2, passes: int = 4, **_):
    """A3 static-top. Top hovers; bottom translates underneath at 0.3-0.5 m/s.

    dz in {0.20, 0.30, 0.40}. The cleanest excitation available: the upper vehicle is a
    fixed wash source and the lower one sweeps through it, so the residual rises and falls
    within a single flight and the transition is captured, not just the steady state.
    Multiple passes give repeats without relanding.
    """
    leg = _line_time(length, speed)
    bottom = C.Shuttle([length, 0.0, 0.0], leg, passes=passes)
    return Scenario(
        'A3', f'Static-top, dz={dz:.2f} m, {passes} passes at {speed:.2f} m/s',
        # The lower vehicle starts half a length back so it passes THROUGH the wash
        # rather than starting in it -- the entry transient is the interesting part.
        [RobotPlan('bottom', _v(-length / 2, 0, 0), bottom),
         RobotPlan('top', _v(0, 0, dz), C.Hover(bottom.duration))],
        tags=['tracking', 'data_collection'],
        params=dict(dz=dz, speed=speed, length=length, passes=passes),
        notes='Upper vehicle fixed; lower sweeps through the wash and out the far side.')


def A4(dz: float = 0.60, offset: float = 0.10, axis: str = 'y', motion: str = 'lemniscate',
       speed: float = 0.4, length: float = 1.2, radius: float = 0.45, period: float = 8.0,
       laps: float = 2.0, **_):
    """A4 offset stack. Top hovers or moves slowly; bottom offset laterally and manoeuvring.

    dz in {0.50, 0.60, 0.80}, lateral offset in {0.10, 0.20} m. The bottom vehicle
    translates or flies a figure-of-eight, so it repeatedly enters and leaves the wash
    edge -- the partial-overlap regime, where the interaction is asymmetric and hardest
    to model.
    """
    lat = _v(offset if axis == 'x' else 0.0, offset if axis == 'y' else 0.0, 0.0)
    if motion == 'lemniscate':
        bottom = C.Lemniscate(radius, period, laps=laps)
    elif motion == 'line':
        bottom = C.Shuttle([length, 0.0, 0.0], _line_time(length, speed), passes=2)
    else:
        raise ValueError("A4 motion must be 'lemniscate' or 'line'")
    return Scenario(
        'A4', f'Offset stack, dz={dz:.2f} m, {axis}-offset {offset:.2f} m, {motion}',
        [RobotPlan('bottom', lat, bottom),
         RobotPlan('top', _v(0, 0, dz), C.Hover(bottom.duration))],
        tags=['tracking', 'data_collection'],
        params=dict(dz=dz, offset=offset, axis=axis, motion=motion, radius=radius),
        notes='Partial wash overlap: the lower vehicle crosses the wash edge repeatedly.')


def A5(dz: float = 0.50, radius: float = 0.75, period: float = 7.5, laps: float = 2.0, **_):
    """A5 reverse-circle stack. Both on a 1.5 m circle, period 7.5 s, opposite directions.

    dz = 0.50 m. Because the senses of rotation oppose, the horizontal offset sweeps from
    zero to a full diameter and back twice per lap: one scenario covers the whole range
    from perfectly stacked to fully separated, at constant speed and constant height
    difference. Alignment happens at the start point and at the antipode.
    """
    return Scenario(
        'A5', f'Reverse-circle stack, dz={dz:.2f} m, d={2 * radius:.2f} m, T={period:.1f} s',
        [RobotPlan('bottom', _v(0, 0, 0), C.Circle(radius, period, laps, direction=+1)),
         RobotPlan('top', _v(0, 0, dz), C.Circle(radius, period, laps, direction=-1))],
        tags=['tracking', 'data_collection'],
        params=dict(dz=dz, radius=radius, period=period, laps=laps),
        notes='Counter-rotating: horizontal offset sweeps 0 -> 2r -> 0 twice per lap.')


def A6(dz: float = 0.10, speed: float = 0.20, length: float = 1.0, passes: int = 2, **_):
    """A6 extreme stack. dz in {0.08, 0.10} m -- straight-line only, low speed. GATED.

    This is where the interaction is strongest and where a tracking error of a few
    centimetres is a collision, so it is refused unless --allow-extreme is passed, and it
    is deliberately restricted to the slowest, simplest motion.
    """
    leg = _line_time(length, speed)
    return Scenario(
        'A6', f'EXTREME stack, dz={dz:.2f} m at {speed:.2f} m/s',
        [RobotPlan('bottom', _v(-length / 2, 0, 0),
                   C.Shuttle([length, 0.0, 0.0], leg, passes=passes)),
         RobotPlan('top', _v(0, 0, dz), C.Hover(passes * leg))],
        tags=['extreme', 'data_collection'],
        extreme=True,
        params=dict(dz=dz, speed=speed, length=length, passes=passes),
        notes='Strongest interaction; smallest error budget. Fly last, and only in sim first.')


def A7(dz_start: float = 1.10, dz_end: float = 0.10, speed: float = 0.30,
       length: float = 1.2, settle: float = 3.0, **_):
    """A7 dynamic merge. dz shrinks 1.1 -> 0.1 m while translating, lateral offset ~0.

    A continuous sweep through the entire separation range in one flight, which is worth
    far more for fitting an interaction model than the same range sampled as discrete
    hovers: the model sees the gradient, not just the levels.

    Gated as extreme, because it ends at 0.1 m by construction.
    """
    move = _line_time(length, speed)
    # The upper vehicle descends onto the lower one while both translate together, so the
    # lateral alignment holds throughout and only the gap changes.
    top = C.Sum(C.Line([length, 0.0, 0.0], move),
                C.Line([0.0, 0.0, dz_end - dz_start], move))
    bottom = C.Line([length, 0.0, 0.0], move)
    return Scenario(
        'A7', f'Dynamic merge, dz {dz_start:.2f} -> {dz_end:.2f} m while translating',
        [RobotPlan('bottom', _v(0, 0, 0), C.Then(C.Pause(settle), bottom)),
         RobotPlan('top', _v(0, 0, dz_start), C.Then(C.Pause(settle), top))],
        tags=['tracking', 'extreme', 'data_collection'],
        extreme=(dz_end < 0.15),
        params=dict(dz_start=dz_start, dz_end=dz_end, speed=speed, length=length),
        notes='Continuous separation sweep -- the gradient, not just the levels.')


def A8(dz: float = 0.25, span: float = 1.0, duration: float = 6.0, settle: float = 2.0, **_):
    """A8 vertical swap, 2 robots. Fixed heights, horizontal side exchange.

    At the crossing the horizontal offset is zero and the vertical separation is dz, in
    {0.20, 0.25} m -- the Neural-Swarm geometry. Unlike the static cases the wash arrives
    as a transient with a known arrival time, which is the hardest thing for a
    feed-forward interaction model to get right.
    """
    half = span / 2.0
    return Scenario(
        'A8', f'Vertical swap (2 robots), dz={dz:.2f} m at crossing, span={span:.2f} m',
        [RobotPlan('bottom', _v(0, -half, 0),
                   C.Then(C.Pause(settle), C.Line([0.0, span, 0.0], duration))),
         RobotPlan('top', _v(0, +half, dz),
                   C.Then(C.Pause(settle), C.Line([0.0, -span, 0.0], duration)))],
        tags=['swap', 'data_collection'],
        params=dict(dz=dz, span=span, duration=duration),
        notes='Vehicles exchange sides; they are exactly stacked at the midpoint.')


# ── Priority B -- three robots ──────────────────────────────────────────────

def B1(dz1: float = 0.20, dz2: float = 0.30, path: str = 'circle', speed: float = 0.4,
       length: float = 1.2, radius: float = 0.75, period: float = 7.5, laps: float = 2.0,
       hold: float = 12.0, **_):
    """B1 I-stack. Three robots in pure vertical alignment, holding formation on a path.

    dz1 = top-to-centre gap (0.20 m), dz2 = centre-to-bottom gap in {0.20, 0.30, 0.40}.
    The bottom vehicle now sits in the wash of TWO others, which is the case that
    separates a model that superposes pairwise interactions from one that does not.
    """
    p = _path(path, speed=speed, length=length, radius=radius, period=period,
              laps=laps, hold=hold)
    return Scenario(
        'B1', f'I-stack (3), dz2={dz2:.2f} m, dz1={dz1:.2f} m, {path}',
        [RobotPlan('bottom', _v(0, 0, 0), p),
         RobotPlan('center', _v(0, 0, dz2), p),
         RobotPlan('top', _v(0, 0, dz2 + dz1), p)],
        tags=['tracking', 'data_collection'],
        params=dict(dz1=dz1, dz2=dz2, path=path, radius=radius, period=period),
        notes='Bottom vehicle is in the combined wash of two -- tests superposition.')


def B2(dz1: float = 0.20, dz2: float = 0.30, r: float = 0.10, path: str = 'circle',
       speed: float = 0.4, length: float = 1.2, radius: float = 0.75, period: float = 7.5,
       laps: float = 2.0, hold: float = 12.0, **_):
    """B2 V-stack. Top and centre laterally offset by r; bottom centred beneath the pair.

    dz1 = 0.20 m between the offset pair, dz2 in {0.30, 0.40} below. With the two upper
    vehicles displaced by r = 0.10 m, the bottom vehicle sits under the overlap of two
    partially offset wakes rather than under one column -- the asymmetric-superposition
    case that B1 cannot produce.
    """
    p = _path(path, speed=speed, length=length, radius=radius, period=period,
              laps=laps, hold=hold)
    return Scenario(
        'B2', f'V-stack (3), r={r:.2f} m, dz2={dz2:.2f} m, dz1={dz1:.2f} m, {path}',
        [RobotPlan('bottom', _v(0, 0, 0), p),
         RobotPlan('center', _v(-r / 2, 0, dz2), p),
         RobotPlan('top', _v(+r / 2, 0, dz2 + dz1), p)],
        tags=['tracking', 'data_collection'],
        params=dict(dz1=dz1, dz2=dz2, r=r, path=path),
        notes='Bottom is centred under a laterally offset pair -- asymmetric overlap.')


def B3(dz: float = 0.22, span: float = 0.55, duration: float = 8.0, settle: float = 2.0, **_):
    """B3 vertical swap, 3 robots. Fixed heights, cyclic horizontal exchange along a line.

    Neighbour gaps are dz, in the specified 0.20-0.25 m. The three vehicles sit in a row
    and shift cyclically: the far one traverses the whole row backwards while the other
    two step forward one place. It therefore passes beneath each of them in turn, giving
    TWO crossings at two different height differences (dz and 2*dz) at two different
    times, within one flight.

    A note on what this is not: the obvious reading -- three vehicles on a circle rotating
    together -- produces no crossings at all. A rigid rotation preserves every mutual
    distance, so the vehicles would hold a constant 0.95 m separation for the whole
    manoeuvre and the scenario would exercise nothing. The exchange has to be non-rigid to
    produce an encounter.
    """
    move = C.Line([0.0, span, 0.0], duration)
    back = C.Line([0.0, -2.0 * span, 0.0], duration)
    roles = ['bottom', 'center', 'top']
    slots = [_v(0, -span, 0), _v(0, 0, dz), _v(0, +span, 2 * dz)]
    curves = [move, move, back]      # 0 and 1 step forward, 2 traverses the row
    return Scenario(
        'B3', f'Vertical swap (3), neighbour dz={dz:.2f} m, span={span:.2f} m',
        [RobotPlan(roles[k], slots[k], C.Then(C.Pause(settle), curves[k])) for k in range(3)],
        tags=['swap', 'data_collection'],
        params=dict(dz=dz, span=span, duration=duration),
        notes='Cyclic shift along a row: the top vehicle passes over the other two in turn.')


# ── Priority C -- coverage extras ───────────────────────────────────────────

def C1(sep: float = 0.30, n: int = 3, hold: float = 12.0, **_):
    """C1 side-by-side coplanar. All at one height, laterally packed.

    The control case for the vertical scenarios: propellers at the same altitude interact
    far more weakly, so a residual seen here and not in A1 is not downwash.
    """
    return Scenario(
        'C1', f'Side-by-side coplanar ({n}), {sep:.2f} m apart',
        [RobotPlan('solo', _v(0, (k - (n - 1) / 2) * sep, 0), C.Hover(hold)) for k in range(n)],
        tags=['hover', 'data_collection'],
        params=dict(sep=sep, n=n),
        notes='Coplanar control case -- weak interaction expected.')


def C2(gap: float = 0.50, n: int = 3, speed: float = 0.4, length: float = 1.2, **_):
    """C2 leader-follower line. Robots in a row along the direction of travel."""
    p = C.Shuttle([length, 0.0, 0.0], _line_time(length, speed), passes=2)
    roles = ['leader'] + ['follower'] * (n - 1)
    return Scenario(
        'C2', f'Leader-follower line ({n}), {gap:.2f} m gap at {speed:.2f} m/s',
        [RobotPlan(roles[k], _v(-k * gap, 0, 0), p) for k in range(n)],
        tags=['tracking'],
        params=dict(gap=gap, n=n, speed=speed),
        notes='Followers fly through the leader\'s wake rather than under it.')


def C3(side: float = 0.50, hold: float = 12.0, **_):
    """C3 equilateral triangle, coplanar."""
    R = side / np.sqrt(3.0)
    return Scenario(
        'C3', f'Equilateral triangle, side {side:.2f} m',
        [RobotPlan('solo', _v(R * np.cos(2 * np.pi * k / 3), R * np.sin(2 * np.pi * k / 3), 0),
                   C.Hover(hold)) for k in range(3)],
        tags=['hover'],
        params=dict(side=side),
        notes='Coplanar three-body case.')


def C4(lateral: float = 1.0, dz_start: float = 0.50, dz_end: float = 0.10,
       duration: float = 8.0, settle: float = 2.0, **_):
    """C4 docking approach, ProxFly-style. Approach from 1 m lateral while closing dz.

    The lower vehicle closes both the lateral gap and the height gap at once, ending
    directly beneath the hovering upper one. Gated as extreme when it ends below 0.15 m.
    """
    top = C.Hover(settle + duration)
    bottom = C.Then(C.Pause(settle),
                    C.Sum(C.Line([-lateral, 0.0, 0.0], duration),
                          C.Line([0.0, 0.0, dz_start - dz_end], duration)))
    return Scenario(
        'C4', f'Docking approach, lateral {lateral:.2f} m, dz {dz_start:.2f} -> {dz_end:.2f} m',
        [RobotPlan('bottom', _v(lateral, 0, 0), bottom),
         RobotPlan('top', _v(0, 0, dz_start), top)],
        tags=['tracking', 'extreme', 'data_collection'],
        extreme=(dz_end < 0.15),
        params=dict(lateral=lateral, dz_start=dz_start, dz_end=dz_end),
        notes='Simultaneous lateral and vertical closure onto a hovering vehicle.')


def C5(z: float = 0.15, speed: float = 0.25, length: float = 1.2, passes: int = 2, **_):
    """C5 near-ground slow pass, single robot. Ground effect, not downwash.

    Included because ground effect contaminates any low-altitude formation measurement,
    and the only way to separate the two is to measure it alone. The base height is
    overridden to `z` by the runner for this scenario.
    """
    leg = _line_time(length, speed)
    return Scenario(
        'C5', f'Near-ground pass at z={z:.2f} m, {speed:.2f} m/s',
        [RobotPlan('solo', _v(-length / 2, 0, 0),
                   C.Shuttle([length, 0.0, 0.0], leg, passes=passes))],
        tags=['data_collection'],
        params=dict(z=z, speed=speed, length=length, passes=passes),
        notes='Single robot. Characterises ground effect so it can be told from downwash.')


# ── Registry ────────────────────────────────────────────────────────────────

BUILDERS = {
    'A1': A1, 'A2': A2, 'A3': A3, 'A4': A4, 'A5': A5, 'A6': A6, 'A7': A7, 'A8': A8,
    'B1': B1, 'B2': B2, 'B3': B3,
    'C1': C1, 'C2': C2, 'C3': C3, 'C4': C4, 'C5': C5,
}

# n_robots per scenario, so the runner can refuse a mismatched roster before flying.
N_ROBOTS = {'A1': 2, 'A2': 2, 'A3': 2, 'A4': 2, 'A5': 2, 'A6': 2, 'A7': 2, 'A8': 2,
            'B1': 3, 'B2': 3, 'B3': 3,
            'C1': 3, 'C2': 3, 'C3': 3, 'C4': 2, 'C5': 1}


def build(sid: str, **params) -> Scenario:
    sid = sid.upper()
    if sid not in BUILDERS:
        raise KeyError(f'unknown scenario {sid!r}; known: {", ".join(sorted(BUILDERS))}')
    params = {k: v for k, v in params.items() if v is not None}
    sc = BUILDERS[sid](**params)

    # Record the FULL effective parameter set, defaults included, rather than trusting
    # each builder to list what it used. A scenario has to be exactly reconstructable
    # from what was recorded, because that is what post-flight verification rebuilds to
    # compare against. Several builders quietly omitted a parameter -- A4 left out
    # `laps`, so a one-lap flight was scored against a two-lap reference and reported
    # 208 mm of horizontal error that did not exist. Deriving this from the signature
    # means a new parameter cannot be forgotten.
    sig = inspect.signature(BUILDERS[sid])
    sc.params = {name: params.get(name, prm.default)
                 for name, prm in sig.parameters.items()
                 if prm.kind is not prm.VAR_KEYWORD and prm.default is not prm.empty}
    return sc


# ── Specification check ─────────────────────────────────────────────────────

def check_spec(scenario: Scenario, tol: float = 2e-3) -> list[str]:
    """Verify the commanded geometry against what each scenario claims.

    This is the check that the library does what its documentation says -- separate from
    safety.check(), which asks whether it is safe. Runs without ROS or a simulator.
    """
    bad = []
    p = scenario.params
    T = scenario.duration
    ts = np.linspace(0.0, T, 400)

    def rel(i, j):
        return np.array([scenario.relative(i, j, t) for t in ts])

    sid = scenario.sid
    if sid in ('A1', 'A2'):
        d = rel(1, 0)
        if abs(d[:, 2].mean() - p['dz']) > tol or np.ptp(d[:, 2]) > tol:
            bad.append(f'{sid}: dz should be constant at {p["dz"]}, got '
                       f'{d[:, 2].min():.4f}..{d[:, 2].max():.4f}')
        if np.abs(d[:, :2]).max() > tol:
            bad.append(f'{sid}: horizontal offset should stay 0, got '
                       f'{np.abs(d[:, :2]).max():.4f}')
    elif sid in ('A3', 'A6'):
        d = rel(1, 0)
        if abs(d[:, 2].mean() - p['dz']) > tol:
            bad.append(f'{sid}: dz should be {p["dz"]}, got {d[:, 2].mean():.4f}')
        if np.abs(d[:, 1]).max() > tol:
            bad.append(f'{sid}: should stay in the x-z plane, got dy '
                       f'{np.abs(d[:, 1]).max():.4f}')
        # Must actually pass under the upper vehicle. Tested as a sign change, not as a
        # sample near zero: the crossing is instantaneous and a finite grid will always
        # straddle it rather than land on it.
        if not (d[:, 0].min() < 0 < d[:, 0].max()):
            bad.append(f'{sid}: lower vehicle never crosses under the upper one '
                       f'(dx range {d[:, 0].min():.3f}..{d[:, 0].max():.3f})')
    elif sid == 'A4':
        d = rel(1, 0)
        if abs(d[:, 2].mean() - p['dz']) > tol:
            bad.append(f'A4: dz should be {p["dz"]}, got {d[:, 2].mean():.4f}')
        ax = 0 if p['axis'] == 'x' else 1
        if abs(abs(d[0, ax]) - p['offset']) > tol:
            bad.append(f'A4: initial lateral offset should be {p["offset"]}, '
                       f'got {abs(d[0, ax]):.4f}')
    elif sid == 'A5':
        d = rel(1, 0)
        if abs(d[:, 2].mean() - p['dz']) > tol or np.ptp(d[:, 2]) > tol:
            bad.append(f'A5: dz should be constant at {p["dz"]}')
        h = np.linalg.norm(d[:, :2], axis=1)
        want = 2 * p['radius']
        if abs(h.max() - want) > 0.05:
            bad.append(f'A5: counter-rotation should reach a full diameter {want:.2f} m, '
                       f'got {h.max():.3f}')
        if h.min() > 0.02:
            bad.append(f'A5: vehicles should align at least once, min offset {h.min():.3f}')
    elif sid == 'A7':
        d = rel(1, 0)
        if abs(d[0, 2] - p['dz_start']) > tol or abs(d[-1, 2] - p['dz_end']) > tol:
            bad.append(f'A7: dz should sweep {p["dz_start"]} -> {p["dz_end"]}, got '
                       f'{d[0, 2]:.3f} -> {d[-1, 2]:.3f}')
        if np.abs(d[:, :2]).max() > tol:
            bad.append(f'A7: lateral alignment should hold, got {np.abs(d[:, :2]).max():.4f}')
    elif sid == 'A8':
        d = rel(1, 0)
        if abs(d[:, 2].mean() - p['dz']) > tol or np.ptp(d[:, 2]) > tol:
            bad.append(f'A8: dz should stay {p["dz"]} throughout')
        if not (d[:, 1].min() < 0 < d[:, 1].max()):
            bad.append(f'A8: vehicles should exchange sides (dy range '
                       f'{d[:, 1].min():.3f}..{d[:, 1].max():.3f})')
        if abs(d[0, 1] + d[-1, 1]) > tol:
            bad.append('A8: start and end lateral offsets should be exact opposites')
    elif sid in ('B1', 'B2'):
        for (i, j, want, name) in ((1, 0, p['dz2'], 'dz2'), (2, 1, p['dz1'], 'dz1')):
            d = rel(i, j)
            if abs(d[:, 2].mean() - want) > tol or np.ptp(d[:, 2]) > tol:
                bad.append(f'{sid}: {name} should be constant at {want}, got '
                           f'{d[:, 2].min():.4f}..{d[:, 2].max():.4f}')
        if sid == 'B2':
            d = rel(2, 1)
            if abs(abs(d[:, 0]).mean() - p['r']) > tol:
                bad.append(f'B2: top/centre lateral offset should be {p["r"]}, '
                           f'got {abs(d[:, 0]).mean():.4f}')
    elif sid == 'B3':
        for i, j in ((1, 0), (2, 1)):
            d = rel(i, j)
            if abs(d[:, 2].mean() - p['dz']) > tol or np.ptp(d[:, 2]) > tol:
                bad.append(f'B3: neighbour dz should stay {p["dz"]}')
        if not 0.20 <= p['dz'] <= 0.25:
            bad.append(f'B3: neighbour dz {p["dz"]} outside the specified 0.20-0.25 m')
        # The whole point is the crossings: the top vehicle must pass BOTH others.
        for j in (0, 1):
            d = rel(2, j)
            if not (d[:, 1].min() < 0 < d[:, 1].max()):
                bad.append(f'B3: robot 2 never crosses robot {j} '
                           f'(dy range {d[:, 1].min():.3f}..{d[:, 1].max():.3f})')
    elif sid == 'C4':
        d = rel(1, 0)
        if abs(d[0, 2] - p['dz_start']) > tol or abs(d[-1, 2] - p['dz_end']) > tol:
            bad.append(f'C4: dz should close {p["dz_start"]} -> {p["dz_end"]}')
        if abs(abs(d[0, 0]) - p['lateral']) > tol:
            bad.append(f'C4: should start {p["lateral"]} m lateral')
        if np.abs(d[-1, :2]).max() > tol:
            bad.append('C4: should end directly beneath the upper vehicle')

    # Universal invariants -- every curve must be rest-to-rest at its own origin,
    # otherwise the vehicle steps when the trajectory starts.
    for k, r in enumerate(scenario.robots):
        if np.abs(np.asarray(r.curve(0.0))[:3]).max() > 1e-9:
            bad.append(f'{sid}: robot {k} curve does not start at its slot')
    return bad


def _self_test() -> int:
    """Check every scenario against its spec, across the separations each one varies."""
    import pathlib

    from . import poly4d, safety
    cases = [
        *[('A1', dict(dz=v)) for v in (0.25, 0.50, 0.75)],
        *[('A2', dict(dz=v, path=p)) for v in (0.20, 0.30, 0.40, 0.50)
          for p in ('line', 'circle')],
        *[('A3', dict(dz=v)) for v in (0.20, 0.30, 0.40)],
        *[('A4', dict(dz=z, offset=o)) for z in (0.50, 0.60, 0.80) for o in (0.10, 0.20)],
        ('A5', dict(dz=0.50)),
        *[('A6', dict(dz=v)) for v in (0.08, 0.10)],
        ('A7', {}), ('A8', dict(dz=0.20)), ('A8', dict(dz=0.25)),
        *[('B1', dict(dz2=v)) for v in (0.20, 0.30, 0.40)],
        *[('B2', dict(dz2=v, r=0.10)) for v in (0.30, 0.40)],
        ('B3', dict(dz=0.20)), ('B3', dict(dz=0.25)),
        ('C1', {}), ('C2', {}), ('C3', {}), ('C4', {}), ('C5', {}),
    ]
    lim = safety.Limits(allow_extreme=True)
    fails = 0
    print(f'{"scenario":<10}{"n":>2} {"T[s]":>6} {"pieces":>7} {"fit[mm]":>8} '
          f'{"min sep[m]":>11}  spec')
    for sid, kw in cases:
        sc = build(sid, **kw)
        spec = check_spec(sc)
        pieces, fit = 0, 0.0
        import tempfile
        tmp = pathlib.Path(tempfile.mkdtemp())
        for k, r in enumerate(sc.robots):
            tab = poly4d.compile_curve(r.curve)
            pieces = max(pieces, len(tab))
            # Round-trip through the CSV, because that is the artefact the vehicle flies
            # and coefficient precision has already bitten once.
            path = poly4d.write_csv(tmp / f'{k}.csv', tab)
            fit = max(fit, poly4d.fit_error(r.curve, poly4d.read_csv(path)))
        sep = safety.min_separation(sc)
        sept = f'{sep[0]:.3f}' if sep else '   -'
        tag = ', '.join(f'{k}={v}' for k, v in kw.items())
        ok = 'OK' if not spec else 'FAIL: ' + '; '.join(spec)
        if spec:
            fails += 1
        print(f'{sid + " " + tag:<10}'[:10] + f'{sc.n_robots:>2} {sc.duration:6.1f} '
              f'{pieces:>7} {fit * 1000:8.2f} {sept:>11}  {ok}')
    print(f'\n{len(cases)} cases, {fails} spec failures')
    return 1 if fails else 0


if __name__ == '__main__':
    ap = argparse.ArgumentParser(description='Formation scenario library')
    ap.add_argument('--self-test', action='store_true')
    ap.add_argument('--list', action='store_true')
    a = ap.parse_args()
    if a.list:
        for k in sorted(BUILDERS):
            sc = build(k)
            print(f'  {k}  n={N_ROBOTS[k]}  {sc.title}')
            print(f'       tags={",".join(sc.tags)}  {sc.notes}')
    else:
        raise SystemExit(_self_test())
