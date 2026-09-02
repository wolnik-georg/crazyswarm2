"""Safety gate for formation scenarios.

Everything here is checked on the COMMANDED geometry, before anything is uploaded. That
is the only thing knowable in advance; the realised geometry is checked after the fact by
the runner's report. A scenario that fails here is refused rather than warned about,
because these are the cases that end with two vehicles in the same place.

The geofence default below is a placeholder -- see FLIGHT_SPACE.
"""

from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np

# Flight volume — TWO separate limits, deliberately.
#
# FLIGHT_SPACE is the PHYSICAL mocap-tracked volume: where the cameras can see the drones at
# all. Tape-measured by the operator 2026-09-02 (4m x by 2m y, 1.7m z) -- matches the operator's
# earlier 2026-08-23 statement and the independent 2026-07-27 one (x and z identical, y then
# quoted as +-2.1) to within rounding. No longer a placeholder.
FLIGHT_SPACE = dict(x=(-1.0, 1.0), y=(-2.0, 2.0), z=(0.0, 1.70))

# The floor we CHOOSE to fly above is a different question from where tracking works.
#
# Near the ground a rotor's own downwash reflects back and pushes the vehicle up -- ground
# effect. It is real, it is roughly the same magnitude as the inter-vehicle downwash this
# thesis measures, and it is NOT the quantity under study. Flying a formation at 0.15 m would
# contaminate every residual with a second unmodelled force that has nothing to do with the
# other drone.
#
# So the default operational floor sits clear of it. Ground effect on a Crazyflie is
# significant below roughly one rotor diameter and largely gone by two to three; 0.30 m is a
# few diameters up and comfortably clear. Consistent with the operator's field observation
# 2026-09-02 that actual flights want more like 0.3-0.4 m to stay clear of ground effect in
# this particular lab -- 0.30 is the low end of that, kept as the default with headroom to
# raise it per-scenario if a specific trajectory still shows floor contamination.
Z_FLOOR_DEFAULT = 0.30

# But ground effect is itself a residual force, and this project measures residual forces. C5
# exists precisely to characterise it in isolation, and that scenario has to go low to see
# anything. Treating it as a second disturbance -- one the same machinery can measure and, in
# principle, the same learned models can compensate -- is a legitimate extension rather than a
# safety exception, so it gets its own floor rather than --allow-extreme.
Z_FLOOR_GROUND_EFFECT = 0.10

# Below this the vehicles are close enough that a tracking error of a few centimetres
# closes the gap. The literature scenarios deliberately go lower; those are gated behind
# --allow-extreme rather than forbidden.
DZ_MIN_DEFAULT = 0.15

# Vertical separation only means anything when one vehicle is actually above the other.
# Two vehicles at the same height but a third of a metre apart are a coplanar formation,
# not a near-miss, and applying dz_min to them would reject every coplanar control case
# in the library. Beyond this horizontal offset the pair is judged on 3-D distance alone.
WASH_RADIUS = 0.20

# Hard floor once --allow-extreme is given. The extreme scenarios exist to fly closer
# than dz_min, so that flag has to relax the distance check too -- but not to zero: below
# roughly a vehicle's own span the two are touching whatever the geometry says.
D_MIN_EXTREME = 0.06


@dataclass
class Limits:
    # Operational floor, separate from the geofence. Raise to Z_FLOOR_GROUND_EFFECT only when
    # ground effect is the thing being measured.
    z_floor: float = Z_FLOOR_DEFAULT
    v_max: float = 1.20            # m/s, commanded
    a_max: float = 2.50            # m/s^2, commanded
    dz_min: float = DZ_MIN_DEFAULT
    d_min: float = 0.12            # m, minimum 3-D separation between any two vehicles
    geofence: dict = field(default_factory=lambda: dict(FLIGHT_SPACE))
    allow_extreme: bool = False


def sample_positions(scenario, n: int = 600, base=None) -> tuple[np.ndarray, np.ndarray]:
    """Commanded ABSOLUTE positions: times (n,) and positions (n, n_robots, 3).

    Slots are offsets from the formation anchor, so `base` -- the anchor itself -- has to
    be added before anything is compared against a geofence. Leaving it out makes every
    robot look like it is sitting on the floor, which turns the fence check into noise.
    """
    T = scenario.duration
    ts = np.linspace(0.0, T, n)
    base = np.zeros(3) if base is None else np.asarray(base, dtype=float)
    pos = np.empty((n, len(scenario.robots), 3))
    for i, r in enumerate(scenario.robots):
        for k, t in enumerate(ts):
            pos[k, i] = base + r.slot + np.asarray(r.curve(t))[:3]
    return ts, pos


def min_separation(scenario, n: int = 600, base=None):
    """Closest commanded approach: (distance, time, (i, j)). None for a single robot."""
    if len(scenario.robots) < 2:
        return None
    ts, pos = sample_positions(scenario, n, base)
    best = (np.inf, 0.0, (0, 0))
    for i in range(len(scenario.robots)):
        for j in range(i + 1, len(scenario.robots)):
            d = np.linalg.norm(pos[:, i] - pos[:, j], axis=1)
            k = int(np.argmin(d))
            if d[k] < best[0]:
                best = (float(d[k]), float(ts[k]), (i, j))
    return best


def check(scenario, limits: Limits, base=None) -> list[str]:
    """Return a list of violations. Empty means the scenario is cleared to fly."""
    problems = []

    for i, r in enumerate(scenario.robots):
        v, a = r.curve.peaks()
        if v > limits.v_max:
            problems.append(f'robot {i} ({r.role}) commanded speed {v:.2f} m/s '
                            f'exceeds v_max {limits.v_max:.2f}')
        if a > limits.a_max:
            problems.append(f'robot {i} ({r.role}) commanded acceleration {a:.2f} m/s^2 '
                            f'exceeds a_max {limits.a_max:.2f}')

    ts, pos = sample_positions(scenario, base=base)
    gf = dict(limits.geofence)
    # The operational floor overrides the physical one, never the reverse: the mocap may see
    # down to the ground, but we choose not to fly a formation into ground effect.
    gf['z'] = (max(gf['z'][0], limits.z_floor), gf['z'][1])
    for i in range(len(scenario.robots)):
        for ax, key in enumerate('xyz'):
            lo, hi = gf[key]
            mn, mx = float(pos[:, i, ax].min()), float(pos[:, i, ax].max())
            if mn < lo - 1e-9 or mx > hi + 1e-9:
                problems.append(f'robot {i} leaves the geofence in {key}: '
                                f'[{mn:.2f}, {mx:.2f}] outside [{lo:.2f}, {hi:.2f}]')

    d_floor = D_MIN_EXTREME if limits.allow_extreme else limits.d_min
    sep = min_separation(scenario, base=base)
    if sep is not None:
        d, t, (i, j) = sep
        if d < d_floor:
            problems.append(f'robots {i} and {j} come within {d * 100:.1f} cm at t={t:.1f}s '
                            f'(minimum {d_floor * 100:.0f} cm)')

    # Vertical separation, judged only while one vehicle is genuinely above another.
    for i in range(len(scenario.robots)):
        for j in range(i + 1, len(scenario.robots)):
            horiz = np.linalg.norm(pos[:, i, :2] - pos[:, j, :2], axis=1)
            over = horiz < WASH_RADIUS
            if not over.any():
                continue                      # never stacked; d_min already covers them
            dz = np.abs(pos[:, i, 2] - pos[:, j, 2])[over].min()
            if dz < limits.dz_min and not limits.allow_extreme:
                problems.append(
                    f'robots {i} and {j} pass within {dz * 100:.1f} cm vertically while '
                    f'overlapping, below dz_min {limits.dz_min * 100:.0f} cm -- pass '
                    f'--allow-extreme if this is intended')

    if scenario.extreme and not limits.allow_extreme:
        problems.append(f'scenario {scenario.sid} is tagged extreme -- pass --allow-extreme')

    return problems


def describe(scenario, limits: Limits, base=None) -> str:
    lines = []
    ts, pos = sample_positions(scenario, base=base)
    for i, r in enumerate(scenario.robots):
        v, a = r.curve.peaks()
        lines.append(f'    robot {i} [{r.role:>7}] slot {np.round(r.slot, 3)}  '
                     f'{type(r.curve).__name__:<11} vmax {v:4.2f} m/s  amax {a:4.2f} m/s^2')
    sep = min_separation(scenario, base=base)
    if sep is not None:
        d, t, (i, j) = sep
        lines.append(f'    closest commanded approach: {d:.3f} m between robots {i} and {j} '
                     f'at t={t:.1f}s')
    return '\n'.join(lines)
