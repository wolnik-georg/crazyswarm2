"""Compile a parametric curve into the Poly4D format Crazyswarm2 already uploads.

Output is the same 33-column CSV the Rust exporter writes
(`duration, x^0..x^7, y^0..y^7, z^0..z^7, yaw^0..yaw^7`, coefficients in physical
seconds), so these trajectories go through `Trajectory.loadcsv()` ->
`uploadTrajectory()` -> `startTrajectory()` unchanged. Nothing about the flight path is
new; only where the polynomial comes from.

Method: split the curve into segments and fit each with a degree-7 polynomial matching
position, velocity, acceleration and jerk at both ends. Eight boundary conditions, eight
coefficients -- exactly determined, no least squares, and C3 continuous across knots
because adjacent segments are matched to the same derivatives. This is the same
boundary-matching approach as `to_hermite_phys7()` in `export_poly4d.rs`; using it here
keeps the two paths conceptually identical.

Accuracy is exact at the knots and interpolates between them, so error is controlled by
segment count, reported by `fit_error()`. The firmware evaluates at most
`HLC_MAX_PIECES` pieces, enforced below.
"""

from __future__ import annotations

from pathlib import Path

import numpy as np

# crtp_commander_high_level.c holds 31 pieces; exceeding it silently truncates the
# trajectory in flight, so it is a hard error here rather than a warning.
HLC_MAX_PIECES = 31

CSV_HEADER = ('duration,' +
              ','.join(f'{ax}^{i}' for ax in ('x', 'y', 'z', 'yaw') for i in range(8)) + ',')


def _derivs(curve, t: float, h: float = 1e-3) -> np.ndarray:
    """Position and first three derivatives at t, as a (4,4) array [p,v,a,j] x [x,y,z,yaw].

    Central differences. The curves are smooth and analytic, so the step size only has to
    balance truncation against round-off; 1e-3 sits comfortably for both. Near the ends
    the stencil is shifted inward rather than clamped -- clamping would silently evaluate
    a one-sided difference as if it were central and report a wrong boundary derivative,
    which is precisely the value the fit depends on.
    """
    T = curve.duration
    t = min(max(t, 2 * h), T - 2 * h) if T > 4 * h else T / 2.0
    f = lambda s: np.asarray(curve.at(min(max(s, 0.0), T)), dtype=float)  # noqa: E731
    fm2, fm1, f0, fp1, fp2 = f(t - 2 * h), f(t - h), f(t), f(t + h), f(t + 2 * h)
    v = (fp1 - fm1) / (2 * h)
    a = (fp1 - 2 * f0 + fm1) / (h * h)
    j = (fp2 - 2 * fp1 + 2 * fm1 - fm2) / (2 * h ** 3)
    return np.stack([f0, v, a, j])


def _hermite7(p0, v0, a0, j0, p1, v1, a1, j1, T):
    """Degree-7 coefficients matching p,v,a,j at t=0 and t=T. Scalar axis."""
    # Start conditions fix the first four coefficients outright.
    c = np.zeros(8)
    c[0], c[1], c[2], c[3] = p0, v0, a0 / 2.0, j0 / 6.0
    # Remaining four follow from the end conditions, after removing what c0..c3 contribute.
    T2, T3, T4, T5, T6, T7 = T**2, T**3, T**4, T**5, T**6, T**7
    known_p = c[0] + c[1] * T + c[2] * T2 + c[3] * T3
    known_v = c[1] + 2 * c[2] * T + 3 * c[3] * T2
    known_a = 2 * c[2] + 6 * c[3] * T
    known_j = 6 * c[3]
    M = np.array([
        [T4,        T5,         T6,          T7],
        [4 * T3,    5 * T4,     6 * T5,      7 * T6],
        [12 * T2,   20 * T3,    30 * T4,     42 * T5],
        [24 * T,    60 * T2,    120 * T3,    210 * T4],
    ])
    rhs = np.array([p1 - known_p, v1 - known_v, a1 - known_a, j1 - known_j])
    c[4:] = np.linalg.solve(M, rhs)
    return c


def _build(curve, knots) -> np.ndarray:
    D = [_derivs(curve, t) for t in knots]
    rows = []
    for k in range(len(knots) - 1):
        dt = knots[k + 1] - knots[k]
        row = [dt]
        for ax in range(4):
            row.extend(_hermite7(D[k][0][ax], D[k][1][ax], D[k][2][ax], D[k][3][ax],
                                 D[k + 1][0][ax], D[k + 1][1][ax], D[k + 1][2][ax],
                                 D[k + 1][3][ax], dt))
        rows.append(row)
    return np.array(rows)


def compile_curve(curve, segments: int | None = None, tol: float = 2e-3) -> np.ndarray:
    """Curve -> (n_pieces, 33) Poly4D table, using as few pieces as meet `tol`.

    Piece count is chosen adaptively rather than from a fixed rate, because the two
    scenarios that need the most time need the fewest pieces: a four-pass straight-line
    shuttle is 26 seconds of motion that a handful of polynomials describe exactly, while
    a two-lap circle is shorter and needs far more. A time-based heuristic gets this
    backwards and pushes long simple trajectories into the firmware's 31-piece ceiling.

    Any interior time the curve reports from `knots()` is forced to be a segment
    boundary, so joins between legs stay sharp instead of being smoothed across.
    """
    T = curve.duration
    if T <= 0:
        raise ValueError('curve has zero duration')

    required = sorted({0.0, T} | {float(k) for k in curve.knots() if 0.0 < k < T})

    if segments is not None:
        segments = max(2, segments)
        if segments > HLC_MAX_PIECES:
            raise ValueError(f'{segments} pieces exceeds the firmware limit of {HLC_MAX_PIECES}')
        return _build(curve, list(np.linspace(0.0, T, segments + 1)))

    # Never emit a single piece. np.loadtxt collapses a one-row CSV to a 1-D array and
    # crazyflie_py's Trajectory.loadcsv then indexes it as if it were 2-D, so a
    # one-segment trajectory -- a plain hover, which several scenarios need -- would fail
    # at upload with an opaque "invalid index to scalar variable". Two pieces cost
    # nothing and keep the file shape unambiguous.
    min_m = 2 if len(required) == 2 else 1

    best = None
    for m in (1, 2, 3, 4, 6, 8, 12, 16, 24, 32):
        if m < min_m:
            continue
        knots = []
        for a, b in zip(required[:-1], required[1:]):
            knots.extend(np.linspace(a, b, m + 1)[:-1])
        knots.append(T)
        if len(knots) - 1 > HLC_MAX_PIECES:
            break
        table = _build(curve, knots)
        best = table
        if fit_error(curve, table) <= tol:
            return table
    if best is None:
        raise ValueError('curve needs more pieces than the firmware allows')
    return best


def evaluate(table: np.ndarray, t: float) -> np.ndarray:
    """Evaluate a compiled table at time t -- the same arithmetic the firmware does."""
    acc = 0.0
    for row in table:
        dur = row[0]
        if t <= acc + dur or row is table[-1]:
            s = min(max(t - acc, 0.0), dur)
            return np.array([np.polyval(row[1 + 8 * ax: 9 + 8 * ax][::-1], s) for ax in range(4)])
        acc += dur
    return np.zeros(4)


def fit_error(curve, table: np.ndarray, n: int = 400) -> float:
    """Max position error between the curve and its compiled form, in metres."""
    ts = np.linspace(0.0, curve.duration, n)
    err = [np.linalg.norm(np.asarray(curve.at(t))[:3] - evaluate(table, t)[:3]) for t in ts]
    return float(np.max(err))


def write_csv(path: Path, table: np.ndarray) -> Path:
    """Write a Poly4D table.

    Significant digits, NOT fixed decimals. A degree-7 coefficient is divided by the piece
    duration to the seventh power, so over a 6.5 s piece c7 lands around 5e-5 while c4 is
    around 2e-2 -- four orders of magnitude apart in one row. Writing that with the
    exporter's `%.6f` leaves c7 with two significant figures, and since c7 multiplies
    t^7 ~ 5e5 the error is enormous: a measured case turned a 0.21 m/s^2 trajectory into
    241 m/s^2 and flew the vehicle into the ground. The Rust exporter never hit this
    because its pieces are under a second, where every coefficient is order 1.

    %.12g round-trips float32 comfortably, which is what the firmware stores.
    """
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, 'w') as fh:
        fh.write(CSV_HEADER + '\n')
        for row in table:
            fh.write(','.join(f'{v:.12g}' for v in row) + ',\n')
    return path


def read_csv(path: Path) -> np.ndarray:
    return np.loadtxt(path, delimiter=',', skiprows=1, usecols=range(33))


def verify_csv(path: Path, curve, tol: float = 5e-3) -> float:
    """Re-read a written trajectory and check it still matches the curve.

    Guards the whole compile-and-write path, not just the fit: the file is what the
    vehicle actually flies, so it is the file that has to be correct.
    """
    err = fit_error(curve, read_csv(path))
    if err > tol:
        raise ValueError(f'{path.name}: round-trip error {err * 1000:.1f} mm exceeds '
                         f'{tol * 1000:.0f} mm -- the written file does not match the curve')
    return err
