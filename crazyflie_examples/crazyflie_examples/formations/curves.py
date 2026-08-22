"""Parametric curves used to build formation scenarios.

A curve maps time to a position OFFSET from the robot's formation slot:

    curve(t) -> np.array([x, y, z, yaw])        t in [0, duration]

Offsets, not absolute positions, because `startTrajectory(relative=True)` anchors a
trajectory to wherever the robot currently is. The robot flies to its slot first, then
the curve moves it from there. Two consequences worth stating once:

* every curve MUST start at the origin and be at rest there, and should end at rest,
  otherwise the vehicle steps or lurches when the trajectory starts or stops;
* the commanded inter-robot geometry is then
  `(slot_i - slot_j) + (curve_i(t) - curve_j(t))`, which is what the scenario
  library checks against the specification.

Frames: world frame throughout, ENU, z up, metres, seconds, yaw in radians. Yaw is
carried through the whole pipeline but every scenario here holds it at zero -- the
downwash geometry is what is under test, and a yawing vehicle would change the
interaction without being the variable of interest.
"""

from __future__ import annotations

import numpy as np


class Curve:
    """Base class. Subclasses implement `at(t)`; `duration` is in seconds."""

    duration: float = 0.0

    def at(self, t: float) -> np.ndarray:
        raise NotImplementedError

    def knots(self) -> list[float]:
        """Interior times that MUST be polynomial segment boundaries.

        A curve made of pieces (out-and-back legs, sequenced sections) is only smooth
        within a piece. Fitting a polynomial across a join would round off the very
        corner the scenario depends on, so the compiler is told where the joins are.
        """
        return []

    def __call__(self, t: float) -> np.ndarray:
        return self.at(float(np.clip(t, 0.0, self.duration)))

    # Peak speed / acceleration, by sampling. Used by the safety gate; sampling rather
    # than analysis so it works for any curve, including composed ones.
    def peaks(self, n: int = 2001) -> tuple[float, float]:
        ts = np.linspace(0.0, self.duration, n)
        p = np.array([self.at(t)[:3] for t in ts])
        dt = ts[1] - ts[0]
        v = np.gradient(p, dt, axis=0)
        a = np.gradient(v, dt, axis=0)
        return float(np.linalg.norm(v, axis=1).max()), float(np.linalg.norm(a, axis=1).max())


def smoothstep7(u: float) -> float:
    """Degree-7 smoothstep: 0->1 with zero velocity, acceleration AND jerk at both ends.

    35u^4 - 84u^5 + 70u^6 - 20u^7. The jerk-free endpoints matter here: these curves are
    compiled into degree-7 polynomials whose boundary derivatives are matched exactly, so
    a profile that is merely acceleration-continuous would hand a nonzero jerk to a
    vehicle sitting at rest.
    """
    u = min(max(u, 0.0), 1.0)
    return u * u * u * u * (35.0 + u * (-84.0 + u * (70.0 - 20.0 * u)))


class Hover(Curve):
    """Stay on the slot. The 'static' half of static-top and the whole of pure hover."""

    def __init__(self, duration: float):
        self.duration = float(duration)

    def at(self, t):
        return np.zeros(4)


class Line(Curve):
    """Rest-to-rest straight translation by `delta` over the full duration."""

    def __init__(self, delta, duration: float):
        self.delta = np.asarray(delta, dtype=float)
        self.duration = float(duration)

    def at(self, t):
        s = smoothstep7(t / self.duration) if self.duration > 0 else 0.0
        return np.array([*(self.delta * s), 0.0])


class Shuttle(Curve):
    """Out and back along `delta`, `passes` times, each leg rest-to-rest.

    The natural motion for 'one robot repeatedly traverses under another': each pass is a
    clean rest-to-rest translation, so the interaction is sampled the same way every time
    instead of drifting with a continuous back-and-forth.
    """

    def __init__(self, delta, leg_duration: float, passes: int = 2, dwell: float = 0.0):
        self.delta = np.asarray(delta, dtype=float)
        self.leg = float(leg_duration)
        self.passes = int(passes)
        self.dwell = float(dwell)
        self.duration = self.passes * (self.leg + self.dwell)

    def knots(self):
        step = self.leg + self.dwell
        return [k * step for k in range(1, self.passes)] + (
            [k * step + self.leg for k in range(self.passes)] if self.dwell > 0 else [])

    def at(self, t):
        step = self.leg + self.dwell
        k = min(int(t // step), self.passes - 1)
        u = (t - k * step) / self.leg
        s = smoothstep7(u)
        # even legs go out, odd legs come back
        base = 0.0 if k % 2 == 0 else 1.0
        sign = 1.0 if k % 2 == 0 else -1.0
        return np.array([*(self.delta * (base + sign * s)), 0.0])


class Circle(Curve):
    """Circle of `radius`, traversed `laps` times with period `period`.

    The offset is measured from the entry point on the circle, so the robot begins where
    it already is. `phase0` picks which point of the circle that is -- needed when several
    robots sit at different phases of one shared circle. `direction` -1 reverses the sense
    of rotation, which is the whole point of the counter-rotating scenario.

    Angular rate is eased in and out with the same degree-7 profile, so the curve starts
    and ends at rest without distorting the circle: only the speed along it is warped,
    never the shape. `ease` is the fraction of total time given to each ramp.
    """

    def __init__(self, radius: float, period: float, laps: float = 1.0,
                 direction: int = +1, phase0: float = 0.0, ease: float = 0.15,
                 plane: str = 'xy'):
        self.radius = float(radius)
        self.period = float(period)
        self.laps = float(laps)
        self.direction = 1 if direction >= 0 else -1
        self.phase0 = float(phase0)
        self.ease = float(np.clip(ease, 0.0, 0.45))
        self.plane = plane
        # Total time = laps at the requested period, plus the ramps, which each cover half
        # the distance a constant-rate ramp would. Solving for duration keeps the
        # commanded period honest during the constant-rate section.
        cruise = self.laps * self.period
        self.duration = cruise / (1.0 - self.ease)

    def _phase_fraction(self, t: float) -> float:
        """Fraction of the total angle swept by time t, with eased ends."""
        T, e = self.duration, self.ease
        if e <= 0:
            return t / T
        te = e * T
        # rate profile: ramp up over te, constant, ramp down over te. Normalised so the
        # integral over [0, T] is exactly 1.
        norm = T - te  # integral of the shape below, before normalisation
        if t <= te:
            # integral of smoothstep7(u) du from 0..t, u = s/te  -> te * S(t/te)
            return _smoothstep7_integral(t / te) * te / norm
        if t >= T - te:
            tail = (T - t) / te
            return (norm - _smoothstep7_integral(tail) * te) / norm
        return (t - te + _smoothstep7_integral(1.0) * te) / norm

    def at(self, t):
        ang = self.phase0 + self.direction * 2.0 * np.pi * self.laps * self._phase_fraction(t)
        a0 = self.phase0
        u = self.radius * (np.cos(ang) - np.cos(a0))
        v = self.radius * (np.sin(ang) - np.sin(a0))
        if self.plane == 'xy':
            return np.array([u, v, 0.0, 0.0])
        if self.plane == 'xz':
            return np.array([u, 0.0, v, 0.0])
        return np.array([0.0, u, v, 0.0])


def _smoothstep7_integral(u: float) -> float:
    """Integral of smoothstep7 from 0 to u, for u in [0,1]. Used for the eased rate ramp."""
    u = min(max(u, 0.0), 1.0)
    return u**5 * (7.0 + u * (-14.0 + u * (10.0 - 2.5 * u)))


class Lemniscate(Curve):
    """Figure-of-eight (Gerono lemniscate), width 2a, height a.

    x = a sin(theta), y = a sin(theta) cos(theta). Passes through the centre twice per
    lap, which is what makes it useful under a hovering robot: the lower vehicle crosses
    directly beneath the upper one repeatedly within a single trajectory.
    """

    def __init__(self, a: float, period: float, laps: float = 1.0, ease: float = 0.15):
        self.a = float(a)
        self.period = float(period)
        self.laps = float(laps)
        self.ease = float(np.clip(ease, 0.0, 0.45))
        self.duration = (self.laps * self.period) / (1.0 - self.ease)
        self._c = Circle(1.0, period, laps, +1, 0.0, ease)   # reuse the eased rate profile
        self._c.duration = self.duration

    def at(self, t):
        th = 2.0 * np.pi * self.laps * self._c._phase_fraction(t)
        x = self.a * np.sin(th)
        y = self.a * np.sin(th) * np.cos(th)
        return np.array([x, y, 0.0, 0.0])


class Sum(Curve):
    """Superpose curves, e.g. translate while changing height (the merge scenario).

    Duration is the longest; shorter curves hold their final value, so a short height
    change followed by continued translation behaves as one would expect.
    """

    def __init__(self, *curves: Curve):
        self.curves = curves
        self.duration = max(c.duration for c in curves)

    def knots(self):
        ks = []
        for c in self.curves:
            ks += c.knots()
            if c.duration < self.duration:
                ks.append(c.duration)
        return ks

    def at(self, t):
        return sum(c.at(min(t, c.duration)) for c in self.curves)


class Then(Curve):
    """Run curves back to back, each continuing from where the previous one ended."""

    def __init__(self, *curves: Curve):
        self.curves = curves
        self.duration = sum(c.duration for c in curves)

    def knots(self):
        ks, off = [], 0.0
        for c in self.curves[:-1]:
            ks += [off + k for k in c.knots()]
            off += c.duration
            ks.append(off)
        ks += [off + k for k in self.curves[-1].knots()]
        return ks

    def at(self, t):
        acc = np.zeros(4)
        for c in self.curves:
            if t <= c.duration:
                return acc + c.at(t)
            acc = acc + c.at(c.duration)
            t -= c.duration
        return acc


class Pause(Curve):
    """Hold position for a while. Useful to give a settled window before motion starts."""

    def __init__(self, duration: float):
        self.duration = float(duration)

    def at(self, t):
        return np.zeros(4)
