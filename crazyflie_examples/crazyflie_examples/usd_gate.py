"""uSD start gate: prove that every deck is actually writing before the flight depends on it.

Why (2026-11-18): S0 round 0 produced no thesis* file on either card. The flight scripts sent the
broadcast ``usd.runTag`` / ``usd.logging = 1`` and recorded "uSD started" (``usd_start_s``) without
any proof: ``crazyflie_py`` ``setParam`` swallows KeyError and every other exception (only a ROS
warning) and uses ``call_async`` without looking at the result.  ``getParam`` is NOT proof either
(it returns the server's cached value, not the drone's).

Proof signals (firmware ``usddeck.c``, LOG group ``usd``), read through the yaml topic ``usd_status``
(vars ``usd.runTag``, ``usd.fatWrBps``, 5 Hz):
  * ``fatWrBps`` (FAT write rate in bytes/s, float) is > 0 only while the deck is writing a file.
  * ``runTag`` echoes the param.  The topic message type is ``float32[]``: a run tag of ~1.8e9 is NOT
    representable in float32 (spacing 128), so the tag is compared at float32 resolution.  It shows
    that the param reached the drone; it cannot tell two flights less than ~128 s apart from each
    other.  The write rate is the real proof.

Pure Python: ROS objects are injected (``spin``, ``clock``, ``set_fn``), so the logic is unit-tested
without ROS (test/test_usd_gate.py).  Nothing here moves a drone and nothing sends ``usec.reset``.
"""
import struct

TOPIC = 'usd_status'
PARAMS = ('usd.logging', 'usd.runTag')
RATE_MIN_BPS = 1.0       # fatWrBps above this = the deck is writing
CONSECUTIVE = 2          # consecutive good samples required


def f32(x):
    """Value as the float32 that travels in LogDataGeneric.values."""
    return struct.unpack('f', struct.pack('f', float(x)))[0]


class StatusTap:
    """Samples (runTag_as_float32, fatWrBps) of one drone's usd_status topic."""

    def __init__(self, name):
        self.name = name
        self.samples = []

    def cb(self, msg):
        v = list(msg.values)
        if len(v) >= 2:
            self.samples.append((float(v[0]), float(v[1])))


def make_taps(node, names, msg_type):
    """Create one StatusTap and subscription per drone name (topic /<name>/usd_status)."""
    taps = {}
    for n in names:
        tap = StatusTap(n)
        node.create_subscription(msg_type, f'/{n}/{TOPIC}', tap.cb, 10)
        taps[n] = tap
    return taps


def set_param_future(target, name, value, broadcast=False):
    """Like crazyflie_py setParam, but returns the future and raises on any problem (no swallowing).

    target: the CrazyflieServer (broadcast=True, 'all.params.<name>') or one Crazyflie.
    """
    from rcl_interfaces.msg import Parameter, ParameterType, ParameterValue
    from rcl_interfaces.srv import SetParameters
    ptype = target.paramTypeDict[name]          # KeyError = the server never saw this param
    if ptype == ParameterType.PARAMETER_INTEGER:
        pv = ParameterValue(type=ptype, integer_value=int(value))
    elif ptype == ParameterType.PARAMETER_DOUBLE:
        pv = ParameterValue(type=ptype, double_value=float(value))
    else:
        raise TypeError(f'unsupported parameter type {ptype} for {name}')
    prefix = 'all' if broadcast else target.prefix[1:]
    req = SetParameters.Request()
    req.parameters = [Parameter(name=f'{prefix}.params.{name}', value=pv)]
    return target.setParamsService.call_async(req)


def _wait_futures(futs, spin, clock, timeout_s):
    """Spin until all futures are done or the timeout passes; returns {key: (ok, reason)}."""
    t_end = clock() + timeout_s
    while clock() < t_end and not all(f.done() for f in futs.values()):
        spin(0.05)
    out = {}
    for k, f in futs.items():
        if not f.done():
            out[k] = (False, 'no response')
            continue
        try:
            res = f.result().results
            ok = bool(res) and all(r.successful for r in res)
            reason = '' if ok else '; '.join(str(getattr(r, 'reason', '')) for r in res) or 'empty result'
            out[k] = (ok, reason)
        except Exception as e:      # noqa: BLE001 - report, never raise
            out[k] = (False, f'exception {e}')
    return out


def _send(set_fn, targets, name, value, spin, clock, timeout_s=1.0):
    """Send one param to {key: (target, broadcast)}; returns {key: (ok, reason)} (no exceptions)."""
    futs, out = {}, {}
    for k, (t, bc) in targets.items():
        try:
            futs[k] = set_fn(t, name, value, broadcast=bc)
        except Exception as e:          # noqa: BLE001 - KeyError etc: this IS the failure to report
            out[k] = (False, f'{type(e).__name__}: {e}')
    if futs:
        out.update(_wait_futures(futs, spin, clock, timeout_s))
    return out


def missing_params(allcfs):
    """usd params the server never listed (the silent-KeyError case of setParam)."""
    return [n for n in PARAMS if n not in allcfs.paramTypeDict]


def _good(sample, run_tag):
    tag, rate = sample
    return tag == f32(run_tag) and rate >= RATE_MIN_BPS


def _wait_taps(taps, names, idx0, ok_fn, spin, clock, timeout_s):
    """Spin until every name has CONSECUTIVE samples after idx0 passing ok_fn, or timeout."""
    t_end = clock() + timeout_s
    done = {n: False for n in names}
    while True:
        for n in names:
            s = taps[n].samples[idx0[n]:]
            done[n] = len(s) >= CONSECUTIVE and all(ok_fn(x) for x in s[-CONSECUTIVE:])
        if all(done.values()) or clock() >= t_end:
            return done
        spin(0.05)


def _drone_summary(tap, idx0, run_tag):
    s = tap.samples[idx0:]
    return {'topic_seen': len(tap.samples) > 0,
            'samples': len(s),
            'runTag_seen': s[-1][0] if s else None,
            'tag_ok': bool(s) and s[-1][0] == f32(run_tag),
            'fatWrBps_max': max((x[1] for x in s), default=0.0)}


def precheck(allcfs, names, taps, spin, clock, wait_s=3.0):
    """GROUND check before takeoff: params known, topic alive on every drone, nobody writing already."""
    res = {'phase': 'precheck', 'passed': False, 'missing_params': missing_params(allcfs), 'drones': {}}
    if res['missing_params']:
        res['reason'] = (f"usd params {res['missing_params']} not in the server's parameter list "
                         f"(setParam would swallow the KeyError)")
        return res
    idx0 = {n: len(taps[n].samples) for n in names}
    done = _wait_taps(taps, names, idx0, lambda s: True, spin, clock, wait_s)
    ok_all = True
    for n in names:
        d = _drone_summary(taps[n], idx0[n], 0)
        d['topic_alive'] = done[n]
        d['already_writing'] = d['fatWrBps_max'] >= RATE_MIN_BPS
        d['passed'] = d['topic_alive'] and not d['already_writing']
        d['reason'] = ('' if d['passed'] else
                       ('usd_status topic silent (yaml without usd_status, or usd.fatWrBps/usd.runTag not in the log TOC)'
                        if not d['topic_alive'] else
                        'deck already writing before the start (logging stuck from an earlier session)'))
        ok_all &= d['passed']
        res['drones'][n] = d
    res['passed'] = ok_all
    if not ok_all:
        res['reason'] = '; '.join(f'{n}: {d["reason"]}' for n, d in res['drones'].items() if not d['passed'])
    return res


def start_and_verify(allcfs, cfs, names, taps, run_tag, spin, clock, monotonic,
                     timeout_s=4.0, retries=1, set_fn=set_param_future):
    """Start uSD logging by broadcast, then PROVE it on every drone; resend per drone on failure.

    Returns a JSON-able dict with 'passed', 't_start_monotonic' (taken right after the broadcast, the
    same instant the scripts used for usd_start_s) and per-drone evidence.
    """
    res = {'phase': 'start', 'passed': False, 'run_tag': int(run_tag), 'drones': {}, 'broadcast': {},
           'missing_params': missing_params(allcfs), 't_start_monotonic': None}
    if res['missing_params']:
        res['reason'] = (f"usd params {res['missing_params']} not in the server's parameter list "
                         f"(the silent-KeyError case)")
        return res
    idx0 = {n: len(taps[n].samples) for n in names}
    bc = {'broadcast': (allcfs, True)}
    r_tag = _send(set_fn, bc, 'usd.runTag', run_tag, spin, clock)
    spin(0.05)
    r_log = _send(set_fn, bc, 'usd.logging', 1, spin, clock)
    res['t_start_monotonic'] = monotonic()
    res['broadcast'] = {'runTag': r_tag['broadcast'], 'logging': r_log['broadcast']}
    done = _wait_taps(taps, names, idx0, lambda s: _good(s, run_tag), spin, clock, timeout_s)
    retried = {n: 0 for n in names}
    for _ in range(retries):
        bad = [n for n in names if not done[n]]
        if not bad:
            break
        per = {n: (c, False) for n, c in zip(names, cfs) if n in bad}
        sent_tag = _send(set_fn, per, 'usd.runTag', run_tag, spin, clock)
        sent_log = _send(set_fn, per, 'usd.logging', 1, spin, clock)
        for n in bad:
            retried[n] += 1
            res['drones'].setdefault(n, {})['retry_set'] = {'runTag': sent_tag.get(n), 'logging': sent_log.get(n)}
        for n in bad:
            idx0[n] = len(taps[n].samples)
        redo = _wait_taps(taps, bad, idx0, lambda s: _good(s, run_tag), spin, clock, timeout_s)
        done.update(redo)
    ok_all = True
    for n in names:
        d = res['drones'].setdefault(n, {})
        d.update(_drone_summary(taps[n], idx0[n], run_tag))
        d['retries'] = retried[n]
        d['passed'] = bool(done[n])
        if not d['passed']:
            d['reason'] = ('usd_status topic silent' if not d['topic_seen'] else
                           f"no sustained write with the right tag (runTag seen {d['runTag_seen']}, "
                           f"max fatWrBps {d['fatWrBps_max']:.0f})")
        ok_all &= d['passed']
    res['passed'] = ok_all
    if not ok_all:
        res['reason'] = '; '.join(f'{n}: {d["reason"]}' for n, d in res['drones'].items() if not d['passed'])
    return res


def stop_and_verify(allcfs, cfs, names, taps, spin, clock, timeout_s=3.0, set_fn=set_param_future):
    """Send usd.logging = 0 (broadcast, per drone if a drone still writes); report, never block or raise."""
    res = {'phase': 'stop', 'passed': False, 'drones': {}}
    try:
        idx0 = {n: len(taps[n].samples) for n in names}
        _send(set_fn, {'broadcast': (allcfs, True)}, 'usd.logging', 0, spin, clock)
        quiet = lambda s: s[1] < RATE_MIN_BPS      # noqa: E731
        done = _wait_taps(taps, names, idx0, quiet, spin, clock, timeout_s)
        bad = [n for n in names if not done[n]]
        if bad:
            per = {n: (c, False) for n, c in zip(names, cfs) if n in bad}
            _send(set_fn, per, 'usd.logging', 0, spin, clock)
            for n in bad:
                idx0[n] = len(taps[n].samples)
            done.update(_wait_taps(taps, bad, idx0, quiet, spin, clock, timeout_s))
        for n in names:
            res['drones'][n] = {'stopped': bool(done[n])}
        res['passed'] = all(done.values())
    except Exception as e:          # noqa: BLE001 - landing must never depend on this
        res['error'] = f'{type(e).__name__}: {e}'
    return res
