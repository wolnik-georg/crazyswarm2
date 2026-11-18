"""Unit tests of usd_gate (pure Python, no ROS): python3 -m pytest crazyflie_examples/test/test_usd_gate.py"""
import os
import sys
import types

sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', 'crazyflie_examples'))
import usd_gate as g  # noqa: E402

TAG = 1795026746


class Fut:
    def __init__(self, ok=True, reason=''):
        self._r = types.SimpleNamespace(results=[types.SimpleNamespace(successful=ok, reason=reason)])

    def done(self):
        return True

    def result(self):
        return self._r


class Sim:
    """Fake drones + clock. spin() advances time and emits usd_status samples at 5 Hz."""

    def __init__(self, names=('cf5', 'cf_second'), topic=True, broadcast_reaches=('cf5', 'cf_second'),
                 per_drone_reaches=('cf5', 'cf_second'), write_rate=177000.0, echo_tag=True,
                 params=('usd.logging', 'usd.runTag')):
        self.t, self.names = 0.0, list(names)
        self.topic, self.bc, self.pd = topic, set(broadcast_reaches), set(per_drone_reaches)
        self.rate, self.echo = write_rate, echo_tag
        self.st = {n: {'logging': 0, 'tag': 0} for n in names}
        self.acc = 0.0
        self.set_calls = []
        self.taps = {n: g.StatusTap(n) for n in names}
        self.allcfs = types.SimpleNamespace(paramTypeDict={p: 1 for p in params})
        self.cfs = [types.SimpleNamespace(prefix='/' + n) for n in names]

    def clock(self):
        return self.t

    def spin(self, dt):
        self.t += dt
        self.acc += dt
        while self.acc >= 0.2 - 1e-9:
            self.acc -= 0.2
            if not self.topic:
                continue
            for n in self.names:
                s = self.st[n]
                tag = g.f32(s['tag']) if (self.echo or s['tag'] == 0) else g.f32(1.0)
                rate = self.rate if s['logging'] else 0.0
                self.taps[n].cb(types.SimpleNamespace(values=[tag, rate]))

    def set_fn(self, target, name, value, broadcast=False):
        if isinstance(target, types.SimpleNamespace) and hasattr(target, 'paramTypeDict') and name not in target.paramTypeDict:
            raise KeyError(name)
        who = self.names if broadcast else [target.prefix[1:]]
        self.set_calls.append((broadcast, name, value, tuple(who)))
        reaches = self.bc if broadcast else self.pd
        for n in who:
            if n in reaches:
                self.st[n]['logging' if name == 'usd.logging' else 'tag'] = int(value)
        return Fut()

    def start(self, **kw):
        return g.start_and_verify(self.allcfs, self.cfs, self.names, self.taps, TAG, self.spin, self.clock,
                                  lambda: self.t, set_fn=self.set_fn, **kw)


def test_1_all_good_passes():
    s = Sim()
    r = s.start()
    assert r['passed'], r
    assert all(d['passed'] and d['retries'] == 0 and d['fatWrBps_max'] > 0 for d in r['drones'].values())
    assert r['t_start_monotonic'] is not None


def test_2_missing_param_fails_fast_without_sending():
    s = Sim(params=('usd.runTag',))
    r = s.start()
    assert not r['passed'] and r['missing_params'] == ['usd.logging']
    assert s.set_calls == [] and s.t == 0.0          # failed before sending anything, no waiting


def test_2b_setparam_keyerror_is_reported_not_swallowed():
    s = Sim()

    def boom(*a, **k):
        raise KeyError('usd.logging')
    r = g.start_and_verify(s.allcfs, s.cfs, s.names, s.taps, TAG, s.spin, s.clock, lambda: s.t, set_fn=boom, retries=0)
    assert not r['passed'] and 'KeyError' in str(r['broadcast'])


def test_3_silent_topic_fails_by_timeout_no_hang():
    s = Sim(topic=False)
    r = s.start(timeout_s=2.0, retries=1)
    assert not r['passed']
    assert all('silent' in d['reason'] for d in r['drones'].values())
    assert s.t < 10.0                                # bounded: 2 waits of 2 s plus set waits


def test_4_wrong_runtag_fails():
    s = Sim(echo_tag=False)
    r = s.start()
    assert not r['passed'] and all(not d['tag_ok'] for d in r['drones'].values())


def test_5_no_write_rate_fails():
    s = Sim(write_rate=0.0)
    r = s.start()
    assert not r['passed'] and all(d['fatWrBps_max'] == 0 for d in r['drones'].values())


def test_6_one_drone_missed_broadcast_retry_per_drone_passes():
    s = Sim(broadcast_reaches=('cf5',))
    r = s.start()
    assert r['passed'], r
    assert r['drones']['cf_second']['retries'] == 1 and r['drones']['cf5']['retries'] == 0
    assert any((not bc) and who == ('cf_second',) for bc, name, v, who in s.set_calls)


def test_7_negative_start_only_cf5_fails_when_retry_cannot_reach():
    s = Sim(broadcast_reaches=('cf5',), per_drone_reaches=('cf5',))
    r = s.start()
    assert not r['passed'] and r['drones']['cf5']['passed'] and not r['drones']['cf_second']['passed']


def test_8_precheck_ok_silent_and_stuck():
    s = Sim()
    assert g.precheck(s.allcfs, s.names, s.taps, s.spin, s.clock)['passed']
    s = Sim(topic=False)
    r = g.precheck(s.allcfs, s.names, s.taps, s.spin, s.clock, wait_s=1.0)
    assert not r['passed'] and 'silent' in r['reason']
    s = Sim()
    s.st['cf5']['logging'] = 1                         # deck already writing before the start
    r = g.precheck(s.allcfs, s.names, s.taps, s.spin, s.clock)
    assert not r['passed'] and 'already writing' in r['reason']


def test_9_stop_verifies_quiet_and_never_raises():
    s = Sim()
    assert s.start()['passed']
    r = g.stop_and_verify(s.allcfs, s.cfs, s.names, s.taps, s.spin, s.clock, set_fn=s.set_fn)
    assert r['passed'], r
    r = g.stop_and_verify(None, None, s.names, s.taps, s.spin, s.clock)    # broken input: no exception
    assert r['phase'] == 'stop'
    r = g.stop_and_verify(s.allcfs, s.cfs, ['nope'], s.taps, s.spin, s.clock, set_fn=s.set_fn)   # unknown drone
    assert 'error' in r and not r['passed']


def test_10_float32_tag_resolution_is_documented_not_exact():
    assert g.f32(1795026746) == g.f32(1795026750)      # same float32 bucket: tag alone cannot prove "this flight"
    assert g.f32(1795026746) != g.f32(1795026746 + 256)
