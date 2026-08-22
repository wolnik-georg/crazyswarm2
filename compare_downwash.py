#!/usr/bin/env python3
"""Diff two recorded sim runs — with and without the downwash model.

Run sim_downwash_test twice, once with crazyflies_sim_np.yaml and once with
crazyflies_sim_dw.yaml, then run this. The two runs are identical in every respect
except the interaction force, so any difference in the LOWER drone's height is the
downwash and nothing else.
"""
import glob
import os
import sys

import numpy as np


def load(d):
    # record_states writes to <output_dir>/<TIMESTAMP>/csv/<name>.csv, so search
    # recursively and keep only the most recent run per drone.
    out = {}
    files = sorted(glob.glob(os.path.join(d, '**', '*.csv'), recursive=True))
    for f in files:
        name = os.path.basename(f)[:-4]
        try:
            a = np.genfromtxt(f, delimiter=',', names=True)
        except Exception:
            continue
        if a.size:
            out[name] = a          # later (newer) runs overwrite earlier ones
    return out


def zcol(a):
    for c in ('z', 'pos_z', 'position_z'):
        if c in (a.dtype.names or ()):
            return a[c]
    return None


def main():
    d_np = sys.argv[1] if len(sys.argv) > 1 else 'state_np'
    d_dw = sys.argv[2] if len(sys.argv) > 2 else 'state_dw'
    A, B = load(d_np), load(d_dw)
    if not A or not B:
        sys.exit(f'[compare] no CSVs in {d_np} and/or {d_dw} - did both runs finish?')

    print('=' * 68)
    print('DOWNWASH EFFECT  (neuralswarm vs np, identical flights)')
    print('=' * 68)

    # Compare on SIM TIME, never by row index. The neuralswarm backend runs several
    # times slower than real time (torch inference every step), so for the same
    # wall-clock run it covers far less simulated time and produces far fewer rows.
    # Index-wise comparison then lines up mid-climb against settled hover.
    any_diff = False
    warned = False
    for name in sorted(set(A) & set(B)):
        a, b = A[name], B[name]
        ta, tb = a['timestamp'], b['timestamp']
        za, zb = zcol(a), zcol(b)
        if za is None or zb is None:
            continue
        lo, hi = max(ta[0], tb[0]), min(ta[-1], tb[-1])
        if hi - lo < 1.0:
            print(f'\n  {name}: overlap only {hi-lo:.1f} s - too short')
            continue
        grid = np.linspace(lo, hi, 2000)
        ia, ib = np.interp(grid, ta, za), np.interp(grid, tb, zb)
        d = ib - ia

        # Compare only the STEADY HOVER, found as the longest stretch where both runs
        # are near-stationary. Takeoff, the goTo climb and the landing are large
        # transients that swamp a centimetre-scale effect, and the two runs do not
        # even reach them at identical times -- averaging over them says nothing.
        va = np.abs(np.gradient(ia, grid))
        vb = np.abs(np.gradient(ib, grid))
        still = (va < 0.02) & (vb < 0.02) & (ia > 0.2) & (ib > 0.2)
        if still.sum() < 50:
            print(f'\n  {name}: no common steady hover found')
            continue
        idx = np.flatnonzero(still)
        # longest contiguous run of `still`
        splits = np.split(idx, np.flatnonzero(np.diff(idx) > 1) + 1)
        settled_idx = max(splits, key=len)
        settled = np.zeros_like(still)
        settled[settled_idx] = True
        t0, t1 = grid[settled_idx[0]], grid[settled_idx[-1]]

        print(f'\n  {name}   [steady hover {t0:.1f}-{t1:.1f} s of {lo:.0f}-{hi:.0f} s]')
        print(f'     height  no-downwash {ia[settled].mean():.4f} m   '
              f'with-downwash {ib[settled].mean():.4f} m')
        print(f'     sag     {1000*d[settled].mean():+.1f} mm')

        if abs(d[settled].mean()) > 0.002:
            any_diff = True
            print('     -> DOWNWASH ACTING')
        else:
            print('     -> no significant difference')

    print()
    if not any_diff:
        print('  RESULT: no measurable difference.')
    else:
        print('  RESULT: downwash is live and measurable in steady hover.')
        print('  The lower drone should sag several times more than the upper one --')
        print('  that asymmetry is the signature that this is really downwash.')
    print()


if __name__ == '__main__':
    main()
