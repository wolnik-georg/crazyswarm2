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
        sys.exit(f'[compare] no CSVs in {d_np} and/or {d_dw} — did both runs finish?')

    print('=' * 66)
    print('DOWNWASH EFFECT  (neuralswarm vs np, identical flights)')
    print('=' * 66)
    any_diff = False
    for name in sorted(set(A) & set(B)):
        za, zb = zcol(A[name]), zcol(B[name])
        if za is None or zb is None:
            continue
        n = min(len(za), len(zb))
        za, zb = za[:n], zb[:n]
        d = zb - za                      # downwash minus baseline
        settled = slice(int(0.5 * n), n)  # ignore takeoff transient
        print(f'\n  {name}')
        print(f'     height  no-downwash {za[settled].mean():.4f} m   '
              f'with-downwash {zb[settled].mean():.4f} m')
        print(f'     delta   mean {1000*d[settled].mean():+.1f} mm   '
              f'max |{1000*np.abs(d).max():.1f}| mm')
        if abs(d[settled].mean()) > 0.002:
            any_diff = True
            print('     -> DOWNWASH IS ACTING on this drone')
        else:
            print('     -> no significant difference')

    print()
    if any_diff:
        print('  RESULT: the downwash model is live and measurable.')
    else:
        print('  RESULT: no difference. Either the drones were not vertically')
        print('  stacked within |dx|,|dy| < 0.2 m, or the backend did not switch.')
    print()


if __name__ == '__main__':
    main()
