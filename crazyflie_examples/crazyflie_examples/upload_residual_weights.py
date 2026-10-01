"""Upload trained residual-network weights to one or more Crazyflies.

    ros2 run crazyflie_examples upload_residual_weights -- --weights weights/a1.npz
    ros2 run crazyflie_examples upload_residual_weights -- --weights w.npz --cf cf231 --enable

Reads the `.npz` written by `flying_drone_stack/tools/residual/train.py` and pushes the 19297
floats through the `rnn.*` parameter protocol, then verifies `rnn.ready` and `rnn.n` read back
correctly before reporting success or honoring `--enable`.

**Uploading does not switch anything on.** `rnn.en` stays at 0 unless `--enable` is passed, and
even then nothing in the current flight code consumes the prediction -- it is computed and logged
either way. That separation is deliberate: the first flights with a trained model are meant to
compare predicted against measured residual, not to hand the model authority over the vehicle.

Timing: each weight costs three synchronous parameter writes, so a full upload is on the order of
half a minute per drone. Do it on the ground, before takeoff -- the service loop runs before the
arming check precisely so that works.
"""

import argparse
import json
import sys
import time

import numpy as np
import rclpy
from rcl_interfaces.msg import Parameter, ParameterType, ParameterValue
from rcl_interfaces.srv import SetParameters

from crazyflie_py import Crazyswarm

N_WEIGHTS = 19297        # must match residual_nn.rs and tools/residual/model.py
                         # (Neural-Swarm2 port, 2026-09-14; was 987 for the old custom deep-sets net)


def _set_sync(cf, th, name, value):
    """Blocking setParam.

    `crazyflie_py.setParam` is call_async; firing wi/wv/wc back to back without waiting can
    reorder or drop writes over the ROS->CRTP path, which for a weight upload means a silently
    corrupted network rather than a visible error. Same reasoning as the trajectory coefficient
    upload in `flight.py`.
    """
    param_type = cf.paramTypeDict[name]
    if param_type == ParameterType.PARAMETER_INTEGER:
        pv = ParameterValue(type=param_type, integer_value=int(value))
    elif param_type == ParameterType.PARAMETER_DOUBLE:
        pv = ParameterValue(type=param_type, double_value=float(value))
    else:
        raise ValueError(f"unsupported parameter type for {name}")
    req = SetParameters.Request()
    req.parameters = [Parameter(name=cf.prefix[1:] + ".params." + name, value=pv)]
    future = cf.setParamsService.call_async(req)
    while rclpy.ok() and not future.done():
        th.sleep(0.001)


def upload(cf, th, w, delay):
    _set_sync(cf, th, "rnn.n", len(w))
    _set_sync(cf, th, "rnn.begin", 1)
    th.sleep(0.05)

    t0 = time.monotonic()
    for i, v in enumerate(w):
        _set_sync(cf, th, "rnn.wi", i)
        _set_sync(cf, th, "rnn.wv", float(v))
        _set_sync(cf, th, "rnn.wc", 1)
        if delay:
            th.sleep(delay)
        if (i + 1) % 200 == 0:
            print(f"    {i + 1}/{len(w)}")

    _set_sync(cf, th, "rnn.end", 1)
    th.sleep(0.2)
    return time.monotonic() - t0


def _read_finite_param(cf, th, name, timeout_s=1.0, poll_s=0.1):
    """Poll ``getParam`` until a finite numeric value or timeout (post-upload cache race)."""
    deadline = time.monotonic() + timeout_s
    while time.monotonic() < deadline:
        try:
            val = cf.getParam(name)
            if val is not None:
                fv = float(val)
                if np.isfinite(fv):
                    return fv
        except Exception:
            pass
        th.sleep(poll_s)
    return None


def main():
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--weights", required=True, help=".npz from tools/residual/train.py")
    ap.add_argument("--cf", action="append", default=None,
                    help="drone name; repeat for several. Default: all in the roster")
    ap.add_argument("--enable", action="store_true",
                    help="also set rnn.en=1. Off by default -- upload and use are separate")
    ap.add_argument("--delay", type=float, default=0.004,
                    help="extra pause per weight [s]")
    args, _ = ap.parse_known_args()

    d = np.load(args.weights, allow_pickle=False)
    w = np.asarray(d["weights"], np.float32)
    if w.size != N_WEIGHTS:
        sys.exit(f"{args.weights}: {w.size} weights, firmware expects {N_WEIGHTS}. "
                 f"The architecture changed on one side only.")
    if not np.all(np.isfinite(w)):
        sys.exit(f"{args.weights}: contains non-finite values; the firmware would refuse them.")

    meta = json.loads(str(d["meta"])) if "meta" in d else {}
    print(f"weights: {args.weights}")
    print(f"  trained {meta.get('created', '?')} from {meta.get('sources', '?')}")
    print(f"  val RMSE {meta.get('val_rmse', float('nan')):.4f} vs baseline "
          f"{meta.get('baseline_rmse', float('nan')):.4f} m/s^2")
    if meta.get("synthetic"):
        print("  *** SYNTHETIC WEIGHTS -- a pipeline rehearsal, not a trained model. ***")

    swarm = Crazyswarm()
    th = swarm.timeHelper
    allcfs = swarm.allcfs
    targets = ([allcfs.crazyfliesByName[n] for n in args.cf] if args.cf
               else list(allcfs.crazyflies))

    ok = True
    for cf in targets:
        name = cf.prefix.lstrip("/")
        if "rnn.wi" not in cf.paramTypeDict:
            print(f"  {name}: no rnn.* parameters in the TOC -- firmware predates the residual "
                  f"network, or is not our out-of-tree build. Skipped.")
            ok = False
            continue
        print(f"  {name}: uploading {len(w)} weights...")
        dt = upload(cf, th, w, args.delay)

        # 2026-10-01 lab: rnn.ready alone can look clean while weights never loaded (rnn_pred_* all 0).
        expected_n = len(w)
        ready = _read_finite_param(cf, th, "rnn.ready")
        if ready is None:
            print(f"  {name}: uploaded in {dt:.1f}s. Could not confirm rnn.ready "
                  f"(timeout or non-finite readback) -- re-upload; do not enable.")
            ok = False
            continue
        if int(ready) != 1:
            # The firmware refuses a partial set outright, so this is a dropped write, not a
            # degraded model. Re-uploading is the fix; flying it is not.
            print(f"  {name}: REJECTED (rnn.ready={int(ready)}) -- weights dropped in transit. "
                  f"Re-run, and raise --delay if it repeats.")
            ok = False
            continue

        n_read = _read_finite_param(cf, th, "rnn.n")
        if n_read is None:
            print(f"  {name}: rnn.ready=1 but could not confirm rnn.n readback -- re-upload; "
                  f"do not enable.")
            ok = False
            continue
        n_read_int = int(n_read)
        if n_read_int != expected_n:
            print(f"  {name}: REJECTED -- rnn.ready=1 but rnn.n={n_read_int} "
                  f"(expected {expected_n}). Weights may not be loaded; re-upload.")
            ok = False
            continue

        print(f"  {name}: uploaded in {dt:.1f}s, verified rnn.ready=1, rnn.n={expected_n}")

        if args.enable:
            _set_sync(cf, th, "rnn.en", 1)
            print(f"  {name}: rnn.en=1")

    if not args.enable:
        print("\nrnn.en left at 0: the prediction is computed and logged but not used. "
              "That is the right setting for the first flights with a new model.")
    return 0 if ok else 1


if __name__ == "__main__":
    sys.exit(main())
