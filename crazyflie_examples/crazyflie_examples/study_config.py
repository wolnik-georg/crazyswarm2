"""Study flight identity and --study validation (read-only helpers + manifest)."""
from __future__ import annotations

import hashlib
import json
import subprocess
from pathlib import Path
from typing import Any

REPO = Path(__file__).resolve().parents[2]
MANIFEST = REPO / "crazyflie" / "config" / "study_manifest.json"
STRATEGIES = REPO / "crazyflie" / "config" / "study_strategies.json"

# Parameter names read back from every drone (2026-10-11). Extracted from the firmware sources
# (traj_iface.c PARAM groups + controller_omar_indi.c); write-only weight-upload params of `rnn`
# (wi, wv, wc, begin, end, rst) are deliberately not read. Missing params (e.g. on stock firmware) read as None.
_INDI_GAINS = ['ctrl_mode', 'kr', 'kw', 'kr_z', 'kw_z', 'kr_geo', 'kw_geo', 'kr_z_geo', 'kw_z_geo', 'kt1', 'kt2', 'kt3', 'kt4',
               'fc_bw', 'fc_bw_yaw', 'fc_iir', 'mass', 'ff_free', 'n1_fix', 'dt_usec', 'filt_order', 'filt_tau', 'j_scale',
               'act_tau', 'clamp_en', 'tau_xy_max', 'tau_z_max', 'tilt_max_deg', 'thrust_max', 'omega_src', 'frame_conv',
               'res_sign', 'filt_dt_us', 'filt_prewarp', 'res_fc', 'res_clamp', 'tau_clamp', 'notch_en', 'notch_f0',
               'notch_bw', 'rpm_source']
_POS_GAINS = ['kp_xy', 'kp_z', 'kv_xy', 'kv_z', 'ki_z', 'ki_z_limit']
_RNN = ['n', 'en', 'ready', 'div']
_OOT5 = ['indi', 'kpos_iz']
_OMAR = ['KR_x', 'KR_y', 'KR_z', 'KI_x', 'KI_y', 'KI_z', 'Kw_x', 'Kw_y', 'Kw_z', 'J_x', 'J_y', 'J_z', 'Kpos_Px', 'Kpos_Py',
         'Kpos_Pz', 'Kpos_P_limit', 'Kpos_Dx', 'Kpos_Dy', 'Kpos_Dz', 'Kpos_D_limit', 'Kpos_Ix', 'Kpos_Iy', 'Kpos_Iz',
         'Kpos_I_limit', 'mass', 'indi']
READBACK_KEYS = (['stabilizer.controller']
                 + [f'indi_gains.{k}' for k in _INDI_GAINS] + [f'pos_gains.{k}' for k in _POS_GAINS]
                 + [f'rnn.{k}' for k in _RNN] + [f'ctrlOot5.{k}' for k in _OOT5]
                 + [f'ctrlOmarIndi.{k}' for k in _OMAR] + ['ctrlFbl.en', 'ctrlFbl.preset', 'ctrlFbl.cd', 'ctrlFbl.ki_z'] + ['deck.bcRpm', 'usd.canLog', 'usd.logging'])

# Frozen-block keys that apply() never writes (they come from the yaml firmware_params at connect time), so they can be
# verified BEFORE takeoff. Everything else (controller, ctrl_mode, gains, res_sign) is set by apply('scenario') after takeoff
# and is verified after it, with a safe landing on mismatch.
STATIC_PREFIXES = ('rnn.', 'ctrlOmarIndi.', 'ctrlOot5.', 'ctrlFbl.', 'deck.')


def load_json(path: Path) -> dict:
    return json.loads(path.read_text())


def git_identity() -> dict[str, Any]:
    try:
        root = REPO
        commit = subprocess.check_output(
            ["git", "-C", str(root), "rev-parse", "HEAD"], text=True
        ).strip()
        dirty = subprocess.call(["git", "-C", str(root), "diff", "--quiet"]) != 0
        tag = subprocess.check_output(
            ["git", "-C", str(root), "describe", "--tags", "--always"],
            text=True,
        ).strip()
        return {"commit": commit, "dirty": dirty, "describe": tag}
    except (subprocess.CalledProcessError, FileNotFoundError):
        return {"commit": None, "dirty": None, "describe": None}


def sha256_file(path: Path) -> str | None:
    if not path.is_file():
        return None
    h = hashlib.sha256()
    with open(path, "rb") as fh:
        for chunk in iter(lambda: fh.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def manifest_ok(manifest: dict | None = None) -> tuple[bool, str]:
    m = manifest if manifest is not None else load_json(MANIFEST)
    for drone in ("cf5", "cf_second"):
        block = m.get(drone) or {}
        if not block.get("firmware_sha256"):
            return False, f"study manifest: {drone} firmware_sha256 missing/empty"
    return True, "ok"


def read_param_safe(cf, name: str, timeout: float = 0.5):
    try:
        val = cf.getParam(name)
        if val is None:
            return None
        return float(val) if isinstance(val, (int, float)) else val
    except Exception:
        return None


def read_drone_block(cf, prefix: str, keys=None) -> dict[str, Any]:
    out: dict[str, Any] = {"_drone": prefix.lstrip("/")}
    for key in (keys if keys is not None else READBACK_KEYS):
        val = read_param_safe(cf, key)
        out[key.replace(".", "_")] = val
    return out


def keys_for_airborne_check(strategy: str) -> list[str]:
    """Keys to read back AFTER apply('scenario') (drones are hovering, keep it short): the frozen-block keys plus
    the parameters apply() writes."""
    base = ["stabilizer.controller", "indi_gains.ctrl_mode", "indi_gains.res_sign", "indi_gains.rpm_source",
            "pos_gains.ki_z", "pos_gains.kp_xy", "pos_gains.kp_z", "pos_gains.kv_xy", "pos_gains.kv_z"]
    return sorted(set(base) | {k for k in strategy_expect(strategy) if not k.startswith("_")})


def strategy_expect(strategy: str) -> dict:
    data = load_json(STRATEGIES)
    if strategy not in data:
        raise KeyError(f"unknown strategy {strategy!r}")
    return data[strategy]


def _key_ok(v, got) -> str | None:
    if got is None:
        return "missing readback (null)"
    try:
        if got != got:   # NaN: crazyflie_py.getParam returns NaN when the server has no such parameter
            return "missing readback (NaN: parameter not available from the server)"
    except Exception:
        return "missing readback (unreadable)"
    if isinstance(v, float):
        return None if abs(float(got) - v) <= 1e-3 else f"got {got} expected {v}"
    return None if int(float(got)) == int(v) else f"got {got} expected {v}"


def validate_readback(strategy: str, readback: dict[str, Any], static_only: bool = False) -> list[str]:
    """Compare readback (keys with underscores) to the frozen strategy block.
    static_only=True checks only the keys apply() never writes (safe to check before takeoff)."""
    expect = strategy_expect(strategy)
    problems = []
    for k, v in expect.items():
        if k.startswith("_"):
            continue
        if static_only and not k.startswith(STATIC_PREFIXES):
            continue
        err = _key_ok(v, readback.get(k.replace(".", "_")))
        if err:
            problems.append(f"{k}: {err}")
    return problems


def validate_intended(strategy: str, effective: dict[str, Any]) -> list[str]:
    """Hardware-free pre-takeoff check: the config the script is about to push (run_formation's `effective`
    block for cf5: controller, ctrl_mode, indi dict, pos dict) against the frozen block."""
    expect = strategy_expect(strategy)
    have = {
        "stabilizer.controller": effective.get("controller"),
        "indi_gains.ctrl_mode": effective.get("ctrl_mode"),
    }
    for k, v in (effective.get("indi") or {}).items():
        have[f"indi_gains.{k}"] = v
    for k, v in (effective.get("pos") or {}).items():
        have[f"pos_gains.{k}"] = v
    problems = []
    for k, v in expect.items():
        if k.startswith("_") or k.startswith(STATIC_PREFIXES):
            continue
        if k not in have:
            continue  # not pushed by the script -> verified from the readback instead
        err = _key_ok(v, have[k])
        if err:
            problems.append(f"intended {k}: {err}")
    return problems


def check_firmware_disk(manifest: dict) -> tuple[list[str], dict[str, Any]]:
    """Compare the manifest hash with the file on this machine where the file exists. Returns (problems, info).
    A file that is not on this machine is recorded as 'unverified' (the hash of the FLASHED image cannot be read from
    the drone; this only guards against a stale manifest)."""
    problems, info = [], {}
    for drone in ("cf5", "cf_second"):
        blk = manifest.get(drone) or {}
        fw = blk.get("firmware_file") or ""
        path = Path(fw) if fw.startswith("/") else (REPO / "crazyflie" / fw if fw else None)
        disk = sha256_file(path) if path else None
        info[f"{drone}_firmware_sha256_disk"] = disk
        if disk is None:
            info[f"{drone}_firmware_disk_check"] = "unverified (file not on this machine)"
        elif disk != blk.get("firmware_sha256"):
            problems.append(f"{drone}: firmware file on disk ({disk[:12]}…) != manifest sha256 ({str(blk.get('firmware_sha256'))[:12]}…)")
            info[f"{drone}_firmware_disk_check"] = "MISMATCH"
        else:
            info[f"{drone}_firmware_disk_check"] = "ok"
    return problems, info
