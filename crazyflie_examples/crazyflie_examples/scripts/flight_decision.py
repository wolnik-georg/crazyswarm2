#!/usr/bin/env python3
"""Append one row to flight_decisions.csv at card pull."""
from __future__ import annotations

import argparse
import csv
from pathlib import Path

HEADER = ["date", "stamp", "scenario", "setting", "strategy", "decision", "reason", "vbat_min", "notes"]


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--csv", type=Path, required=True)
    ap.add_argument("--date", required=True)
    ap.add_argument("--stamp", required=True)
    ap.add_argument("--scenario", required=True)
    ap.add_argument("--setting", default="")
    ap.add_argument("--strategy", required=True)
    ap.add_argument("--decision", choices=["clean", "excluded", "attempt-aborted"], required=True)
    ap.add_argument("--reason", default="")
    ap.add_argument("--vbat-min", dest="vbat_min", default="")
    ap.add_argument("--notes", default="")
    args = ap.parse_args()
    row = {
        "date": args.date,
        "stamp": args.stamp,
        "scenario": args.scenario,
        "setting": args.setting,
        "strategy": args.strategy,
        "decision": args.decision,
        "reason": args.reason,
        "vbat_min": args.vbat_min,
        "notes": args.notes,
    }
    args.csv.parent.mkdir(parents=True, exist_ok=True)
    write_header = not args.csv.is_file() or args.csv.stat().st_size == 0
    with open(args.csv, "a", newline="") as fh:
        w = csv.DictWriter(fh, fieldnames=HEADER)
        if write_header:
            w.writeheader()
        w.writerow(row)
    print(f"appended to {args.csv}")


if __name__ == "__main__":
    main()
