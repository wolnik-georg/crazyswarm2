#!/usr/bin/env bash
# Clean rebuild of the crazyflie_examples ROS2 package with --symlink-install.
#
# Fixes stale-install issues where `ros2 run` keeps executing an old compiled
# snapshot of flight.py instead of the latest source edits (confirmed cause
# of a hard-to-diagnose "trajectory just hovers" bug this session — a plain
# `colcon build` without --symlink-install leaves a static copy in install/
# that silently goes stale on every subsequent source edit).
#
# Captures full build output (stdout+stderr, unbuffered, no hidden summary)
# to a timestamped log file so failures are diagnosable without re-running.
#
# Usage: bash rebuild_crazyflie_examples.sh   (run from anywhere, cd's itself)
set -uo pipefail

CS2_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
cd "$CS2_DIR"

LOG_DIR="$CS2_DIR/rebuild_logs"
mkdir -p "$LOG_DIR"
LOG_FILE="$LOG_DIR/rebuild_$(date +%Y%m%d_%H%M%S).log"

{
  echo "=== $(date) ==="
  echo "=== Step 1: clean stale build/install artifacts ==="
  rm -rf build/crazyflie_examples install/crazyflie_examples log/latest_build
  echo "cleaned."

  echo
  echo "=== Step 2: colcon build (verbose, unbuffered) ==="
  colcon build --symlink-install --packages-select crazyflie_examples \
    --event-handlers console_direct+
  BUILD_STATUS=$?

  echo
  echo "=== Step 3: build status ==="
  if [ "$BUILD_STATUS" -eq 0 ]; then
    echo "BUILD OK"
  else
    echo "BUILD FAILED (exit $BUILD_STATUS)"
  fi

  echo
  echo "=== Step 4: verify installed flight.py is a symlink ==="
  INSTALLED=$(find "$CS2_DIR/install" -name "flight.py" 2>/dev/null | head -1)
  if [ -z "$INSTALLED" ]; then
    echo "No installed flight.py found — build did not produce it."
  elif [ -L "$INSTALLED" ]; then
    echo "OK: $INSTALLED -> $(readlink -f "$INSTALLED")"
  else
    echo "WARNING: $INSTALLED exists but is NOT a symlink (stale-copy risk remains)."
  fi

  echo
  echo "=== Step 5: source workspace ==="
  # shellcheck disable=SC1091
  source install/setup.bash 2>&1
  echo "sourced $CS2_DIR/install/setup.bash in this script's shell."
  echo "You must also run this in YOUR shell before flying:"
  echo "  source $CS2_DIR/install/setup.bash"

  echo
  echo "=== Done. Full log: $LOG_FILE ==="
} 2>&1 | tee "$LOG_FILE"

echo
echo "Log saved to: $LOG_FILE"
