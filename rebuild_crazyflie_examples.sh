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
  echo "=== Step 0: ensure ROS2 environment is sourced ==="
  # Required because this script runs via `bash script.sh`, which many .bashrc
  # setups skip ROS sourcing for (common guard: `[ -z "$PS1" ] && return`
  # short-circuits before the ROS source line in non-interactive shells).
  # Without this, CMake can't find ament_cmake and every package fails to
  # configure (confirmed root cause of the crazyflie_interfaces build failure).
  if [ -z "${ROS_DISTRO:-}" ]; then
    ROS_SETUP=$(ls /opt/ros/*/setup.bash 2>/dev/null | head -1)
    if [ -n "$ROS_SETUP" ]; then
      echo "ROS_DISTRO not set — sourcing $ROS_SETUP"
      # shellcheck disable=SC1090
      source "$ROS_SETUP"
    else
      echo "ERROR: ROS_DISTRO not set and no /opt/ros/*/setup.bash found."
      echo "Source your ROS2 environment manually first, e.g.:"
      echo "  source /opt/ros/humble/setup.bash"
      exit 1
    fi
  else
    echo "ROS_DISTRO=$ROS_DISTRO already set."
  fi

  echo
  echo "=== Step 1: clean stale build/install artifacts ==="
  rm -rf build/crazyflie_examples install/crazyflie_examples log/latest_build
  echo "cleaned."

  echo
  echo "=== Step 2: colcon build (verbose, unbuffered) ==="
  # --packages-up-to (not --packages-select): also builds crazyflie_interfaces /
  # crazyflie_py if they aren't already present in this workspace's install/ —
  # --packages-select assumes dependencies already exist and fails outright if
  # colcon can't discover them (confirmed root cause of the "1 package failed"
  # error: crazyflie_interfaces/crazyflie_py package.sh files were missing).
  colcon build --symlink-install --packages-up-to crazyflie_examples \
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
