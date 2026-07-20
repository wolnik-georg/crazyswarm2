#!/usr/bin/env bash
# Clean rebuild of the crazyflie_examples ROS2 package with --symlink-install.
#
# Fixes stale-install issues where `ros2 run` keeps executing an old compiled
# snapshot of flight.py instead of the latest source edits (confirmed cause
# of a hard-to-diagnose "trajectory just hovers" bug this session — a plain
# `colcon build` without --symlink-install leaves a static copy in install/
# that silently goes stale on every subsequent source edit).
#
# Sourcing sequence mirrors the lab PC's own known-working manual sequence:
#   . install/local_setup.bash        (existing overlay, if present)
#   source /opt/ros/jazzy/setup.bash  (ROS2 underlay)
#   source $VENV/bin/activate         (Python venv)
#   colcon build ...
#   . install/local_setup.bash        (re-source the freshly-built overlay)
#
# set -u is deliberately OFF around all sourcing steps: ROS2/venv setup
# scripts reference unset variables internally (e.g. AMENT_TRACE_SETUP_FILES)
# and are not written to be nounset-safe — this is a well-known ROS2 gotcha,
# not a bug in this script.
#
# Captures full build output (stdout+stderr, unbuffered, no hidden summary)
# to a timestamped log file so failures are diagnosable without re-running.
#
# Usage: bash rebuild_crazyflie_examples.sh   (run from anywhere, cd's itself)
set -o pipefail

CS2_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
cd "$CS2_DIR"

ROS_SETUP_CANDIDATES=(
  "/opt/ros/jazzy/setup.bash"
  "/opt/ros/humble/setup.bash"
)
VENV_ACTIVATE_CANDIDATES=(
  "/home/flyingrobots/venv/bin/activate"
  "$HOME/venv/bin/activate"
)

LOG_DIR="$CS2_DIR/rebuild_logs"
mkdir -p "$LOG_DIR"
LOG_FILE="$LOG_DIR/rebuild_$(date +%Y%m%d_%H%M%S).log"

{
  echo "=== $(date) ==="
  echo "=== Step 0: source workspace environment (matches known-good manual sequence) ==="
  set +u

  if [ -f "$CS2_DIR/install/local_setup.bash" ]; then
    echo "sourcing existing install/local_setup.bash (pre-build overlay)..."
    . "$CS2_DIR/install/local_setup.bash"
  else
    echo "no existing install/local_setup.bash yet (first build) — skipping."
  fi

  ROS_SETUP=""
  for c in "${ROS_SETUP_CANDIDATES[@]}"; do
    [ -f "$c" ] && ROS_SETUP="$c" && break
  done
  if [ -z "$ROS_SETUP" ]; then
    ROS_SETUP=$(ls /opt/ros/*/setup.bash 2>/dev/null | head -1)
  fi
  if [ -n "$ROS_SETUP" ]; then
    echo "sourcing $ROS_SETUP..."
    source "$ROS_SETUP"
  else
    echo "ERROR: no ROS2 setup.bash found (checked jazzy, humble, /opt/ros/*)."
    set -u
    exit 1
  fi

  VENV_ACTIVATE=""
  for c in "${VENV_ACTIVATE_CANDIDATES[@]}"; do
    [ -f "$c" ] && VENV_ACTIVATE="$c" && break
  done
  if [ -n "$VENV_ACTIVATE" ]; then
    echo "activating venv: $VENV_ACTIVATE..."
    source "$VENV_ACTIVATE"
  else
    echo "no venv found (checked ${VENV_ACTIVATE_CANDIDATES[*]}) — continuing without one."
  fi
  set -u

  echo
  echo "=== Step 1: clean stale build/install artifacts ==="
  rm -rf build/crazyflie_examples install/crazyflie_examples log/latest_build
  echo "cleaned."

  echo
  echo "=== Step 2: colcon build (verbose, unbuffered) ==="
  # --packages-up-to (not --packages-select): also builds crazyflie_interfaces /
  # crazyflie_py if they aren't already present in this workspace's install/ —
  # --packages-select assumes dependencies already exist and fails outright if
  # colcon can't discover them (confirmed root cause of an earlier failure:
  # crazyflie_interfaces/crazyflie_py package.sh files were missing).
  set +u
  colcon build --symlink-install --packages-up-to crazyflie_examples \
    --event-handlers console_direct+
  BUILD_STATUS=$?
  set -u

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
  echo "=== Step 5: re-source install/local_setup.bash ==="
  set +u
  if [ -f "$CS2_DIR/install/local_setup.bash" ]; then
    . "$CS2_DIR/install/local_setup.bash"
    echo "sourced $CS2_DIR/install/local_setup.bash in this script's shell."
  else
    echo "install/local_setup.bash still missing — build did not complete."
  fi
  set -u
  echo "You must also run this in YOUR shell before flying:"
  echo "  . $CS2_DIR/install/local_setup.bash"

  echo
  echo "=== Done. Full log: $LOG_FILE ==="
} 2>&1 | tee "$LOG_FILE"

echo
echo "Log saved to: $LOG_FILE"
