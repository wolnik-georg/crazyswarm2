#!/usr/bin/env bash
# Clean rebuild of the crazyflie_examples ROS2 package with --symlink-install.
#
# Fixes stale-install issues where `ros2 run` keeps executing an old compiled
# snapshot of flight.py instead of the latest source edits (confirmed cause
# of a hard-to-diagnose "trajectory just hovers" bug this session — a plain
# `colcon build` without --symlink-install leaves a static copy in install/
# that silently goes stale on every subsequent source edit).
#
# Usage: bash rebuild_crazyflie_examples.sh   (run from anywhere, cd's itself)
set -euo pipefail

CS2_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
cd "$CS2_DIR"

rm -rf build/crazyflie_examples install/crazyflie_examples log/latest_build
colcon build --symlink-install --packages-select crazyflie_examples
source install/setup.bash

echo
echo "Rebuilt and sourced. Remember to 'source $CS2_DIR/install/setup.bash' in any NEW shell before flying."
