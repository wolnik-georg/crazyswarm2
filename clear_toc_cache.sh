#!/usr/bin/env bash
# Find and clear crazyflie_py/cflib's cached parameter+log TOC ("./cache",
# see crazyflie/scripts/crazyflie_server.py: CachedCfFactory(rw_cache="./cache")).
#
# Why this matters today specifically: the drone's firmware was rebuilt/reflashed
# multiple times this session (original -> widened traj.ci/aci to u16 -> reverted
# back to the original). The TOC cache is keyed by firmware CRC, so in principle a
# genuinely different firmware build should trigger a fresh download automatically
# — but if any stale/corrupted cache entry survives (e.g. from the intermediate
# u16 build), parameter writes like traj.ci/traj.mode/traj.start could silently
# target the wrong param IDs on the currently-flashed firmware while unrelated
# params (ramp gains, position hold) keep working — which matches the observed
# symptom (normal takeoff/hold, but trajectory motion never actually happens).
#
# Usage: bash clear_toc_cache.sh
set -uo pipefail

echo "Searching common locations for TOC cache directories named 'cache'..."
echo "(this can take a few seconds)"
echo

FOUND=0
for base in "$HOME" "$HOME/Desktop" "$HOME/Desktop/crazyswarm2" "$HOME/ros2_ws" "/home/flyingrobots"; do
  [ -d "$base" ] || continue
  while IFS= read -r -d '' d; do
    FOUND=1
    echo "Found: $d"
    ls -la "$d" 2>/dev/null | head -5
    echo
  done < <(find "$base" -maxdepth 6 -type d -name "cache" -print0 2>/dev/null)
done

if [ "$FOUND" -eq 0 ]; then
  echo "No 'cache' directories found under the searched locations."
  echo "You likely need to run this from (or point it at) whatever directory"
  echo "you normally run 'ros2 launch crazyflie launch.py' from — that's the"
  echo "CWD the cache gets created relative to."
  exit 0
fi

echo "=== Delete all of the above? [y/N] ==="
read -r ans
if [ "$ans" = "y" ] || [ "$ans" = "Y" ]; then
  for base in "$HOME" "$HOME/Desktop" "$HOME/Desktop/crazyswarm2" "$HOME/ros2_ws" "/home/flyingrobots"; do
    [ -d "$base" ] || continue
    find "$base" -maxdepth 6 -type d -name "cache" -exec rm -rf {} + 2>/dev/null
  done
  echo "Cleared. Restart the CS2 server (re-run your ros2 launch command) before the next flight —"
  echo "it will re-download the parameter TOC fresh from the currently-flashed firmware."
else
  echo "Skipped deletion."
fi
