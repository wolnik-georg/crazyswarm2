#!/usr/bin/env python3
"""
Stacked-hover downwash test for the simulator.

Both drones take off and hold station at the SAME x,y with a vertical offset, which
is the only geometry in which the Neural-Swarm2 backend's interaction term is active:
`compute_Fa` gates on |dx| < 0.2 and |dy| < 0.2 m. `multi_trajectory` and `figure8`
fly the drones apart horizontally, so they show no downwash at all — which is why
np and neuralswarm look identical there.

No joystick needed (unlike nice_hover) and no button press, so it can be run
unattended twice: once per backend. Pair it with the record_states visualization
and diff the two runs — the difference IS the downwash.

Measured from the model directly, force on the lower drone, neighbour straight above:
    0.30 m -> -8.8 g     0.40 m -> -7.8 g     0.50 m -> -6.4 g
against a 27-41 g vehicle, i.e. up to a quarter of its weight. Sizeable in the data;
still essentially invisible in RViz, which is why this test writes numbers.
"""
from crazyflie_py import Crazyswarm
import numpy as np

HOVER_S = 12.0     # seconds of stacked hover
Z = 1.0            # height added to each drone's initial position


def main():
    swarm = Crazyswarm()
    timeHelper = swarm.timeHelper
    allcfs = swarm.allcfs

    print(f'[downwash] {len(allcfs.crazyflies)} drone(s)')
    for cf in allcfs.crazyflies:
        print(f'   {cf.prefix.lstrip("/")} start {np.array(cf.initialPosition)}')

    allcfs.takeoff(targetHeight=Z, duration=3.0)
    timeHelper.sleep(3.5)

    # Each drone holds its own x,y -> they stay vertically stacked.
    for cf in allcfs.crazyflies:
        cf.goTo(np.array(cf.initialPosition) + np.array([0, 0, Z]), 0, 3.0)
    timeHelper.sleep(3.5)

    print(f'[downwash] holding stacked hover for {HOVER_S:.0f} s ...')
    timeHelper.sleep(HOVER_S)

    print('[downwash] landing')
    allcfs.land(targetHeight=0.03, duration=3.0)
    timeHelper.sleep(3.5)
    print('[downwash] done')


if __name__ == '__main__':
    main()
