"""Formation flight scenarios for multi-robot interaction-force experiments.

ROS-free by design: the scenario definitions, the curve-to-Poly4D compiler and the safety
gate all import only numpy, so they can be unit-tested, used from analysis code, and
checked without a simulator. The ROS runner is `crazyflie_examples/run_formation.py`.
"""

from . import curves, poly4d, safety, scenarios  # noqa: F401
from .scenarios import BUILDERS, N_ROBOTS, Scenario, build, check_spec  # noqa: F401
