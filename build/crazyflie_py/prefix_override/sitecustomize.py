import sys
if sys.prefix == '/usr':
    sys.real_prefix = sys.prefix
    sys.prefix = sys.exec_prefix = '/home/flyingrobots/georg/ros2_ws/src/crazyswarm2/install/crazyflie_py'
