from __future__ import annotations

import numpy as np
from rclpy.node import Node
from rclpy.time import Time
from rosgraph_msgs.msg import Clock
import rowan

from ..sim_data_types import Action, State


class Backend:
    """Backend that uses newton-euler rigid-body dynamics implemented in numpy."""

    def __init__(self, node: Node, names: list[str], states: list[State]):
        self.node = node
        self.names = names
        self.clock_publisher = node.create_publisher(Clock, 'clock', 10)
        self.t = 0
        self.dt = 0.0005

        params = uav_params(node)
        self.uavs = []
        for state in states:
            uav = Quadrotor(state, params)
            self.uavs.append(uav)

    def time(self) -> float:
        return self.t

    def step(self, states_desired: list[State], actions: list[Action]) -> list[State]:
        # advance the time
        self.t += self.dt

        next_states = []

        for uav, action in zip(self.uavs, actions):
            uav.step(action, self.dt)
            next_states.append(uav.state)

        # print(states_desired, actions, next_states)
        # publish the current clock
        clock_message = Clock()
        clock_message.clock = Time(seconds=self.time()).to_msg()
        self.clock_publisher.publish(clock_message)

        return next_states

    def shutdown(self):
        pass


def uav_params(node):
    """Read the optional sim.physics block. None means 'keep the built-in CF2.0 model'."""
    try:
        return dict(node._ros_parameters['sim']['physics'])
    except (AttributeError, KeyError, TypeError):
        return None


class Quadrotor:
    """Basic rigid body quadrotor model (no drag) using numpy and rowan."""

    def __init__(self, state, params=None):
        # parameters (Crazyflie 2.0 quadrotor)
        self.mass = 0.034  # kg
        # self.J = np.array([
        # 	[16.56,0.83,0.71],
        # 	[0.83,16.66,1.8],
        # 	[0.72,1.8,29.26]
        # 	]) * 1e-6  # kg m^2
        self.J = np.array([16.571710e-6, 16.655602e-6, 29.261652e-6])

        # Note: we assume here that our control is forces
        arm_length = 0.046  # m
        arm = 0.707106781 * arm_length
        t2t = 0.006  # thrust-to-torque ratio
        self.B0 = np.array([
            [1, 1, 1, 1],
            [-arm, -arm, arm, arm],
            [-arm, arm, arm, -arm],
            [-t2t, t2t, -t2t, t2t]
            ])
        self.g = 9.81  # not signed

        if self.J.shape == (3, 3):
            self.inv_J = np.linalg.pinv(self.J)  # full matrix -> pseudo inverse
        else:
            self.inv_J = 1 / self.J  # diagonal matrix -> division

        # Optional override of the vehicle being simulated. The defaults above are a
        # brushed CF2.0; our thesis drone is a brushless CF21BL, and a model-inverting
        # controller (INDI) reconstructs the thrust from RPM using ITS OWN constants.
        # If the plant and the controller disagree about kt or mass, the mismatch shows
        # up as a residual force that no amount of tuning removes -- and residual force
        # is the thing this thesis measures. So the plant is made to match the firmware.
        self.kt = None
        # Rotor drag, body-frame diagonal, in 1/s: a_drag = -R D R^T v. Absent from this
        # model entirely ("no drag" in the class docstring), so it is opt-in rather than
        # assumed. Rotor/induced drag dominates body drag at the speeds flown here.
        self.drag = None
        # First-order motor lag [s]: the rotor cannot change speed instantly. Absent from
        # this model (thrust is applied the instant it is commanded), though the project's
        # own Rust simulator carries motor_time_constant = 0.03 and the measured hardware
        # value is 44 ms brushless / 71 ms upgraded. Opt-in; None means the original
        # instantaneous behaviour.
        self.motor_tau = None
        self._rpm = None
        if params:
            self.mass = float(params.get('mass', self.mass))
            if 'inertia' in params:
                self.J = np.array(params['inertia'], dtype=float)
                self.inv_J = np.linalg.pinv(self.J) if self.J.shape == (3, 3) else 1 / self.J
            if 'kt' in params:
                kt = params['kt']
                self.kt = np.full(4, float(kt)) if np.isscalar(kt) \
                    else np.array(kt, dtype=float)
            if params.get('motor_tau'):
                self.motor_tau = float(params['motor_tau'])
            if 'drag' in params and params['drag'] is not None:
                d = params['drag']
                self.drag = np.full(3, float(d)) if np.isscalar(d) \
                    else np.array(d, dtype=float)
            arm_length = float(params.get('arm_length', 0.046))
            t2t = float(params.get('t2t', 0.006))
            arm = 0.707106781 * arm_length
            self.B0 = np.array([
                [1, 1, 1, 1],
                [-arm, -arm, arm, arm],
                [-arm, arm, arm, -arm],
                [-t2t, t2t, -t2t, t2t]
                ])

        self.state = state

    def step(self, action, dt, f_a=np.zeros(3)):

        # convert RPM -> Force
        def rpm_to_force(rpm):
            if self.kt is not None:
                # Quadratic propeller model, thrust = kt * RPM^2 -- the same form the
                # firmware inverts (indi_gains.kt1..kt4).
                return self.kt * np.square(np.asarray(rpm, dtype=float))
            # polyfit using data and scripts from https://github.com/IMRCLab/crazyflie-system-id
            p = [2.55077341e-08, -4.92422570e-05, -1.51910248e-01]
            force_in_grams = np.polyval(p, rpm)
            force_in_newton = force_in_grams * 9.81 / 1000.0
            return np.maximum(force_in_newton, 0)

        rpm = np.asarray(action.rpm, dtype=float)
        if self.motor_tau:
            # Rotor speed chases the commanded speed with a first-order lag.
            if self._rpm is None:
                self._rpm = rpm.copy()
            alpha = dt / (self.motor_tau + dt)
            self._rpm = self._rpm + alpha * (rpm - self._rpm)
            rpm = self._rpm

        force = rpm_to_force(rpm)

        # compute next state
        eta = np.dot(self.B0, force)
        f_u = np.array([0, 0, eta[0]])
        tau_u = np.array([eta[1], eta[2], eta[3]])

        # Rotor drag, body frame, opposing the body-frame velocity. Folded into f_u so it
        # reaches the accelerometer as well as the dynamics -- an IMU measures every force
        # except gravity, and a drag term that moved the vehicle without being felt would
        # show up as a spurious residual, which is the one quantity this thesis measures.
        if self.drag is not None:
            v_body = rowan.rotate(rowan.inverse(self.state.quat), self.state.vel)
            f_u = f_u - self.mass * self.drag * v_body

        # dynamics
        # dot{p} = v
        pos_next = self.state.pos + self.state.vel * dt
        # mv = mg + R f_u + f_a
        vel_next = self.state.vel + (
            np.array([0, 0, -self.g]) +
            (rowan.rotate(self.state.quat, f_u) + f_a) / self.mass) * dt

        # dot{R} = R S(w)
        # to integrate the dynamics, see
        # https://www.ashwinnarayan.com/post/how-to-integrate-quaternions/, and
        # https://arxiv.org/pdf/1604.08139.pdf
        # Sec 4.5, https://arxiv.org/pdf/1711.02508.pdf
        omega_global = rowan.rotate(self.state.quat, self.state.omega)
        q_next = rowan.normalize(
            rowan.calculus.integrate(
                self.state.quat, omega_global, dt))

        # mJ = Jw x w + tau_u
        omega_next = self.state.omega + (
            self.inv_J * (np.cross(self.J * self.state.omega, self.state.omega) + tau_u)) * dt

        # What an onboard IMU would measure: specific force, body frame, in g.
        # Gravity is excluded by construction (an accelerometer measures every force
        # EXCEPT gravity), so this is (thrust + f_a) / m expressed in the body frame.
        # f_a is the aerodynamic interaction force -- so with the neuralswarm backend
        # the simulated accelerometer genuinely feels the downwash, which is what
        # makes the residual a_meas - a_model measurable in simulation at all.
        acc_body = (f_u + rowan.rotate(rowan.inverse(self.state.quat), f_a)) \
            / (self.mass * self.g)

        self.state.pos = pos_next
        self.state.vel = vel_next
        self.state.quat = q_next
        self.state.omega = omega_next
        self.state.acc = acc_body
        self.state.rpm = np.asarray(rpm, dtype=float)   # post-lag, i.e. measured

        # if we fall below the ground, set velocities to 0
        if self.state.pos[2] < 0:
            self.state.pos[2] = 0
            self.state.vel = [0, 0, 0]
            self.state.omega = [0, 0, 0]
