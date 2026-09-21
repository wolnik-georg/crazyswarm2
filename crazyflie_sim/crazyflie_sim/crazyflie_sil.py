#!/usr/bin/env python3

"""
Crazyflie Software-In-The-Loop Wrapper that uses the firmware Python bindings.

    2022 - Wolfgang Hönig (TU Berlin)
"""
from __future__ import annotations

import csv
import os
import time

import cffirmware as firm
import numpy as np
import rowan

from . import sim_data_types

# 2026-09-18: opt-in per-tick debug log for the controller=7/8 real-SIL investigation
# (flying_robot_course docs/07, firmware_app/host/naindi_reference_build_notes.md
# "Investigation plan"). Logs every Python-level executeController() call attempt for
# oot2/oot3 -- both genuine 500 Hz Rust-side computes and the ones the Rust-side
# RATE_DO_EXECUTE gate silently no-ops -- so the real dt-between-computes, setpoint
# trajectory shape, and any x/y motion become directly observable instead of assumed. Off
# by default; set NAINDI_DEBUG_LOG=<path> to enable. Writes are append-only per process.
_naindi_debug_writer = None
_naindi_debug_file = None
_naindi_debug_last_thrust = None


def _naindi_debug_log(cf, tick):
    global _naindi_debug_writer, _naindi_debug_file, _naindi_debug_last_thrust
    path = os.environ.get('NAINDI_DEBUG_LOG')
    if not path:
        return
    if _naindi_debug_writer is None:
        _naindi_debug_file = open(path, 'w', newline='')
        _naindi_debug_writer = csv.writer(_naindi_debug_file)
        _naindi_debug_writer.writerow([
            'wall_time', 'tick', 'real_compute',
            'sp_x', 'sp_y', 'sp_z', 'sp_vx', 'sp_vy', 'sp_vz', 'sp_ax', 'sp_ay', 'sp_az',
            'st_x', 'st_y', 'st_z', 'st_vx', 'st_vy', 'st_vz',
            'gyro_x', 'gyro_y', 'gyro_z',
            'thrust', 'tau_x', 'tau_y', 'tau_z',
        ])
    thrust = cf.control.thrustSi
    # The Rust-side RATE_DO_EXECUTE-equivalent gate leaves control_t entirely unchanged on
    # a no-op tick -- an EXACT repeat of the previous thrust value (not just "close") is the
    # signature of a gated-out call, distinct from a genuine recompute that happens to
    # command similar thrust between ticks (float equality is intentional here, not sloppy).
    real_compute = (thrust != _naindi_debug_last_thrust)
    _naindi_debug_last_thrust = thrust
    sp, st, se, c = cf.setpoint, cf.state, cf.sensors, cf.control
    _naindi_debug_writer.writerow([
        time.time(), tick, int(real_compute),
        sp.position.x, sp.position.y, sp.position.z,
        sp.velocity.x, sp.velocity.y, sp.velocity.z,
        sp.acceleration.x, sp.acceleration.y, sp.acceleration.z,
        st.position.x, st.position.y, st.position.z,
        st.velocity.x, st.velocity.y, st.velocity.z,
        se.gyro.x, se.gyro.y, se.gyro.z,
        c.thrustSi, c.torqueX, c.torqueY, c.torqueZ,
    ])
    _naindi_debug_file.flush()


class TrajectoryPolynomialPiece:

    def __init__(self, poly_x, poly_y, poly_z, poly_yaw, duration):
        self.poly_x = poly_x
        self.poly_y = poly_y
        self.poly_z = poly_z
        self.poly_yaw = poly_yaw
        self.duration = duration


def copy_svec(v):
    return firm.mkvec(v.x, v.y, v.z)


class CrazyflieSIL:

    # Flight modes.
    _oot_count = 0
    # 2026-09-16: separate counters for controller=7/8 (naindi.rs / naindi_hybrid.rs).
    # 2026-09-18: each now has its own per-vehicle state-swap hook (naindi_select_drone /
    # naindi_hybrid_select_drone in oot_host.c), same shape as controllerOutOfTree's own
    # oot_select_drone -- these counters assign each vehicle its slot index (self._naindi_index),
    # same pattern as _oot_index below.
    _oot2_count = 0
    _oot3_count = 0

    MODE_IDLE = 0
    MODE_HIGH_POLY = 1
    MODE_LOW_FULLSTATE = 2
    MODE_LOW_POSITION = 3
    MODE_LOW_VELOCITY = 4

    def __init__(self, name, initialPosition, controller_name, time_func):
        # Core.
        self.name = name
        self.groupMask = 0
        self.initialPosition = np.array(initialPosition)
        self.time_func = time_func

        # Commander.
        self.mode = CrazyflieSIL.MODE_IDLE
        self.planner = firm.planner()
        firm.plan_init(self.planner)
        self.trajectories = {}

        # previous state for HL commander
        self.cmdHl_pos = firm.mkvec(*initialPosition)
        self.cmdHl_vel = firm.vzero()
        self.cmdHl_yaw = 0

        # current setpoint
        self.setpoint = firm.setpoint_t()

        # latest sensor values.
        self.state = firm.state_t()
        self.state.position.x = self.initialPosition[0]
        self.state.position.y = self.initialPosition[1]
        self.state.position.z = self.initialPosition[2]
        self.state.velocity.x = 0
        self.state.velocity.y = 0
        self.state.velocity.z = 0
        self.state.attitude.roll = 0
        self.state.attitude.pitch = -0  # WARNING: this is in the legacy coordinate system
        self.state.attitude.yaw = 0

        self.sensors = firm.sensorData_t()
        self.sensors.gyro.x = 0
        self.sensors.gyro.y = 0
        self.sensors.gyro.z = 0

        self.motors_rpm = [0.0, 0.0, 0.0, 0.0]
        self.kt = None  # set only by the 'oot' controller
        self.motors_rpm_meas = None   # actual rotor speed reported by the plant
        self.thrust_max = None
        self._last_ctrl_tick = -1
        self._last_action = None
        # Index into the simulator's per-vehicle controller state. The compiled
        # controller keeps its filters and integrators in one static -- correct on a
        # drone, where there is one vehicle per MCU, but every simulated drone runs
        # through that same static here. See oot_select_drone in oot_host.c.
        self._oot_index = CrazyflieSIL._oot_count
        # Positions of the other vehicles, refreshed by the server each tick. On the drone these
        # arrive through peer_localization from Crazyswarm2's pose broadcast; the simulator has
        # no such module, so they are injected instead. Empty means the residual network sees no
        # neighbours and predicts exactly zero -- which is correct for a single drone and wrong
        # for every formation, so the server must actually set this.
        self.peers = []
        # Host-side Kalman core: sim mocap -> extpose equivalent (estimator_kalman.c).
        self._kalman = firm.kalmanCoreData_t()
        self._kalman_params = firm.kalmanCoreParams_t()
        firm.kalmanCoreDefaultParams(self._kalman_params)
        self._kalman_ms = 0
        self._kalman_next_predict_ms = 0
        firm.kalmanCoreInit(self._kalman, self._kalman_params, 0)
        # stateEstimate.* published to ROS /state (pos + vel, 6 floats).
        self.state_estimate = (
            float(self.initialPosition[0]),
            float(self.initialPosition[1]),
            float(self.initialPosition[2]),
            0.0, 0.0, 0.0,
        )
        # Captured immediately after the controller runs, while the shared out-of-tree statics
        # still belong to THIS vehicle. Reading them later would return the last drone stepped.
        self.a_res = [0.0, 0.0, 0.0]
        self.rnn_pred = [0.0, 0.0, 0.0]
        CrazyflieSIL._oot_count += 1

        # current controller output
        self.control = firm.control_t()
        self.motors_thrust_uncapped = firm.motors_thrust_uncapped_t()
        self.motors_thrust_pwm = firm.motors_thrust_pwm_t()

        self.controller_name = controller_name

        # set up controller
        if controller_name == 'none':
            self.controller = None
        elif controller_name == 'pid':
            firm.controllerPidInit()
            self.controller = firm.controllerPid
        elif controller_name == 'mellinger':
            self.mellinger_control = firm.controllerMellinger_t()
            firm.controllerMellingerInit(self.mellinger_control)
            self.controller = firm.controllerMellinger
        elif controller_name == 'brescianini':
            firm.controllerBrescianiniInit()
            self.controller = firm.controllerBrescianini
        elif controller_name == 'lee':
            # Stock geometric SE(3) (controller_lee.c, mainline crazyflie-firmware --
            # ControllerTypeLee=5), wired in 2026-09-18 as a third-party sanity baseline for
            # the strategy comparison: isolates whether any issue with our own geometric
            # (controller=6, ctrl_mode=0) is specific to our implementation or general to the
            # control law. This is `cf_second`'s permanent real-hardware pin, never before run
            # through this SIL. Like Mellinger, `controllerLee()` takes an explicit `self`
            # struct (not a hidden static) -- the CRAZYFLIE_FW-gated `controllerLeeFirmware`
            # wrapper that normally hides this on real hardware isn't built for the host, so
            # the struct is carried here per-vehicle, same pattern as `mellinger_control`.
            self.lee_control = firm.controllerLee_t()
            firm.controllerLeeInit(self.lee_control)
            self.controller = firm.controllerLee
        elif controller_name == 'indi':
            # Bitcraze's own stock INDI (controller_indi.c + position_controller_indi.c,
            # mainline crazyflie-firmware -- ControllerTypeINDI=3), wired in 2026-09-18 as a
            # third reference point for the controller=7 SIL investigation (docs/07). A
            # structurally DIFFERENT INDI from both this project's own (lib.rs, Tal & Karaman)
            # and naindi.rs (Cobo-Briesewitz's controller_lee.c port): pure gyro-differentiation,
            # no RPM feedback at all -- confirmed by grep, no rpm/Rpm/RPM/motorsGetRatio symbol
            # anywhere in controller_indi.c. Self-contained from sensors/state/setpoint alone,
            # same shape as 'pid'/'mellinger'/'brescianini' above -- no injection plumbing
            # needed, unlike 'oot'/'oot2'/'oot3'. Flew clean on real hardware (operator report,
            # 2026-09-17) with no oscillation, just worse tracking than either INDI above.
            firm.controllerINDIInit()
            self.controller = firm.controllerINDI
        elif controller_name == 'oot':
            # The thesis controller: the SAME geometric SE(3) / INDI Rust source that
            # flies on the drone (flying_drone_stack/firmware_app), compiled for the
            # host and linked into cffirmware -- not a reimplementation. Which law runs
            # is indi_gains.ctrl_mode, exactly as on hardware:
            #   0 geometric | 1 position INDI | 2 attitude INDI | 3 full INDI
            # Gains come from traj_iface.c's defaults and can be overridden through
            # firm.cvar.g_indi_* / g_kp_* the same way crazyflies.yaml does over CRTP.
            if not hasattr(firm, 'controllerOutOfTree'):
                raise ValueError(
                    "controller 'oot' needs cffirmware built with the out-of-tree "
                    'controller. Build it with:\n'
                    '  cd flying_drone_stack/firmware_app && RUSTFLAGS="-C panic=abort" \\\n'
                    '      cargo build --release --target x86_64-unknown-linux-gnu\n'
                    '  cd crazyflie-firmware && make bindings_python')
            firm.controllerOutOfTreeInit()
            self.controller = firm.controllerOutOfTree
            # Thrust constants the controller inverts. Used below to make the
            # PWM -> RPM -> force chain self-consistent; see pwm_to_rpm.
            self.kt = [firm.cvar.g_indi_kt1, firm.cvar.g_indi_kt2,
                       firm.cvar.g_indi_kt3, firm.cvar.g_indi_kt4]
            self.thrust_max = firm.oot_thrust_max()
        elif controller_name in ('oot2', 'oot3'):
            # controller=7 (naindi.rs, use_nn=0) / controller=8 (naindi_hybrid.rs,
            # use_nn=1): faithful ports of Cobo & Briesewitz's NA-INDI, numerically
            # verified against their own compiled controller_lee.c (see
            # firmware_app/host/test_naindi_reference.py / test_naindi_hybrid_
            # reference.py) but, until now, never run through this closed-loop
            # simulator -- only through hand-built static test vectors. Both read
            # g_indi_mass/g_indi_kt1-4 for their position-INDI term, same as 'oot',
            # so the plant is still built from the firmware's own constants (see
            # crazyflie_server.py's _setup_oot).
            attr = 'controllerOutOfTree2' if controller_name == 'oot2' else 'controllerOutOfTree3'
            if not hasattr(firm, attr):
                raise ValueError(
                    "controller '{}' needs cffirmware built with that out-of-tree "
                    'controller. Build it with:\n'
                    '  cd flying_drone_stack/firmware_app && RUSTFLAGS="-C panic=abort" \\\n'
                    '      cargo build --release --target x86_64-unknown-linux-gnu\n'
                    '  cd crazyflie-firmware && make bindings_python'.format(controller_name))
            # 2026-09-18: naindi.rs/naindi_hybrid.rs each now have their own state-swap hook
            # (naindi_select_drone / naindi_hybrid_select_drone in oot_host.c, mirroring
            # controllerOutOfTree's oot_select_drone -- see that file's oot_swap_select).
            # This index is this vehicle's slot in that controller's own swap pool, same
            # pattern as _oot_index above.
            count_attr = '_oot2_count' if controller_name == 'oot2' else '_oot3_count'
            self._naindi_index = getattr(CrazyflieSIL, count_attr)
            setattr(CrazyflieSIL, count_attr, getattr(CrazyflieSIL, count_attr) + 1)
            getattr(firm, attr + 'Init')()
            # 2026-09-17: opt-in test of the inertia-mismatch hypothesis for the CS2 SIL
            # divergence found 2026-09-16 (docs/07 History (39)/(40)) -- KR/KOMEGA are the
            # reference's own torque gains, tuned for their J, applied unchanged to this
            # project's real, ~44.5%/43.8%/10.6% (x/y/z) heavier CF21BL inertia. Scaling both
            # gains by the same per-axis J_real/J_ref ratio restores the reference's intended
            # omega_n AND damping ratio (docs/22 §2h-style analysis, worked out for this
            # controller in naindi.rs's GAIN_TEST_OVERRIDE comment). Off by default -- the
            # numerically-verified-to-1e-9 reference gains are untouched unless this env var
            # is set, so nothing about the default byte-for-bit port changes.
            if os.environ.get('NAINDI_SCALED_GAINS') == '1':
                setter = 'naindi_test_set_gains' if controller_name == 'oot2' \
                    else 'naindi_hybrid_test_set_gains'
                getattr(firm, setter)(0.010117, 0.010066, 0.011054,
                                       0.0028906, 0.0028760, 0.0022109)
            # 2026-09-18: diagnostic only, NOT a proposed flying config -- naindi.rs's
            # POS_GAIN_TEST_OVERRIDE comment. By the time this was added, mass/kt/arm/t2t/J/
            # KR/KOMEGA/KPOS_P/KPOS_D/KPOS_I had ALL already been set to the reference
            # authors' own values in one run and it still crashed (docs/07, 2026-09-17), so
            # there's no "which number is wrong" question left -- this scales KPOS_P/KPOS_D/
            # KPOS_I together by NAINDI_POS_GAIN_SCALE (env var, float; e.g. "0.25") to test
            # whether the ~1.4s oscillation period found in that run tracks the position
            # gain the way a real position-loop resonance would. controller=8 has no
            # equivalent setter yet (naindi_hybrid_test_set_pos_gains doesn't exist).
            pos_scale = os.environ.get('NAINDI_POS_GAIN_SCALE')
            if pos_scale is not None and controller_name == 'oot2':
                f = float(pos_scale)
                kp, kd, ki = 12.0 * f, 10.5 * f, 2.0 * f
                firm.naindi_test_set_pos_gains(kp, kp, kp, kd, kd, kd, ki, ki, ki)
                # CrazyflieSIL has no get_logger() (that's the server Node's method, not
                # this per-vehicle class's) -- plain print is what the rest of this
                # __init__ path uses for anything printed before the node exists.
                print('NAINDI_POS_GAIN_SCALE=%s: KPOS_P/D/I -> %.3f/%.3f/%.3f '
                      '(reference values x%s)' % (pos_scale, kp, kd, ki, pos_scale))
            self.controller = getattr(firm, attr)
            self.kt = [firm.cvar.g_indi_kt1, firm.cvar.g_indi_kt2,
                       firm.cvar.g_indi_kt3, firm.cvar.g_indi_kt4]
            self.thrust_max = firm.oot_thrust_max()
        else:
            raise ValueError('Unknown controller {}'.format(controller_name))

    def setGroupMask(self, groupMask):
        self.groupMask = groupMask

    def takeoff(self, targetHeight, duration, groupMask=0):
        if self._isGroup(groupMask):
            self.mode = CrazyflieSIL.MODE_HIGH_POLY
            targetYaw = 0.0
            firm.plan_takeoff(
                self.planner,
                self.cmdHl_pos,
                self.cmdHl_yaw,
                targetHeight, targetYaw, duration, self.time_func())

    def land(self, targetHeight, duration, groupMask=0):
        if self._isGroup(groupMask):
            self.mode = CrazyflieSIL.MODE_HIGH_POLY
            targetYaw = 0.0
            firm.plan_land(
                self.planner,
                self.cmdHl_pos,
                self.cmdHl_yaw,
                targetHeight, targetYaw, duration, self.time_func())

    # def stop(self, groupMask = 0):
    #     if self._isGroup(groupMask):
    #         self.mode = CrazyflieSIL.MODE_IDLE
    #         firm.plan_stop(self.planner)

    def goTo(self, goal, yaw, duration, relative=False, groupMask=0):
        if self._isGroup(groupMask):
            if self.mode != CrazyflieSIL.MODE_HIGH_POLY:
                # We need to update to the latest firmware that has go_to_from.
                raise ValueError('goTo from low-level modes not yet supported.')
            self.mode = CrazyflieSIL.MODE_HIGH_POLY
            try:
                firm.plan_go_to(
                    self.planner,
                    relative,
                    False,
                    firm.mkvec(*goal),
                    yaw, duration, self.time_func())
            except TypeError:
                message = ('Warning: Your Crazyflie firmware is outdated. '
                           ' Please update to the latest version.')
                print(message)
                firm.plan_go_to(
                    self.planner,
                    relative,
                    firm.mkvec(*goal),
                    yaw, duration, self.time_func())

    def uploadTrajectory(self,
                         trajectoryId: int,
                         pieceOffset: int,
                         pieces: list[TrajectoryPolynomialPiece]):
        traj = firm.piecewise_traj()
        traj.t_begin = 0
        traj.timescale = 1.0
        traj.shift = firm.mkvec(0, 0, 0)
        traj.n_pieces = len(pieces)
        traj.pieces = firm.poly4d_malloc(traj.n_pieces)
        for i, piece in enumerate(pieces):
            fwpiece = firm.piecewise_get(traj, i)
            fwpiece.duration = piece.duration
            for coef in range(0, 8):
                firm.poly4d_set(fwpiece, 0, coef, piece.poly_x[coef])
                firm.poly4d_set(fwpiece, 1, coef, piece.poly_y[coef])
                firm.poly4d_set(fwpiece, 2, coef, piece.poly_z[coef])
                firm.poly4d_set(fwpiece, 3, coef, piece.poly_yaw[coef])
        self.trajectories[trajectoryId] = traj

    def startTrajectory(self,
                        trajectoryId: int,
                        timescale: float = 1.0,
                        reverse: bool = False,
                        relative: bool = True,
                        groupMask: int = 0):
        if self._isGroup(groupMask):
            self.mode = CrazyflieSIL.MODE_HIGH_POLY
            traj = self.trajectories[trajectoryId]
            traj.t_begin = self.time_func()
            traj.timescale = timescale
            startfrom = self.cmdHl_pos
            # Signature fix (2026-08-22): this firmware's plan_start_trajectory takes 7 args
            #   (planner, traj, reversed, relative_position, relative_yaw, start_from, start_yaw)
            # while upstream crazyflie_sim still calls the older 5-arg form, which put
            # `startfrom` in the relative_yaw slot -> TypeError, killing the sim server the
            # moment a trajectory started. Argument order mirrors the real firmware call in
            # crtp_commander_high_level.c:750, which passes relative_yaw=false.
            firm.plan_start_trajectory(self.planner, traj, reverse, relative, False,
                                       startfrom, self.cmdHl_yaw)

    # def notifySetpointsStop(self, remainValidMillisecs=100):
    #     # No-op - the real Crazyflie prioritizes streaming setpoints over
    #     # high-level commands. This tells it to stop doing that. We don't
    #     # simulate this behavior.
    #     pass

    def cmdFullState(self, pos, vel, acc, yaw, omega):
        self.mode = CrazyflieSIL.MODE_LOW_FULLSTATE
        self.setpoint.position.x = pos[0]
        self.setpoint.position.y = pos[1]
        self.setpoint.position.z = pos[2]
        self.setpoint.velocity.x = vel[0]
        self.setpoint.velocity.y = vel[1]
        self.setpoint.velocity.z = vel[2]
        self.setpoint.attitude.yaw = np.degrees(yaw)
        self.setpoint.attitudeRate.roll = np.degrees(omega[0])
        self.setpoint.attitudeRate.pitch = np.degrees(omega[1])
        self.setpoint.attitudeRate.yaw = np.degrees(omega[2])
        self.setpoint.mode.x = firm.modeAbs
        self.setpoint.mode.y = firm.modeAbs
        self.setpoint.mode.z = firm.modeAbs
        self.setpoint.mode.roll = firm.modeDisable
        self.setpoint.mode.pitch = firm.modeDisable
        self.setpoint.mode.yaw = firm.modeAbs
        self.setpoint.mode.quat = firm.modeDisable
        self.setpoint.acceleration.x = acc[0]
        self.setpoint.acceleration.y = acc[1]
        self.setpoint.acceleration.z = acc[2]

        self.cmdHl_pos = copy_svec(self.setpoint.position)
        self.cmdHl_vel = copy_svec(self.setpoint.velocity)
        self.cmdHl_yaw = yaw

    # def cmdPosition(self, pos, yaw = 0):
    #     self.mode = CrazyflieSIL.MODE_LOW_POSITION
    #     self.setState.pos = firm.mkvec(*pos)
    #     self.setState.yaw = yaw
    #     # TODO: should we set vel, acc, omega to zero, or rely on modes to not read them?

    # def cmdVelocityWorld(self, vel, yawRate):
    #     self.mode = CrazyflieSIL.MODE_LOW_VELOCITY
    #     self.setState.vel = firm.mkvec(*vel)
    #     self.setState.omega = firm.mkvec(0.0, 0.0, yawRate)
    #     # TODO: should we set pos, acc, yaw to zero, or rely on modes to not read them?

    # def cmdStop(self):
    #     # TODO: set mode to MODE_IDLE?
    #     pass

    def getSetpoint(self):
        if self.mode == CrazyflieSIL.MODE_HIGH_POLY:
            # See logic in crtp_commander_high_level.c
            ev = firm.plan_current_goal(self.planner, self.time_func())
            if firm.is_traj_eval_valid(ev):
                self.setpoint.position.x = ev.pos.x
                self.setpoint.position.y = ev.pos.y
                self.setpoint.position.z = ev.pos.z
                self.setpoint.velocity.x = ev.vel.x
                self.setpoint.velocity.y = ev.vel.y
                self.setpoint.velocity.z = ev.vel.z
                self.setpoint.attitude.yaw = np.degrees(ev.yaw)
                self.setpoint.attitudeRate.roll = np.degrees(ev.omega.x)
                self.setpoint.attitudeRate.pitch = np.degrees(ev.omega.y)
                self.setpoint.attitudeRate.yaw = np.degrees(ev.omega.z)
                self.setpoint.mode.x = firm.modeAbs
                self.setpoint.mode.y = firm.modeAbs
                self.setpoint.mode.z = firm.modeAbs
                self.setpoint.mode.roll = firm.modeDisable
                self.setpoint.mode.pitch = firm.modeDisable
                self.setpoint.mode.yaw = firm.modeAbs
                self.setpoint.mode.quat = firm.modeDisable
                self.setpoint.acceleration.x = ev.acc.x
                self.setpoint.acceleration.y = ev.acc.y
                self.setpoint.acceleration.z = ev.acc.z

                # Jerk and snap, exactly as crtp_commander_high_level.c does on the drone
                # (see setpoint->jerk there). traj_eval carries both and the setpoint
                # struct has fields for both, but this function used to stop at
                # acceleration -- so in simulation a flatness-based controller received a
                # permanently zero jerk. Position and velocity still looked right, which
                # is what makes it hard to spot: the attitude FEEDFORWARD is what breaks.
                # A straight line has almost no jerk and flies fine; a circle is nothing
                # but jerk and diverges. Measured on the stock circle trajectory: 139 deg
                # of tilt with jerk dropped, against 27 deg for the in-tree controllers
                # that do not use it.
                for src, dst in ((ev.jerk, self.setpoint.jerk),
                                 (ev.snap, self.setpoint.snap)):
                    dst.x, dst.y, dst.z = src.x, src.y, src.z

                self.cmdHl_pos = copy_svec(ev.pos)
                self.cmdHl_vel = copy_svec(ev.vel)
                self.cmdHl_yaw = ev.yaw

        return self._fwsetpoint_to_sim_data_types_state(self.setpoint)

        # # else:
        #     # return self._fwstate_to_sim_data_types_state(self.setState)
        # setState = firm.traj_eval(self.setState)
        # if not firm.is_traj_eval_valid(setState):
        #     return self._fwstate_to_sim_data_types_state(self.state)

        # if self.mode == CrazyflieSIL.MODE_IDLE:
        #     return self._fwstate_to_sim_data_types_state(self.state)

        # self.state = setState
        # return self._fwstate_to_sim_data_types_state(setState)

    def setState(self, state: sim_data_types.State):
        self.state.position.x = state.pos[0]
        self.state.position.y = state.pos[1]
        self.state.position.z = state.pos[2]

        self.state.velocity.x = state.vel[0]
        self.state.velocity.y = state.vel[1]
        self.state.velocity.z = state.vel[2]

        rpy = np.degrees(rowan.to_euler(state.quat, convention='xyz'))
        # Note, legacy coordinate system, so invert pitch
        self.state.attitude.roll = rpy[0]
        self.state.attitude.pitch = -rpy[1]
        self.state.attitude.yaw = rpy[2]

        self.state.attitudeQuaternion.w = state.quat[0]
        self.state.attitudeQuaternion.x = state.quat[1]
        self.state.attitudeQuaternion.y = state.quat[2]
        self.state.attitudeQuaternion.z = state.quat[3]

        # omega is part of sensors, not of the state
        self.sensors.gyro.x = np.degrees(state.omega[0])
        self.sensors.gyro.y = np.degrees(state.omega[1])
        self.sensors.gyro.z = np.degrees(state.omega[2])

        # gyroNoLpf: the sim applies no software LPF to begin with, so the unfiltered
        # signal controller=7 (naindi.rs) reads is the same ground-truth omega as gyro.
        self.sensors.gyroNoLpf.x = self.sensors.gyro.x
        self.sensors.gyroNoLpf.y = self.sensors.gyro.y
        self.sensors.gyroNoLpf.z = self.sensors.gyro.z

        # Actual rotor speeds, if this backend reports them. Backends that do not still
        # fall back to the commanded value below, so nothing changes for them.
        rpm = getattr(state, 'rpm', None)
        if rpm is not None and np.any(rpm):
            self.motors_rpm_meas = list(rpm)

        # Accelerometer. INDI inverts the measured acceleration, so without this it
        # reads a permanent free-fall and the controller is not the one that flies.
        self.sensors.acc.x = state.acc[0]
        self.sensors.acc.y = state.acc[1]
        self.sensors.acc.z = state.acc[2]

        # 2026-09-18: REAL BUG FIX -- self.state.acc (state_t's own field) was never set
        # anywhere in this class, only self.sensors.acc above. naindi.rs (controller=7)
        # reads state->acc, NOT sensors->acc, for its position-INDI residual term (a_imu) --
        # confirmed directly in naindi.rs (module doc + line ~389, `let acc = &st.acc`).
        # So a_imu was ALWAYS EXACTLY ZERO for the entire flight, every controller=7/8 SIL
        # run to date, making the INDI residual (a_res = a_imu - a_rpm) equal -a_rpm instead
        # of a genuine measured-vs-modeled comparison -- roughly constant (harmless) during
        # steady hover, but large and dynamically varying whenever commanded thrust changes
        # (climb, landing), exactly the failure window every crash to date was observed in.
        # Root-caused via firmware_app/host/naindi_reference_closed_loop.py's
        # --zero-state-acc flag, which reproduces this exact bug standalone and diverges at
        # the same ~10s mark a real crashing SIL run does (docs/07, 2026-09-18).
        #
        # state->acc's convention (controller_lee.c: a_imu = 9.81 * state->acc, compared
        # directly against a_rpm = f_thrust_world/mass - g_vec, i.e. world-frame,
        # gravity-EXCLUDED, reads (0,0,0) at hover) differs from sensors.acc's convention
        # (body-frame specific force, gravity implicitly present via the thrust reaction,
        # reads (0,0,1) at hover) -- rotate into world frame, then subtract the
        # gravity-cancelling (0,0,1) hover offset to get the gravity-excluded quantity
        # state->acc actually means.
        acc_world = rowan.rotate(state.quat, state.acc)
        self.state.acc.x = acc_world[0]
        self.state.acc.y = acc_world[1]
        self.state.acc.z = acc_world[2] - 1.0

        self._sync_kalman_from_mocap(int(self.time_func() * 1000))

    def reset_kalman_estimator(self, now_ms=None):
        """Hardware kalman.resetEstimation + complementary re-init equivalent."""
        if now_ms is None:
            now_ms = int(self.time_func() * 1000)
        firm.kalmanCoreInit(self._kalman, self._kalman_params, now_ms)
        self._kalman_ms = now_ms
        self._kalman_next_predict_ms = now_ms
        self._sync_kalman_from_mocap(now_ms)

    def _sync_kalman_from_mocap(self, now_ms: int):
        """Feed ground-truth pose into the linked firmware Kalman core each tick."""
        self._kalman_ms = now_ms
        acc = self.sensors.acc
        gyro = self.sensors.gyro
        flying = self.mode != CrazyflieSIL.MODE_IDLE
        if now_ms >= self._kalman_next_predict_ms:
            firm.kalmanCorePredict(
                self._kalman, self._kalman_params, acc, gyro, now_ms, flying)
            self._kalman_next_predict_ms = now_ms + 10
        firm.kalmanCoreAddProcessNoise(self._kalman, self._kalman_params, now_ms)

        pose = firm.poseMeasurement_t()
        pose.x = self.state.position.x
        pose.y = self.state.position.y
        pose.z = self.state.position.z
        q = self.state.attitudeQuaternion
        pose.quat.w = q.w
        pose.quat.x = q.x
        pose.quat.y = q.y
        pose.quat.z = q.z
        pose.stdDevPos = 0.005
        pose.stdDevQuat = 0.01
        firm.kalmanCoreUpdateWithPose(self._kalman, pose)
        firm.kalmanCoreFinalize(self._kalman)

        est = firm.state_t()
        firm.kalmanCoreExternalizeState(self._kalman, est, acc)
        # Tight mocap fusion: publish physics truth as stateEstimate (matches hardware
        # when extpose has converged). Externalize can drift in host-only acc convention;
        # gate and logs must reflect the pose the Kalman was fed.
        self.state_estimate = (
            float(self.state.position.x),
            float(self.state.position.y),
            float(self.state.position.z),
            float(self.state.velocity.x),
            float(self.state.velocity.y),
            float(self.state.velocity.z),
        )

    def executeController(self):
        if self.controller is None:
            return None

        if self.mode == CrazyflieSIL.MODE_IDLE:
            return sim_data_types.Action([0, 0, 0, 0])

        time_in_seconds = self.time_func()
        # ticks is essentially the time in milliseconds as an integer
        tick = int(time_in_seconds * 1000)

        # The backend substeps at 2 kHz but the tick above only advances every second
        # call, and the firmware derives its dt from the tick delta -- so half the
        # invocations would run with dt = 0. A controller that differentiates (INDI
        # takes the angular acceleration from the gyro) is destroyed by that; on the
        # ROS side full INDI never left the ground while geometric, which does not
        # differentiate, looked fine. Hardware does not work this way: the stabilizer
        # loop is discrete at 1 kHz while the physics is continuous. So run the
        # controller once per millisecond and hold the command over the substeps.
        # Only the out-of-tree controller is affected -- every other controller keeps
        # its original call pattern so existing simulation results still reproduce.
        if self.controller_name in ('oot', 'oot2', 'oot3'):
            if tick == self._last_ctrl_tick and self._last_action is not None:
                return self._last_action
            self._last_ctrl_tick = tick

        if self.controller_name == 'oot':
            # Hand the controller this vehicle's own state before it runs.
            firm.oot_select_drone(self._oot_index)
            # Attitude INDI derives tau_current from measured RPM^2. On the drone that
            # comes from the RPM deck / DShot telemetry; here the simulator knows the
            # true motor speeds, so hand them over. Without this the controller silently
            # falls back to tau_prev and the INDI being tested is the degraded variant.
            # Prefer the MEASURED rotor speed over the commanded one. With actuator lag
            # in the plant the two differ, and attitude INDI reconstructs its applied
            # torque from this number: feeding it the command makes the reconstruction
            # wrong by exactly the lag, the increment is computed against a torque that
            # was never applied, and the vehicle does not lift off at all. Geometric
            # control never touches RPM, which is why it is unaffected.
            r = self.motors_rpm_meas or getattr(self, 'motors_rpm', [0, 0, 0, 0])
            firm.oot_set_rpm(int(r[0]), int(r[1]), int(r[2]), int(r[3]))
            # Peer positions for the residual network, timestamped with the same tick the
            # controller sees. The firmware differences relative velocity from consecutive
            # timestamps, so the stamp has to advance exactly as it does in flight.
            if hasattr(firm, 'oot_set_peer'):
                for k, p in enumerate(self.peers[:3]):
                    firm.oot_set_peer(k, float(p[0]), float(p[1]), float(p[2]), int(tick))
                firm.oot_set_peer_count(min(len(self.peers), 3))
            self.controller(self.control, self.setpoint, self.sensors, self.state, tick)
            if hasattr(firm, 'oot_get_a_res'):
                self.a_res = [firm.oot_get_a_res(i) for i in range(3)]
                self.rnn_pred = [firm.cvar.g_rnn_pred_x, firm.cvar.g_rnn_pred_y,
                                 firm.cvar.g_rnn_pred_z]
        elif self.controller_name in ('oot2', 'oot3'):
            # 2026-09-18: hand the controller this vehicle's own state, same pattern as
            # 'oot' above (firm.oot_select_drone).
            if self.controller_name == 'oot2':
                firm.naindi_select_drone(self._naindi_index)
            else:
                firm.naindi_hybrid_select_drone(self._naindi_index)
            # Same RPM injection as 'oot' above, and for the same reason -- attitude
            # INDI derives tau_current from measured RPM^2 and silently falls back to
            # tau_prev without it. rpm_get_all() is a single global shared by every
            # out-of-tree controller (not per-vehicle-swapped like controllerOutOfTree's
            # own static) -- unlike the two controller-state swaps above, this one global
            # is intentionally shared, since each vehicle sets it immediately before its
            # own controller() call and nothing reads it in between.
            r = self.motors_rpm_meas or getattr(self, 'motors_rpm', [0, 0, 0, 0])
            firm.oot_set_rpm(int(r[0]), int(r[1]), int(r[2]), int(r[3]))
            # controller=8 only: its NN reads commanded PWM ratio (motorsGetRatio), not
            # RPM -- see naindi_hybrid.rs's module doc. The previous tick's own PWM
            # command is the SIL's equivalent of "commanded", same pattern as the RPM
            # injection above (last tick's output, not this tick's not-yet-computed one).
            if self.controller_name == 'oot3' and hasattr(firm, 'oot_set_pwm_ratio'):
                pwm = self.motors_thrust_pwm
                firm.oot_set_pwm_ratio(int(pwm.motors.m1), int(pwm.motors.m2),
                                        int(pwm.motors.m3), int(pwm.motors.m4))
            self.controller(self.control, self.setpoint, self.sensors, self.state, tick)
            _naindi_debug_log(self, tick)
        elif self.controller_name == 'mellinger':
            self.controller(
                self.mellinger_control,
                self.control,
                self.setpoint,
                self.sensors,
                self.state,
                tick)
        elif self.controller_name == 'lee':
            self.controller(
                self.lee_control,
                self.control,
                self.setpoint,
                self.sensors,
                self.state,
                tick)
        else:
            self.controller(self.control, self.setpoint, self.sensors, self.state, tick)
        self._last_action = self._fwcontrol_to_sim_data_types_action()
        return self._last_action

    # 'private' methods
    def _isGroup(self, groupMask):
        return groupMask == 0 or (self.groupMask & groupMask) > 0

    def _fwcontrol_to_sim_data_types_action(self):

        firm.powerDistribution(self.control, self.motors_thrust_uncapped)
        firm.powerDistributionCap(self.motors_thrust_uncapped, self.motors_thrust_pwm)

        # self.motors_thrust_pwm.motors.m{1,4} contain the PWM
        # convert PWM -> RPM
        def pwm_to_force(pwm):
            if self.kt is not None:
                # Exact inverse of powerDistributionForceTorque, which does
                # pwm = force / THRUST_MAX * UINT16_MAX. Using the CF2.0 system-id
                # polynomial here instead would deliver roughly a third of the
                # commanded thrust, which every controller then shows as the same
                # large steady-state height droop -- a plant error that reads as a
                # tuning problem.
                return pwm / 65535.0 * self.thrust_max
            # polyfit using data and scripts from https://github.com/IMRCLab/crazyflie-system-id
            p = [1.71479058e-09,  8.80284482e-05, -2.21152097e-01]
            force_in_grams = np.polyval(p, pwm)
            force_in_newton = force_in_grams * 9.81 / 1000.0
            return np.maximum(force_in_newton, 0)

        def pwm_to_rpm(pwm, i=0):
            if self.kt is not None:
                # Exact inverse of the force model above, so the plant produces the force
                # powerDistribution asked for and the RPM handed back to INDI agrees with
                # it -- one force model end to end.
                #
                # Deliberately NO idle deadband on this path. The 10000-PWM cutoff below
                # exists because the CF2.0 system-id polynomial is not valid at low PWM;
                # the kt inversion is exact down to zero. Applying the cutoff here zeroes
                # a motor that was legitimately given a small force, which loses thrust
                # AND delivers more torque than commanded -- measured at +21% for a
                # 10 mNm request. An attitude loop whose gain is a fifth higher than the
                # controller believes, by an amount that grows with the command, turns an
                # adequately damped cascade into an oscillating one.
                return float(np.sqrt(pwm_to_force(pwm) / self.kt[i]))
            # polyfit using data and scripts from https://github.com/IMRCLab/crazyflie-system-id
            if pwm < 10000:
                return 0
            p = [3.26535711e-01, 3.37495115e+03]
            return np.polyval(p, pwm)

        # Latch the resulting motor speeds. The out-of-tree controller reads these back
        # on the next step as its RPM measurement -- the simulator's stand-in for the
        # RPM deck / DShot telemetry. One step of delay is the honest model: on
        # hardware the measurement also lags the command.
        self.motors_rpm = [pwm_to_rpm(self.motors_thrust_pwm.motors.m1, 0),
                           pwm_to_rpm(self.motors_thrust_pwm.motors.m2, 1),
                           pwm_to_rpm(self.motors_thrust_pwm.motors.m3, 2),
                           pwm_to_rpm(self.motors_thrust_pwm.motors.m4, 3)]

        return sim_data_types.Action(list(self.motors_rpm))

    @staticmethod
    def _fwsetpoint_to_sim_data_types_state(fwsetpoint):
        pos = np.array([fwsetpoint.position.x, fwsetpoint.position.y, fwsetpoint.position.z])
        vel = np.array([fwsetpoint.velocity.x, fwsetpoint.velocity.y, fwsetpoint.velocity.z])
        acc = np.array([
            fwsetpoint.acceleration.x,
            fwsetpoint.acceleration.y,
            fwsetpoint.acceleration.z])
        omega = np.radians(np.array([
            fwsetpoint.attitudeRate.roll,
            fwsetpoint.attitudeRate.pitch,
            fwsetpoint.attitudeRate.yaw]))

        if fwsetpoint.mode.quat == firm.modeDisable:
            # compute rotation based on differential flatness
            thrust = acc + np.array([0, 0, 9.81])
            z_body = thrust / np.linalg.norm(thrust)
            yaw = np.radians(fwsetpoint.attitude.yaw)
            x_world = np.array([np.cos(yaw), np.sin(yaw), 0])
            y_body = np.cross(z_body, x_world)
            # Mathematically not needed. This addresses numerical issues to ensure R is orthogonal
            y_body /= np.linalg.norm(y_body)
            x_body = np.cross(y_body, z_body)
            # Mathematically not needed. This addresses numerical issues to ensure R is orthogonal
            x_body /= np.linalg.norm(x_body)
            R = np.column_stack([x_body, y_body, z_body])
            quat = rowan.from_matrix(R)
        else:
            quat = fwsetpoint.attitudeQuaternion

        return sim_data_types.State(pos, vel, quat, omega)
