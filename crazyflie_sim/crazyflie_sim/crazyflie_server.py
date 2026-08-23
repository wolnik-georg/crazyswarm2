#!/usr/bin/env python3

"""
A crazyflie server for simulation.

    2022 - Wolfgang Hönig (TU Berlin)
    2025 - Updated by Kimberly N. McGuire (Independent)
"""

from functools import partial
import importlib
import os

from crazyflie_interfaces.msg import FullState, Hover
from crazyflie_interfaces.srv import GoTo, Land, Takeoff
from crazyflie_interfaces.srv import NotifySetpointsStop, StartTrajectory, UploadTrajectory
from geometry_msgs.msg import Twist
import rclpy
from rclpy.node import Node
import rowan
from std_msgs.msg import String
from std_srvs.srv import Empty


# import BackendRviz from .backend_rviz
# from .backend import *
# from .backend.none import BackendNone
from .crazyflie_sil import CrazyflieSIL, TrajectoryPolynomialPiece
from .sim_data_types import State


class CrazyflieServer(Node):

    def __init__(self):
        super().__init__(
            'crazyflie_server',
            allow_undeclared_parameters=True,
            automatically_declare_parameters_from_overrides=True,
        )

        # Turn ROS parameters into a dictionary
        self._ros_parameters = self._param_to_dict(self._parameters)
        self.cfs = {}

        world_tf_name = 'world'
        robot_yaml_version = 0

        try:
            robot_yaml_version = self._ros_parameters['fileversion']
        except KeyError:
            self.get_logger().info('No fileversion found in crazyflies.yaml, assuming version 0')

        robot_data = self._ros_parameters['robots']

        # Parse robots
        names = []
        initial_states = []
        reference_frames = []
        for cfname in robot_data:
            if robot_data[cfname]['enabled']:
                type_cf = robot_data[cfname]['type']
                # do not include virtual objects
                connection = self._ros_parameters['robot_types'][type_cf].get(
                    'connection', 'crazyflie')
                if connection == 'crazyflie':
                    names.append(cfname)
                    pos = robot_data[cfname]['initial_position']
                    initial_states.append(State(pos))
                    # Get the current reference frame for the robot
                    reference_frame = world_tf_name
                    if robot_yaml_version >= 3:
                        try:
                            reference_frame = self._ros_parameters['all']['reference_frame']
                        except KeyError:
                            pass
                        try:
                            reference_frame = self._ros_parameters['robot_types'][
                                robot_data[cfname]['type']]['reference_frame']
                        except KeyError:
                            pass
                        try:
                            reference_frame = self._ros_parameters['robots'][
                                cfname]['reference_frame']
                        except KeyError:
                            pass
                    reference_frames.append(reference_frame)

        # Configure the out-of-tree controller BEFORE the backend exists, because the
        # backend builds its vehicle model from these numbers.
        self._oot_active = self._ros_parameters['sim'].get('controller') == 'oot'
        if self._oot_active:
            self._setup_oot()

        # initialize backend by dynamically loading the module
        backend_name = self._ros_parameters['sim']['backend']
        module = importlib.import_module(
            '.backend.' + backend_name, package='crazyflie_sim'
        )
        class_ = getattr(module, 'Backend')
        self.backend = class_(self, names, initial_states)

        # initialize visualizations by dynamically loading the modules
        self.visualizations = []
        for vis_key in self._ros_parameters['sim']['visualizations']:
            if self._ros_parameters['sim']['visualizations'][vis_key]['enabled']:
                module = importlib.import_module(
                    '.visualization.' + str(vis_key), package='crazyflie_sim'
                )
                class_ = getattr(module, 'Visualization')
                if vis_key == 'rviz':
                    # special case for rviz, which needs the reference frames
                    vis = class_(
                        self,
                        self._ros_parameters['sim']['visualizations'][vis_key],
                        names,
                        initial_states,
                        reference_frames
                    )
                else:
                    vis = class_(
                        self,
                        self._ros_parameters['sim']['visualizations'][vis_key],
                        names,
                        initial_states
                    )
                self.visualizations.append(vis)

        controller_name = backend_name = self._ros_parameters['sim']['controller']

        # create robot SIL objects
        for name, initial_state in zip(names, initial_states):
            self.cfs[name] = CrazyflieSIL(
                name,
                initial_state.pos,
                controller_name,
                self.backend.time)

        self._init_residual_log()

        for name, _ in self.cfs.items():
            pub = self.create_publisher(
                    String,
                    name + '/robot_description',
                    rclpy.qos.QoSProfile(
                        depth=1,
                        durability=rclpy.qos.QoSDurabilityPolicy.TRANSIENT_LOCAL))

            msg = String()
            msg.data = self._ros_parameters['robot_description'].replace('$NAME', name)
            pub.publish(msg)

            self.create_service(
                Empty,
                name + '/emergency',
                partial(self._emergency_callback, name=name)
            )
            self.create_service(
                Takeoff,
                name + '/takeoff',
                partial(self._takeoff_callback, name=name)
            )
            self.create_service(
                Land,
                name + '/land',
                partial(self._land_callback, name=name)
            )
            self.create_service(
                GoTo,
                name + '/go_to',
                partial(self._go_to_callback, name=name)
            )
            self.create_service(
                StartTrajectory,
                name + '/start_trajectory',
                partial(self._start_trajectory_callback, name=name)
            )
            self.create_service(
                UploadTrajectory,
                name + '/upload_trajectory',
                partial(self._upload_trajectory_callback, name=name)
            )
            self.create_service(
                NotifySetpointsStop,
                name + '/notify_setpoints_stop',
                partial(self._notify_setpoints_stop_callback, name=name)
            )
            self.create_subscription(
                Twist,
                name + '/cmd_vel_legacy',
                partial(self._cmd_vel_legacy_changed, name=name),
                10
            )
            self.create_subscription(
                Hover,
                name + '/cmd_hover',
                partial(self._cmd_hover_changed, name=name),
                10
            )
            self.create_subscription(
                FullState,
                name + '/cmd_full_state',
                partial(self._cmd_full_state_changed, name=name),
                10
            )

        # Create services for the entire swarm and each individual crazyflie
        self.create_service(Takeoff, 'all/takeoff', self._takeoff_callback)
        self.create_service(Land, 'all/land', self._land_callback)
        self.create_service(GoTo, 'all/go_to', self._go_to_callback)
        self.create_service(StartTrajectory,
                            'all/start_trajectory',
                            self._start_trajectory_callback)

        # This is the last service to announce.
        # Can be used to check if the server is fully available.
        self.create_service(Empty, 'all/emergency', self._emergency_callback)

        # step as fast as possible
        max_dt = 0.0 if 'max_dt' not in self._ros_parameters['sim'] \
            else self._ros_parameters['sim']['max_dt']
        self.timer = self.create_timer(max_dt, self._timer_callback)
        self.is_shutdown = False

    # indi_gains.<key> / pos_gains.<key> -> firmware global. Only these are settable;
    # anything else in the yaml is either not a controller gain or not exposed by the
    # bindings, and is reported rather than silently ignored.
    _OOT_PARAM_MAP = {
        'indi_gains': {
            'ctrl_mode': ('g_controller_mode', int),
            'kr': ('g_indi_kr', float), 'kw': ('g_indi_kw', float),
            'kr_z': ('g_indi_kr_z', float), 'kw_z': ('g_indi_kw_z', float),
            'fc_bw': ('g_indi_fc_bw', float), 'mass': ('g_indi_mass', float),
            'kt1': ('g_indi_kt1', float), 'kt2': ('g_indi_kt2', float),
            'kt3': ('g_indi_kt3', float), 'kt4': ('g_indi_kt4', float),
            'ff_free': ('g_indi_ff_free', int),
            'filt_order': ('g_indi_filt_order', int),
            'filt_tau': ('g_indi_filt_tau', int),
            'j_scale': ('g_indi_j_scale', float),
            'act_tau': ('g_indi_act_tau', float),
            'clamp_en': ('g_indi_clamp_en', int),
            'tau_xy_max': ('g_indi_tau_xy_max', float),
            'tau_z_max': ('g_indi_tau_z_max', float),
            'tilt_max_deg': ('g_indi_tilt_max_deg', float),
            'thrust_max': ('g_indi_thrust_max', float),
            'notch_en': ('g_indi_notch_en', int),
            'notch_f0': ('g_indi_notch_f0', float),
            'notch_bw': ('g_indi_notch_bw', float),
            'omega_src': ('g_indi_omega_src', int),
            'frame_conv': ('g_indi_frame_conv', int),
        },
        'pos_gains': {
            'kp_xy': ('g_kp_xy', float), 'kp_z': ('g_kp_z', float),
            'kv_xy': ('g_kv_xy', float), 'kv_z': ('g_kv_z', float),
        },
    }

    def _apply_oot_firmware_params(self, firm):
        """Push crazyflies.yaml firmware_params into the controller, as CRTP does on hardware.

        There is one compiled controller for every simulated drone, so these gains are
        necessarily shared. That matches the yaml, where the block lives under `all:`
        and is pushed to every robot; a per-robot override cannot be honoured here and
        is reported rather than quietly dropped.
        """
        applied, skipped = [], []
        blocks = dict(self._ros_parameters.get('all', {}).get('firmware_params', {}))
        for group, keys in self._OOT_PARAM_MAP.items():
            for key, value in dict(blocks.get(group, {})).items():
                if key not in keys:
                    skipped.append('%s.%s' % (group, key))
                    continue
                name, cast = keys[key]
                if not hasattr(firm.cvar, name):
                    skipped.append('%s.%s' % (group, key))
                    continue
                setattr(firm.cvar, name, cast(value))
                applied.append('%s=%s' % (key, value))

        for robot, cfg in self._ros_parameters.get('robots', {}).items():
            if any(g in dict(cfg.get('firmware_params', {}))
                   for g in self._OOT_PARAM_MAP):
                self.get_logger().warn(
                    'robot %s has per-robot controller gains; the simulator has one '
                    'shared controller and cannot apply them' % robot)

        if applied:
            self.get_logger().info(
                'applied %d firmware_params to the out-of-tree controller: %s'
                % (len(applied), ', '.join(applied)))
        if skipped:
            self.get_logger().info(
                'firmware_params not applicable to the controller: %s'
                % ', '.join(sorted(set(skipped))))

    def _setup_oot(self):
        """Configure the out-of-tree controller and match the plant to it.

        Two things have to agree or the simulation is not of our drone:

        1. Which control law runs -- sim.oot_ctrl_mode, the same 0/1/2/3 selector as
           indi_gains.ctrl_mode on hardware.
        2. The airframe. INDI inverts the vehicle model, so if the simulated plant has
           a different mass, thrust constant or arm length than the firmware was
           compiled with, the mismatch appears as a residual force -- and residual
           force is exactly what this thesis measures. The plant is therefore built
           from the firmware's own constants rather than from a second set of numbers
           in the yaml, so the two cannot drift apart. An explicit sim.physics block
           still wins if one is given.
        """
        import cffirmware as _firm
        sim = self._ros_parameters['sim']

        # Apply firmware_params exactly as hardware does. On a real flight
        # crazyflies.yaml is pushed to the drone over CRTP at connect; nothing did that
        # in simulation, so the controller ran on traj_iface.c compile-time defaults --
        # a different mass, different kt, kr=100/kw=30 instead of 2400/170, and
        # filt_order/filt_tau off. Those last two are the phase-matching filters whose
        # absence the firmware comments identify as driving a limit cycle, and the
        # attitude INDI loop duly diverged in sim while flying fine on the drone.
        # Simulating gains nobody flies is worse than not simulating at all.
        self._apply_oot_firmware_params(_firm)

        if 'oot_ctrl_mode' in sim:
            # The per-run selector wins over the yaml's ctrl_mode, so one config file
            # can be flown under several control laws.
            _firm.cvar.g_controller_mode = int(sim['oot_ctrl_mode'])
        names = {0: 'geometric SE(3)', 1: 'position INDI',
                 2: 'attitude INDI', 3: 'full INDI'}
        mode = int(_firm.cvar.g_controller_mode)
        self.get_logger().info(
            'out-of-tree controller: ctrl_mode=%d (%s)' % (mode, names.get(mode, '?')))

        if 'physics' not in sim:
            sim['physics'] = {
                'mass': float(_firm.cvar.g_indi_mass),
                'kt': [float(_firm.cvar.g_indi_kt1), float(_firm.cvar.g_indi_kt2),
                       float(_firm.cvar.g_indi_kt3), float(_firm.cvar.g_indi_kt4)],
                'arm_length': float(_firm.oot_arm_length()),
                't2t': float(_firm.oot_thrust2torque()),
                # First-order motor lag. This is a property of the PLANT, so unlike the
                # others it cannot come from the controller -- indi_gains.act_tau is the
                # controller's *model* of actuator lag and is deliberately 0. The value
                # here is the measured brushless rotor time constant. The model had no
                # actuator dynamics at all, which is both unfaithful and optimistic:
                # adding it moves the simulator's stability wall from kv_xy 10 to 8,
                # closing roughly 40% of the gap to hardware's 5. Use 0.071 for the
                # upgraded CF2.1.
                'motor_tau': 0.044,
            }
            self.get_logger().info(
                'simulated airframe taken from firmware: mass=%.4f kg, '
                'THRUST_MAX=%.3f N/motor, arm=%.3f m'
                % (sim['physics']['mass'], _firm.oot_thrust_max(),
                   sim['physics']['arm_length']))

        self._setup_residual_nn(_firm, sim)

    def _setup_residual_nn(self, _firm, sim):
        """Load trained residual-network weights and set rnn.en, from the sim config.

        `sim.rnn_weights` is a path to the .npz written by tools/residual/train.py;
        `sim.rnn_enable` decides whether the prediction is fed into the control law or merely
        logged. They are deliberately separate: the useful first run with a new model records
        predicted against measured residual WITHOUT the model touching the vehicle, which is the
        only way to tell a good prediction from a lucky feedback loop.

        Weights go in through the same rnn.* protocol the drone uses -- one index/value/commit
        per controller tick -- rather than a back door, so the simulator exercises the code path
        that will run in the lab. `rnn.ready` is checked, because a partial upload is refused
        onboard and would otherwise look like "the model did nothing".
        """
        if not hasattr(_firm.cvar, 'g_rnn_wi'):
            return
        path = sim.get('rnn_weights', '')
        if not path:
            return
        import json
        import numpy as _np
        d = _np.load(path, allow_pickle=False)
        w = _np.asarray(d['weights'], _np.float32)
        meta = json.loads(str(d['meta'])) if 'meta' in d else {}

        _firm.cvar.g_rnn_n = int(w.size)
        _firm.cvar.g_rnn_begin = 1
        _firm.oot_rnn_service()
        self._rnn_pending = [(i, float(v)) for i, v in enumerate(w)]
        self._rnn_enable = bool(sim.get('rnn_enable', False))
        self.get_logger().info(
            'residual network: uploading %d weights from %s (trained %s, val RMSE %s), '
            'rnn.en will be %d'
            % (w.size, path, meta.get('created', '?'), meta.get('val_rmse', '?'),
               int(self._rnn_enable)))
        if meta.get('synthetic'):
            self.get_logger().warn('residual weights are SYNTHETIC -- a pipeline rehearsal, '
                                   'not a trained model')

    def _service_rnn_upload(self, _firm):
        """Feed one queued weight per tick, then commit and set rnn.en."""
        pending = getattr(self, '_rnn_pending', None)
        if pending is None:
            return
        if pending:
            # One weight per tick, through oot_rnn_service -- the same routine the drone runs
            # from its control loop. Driving it directly is necessary because the SIL layer
            # returns early while a vehicle is idle and never calls the controller, so the
            # upload would otherwise stall until takeoff and then land weights mid-flight.
            i, v = pending.pop(0)
            _firm.cvar.g_rnn_wi = i
            _firm.cvar.g_rnn_wv = v
            _firm.cvar.g_rnn_wc = 1
            _firm.oot_rnn_service()
            return
        _firm.cvar.g_rnn_end = 1
        _firm.oot_rnn_service()
        self._rnn_pending = None
        if _firm.cvar.g_rnn_ready != 1:
            self.get_logger().error('residual network upload REJECTED (rnn.ready=0) -- the '
                                    'weights are not loaded and nothing will use them')
            return
        _firm.cvar.g_rnn_en = 1 if getattr(self, '_rnn_enable', False) else 0
        self.get_logger().info('residual network ready, rnn.en=%d' % _firm.cvar.g_rnn_en)

    def on_shutdown_callback(self):
        if not self.is_shutdown:
            if getattr(self, '_res_log', None) is not None:
                self._res_log.close()
                self._res_log = None
            self.backend.shutdown()
            for visualization in self.visualizations:
                visualization.shutdown()

            self.is_shutdown = True

    def _timer_callback(self):
        # update setpoint
        states_desired = [cf.getSetpoint() for _, cf in self.cfs.items()]

        cfs = list(self.cfs.values())
        if self._oot_active:
            import cffirmware as _firm
            self._service_rnn_upload(_firm)
            # Every vehicle's view of the others. On the drone this arrives by pose broadcast;
            # here it is handed over directly. Positions only, matching the peer API exactly --
            # feeding a velocity the firmware cannot have would make the simulation easier than
            # reality in the one place the residual model depends on.
            for i, cf in enumerate(cfs):
                cf.peers = [(o.state.position.x, o.state.position.y, o.state.position.z)
                            for j, o in enumerate(cfs) if j != i]

        # execute the control loop
        actions = [cf.executeController() for _, cf in self.cfs.items()]

        # execute the physics simulator
        states_next = self.backend.step(states_desired, actions)

        # update the resulting state
        for state, (_, cf) in zip(states_next, self.cfs.items()):
            cf.setState(state)

        for vis in self.visualizations:
            vis.step(self.backend.time(), states_next, states_desired, actions)

        self._log_residual(states_next, states_desired)

    def _init_residual_log(self):
        """Open the residual CSV, if sim.residual_log names one.

        Columns are exactly what `merge_usd_logs.py` produces from real uSD logs, so
        `tools/residual/dataset.py` reads simulation and flight data through the same path
        with no special cases -- which also means the sim dry-run genuinely tests the loader
        that will be used on hardware data.
        """
        self._res_log = None
        path = self._ros_parameters['sim'].get('residual_log', '')
        if not path or not self._oot_active:
            return
        os.makedirs(os.path.dirname(os.path.abspath(path)), exist_ok=True)
        names = list(self.cfs.keys())
        cols = ['t']
        for n in names:
            cols += [f'{n}.{c}' for c in
                     ('x', 'y', 'z', 'vx', 'vy', 'vz',
                      'a_res_x', 'a_res_y', 'a_res_z',
                      'rnn_pred_x', 'rnn_pred_y', 'rnn_pred_z',
                      # What the trajectory ASKED for. Without it, "is the formation holding
                      # its geometry" cannot be answered from this log at all: the realised
                      # separation mixes commanded motion with error, and no threshold on the
                      # realised signal alone can separate them.
                      'cmd_x', 'cmd_y', 'cmd_z')]
        self._res_log = open(path, 'w')
        self._res_log.write(','.join(cols) + '\n')
        self._res_names = names
        self._res_next_t = 0.0
        # 100 Hz. The controller runs at 1 kHz but the residual is smooth compared with that,
        # and a 60 s three-drone flight at full rate is a 200 MB CSV for no extra information.
        self._res_period = float(self._ros_parameters['sim'].get('residual_log_hz', 100.0))
        self._res_period = 1.0 / max(self._res_period, 1.0)
        self.get_logger().info('residual log: %s' % path)

    def _log_residual(self, states, desired):
        if getattr(self, '_res_log', None) is None:
            return
        t = self.backend.time()
        if t < self._res_next_t:
            return
        self._res_next_t = t + self._res_period
        row = [f'{t:.6f}']
        for name, st in zip(self._res_names, states):
            cf = self.cfs[name]
            row += [f'{st.pos[0]:.6f}', f'{st.pos[1]:.6f}', f'{st.pos[2]:.6f}',
                    f'{st.vel[0]:.6f}', f'{st.vel[1]:.6f}', f'{st.vel[2]:.6f}']
            row += [f'{v:.6f}' for v in cf.a_res]
            row += [f'{v:.6f}' for v in cf.rnn_pred]
            row += [f'{d:.6f}' for d in desired[self._res_names.index(name)].pos]
        self._res_log.write(','.join(row) + '\n')

    def _param_to_dict(self, param_ros):
        """Turn ROS 2 parameters from the node into a dict."""
        tree = {}
        for item in param_ros:
            t = tree
            for part in item.split('.'):
                if part == item.split('.')[-1]:
                    t = t.setdefault(part, param_ros[item].value)
                else:
                    t = t.setdefault(part, {})
        return tree

    def _emergency_callback(self, request, response, name='all'):
        self.get_logger().info(f'[{name}] emergency not yet implemented')

        return response

    def _takeoff_callback(self, request, response, name='all'):
        """Service callback to takeoff the crazyflie."""
        duration = float(request.duration.sec) + \
            float(request.duration.nanosec / 1e9)
        self.get_logger().info(
            f'[{name}] takeoff(height={request.height} m,'
            + f'duration={duration} s,'
            + f'group_mask={request.group_mask})'
        )
        cfs = self.cfs if name == 'all' else {name: self.cfs[name]}
        for _, cf in cfs.items():
            cf.takeoff(request.height, duration, request.group_mask)

        return response

    def _land_callback(self, request, response, name='all'):
        """Service callback to land the crazyflie."""
        duration = float(request.duration.sec) + \
            float(request.duration.nanosec / 1e9)
        self.get_logger().info(
            f'[{name}] land(height={request.height} m,'
            + f'duration={duration} s,'
            + f'group_mask={request.group_mask})'
        )
        cfs = self.cfs if name == 'all' else {name: self.cfs[name]}
        for _, cf in cfs.items():
            cf.land(request.height, duration, request.group_mask)

        return response

    def _go_to_callback(self, request, response, name='all'):
        """Service callback to have the crazyflie go to a position."""
        duration = float(request.duration.sec) + \
            float(request.duration.nanosec / 1e9)

        self.get_logger().info(
            """[%s] go_to(position=%f,%f,%f m,
             yaw=%f rad,
             duration=%f s,
             relative=%d,
             group_mask=%d)"""
            % (
                name,
                request.goal.x,
                request.goal.y,
                request.goal.z,
                request.yaw,
                duration,
                request.relative,
                request.group_mask,
            )
        )
        cfs = self.cfs if name == 'all' else {name: self.cfs[name]}
        for _, cf in cfs.items():
            cf.goTo([request.goal.x, request.goal.y, request.goal.z],
                    request.yaw, duration, request.relative, request.group_mask)

        return response

    def _notify_setpoints_stop_callback(self, request, response, name='all'):
        self.get_logger().info(f'[{name}] Notify setpoint stop not yet implemented')
        return response

    def _upload_trajectory_callback(self, request, response, name='all'):
        self.get_logger().info('[%s] Upload trajectory(id=%d)' % (name, request.trajectory_id))

        cfs = self.cfs if name == 'all' else {name: self.cfs[name]}
        for _, cf in cfs.items():
            pieces = []
            for piece in request.pieces:
                poly_x = piece.poly_x
                poly_y = piece.poly_y
                poly_z = piece.poly_z
                poly_yaw = piece.poly_yaw
                duration = float(piece.duration.sec) + \
                    float(piece.duration.nanosec / 1e9)
                pieces.append(TrajectoryPolynomialPiece(
                    poly_x,
                    poly_y,
                    poly_z,
                    poly_yaw,
                    duration))
            cf.uploadTrajectory(request.trajectory_id, request.piece_offset, pieces)

        return response

    def _start_trajectory_callback(self, request, response, name='all'):
        self.get_logger().info(
            '[%s] start_trajectory(id=%d, timescale=%f, reverse=%d, relative=%d, group_mask=%d)'
            % (
                name,
                request.trajectory_id,
                request.timescale,
                request.reversed,
                request.relative,
                request.group_mask,
            )
        )
        cfs = self.cfs if name == 'all' else {name: self.cfs[name]}
        for _, cf in cfs.items():
            cf.startTrajectory(
                request.trajectory_id,
                request.timescale,
                request.reversed,
                request.relative,
                request.group_mask)

        return response

    def _cmd_vel_legacy_changed(self, msg, name=''):
        """
        Topic update callback.

        Controls the attitude and thrust of the crazyflie with teleop.
        """
        self.get_logger().info('cmd_vel_legacy not yet implemented')

    def _cmd_hover_changed(self, msg, name=''):
        """
        Topic update callback for hover command.

        Used from the velocity multiplexer (vel_mux).
        """
        self.get_logger().info('cmd_hover not yet implemented')

    def _cmd_full_state_changed(self, msg, name):
        q = [msg.pose.orientation.w,
             msg.pose.orientation.x,
             msg.pose.orientation.y,
             msg.pose.orientation.z]
        rpy = rowan.to_euler(q, convention='xyz')

        self.cfs[name].cmdFullState(
            [msg.pose.position.x, msg.pose.position.y, msg.pose.position.z],
            [msg.twist.linear.x, msg.twist.linear.y, msg.twist.linear.z],
            [msg.acc.x, msg.acc.y, msg.acc.z],
            rpy[2],
            [msg.twist.angular.x, msg.twist.angular.y, msg.twist.angular.z])


def main(args=None):

    rclpy.init(args=args)
    crazyflie_server = CrazyflieServer()
    rclpy.get_default_context().on_shutdown(crazyflie_server.on_shutdown_callback)

    try:
        rclpy.spin(crazyflie_server)
    except KeyboardInterrupt:
        crazyflie_server.on_shutdown_callback()
    finally:
        rclpy.try_shutdown()
        crazyflie_server.destroy_node()


if __name__ == '__main__':
    main()
