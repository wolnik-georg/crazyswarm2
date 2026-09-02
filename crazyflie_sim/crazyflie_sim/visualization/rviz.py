from __future__ import annotations

import math

from geometry_msgs.msg import Point, TransformStamped
import numpy as np
from rclpy.node import Node
from std_msgs.msg import ColorRGBA
from tf2_ros import TransformBroadcaster
from visualization_msgs.msg import Marker, MarkerArray

from ..sim_data_types import Action, State

# Shipped with the crazyflie package; textured, with propellers.
DEFAULT_MESH = 'package://crazyflie/urdf/cf2_assembly_with_props.dae'

# Distinguishable per-vehicle colours, in roster order.
PALETTE = [
    (0.20, 0.65, 0.64),   # teal
    (0.90, 0.45, 0.20),   # orange
    (0.45, 0.55, 0.85),   # blue
    (0.55, 0.75, 0.35),   # green
    (0.80, 0.35, 0.55),   # magenta
]


class Visualization:
    """Publish ROS 2 transforms and RViz markers for the simulated vehicles.

    The transforms alone render as bare coordinate triads, which makes a formation hard to
    read at a glance. This adds two things on top, both switchable from the server yaml:

    ``mesh``  the Crazyflie model already shipped in the crazyflie package, so a vehicle
              looks like a vehicle and its attitude is visible without reading axes.

    ``cone``  a translucent cone under each rotor plane, standing in for the downwash the
              vehicle is producing.

    THE CONE IS AN INDICATOR, NOT A FIELD. Its geometry is fixed (a half-angle chosen for
    legibility, not measured), and only its opacity carries data: rotor thrust, summed over
    the four rotors from the RPM the simulator is actually applying. It shows where wash goes
    and how hard the rotors are working. It is NOT the interaction force, it is not
    ``a_res``, and it must never be read as either. The residual is measured in flight and
    logged; see docs/03 in the thesis repository.
    """

    def __init__(
        self,
        node: Node,
        params: dict,
        names: list[str],
        states: list[State],
        reference_frames: list[str],
    ):
        self.node = node
        self.names = names
        self.reference_frames = reference_frames
        self.tfbr = TransformBroadcaster(self.node)

        self.show_mesh = bool(params.get('mesh', True))
        self.show_cone = bool(params.get('cone', True))
        self.mesh_resource = str(params.get('mesh_resource', DEFAULT_MESH))
        self.cone_length = float(params.get('cone_length', 0.45))     # m below the rotor plane
        self.cone_half_angle = float(params.get('cone_half_angle', 22.0))  # degrees
        # Display scale only: the thrust that maps to the most opaque cone. The default
        # covers both airframes in this project with headroom, a CF2 hovering near 0.27 N
        # and the brushless near 0.40 N, so hover sits mid-scale on either.
        self.thrust_full = float(params.get('cone_thrust_full', 0.8))
        # Rotor thrust constant, N per (rad/s)^2. Only sets the opacity scale.
        self.kt = float(params.get('cone_kt', 2.13e-8))

        self.pub = None
        if self.show_mesh or self.show_cone:
            self.pub = self.node.create_publisher(MarkerArray, 'sim_markers', 1)
            self._cone_points = self._make_cone(self.cone_length, self.cone_half_angle)

    # ── geometry ────────────────────────────────────────────────────────────────────
    @staticmethod
    def _make_cone(length: float, half_angle_deg: float, segments: int = 24) -> list[Point]:
        """A cone with its apex at the body origin, opening downward along -z.

        RViz has no cone primitive, so it is built as a triangle list: apex to each pair of
        adjacent rim points, then the rim capped back to its own centre.
        """
        radius = length * math.tan(math.radians(half_angle_deg))
        apex = Point(x=0.0, y=0.0, z=0.0)
        base_c = Point(x=0.0, y=0.0, z=-length)
        rim = []
        for i in range(segments):
            th = 2.0 * math.pi * i / segments
            rim.append(Point(x=radius * math.cos(th), y=radius * math.sin(th), z=-length))
        pts = []
        for i in range(segments):
            a, b = rim[i], rim[(i + 1) % segments]
            pts += [apex, a, b]          # side
            pts += [base_c, b, a]        # cap, wound the other way
        return pts

    def _thrust(self, action: Action) -> float:
        """Total rotor thrust in newtons, from the RPM actually applied this step."""
        try:
            rpm = np.asarray(action.rpm, dtype=float)
        except (AttributeError, TypeError, ValueError):
            return 0.0
        omega = rpm * 2.0 * math.pi / 60.0
        return float(self.kt * np.sum(omega ** 2))

    # ── per-step publishing ─────────────────────────────────────────────────────────
    def step(self, t, states: list[State], states_desired: list[State], actions: list[Action]):
        sec = math.floor(t)
        nanosec = int((t - sec) * 1e9)

        msgs = []
        for name, state, reference_frame in zip(self.names, states, self.reference_frames):
            msg = TransformStamped()
            msg.header.stamp.sec = sec
            msg.header.stamp.nanosec = nanosec
            msg.header.frame_id = reference_frame
            msg.child_frame_id = name
            msg.transform.translation.x = state.pos[0]
            msg.transform.translation.y = state.pos[1]
            msg.transform.translation.z = state.pos[2]
            msg.transform.rotation.x = state.quat[1]
            msg.transform.rotation.y = state.quat[2]
            msg.transform.rotation.z = state.quat[3]
            msg.transform.rotation.w = state.quat[0]
            msgs.append(msg)
        self.tfbr.sendTransform(msgs)

        if self.pub is None:
            return

        # Markers live in each vehicle's own frame with an identity pose, so they inherit the
        # transform published above. Nothing here has to repeat the pose, and nothing can
        # drift out of step with it.
        arr = MarkerArray()
        for i, name in enumerate(self.names):
            r, g, b = PALETTE[i % len(PALETTE)]

            if self.show_mesh:
                m = Marker()
                m.header.frame_id = name
                m.header.stamp.sec = sec
                m.header.stamp.nanosec = nanosec
                m.ns = 'body'
                m.id = i
                m.type = Marker.MESH_RESOURCE
                m.action = Marker.ADD
                m.mesh_resource = self.mesh_resource
                m.mesh_use_embedded_materials = False
                m.pose.orientation.w = 1.0
                m.scale.x = m.scale.y = m.scale.z = 1.0
                m.color = ColorRGBA(r=r, g=g, b=b, a=1.0)
                arr.markers.append(m)

            if self.show_cone:
                thrust = self._thrust(actions[i]) if i < len(actions) else 0.0
                alpha = 0.08 + 0.37 * min(max(thrust / self.thrust_full, 0.0), 1.0)
                c = Marker()
                c.header.frame_id = name
                c.header.stamp.sec = sec
                c.header.stamp.nanosec = nanosec
                c.ns = 'downwash'
                c.id = i
                c.type = Marker.TRIANGLE_LIST
                c.action = Marker.ADD
                c.pose.orientation.w = 1.0
                c.scale.x = c.scale.y = c.scale.z = 1.0
                c.points = self._cone_points
                c.color = ColorRGBA(r=r, g=g, b=b, a=alpha)
                arr.markers.append(c)

        self.pub.publish(arr)

    def shutdown(self):
        # Clear the markers, so a stale formation is not left hanging in RViz after the run.
        if self.pub is None:
            return
        arr = MarkerArray()
        for ns in ('body', 'downwash'):
            m = Marker()
            m.ns = ns
            m.action = Marker.DELETEALL
            arr.markers.append(m)
        self.pub.publish(arr)
