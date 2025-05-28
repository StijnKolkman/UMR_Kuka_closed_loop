import math
import rclpy
from rclpy.node import Node
from std_msgs.msg import Bool, Float32
from geometry_msgs.msg import Point, Vector3
from kukapythonvarproxy import KRL_Pos
from kuka_interfaces.msg import KukaWriteVariable, KukaAction
from kuka_interfaces.srv import KukaReadVariable
from tf2_ros.buffer import Buffer
from tf2_ros.transform_listener import TransformListener
from scipy.spatial.transform import Rotation as R
import numpy as np


class MagnetControl(Node):
    def __init__(self):
        super().__init__("magnet_control")

        # Store the last received magnet position (x, y, z)
        self.magnet_pos = Point()

        # # Subscription to 'magnet_pos' to get the 3D point
        # self.magnet_pos_sub = self.create_subscription(
        #     Point, "magnet_pos", self.magnet_pos_callback, 10
        # )
        self.capsule_yaw = 0
        self.capsule_pitch = 0
        self.base_yaw = 0
        # Subscription to 'position_ready' to trigger sending KukaAction
        self.position_ready_sub = self.create_subscription(
            Bool, "kuka/position_ready", self.position_ready_callback, 10
        )
        self.yaw_pitch_sub = self.create_subscription(
            Vector3, "capsule/yaw_pitch", self.yaw_pitch_callback, 10
        )

        self.declare_parameter("kuka_base", "kuka_base")
        self.declare_parameter("capsule", "capsule")
        self.declare_parameter("capsule_maxon_z", 0.14)

        # Publisher for KukaAction
        self.kuka_action_pub = self.create_publisher(KukaAction, "kuka/action", 10)
        self.maxon_angle_pub = self.create_publisher(
            Float32, "maxon/target_position", 10
        )
        # Set up the tf2 Buffer and TransformListener
        self.tf_buffer = Buffer()
        self.tf_listener = TransformListener(self.tf_buffer, self)

    def yaw_pitch_callback(self, msg):
        """Callback to receive yaw and pitch from capsule"""
        self.capsule_yaw = msg.x
        self.capsule_pitch = msg.y

    def position_ready_callback(self, msg):
        """Callback to send KukaAction when position is ready"""
        if msg.data:  # If position is ready
            ## Do control logic here.
            # Get position of magnetself.get_logger().info("Position is ready. Sending KukaAction.")
            capsule_pos = self.get_capsule_position()
            if capsule_pos is None:
                # self.get_logger().error("Capsule position is None. Skipping action.")
                return
            self.get_logger().info(
                f"Received yaw: {math.degrees(self.capsule_yaw)}, pitch: {math.degrees(self.capsule_pitch)}"
            )
            # Set rpm position 10 cm above capsule position.
            # Convert meters to millimeters
            rpm_pos = KRL_Pos.KRLPos("RPM")
            rpm_pos.set_all(
                capsule_pos.x * 1e3,
                capsule_pos.y * 1e3,
                (
                    capsule_pos.z
                    + +self.get_parameter("capsule_maxon_z")
                    .get_parameter_value()
                    .double_value
                )
                * 1e3,
                0,
                0,
                0,
            )
            # Get joystick input/path input.
            # TODO: Add joystick input to control orientation of magnet.
            # For now, set orientation to 0, 0, 0
            # Joystick range from -1 to 1. This maps from -90 to +90 degrees.
            # Up and down maps to 0 and + 90
            joystick_l_r = 0
            joystick_u_d = 0
            # Ignore negative values for pitch

            pitch = math.radians(joystick_u_d * 90)
            pitch = max(0, pitch)
            yaw = math.radians(joystick_l_r * 90)
            # Pitch is absolute respect to x-axis.
            # TODO Convert pitch to maxon angle
            maxon_angle = self.convert_pitch_to_maxon(pitch)
            msg = Float32()
            msg.data = maxon_angle
            self.maxon_angle_pub.publish(msg)
            self.get_logger().info(f"Controlling to maxon angle: {maxon_angle}")
            # yaw is relative to what? Define some start angle. yaw relative to capsule?
            # If joystick is zero, get base value for yaw.
            # if abs(yaw) < 0.1 and math.degrees(abs(self.capsule_pitch)) < 5:
            #     self.base_yaw = math.degrees(self.capsule_yaw)
            #     yaw = 0
            # yaw = self.base_yaw + yaw
            rpm_pos.set_a(yaw)

            # Based on desired orientation control, set angles of Kuka and Maxon. Kuka controls movement around z-axis. Maxon around x-axis

            # Create KukaAction message
            kuka_action = KukaAction()
            kuka_action.com_action = 15
            COM_E6 = KukaWriteVariable(name="COM_E6POS", value=rpm_pos.get_KRL_string())

            kuka_action.variable = [
                COM_E6,
            ]  # Add the variables to the action

            # Publish the KukaAction
            self.kuka_action_pub.publish(kuka_action)
            self.get_logger().info(f"Joint movement sent with target: {COM_E6}")

    def get_capsule_position(self):
        try:
            t = self.tf_buffer.lookup_transform(
                self.get_parameter("kuka_base").get_parameter_value().string_value,
                self.get_parameter("capsule").get_parameter_value().string_value,
                rclpy.time.Time(),
            )
        except Exception as e:
            self.get_logger().error(f"Error looking up transform: {e}")
            return None
        capsule_t = t.transform
        capsule_pos = capsule_t.translation
        capsule_pos = self.clamp_capsule(capsule_pos)
        self.get_logger().info(
            f"Received magnet position: {capsule_pos.x}, {capsule_pos.y}, {capsule_pos.z}, "
        )
        return capsule_pos

    def convert_pitch_to_maxon(self, pitch):
        if abs(pitch) < 0.1:
            return math.pi / 2
        # Convert pitch to maxon angle
        L = 0.01  # Length of capsule
        m = 0.03  # Mass of capsule
        g = 9.81  # Gravitational acceleration
        dz = (
            self.get_parameter("capsule_distance_z").get_parameter_value().double_value
        )  # Distance from capsule to maxon
        mu0 = 1e-7
        m_rpm = 50.118
        m_cap = 0.3

        pitch = math.pi + math.acos(
            (L * g * m * abs(dz) ^ 5) / (18 * dz ^ 2 * m_cap * mu0 * (m_rpm))
        )
        return pitch

    def clamp_capsule(self, capsule_pos):
        # Clamp the capsule position to a certain range
        # For example, let's say we want to clamp it within -1.0 to 1.0 for x, y, z
        t = self.tf_buffer.lookup_transform(
            self.get_parameter("kuka_base").get_parameter_value().string_value,
            "sensor_board",
            rclpy.time.Time(),
        )
        sensor_pos = t.transform.translation
        sx = sensor_pos.x
        sy = sensor_pos.y
        sz = sensor_pos.z
        capsule_pos.x = max(sx - 0.1, min(sx + 0.2, capsule_pos.x))
        capsule_pos.y = max(sy - 0.1, min(sy + 0.2, capsule_pos.y))
        capsule_pos.z = max(sz + 0.05, min(sz + 0.2, capsule_pos.z))
        return capsule_pos


def main():
    rclpy.init()
    magnet_control_node = MagnetControl()
    rclpy.spin(magnet_control_node)
    rclpy.shutdown()


if __name__ == "__main__":
    main()
