#!/usr/bin/env python3
import rclpy
from rclpy.node import Node
from std_msgs.msg import Float32
from tf2_ros import TransformBroadcaster
from geometry_msgs.msg import TransformStamped
import math
from .motor import MotorMode
from .motor import Motor


# The ROS2 Node that wraps the Motor functionality
class MotorController(Node):
    def __init__(self):
        super().__init__("motor_controller")
        # Declare configurable parameters.
        self.declare_parameter("node_id", 1)
        self.declare_parameter("usb_id", 0)
        self.declare_parameter("velocity_profile", 100)
        self.declare_parameter("acceleration", 8000)
        self.declare_parameter("deceleration", 8000)

        node_id = self.get_parameter("node_id").get_parameter_value().integer_value
        usb_id = self.get_parameter("usb_id").get_parameter_value().integer_value
        self.velocity_profile = (
            self.get_parameter("velocity_profile").get_parameter_value().integer_value
        )
        self.acceleration = (
            self.get_parameter("acceleration").get_parameter_value().integer_value
        )
        self.deceleration = (
            self.get_parameter("deceleration").get_parameter_value().integer_value
        )

        # Instantiate the Motor object.
        # self.motor = None

        self.motor = Motor(NodeID=node_id, USBID=usb_id)
        self.motor.OpenCommunication()
        self.motor.EnableMotor()

        self.motor.SetPositionProfile(
            self.velocity_profile, self.acceleration, self.deceleration
        )
        # Create a publisher for the motor position.
        self.position_broadcaster = TransformBroadcaster(self)
        # Create a subscriber to command a new target position.
        self.create_subscription(
            Float32, "maxon/target_position", self.target_position_callback, 5
        )

        self.create_subscription(
            Float32, "maxon/target_velocity", self.target_velocity_callback, 5
        )
        # Timer callback to periodically read and publish the motor position.
        timer_period = 0.05  # seconds
        self.timer = self.create_timer(timer_period, self.publish_joint_state)

    def target_position_callback(self, msg):
        target = msg.data

        target_count = self.motor.ConvertRadiansToCount(target)
        self.get_logger().info(
            f"Received target position: {target}, count: {target_count}"
        )
        # Command the motor to move by setting the operation mode to Position and configuring the profile.
        self.motor.SetOperationMode(MotorMode.ProfilePosition)
        self.motor.SetPositionProfile(
            self.velocity_profile, self.acceleration, self.deceleration
        )
        self.motor.SetPosition(target_count, True, False)
        self.get_logger().info(f"Moving motor to position: {target}")

    def target_velocity_callback(self, msg):
        target = msg.data
        target_count = self.motor.ConvertRadiansToCount(target)
        self.get_logger().info(
            f"Received target position: {target}, count: {target_count}"
        )
        # Command the motor to move by setting the operation mode to Veloity and configuring the profile.
        self.motor.SetOperationMode(MotorMode.ProfileVelocity)
        self.motor.RunSetVelocity(target_count)
        self.get_logger().info(f"Moving motor with: {target} rad/s")

    def publish_joint_state(self):
        try:
            if self.motor is not None:
                angle = self.motor.GetPositionRadians()
            else:
                angle = 0
            if angle is None:
                self.get_logger().warn("Failed to get motor position.")
                return
            # self.get_logger().info(f"Current motor angle: {angle}")
            # Create TransformStamped message
            t = TransformStamped()
            t.header.stamp = self.get_clock().now().to_msg()
            t.header.frame_id = "kuka_tool"
            t.child_frame_id = "magnet"
            t.transform.translation.x = 0.0
            t.transform.translation.y = 0.0
            t.transform.translation.z = 0.0
            t.transform.rotation.x = math.sin(angle / 2.0)
            t.transform.rotation.y = 0.0
            t.transform.rotation.z = 0.0
            t.transform.rotation.w = math.cos(angle / 2.0)

            # Publish the transform
            self.position_broadcaster.sendTransform(t)
        except Exception as e:
            self.get_logger().error(f"Failed to publish transform: {e}")

    def destroy_node(self):
        self.get_logger().info("Shutting down motor controller...")
        if Motor is not None:
            self.motor.DisableMotor()
            self.motor.CloseCommunication()
        super().destroy_node()


def main(args=None):
    rclpy.init(args=args)
    motor_controller = MotorController()
    try:
        rclpy.spin(motor_controller)
    except KeyboardInterrupt:
        pass
    finally:
        motor_controller.destroy_node()
        rclpy.shutdown()


if __name__ == "__main__":
    main()
