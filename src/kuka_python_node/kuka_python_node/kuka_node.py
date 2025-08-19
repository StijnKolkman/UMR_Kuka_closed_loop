import rclpy
from rclpy.node import Node

import numpy as np
from threading import Thread

from std_msgs.msg import Float32
from kuka_interfaces.msg import KukaPos, KukaAction
from scipy.spatial.transform import Rotation as R

from tf2_ros import Buffer, TransformException
from tf2_ros.transform_listener import TransformListener


class kuka_python(Node):
    """ROS2 helper node for:
    - publishing target end-effector poses to KUKA bridge (KukaPos: mm & deg),
    - commanding motor velocity (Float32: rad/s),
    - reading current tool pose via TF2 ('base_link' ← 'tool').

    Note: This node is spun in a **background thread** by `start_kuka_node()`.
    """
    def __init__(self):
        super().__init__('kuka_standalone_publisher')

        # Publishers
        self.publisher_target_position = self.create_publisher(KukaPos, 'python/target_position', 1)
        self.publish_action = self.create_publisher(KukaAction, "kuka/action", 10)
        self.publisher_motor_velocity = self.create_publisher(Float32, 'maxon/target_velocity', 1)

        # Subscribers
        self.tf_buffer = Buffer()
        self.tf_listener = TransformListener(self.tf_buffer,self)

        # Initialize kuka_pose with zeros
        self.kuka_pose = np.zeros(6)

    def get_position(self):
        """Get current tool pose from TF2 and update kuka_pose."""
        try:
            transform = self.tf_buffer.lookup_transform(
                "base_link",  # target frame
                "tool",       # source frame
                rclpy.time.Time()
            )

            pos = transform.transform.translation
            rot = transform.transform.rotation

            r = R.from_quat([rot.x, rot.y, rot.z, rot.w])
            c, b, a = r.as_euler("xyz", degrees=False)

            self.kuka_pose = (pos.x, pos.y, pos.z, a, b, c)
            self.get_logger().info(f"kuka_pose updated: {self.kuka_pose}")
            return self.kuka_pose

        except TransformException as e:
            self.get_logger().warn(f"TF lookup failed: {str(e)}")
            return self.kuka_pose

    def publish_pose(self, pose):
        """Publish target pose to KUKA bridge."""
        msg = KukaPos()
        msg.x, msg.y, msg.z, msg.a, msg.b, msg.c = pose
        self.publisher_target_position.publish(msg)
        #self.get_logger().info(f"Publishing KukaPos: {pose}")

    def set_motor_speed(self,velocity):
        """Publish motor velocity command."""
        msg = Float32()
        msg.data = velocity
        self.publisher_motor_velocity.publish(msg)
        self.get_logger().info(f"Published velocity command: {msg.data}")

    def shutdown_publisher(self) -> None:
        """Shutdown publishers and node."""
        self.get_logger().info("Shutting down publishers...")
        self.destroy_node()
        if rclpy.ok():
            rclpy.shutdown()

def ros_thread(node):
    """Run the ROS2 node in a separate thread."""
    rclpy.spin(node)

def start_kuka_node():
    """Initialize ROS2 and start the kuka_python node in a background thread."""
    rclpy.init()
    node = kuka_python()
    thread = Thread(target=rclpy.spin, args=(node,), daemon=True)
    thread.start()
    return node
        
if __name__=="__main__":
    node = kuka_python()
    thread = Thread(target=ros_thread, args=(node,), daemon=True)
    thread.start()

    try:
       input("Kuka_cummunication node is running\n") 
    except KeyboardInterrupt:
        node.get_logger().info("Shutting down node...")
    finally:
        node.shutdown_publisher()