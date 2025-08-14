import rclpy
from rclpy.node import Node
from threading import Thread
from kuka_interfaces.msg import KukaPos, KukaAction
from scipy.spatial.transform import Rotation as R
from geometry_msgs.msg import PoseStamped
from std_msgs.msg import Float32
import time
import numpy as np

from tf2_ros import TransformException
from tf2_ros import Buffer
from tf2_ros.transform_listener import TransformListener

# Initialize ROS 2 context, node, and publisher exactly once
class kuka_python(Node):
    def __init__(self):
        super().__init__('kuka_standalone_publisher')

        #publishers
        self.publisher_target_position = self.create_publisher(KukaPos, 'python/target_position', 1)
        self.publish_action = self.create_publisher(KukaAction, "kuka/action", 10)
        self.publisher_motor_velocity = self.create_publisher(Float32, 'maxon/target_velocity', 1)

        # Subscribers
        self.tf_buffer = Buffer()
        self.tf_listener = TransformListener(self.tf_buffer,self)

        # Initialize kuka_pose with zeros
        self.kuka_pose = np.zeros(6)

    def get_position(self):
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
        msg = KukaPos()
        msg.x, msg.y, msg.z, msg.a, msg.b, msg.c = pose
        self.publisher_target_position.publish(msg)
        #self.get_logger().info(f"Publishing KukaPos: {pose}")

    def set_motor_speed(self,velocity):
        msg = Float32()
        msg.data = velocity
        self.publisher_motor_velocity.publish(msg)
        self.get_logger().info(f"Published velocity command: {msg.data}")

    def shutdown_publisher(self) -> None:
        self.get_logger().info("Shutting down publishers...")
        self.destroy_node()
        if rclpy.ok():
            rclpy.shutdown()

def ros_thread(node):
    rclpy.spin(node)

def start_kuka_node():
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

    #def send_action(self, com_action: int, variable_list: list(KukaWriteVariable)):
    #    action = KukaAction()
    #    action.com_action = com_action
    #    action.variables = variable_list
    #    self.publish_action.publish(action)
    #    return