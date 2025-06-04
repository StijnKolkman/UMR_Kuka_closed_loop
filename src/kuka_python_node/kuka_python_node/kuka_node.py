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
        self.publisher_target_position = self.create_publisher(KukaPos, 'python/target_position', 1)
        self.publish_action = self.create_publisher(KukaAction, "kuka/action", 10)
        #self.position_sub = self.create_subscription(PoseStamped, "kuka/pose", self.read_position, 10)
        self.publisher_motor_velocity = self.create_publisher(Float32, 'kuka/command_velocity', 1)

        self.tf_buffer = Buffer()
        self.tf_listener = TransformListener(self.tf_buffer,self)

        self.x = self.y = self.z = 0.0  # Init default position
        self.a = self.b = self.c = 0.0
        self.kuka_pose = np.zeros(6)

    #def send_action(self, com_action: int, variable_list: list(KukaWriteVariable)):
    #    action = KukaAction()
    #    action.com_action = com_action
    #    action.variables = variable_list
    #    self.publish_action.publish(action)
    #    return

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

    # def read_position(self, msg):
    #     self.get_logger().info(f"i read a new position")
    #     pose = msg.pose
    #     self.x = pose.position.x
    #     self.y = pose.position.y
    #     self.z = pose.position.z

    #     qx = pose.orientation.x
    #     qy = pose.orientation.y
    #     qz = pose.orientation.z
    #     qw = pose.orientation.w

    #     # Quaternion → Euler (roll, pitch, yaw) in radians
    #     r = R.from_quat([qx, qy, qz, qw])
    #     self.a, self.b, self.c = r.as_euler('xyz', degrees=False)

    #     # Maak de gecombineerde tuple (x, y, z, a, b, c)
    #     self.kuka_pose = (self.x, self.y, self.z, self.a, self.b, self.c)

    #def read_position(self)
    #t = 

    def publish_pose(self, pose):
        msg = KukaPos()
        # Assign tuple values to message fields
        msg.x, msg.y, msg.z, msg.a, msg.b, msg.c = pose
        self.publisher_target_position.publish(msg)
        self.get_logger().info(f"Publishing KukaPos: {pose}")

    def set_motor_speed(self,velocity):
        msg = Float32()
        msg.data = velocity
        self.publisher_motor_velocity.publish(msg)
        self.get_logger().info(f"Published velocity command: {msg.data}")

    def shutdown_publisher(self) -> None:
        """
        Clean up ROS 2 node and context. Call when completely done.
        """
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