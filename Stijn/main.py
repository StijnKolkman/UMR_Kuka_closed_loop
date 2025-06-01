import rclpy
from rclpy.node import Node
from kuka_interfaces.msg import KukaPos, KukaAction
from geometry_msgs.msg import PoseStamped
from std_msgs.msg import Float32

# Initialize ROS 2 context, node, and publisher exactly once
class kuka_python(Node):
    def __init__(self):
        super().__init__('kuka_standalone_publisher')
        self.publisher_target_position = self.create_publisher(KukaPos, 'python/target_position', 1)
        self.publish_action = self.create_publisher(KukaAction, "kuka/action", 10)
        self.position_sub = self.create_subscription(PoseStamped, "kuka/pose", self.read_position, 1)
        self.publisher_motor_velocity = self.create_publisher(Float32, 'kuka/command_velocity', 1)

    #def send_action(self, com_action: int, variable_list: list(KukaWriteVariable)):
    #    action = KukaAction()
    #    action.com_action = com_action
    #    action.variables = variable_list
    #    self.publish_action.publish(action)
    #    return

    def get_position(self):
        rclpy.spin_once(self)
        return self.x, self.y, self.z

    def read_position(self, msg):
        pose = msg.pose
        self.x = pose.position.x
        self.y = pose.position.y
        self.z = pose.position.z

    def publish_pose(self, pose):
        msg = KukaPos()
        # Assign tuple values to message fields
        msg.x, msg.y, msg.z, msg.a, msg.b, msg.c = pose
        self.publisher_target_position.publish(msg)
        self.get_logger().info(f"Publishing KukaPos: {pose}")
        rclpy.spin_once(node, timeout_sec=0.1)

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
        rclpy.shutdown()
if __name__=="main":
    rclpy.init()
    node = kuka_python()
