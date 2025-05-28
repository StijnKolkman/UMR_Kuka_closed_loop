import rclpy
from rclpy.node import Node
from kuka_interfaces.msg import KukaPos  # replace with your actual message package

# Initialize ROS 2 context, node, and publisher exactly once
rclpy.init()
node = Node('kuka_standalone_publisher')
publisher = node.create_publisher(KukaPos, 'python/target_position', 1)

def publish_pose(pose):
    msg = KukaPos()
    # Assign tuple values to message fields
    msg.x, msg.y, msg.z, msg.a, msg.b, msg.c = pose
    publisher.publish(msg)
    node.get_logger().info(f"Publishing KukaPos: {pose}")
    rclpy.spin_once(node, timeout_sec=0.1)

def shutdown_publisher() -> None:
    """
    Clean up ROS 2 node and context. Call when completely done.
    """
    node.destroy_node()
    rclpy.shutdown()

