import rclpy
from rclpy.node import Node
from kuka_interfaces.msg import KukaPos  # replace with your actual message package
import time

def publish_callback(node, publisher):
    # Example target pose (x, y, z, a, b, c)
    pose = (400.0, 0.0, 200.0, 0.0, 0.0, 0.0)
    msg = KukaPos()
    # Assign tuple values to message fields
    msg.x, msg.y, msg.z, msg.a, msg.b, msg.c = pose
    publisher.publish(msg)
    node.get_logger().info(f"Publishing KukaPos: {pose}")
    rclpy.spin_once(node, timeout_sec=0.1)


def main(args=None):
    # Initialize rclpy
    rclpy.init(args=args)
    # Create a node (no subclassing)
    node = Node('kuka_standalone_publisher')
    # Create publisher for KukaPos messages
    publisher = node.create_publisher(KukaPos, 'python/target_position', 1)


    try:
        while True:
            publish_callback(node, publisher)
            time.sleep(1)  # Sleep for 1 second to simulate 1 Hz publishing
    except KeyboardInterrupt:
        rclpy.shutdown()


if __name__ == '__main__':
    main()

