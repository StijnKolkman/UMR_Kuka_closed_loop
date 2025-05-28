import rclpy
from rclpy.node import Node
from kukapythonvarproxy.KRL_Pos import KRLPos
from kuka_interfaces.msg import KukaWriteVariable, KukaAction
from rclpy.executors import SingleThreadedExecutor
from kuka_interfaces.srv import KukaReadVariable
from kuka_interfaces.msg import KukaPos
from std_msgs.msg import Bool
import numpy as np

import time

class KukaCommander(Node):
    def __init__(self):
        super().__init__("kuka_commander")
        self.position_ready = self.create_subscription(Bool, "kuka/position_ready", self.publish_joint_move, 1)

        self.newest_position_sub = self.create_subscription(KukaPos, "python/target_position",self.update_position, 1)
        self.newest_position = None

        self.kuka_action_pub = self.create_publisher(KukaAction, "kuka/action", 10)

    def update_position(self, msg: KukaPos):
        # Callback to receive new target pose
        self.newest_position = (msg.x, msg.y, msg.z, msg.a, msg.b, msg.c)
        self.get_logger().info(f'Received new target pose: {self.newest_position}')

    def publish_joint_move(self, msg):
        if not msg.data:
            return
        
        target_pos = KukaWriteVariable()
        target_pos.name = "COM_E6POS"
        if self.newest_position is None:
            return
        x, y, z, a, b, c = self.newest_position
        target = KRLPos("COM_E6POS")
        target.set_all(x, y, z, a, b, c)
        target_pos.value = target.get_KRL_string()
        self.get_logger().info(f"Target position set to: {target_pos.value}")

        # Create the action message to send to the KUKA
        action = KukaAction()
        action.com_action = 15  # Move to position
        action.variable = [target_pos]
        self.kuka_action_pub.publish(action)
        self.get_logger().info("Published joint move command.")


def main(args=None):
    rclpy.init(args=args)
    node = KukaCommander()

    executor = SingleThreadedExecutor()
    executor.add_node(node)

    try:
        executor.spin()
    except KeyboardInterrupt:
        pass
    finally:
        executor.remove_node(node)
        node.destroy_node()
        rclpy.shutdown()

if __name__ == '__main__':
    main()
