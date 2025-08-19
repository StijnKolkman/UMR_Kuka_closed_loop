import rclpy
from rclpy.node import Node
from std_msgs.msg import Bool

from kukapythonvarproxy.KRL_Pos import KRLPos
from kukapythonvarproxy.KRL_Parameter import KRLParam

from kuka_interfaces.msg import KukaWriteVariable, KukaAction, KukaPos
from kuka_interfaces.srv import KukaReadVariable

class KukaCommander(Node):
    """ROS2 node that listens for target poses and triggers KUKA moves when ready."""

    def __init__(self):
        super().__init__("kuka_commander")

        # Initialize the node
        self.position_ready_sub = self.create_subscription(Bool, "kuka/position_ready", self.publish_joint_move, 1)
        self.newest_position_sub = self.create_subscription(KukaPos, "python/target_position",self.update_position, 1)
        self.kuka_action_pub = self.create_publisher(KukaAction, "kuka/action", 10)

        # Set the correct rounding value
        self.set_rounding("20")

        # Initialize newest_position
        self.newest_position = None

    def update_position(self, msg: KukaPos):
        """Update the newest position from the received KukaPos message."""

        self.newest_position = (msg.x, msg.y, msg.z, msg.a, msg.b, msg.c)
        #self.get_logger().info(f'Received new target pose: {self.newest_position}')

    def publish_joint_move(self, msg):
        """Publish a joint move command to the KUKA when the position is ready."""

        if not msg.data:
            return
    
        if self.newest_position is None:
            return

        if any(v is None for v in self.newest_position):
            self.get_logger().warn("Received incomplete or missing position data.")
            return
        
        target_pos = KukaWriteVariable()
        target_pos.name = "COM_E6POS"
        target = KRLPos("COM_E6POS")
        x, y, z, a, b, c = self.newest_position
        target.set_all(x, y, z, a, b, c)
        target_pos.value = target.get_KRL_string()
        #self.get_logger().info(f"Target position set to: {target_pos.value}")

        # Create the action message to send to the KUKA
        action = KukaAction()
        action.com_action = 15  # Move to position
        action.variable = [target_pos]
        self.kuka_action_pub.publish(action)
        #self.get_logger().info("Published joint move command.")

    def set_rounding(self, rounding_value):
        """Set the rounding value for KUKA commands."""
        
        action = KukaAction()
        rounding = KRLParam("COM_ROUNDM")
        rounding.set_value(rounding_value)
        action.com_action = 8
        rounding_var = KukaWriteVariable()
        rounding_var.name = "COM_ROUNDM"
        rounding_var.value = rounding.get_KRL_string()
        action.variable = [rounding_var]
        self.kuka_action_pub.publish(action)
      
def main(args=None):
    rclpy.init(args=args)
    node = KukaCommander()
    try:
        rclpy.spin(node)
    except KeyboardInterrupt:
        pass
    finally: 
        node.destroy_node()
        rclpy.shutdown()

if __name__ == '__main__':
    main()
