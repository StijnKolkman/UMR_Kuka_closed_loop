import rclpy
from rclpy.node import Node
from kuka_interfaces.msg import KukaAction, KukaWriteVariable


class KukaActionPublisher(Node):
    def __init__(self):
        super().__init__("kuka_action_publisher")

        # Create publisher
        self.publisher_ = self.create_publisher(KukaAction, "kuka/action", 10)

        # Timer to publish the action every 1 second
        self.timer = self.create_timer(1.0, self.publish_kuka_action)

        # Creating a sample action message
        self.action_msg = KukaAction()
        self.action_msg.com_action = 4  # Setting initial com_action to 0

        # Fill in KukaWriteVariable values (example)
        var1 = KukaWriteVariable(name="VAR1", value="100")
        var2 = KukaWriteVariable(name="VAR2", value="200")
        var3 = KukaWriteVariable(name="VAR3", value="1")

        # Add the variables to the KukaAction message
        self.action_msg.variable = [var1, var2, var3]

    def publish_kuka_action(self):
        """Publish a KukaAction message."""
        self.publisher_.publish(self.action_msg)
        self.get_logger().info(
            f"Publishing: com_action={self.action_msg.com_action}, variables={len(self.action_msg.variable)}"
        )


def main():
    rclpy.init()

    kuka_action_publisher = KukaActionPublisher()
    rclpy.spin(kuka_action_publisher)

    rclpy.shutdown()


if __name__ == "__main__":
    main()
