import threading
from collections import deque
import time

import rclpy
from rclpy.node import Node
from std_msgs.msg import Bool
from geometry_msgs.msg import (
    Pose,
    Quaternion,
    Point,
    PoseStamped,
    TransformStamped,
    Vector3,
)
from scipy.spatial.transform import Rotation as R
import math

from kukapythonvarproxy.KRL_Parameter import KRLParam

from kukapythonvarproxy.mock_kuka import fakeKUKA
from kukapythonvarproxy.KRL_Pos import KRLPos
from kukapythonvarproxy.KRL_Axis import KRLAxis
from kukapythonvarproxy.kukavarproxy import KUKA
from kuka_interfaces.msg import KukaPos, KukaWriteVariable, KukaAction
from kuka_interfaces.srv import KukaReadVariable
from rclpy.callback_groups import ReentrantCallbackGroup

from tf2_ros import TransformBroadcaster


# Connect to kuka with following steps on linux:
# sudo ip addr add 172.31.1.100/24 dev eth0
# sudo ip link set eth0 up
def main():
    rclpy.init()
    #kuka = fakeKUKA()
    kuka = KUKA("172.31.1.147")
    kuka_node = KukaControl(kuka)
    kuka_node.get_logger().info("Kuka control node started")
    rclpy.spin(kuka_node)

    rclpy.shutdown()

    if __name__ == "__main__":
        main()


class KukaControl(Node):
    com_action = KRLParam("COM_ACTION")
    target_pos = KRLPos("COM_E6POS")
    current_pos = KRLPos("$POS_ACT_MES")
    current_joint = KRLAxis("$AXIS_ACT")
    tool_frame = KRLPos("COM_FRAME")

    def __init__(self, kuka_):
        super().__init__("kuka_control")

        # self.kuka = KUKA(address)
        self.kuka = kuka_
        self.current_pos.read(self.kuka)
        self.target_pos.values = self.current_pos.values
        self.target_pos.send(self.kuka)
        self.declare_parameter("tool_frame", [0.0, 140.0, 40.0, 90.0, 0.0, -179.0])
        # KukaAction queuing and control thread
        self.action_queue = deque(maxlen=3)
        self.queue_lock = threading.Lock()
        self.condition = threading.Condition(self.queue_lock)
        self.kuka_action_sub = self.create_subscription(
            KukaAction, "kuka/action", self.kuka_action, 10
        )
        self.control_thread = threading.Thread(target=self.kuka_control, daemon=True)

        # General read and write of variables.
        self.read_srv = self.create_service(
            KukaReadVariable,
            "kuka/read_variable",
            self.read_variable,
            callback_group=ReentrantCallbackGroup(),
        )
        self.sub_write = self.create_subscription(
            KukaWriteVariable,
            "kuka/write_variable",
            self.write_variable,
            10,
            callback_group=ReentrantCallbackGroup(),
        )

        # Publish when a new target position can be sent.
        self.pub_position_ready = self.create_publisher(Bool, "kuka/position_ready", 1)

        # Publish current position
        self.pub_state = self.create_publisher(PoseStamped, "kuka/pose", 1)
        self.br = TransformBroadcaster(self)

        # self.pub_joint = self.create_publisher(JointState, "joint_states")
        self.state_timer = self.create_timer(0.05, self.publish_current_position)

        x, y, z, a, b, c = (
            self.get_parameter("tool_frame").get_parameter_value().double_array_value
        )
        self.tool_frame.set_all(x, y, z, a, b, c)

        self.load_config()

        self.control_thread.start()

    def new_target_position(self, msg):
        """Saves given target position

        Args:
            msg (KukaPos): Next target position for the kuka
        """
        self.get_logger().info(f"Received target position {msg.x}, {msg.y}")
        self.target_pos.set_all(msg.x, msg.y, msg.z, msg.a, msg.b, msg.c)
        # self.target_pos.send(self.kuka)

    def load_config(self):
        """Load kuka configuration from param file to kuka"""
        action = KukaAction()
        var = KukaWriteVariable()
        # Load kuka tool
        action.com_action = 5

        var.name = "COM_FRAME"
        var.value = self.tool_frame.get_KRL_string()
        action.variable = [var]
        self.kuka_action(action)

    def check_config(self):
        """Check if kuka configuration is correct"""
        tool_kuka = KRLPos("$TOOL")
        tool_kuka.read(self.kuka)
        if tool_kuka.values != self.tool_frame.values:
            # self.tool_frame.send(self.kuka)
            tool_var = KukaWriteVariable()
            tool_var.name = "COM_FRAME"
            tool_var.value = self.tool_frame.get_KRL_string()
            action = KukaAction()
            action.com_action = 5
            action.variable = [tool_var]
            self.kuka_action(action)
            self._wait_for_com_action_reset()

    def kuka_action(self, msg):
        with self.queue_lock:
            self.action_queue.append(msg)
            self.get_logger().info(
                f"Received new KukaAction {msg.com_action}, variable: {msg.variable}"
            )

    def kuka_control(self):
        # Wait until com_action becomes "0"
        self._wait_for_com_action_reset()
        self.pub_position_ready.publish(Bool(data=True))
        """Control loop function for Kuka. Reads com_action value"""
        while rclpy.ok():
            self.check_config()
            # If there are new KukaActions, send them
            if self.action_queue:
                self.get_logger().info(
                    f"Sending {len(self.action_queue)} KukaAction(s)"
                )
                # List copy to quickly allow new queue callbacks
                with self.queue_lock:
                    action_list = list(self.action_queue)
                # Send actions to Kuka
                for action in action_list:
                    for var in action.variable:
                        if var.name == "COM_E6POS":
                            # TODO : Check if the position is within limits
                            pass
                        self.kuka.write(var.name, var.value)
                    self.com_action.value = str(action.com_action)
                    self.com_action.send(self.kuka)
                    self._wait_for_com_action_reset()
                self.action_queue.clear()
                self.pub_position_ready.publish(Bool(data=True))

            else:
                self.check_config()
                self._wait_for_com_action_reset()
                if not self.action_queue:
                    self.pub_position_ready.publish(Bool(data=True))
            # Give others some time to publish a new action
            time.sleep(0.01)

    def _wait_for_com_action_reset(self):
        self.com_action.read(self.kuka)
        # self.get_logger().info(f"Read com_action {self.com_action.value}")
        while self.com_action.value != "0":
            time.sleep(0.01)
            self.com_action.read(self.kuka)

    def read_variable(self, request, response):
        response.value = "1"
        if hasattr(self, "kuka"):
            response.value = self.kuka.read(request.name)
        self.get_logger().info(
            f"read_variable request: {request.name} : {response.value}"
        )
        return response

    def write_variable(self, msg):
        response = self.kuka.write(msg.name, msg.value)
        self.get_logger().info(f"Wrote variable {msg.name} to value {msg.value}")
        self.get_logger().info(f"Response: {response}")
        return response

    def publish_current_position(self):
        # self.get_logger().info("Reading current position")
        try:
            self.current_pos.read(self.kuka)
        except Exception as e:
            self.get_logger().error(f"Failed to read current position: {e}")
            return

        position = Vector3()
        ori = Quaternion()
        # self.get_logger().info(f"Current position: {self.current_pos.values}")
        # Kuka messages are in mm, convert to meters
        position.x = self.current_pos.values["X"] / 1000
        position.y = self.current_pos.values["Y"] / 1000
        position.z = self.current_pos.values["Z"] / 1000

        # Degrees to quaternion
        A = math.radians(self.current_pos.values["A"])
        B = math.radians(self.current_pos.values["B"])
        C = math.radians(self.current_pos.values["C"])
        rot = R.from_euler("zyx", [A, B, C])
        ori.x, ori.y, ori.z, ori.w = rot.as_quat()
        # ori.x, ori.y, ori.z, ori.w = [0.0, 0.0, 0.0, 1.0]

        # Publish transform
        transform = TransformStamped()
        transform.header.stamp = self.get_clock().now().to_msg()
        transform.header.frame_id = "kuka_base"
        transform.child_frame_id = "kuka_tool"
        transform.transform.translation = position
        transform.transform.rotation = ori
        self.br.sendTransform(transform)
        # self.get_logger().info(f"Publishing transform {transform}")

    def destroy_node(self):
        super().destroy_node()
