import rclpy
from rclpy.node import Node
from kukapythonvarproxy.KRL_Pos import KRLPos
from kuka_interfaces.msg import KukaWriteVariable, KukaAction
from kuka_interfaces.srv import KukaReadVariable
from std_msgs.msg import Bool
import numpy as np

import time


class CSVKukaPath(Node):
    target_pos = KRLPos("COM_E6POS")

    def __init__(self, csv_file):
        super().__init__("csv_path_follow")
        self.declare_parameter("csv_file", "measure_path.csv")
        self.path = np.genfromtxt(self.get_parameter("csv_file").value, delimiter=",")
        self.index = 0
        self.pos_ready = self.create_subscription(
            Bool, "kuka/position_ready", self.next_position, 1
        )
        #self.target_subscription =  telkens bij inkomende waarde uitlezen (en die sturen we dan alleen door als pos_ready)
        self.read_cli = self.create_client(KukaReadVariable, "kuka/read_variable")
        self.robo_action = self.create_publisher(KukaAction, "kuka/action", 3)

        req = KukaReadVariable.Request()
        req.name = "$POS_ACT"
        self.base_position = None
        future = self.read_cli.call_async(req)
        future.add_done_callback(self.get_base_position)

        rounding = KukaAction()
        roundm = KukaWriteVariable()
        roundm.name = "COM_ROUNDM"
        roundm.value = "2"
        rounding.variable = [roundm]
        rounding.com_action = 8
        self.robo_action.publish(rounding)

    def next_position(self, msg):
        if msg.data == False:
            return
        if self.base_position == None:
            return
        self.index += 1
        self.index = max(0, min(self.index, self.path.shape[0]))
        x, y, z, a, b, c = self.get_corresponding_actuator_pose()
        self.target_pos.set_all(x, y, z, a, b, c)
        self.get_logger().info(f"Target: {self.target_pos.values}")
        self.target_pos += self.base_position
        self.get_logger().info(f"Target + base: {self.target_pos.values}")
        msg = KukaWriteVariable()
        msg.name = "COM_E6POS"
        msg.value = self.target_pos.get_KRL_string()
        action = KukaAction()
        action.variable = [msg]
        action.com_action = 15
        self.robo_action.publish(action)
        # self.get_logger().info(f"Sent new target position {self.index}")
        return

    def get_corresponding_actuator_pose(self):
        # Get the actuator pose corresponding to the vessel position, based on the z-offset
        targets = self.path[self.index, 0:3]  # position vessel wrt actuator base
        angles = self.path[self.index, 3:6]
        t_matrix = self.transformation_matrix(targets, angles)
        actuator_pose = t_matrix @ [0, 0, 0, 1]
        values = [
            actuator_pose[0],
            actuator_pose[1],
            actuator_pose[2],
            np.rad2deg(angles[2]),
            np.rad2deg(angles[1]),
            np.rad2deg(angles[0]),
        ]
        return values

    def update_path_orientation(self, angles):
        if self.pulling:
            angles[2] = (
                angles[2] + np.pi
            )  # If pulling, angles and orientation are flipped

        for i in range(0, self.lenght_path):
            vessel_target = self.vesselPath[i, 0:3]
            vessel_angle = self.vesselPath[i, 3:6]
            t_matrix = CSVKukaPath.transformation_matrix(
                [0, 0, 0], angles
            )  # only orientation change, no position
            new_vessel_target = t_matrix @ np.append(vessel_target, 1)
            self.path[i, 0:3] = new_vessel_target[0:3]
            self.path[i, 3:6] = -(vessel_angle + angles)
            if self.pulling:
                self.path[i, 5] = (
                    self.path[i, 5] - np.pi
                )  # If pulling, angles and orientation are flipped

    def get_base_position(self, future):
        response = future.result()
        self.base_position = KRLPos("Base")
        self.base_position.set_all(
            570.64,
            469.6,
            605.81,
            90,
            0,
            180,
        )
        # self.base_position.from_KRL_string(response.value)
        self.get_logger().info(f"Start position{self.base_position.values}")

    def update_path_position(self, position):
        self.base_position = np.array(position)

    def update_base_frame(self, position, angles):
        self.update_path_position(position)
        self.update_path_orientation(angles)

    def transformation_matrix(self, translation, rotation):
        """
        Create a 4x4 transformation matrix for 3D transformations.

        :param scale: (sx, sy, sz) scaling factors
        :param rotation: (rx, ry, rz) rotation angles in radians
        :param translation: (tx, ty, tz) translation values
        :return: 4x4 transformation matrix
        """
        tx, ty, tz = translation
        rx, ry, rz = rotation

        # Rotation matrices around X, Y, Z axes
        Rx = np.array(
            [[1, 0, 0], [0, np.cos(rx), -np.sin(rx)], [0, np.sin(rx), np.cos(rx)]]
        )

        Ry = np.array(
            [[np.cos(ry), 0, np.sin(ry)], [0, 1, 0], [-np.sin(ry), 0, np.cos(ry)]]
        )

        Rz = np.array(
            [[np.cos(rz), -np.sin(rz), 0], [np.sin(rz), np.cos(rz), 0], [0, 0, 1]]
        )

        # Combined rotation matrix R = Rz * Ry * Rx
        R = Rz @ Ry @ Rx

        # Create 4x4 transformation matrix
        T = np.eye(4)
        T[:3, :3] = R  # Apply rotation
        T[:3, 3] = [tx, ty, tz]  # Apply translation

        return T


def main():
    rclpy.init()
    file_name = "measure_path.csv"
    csv = CSVKukaPath(file_name)
    time.sleep(0.1)
    rclpy.spin(csv)

    rclpy.shutdown()

    if __name__ == "__main__":
        main()
