import rclpy
from rclpy.node import Node
# from KRL_Pos import KRLPos
from kuka_interfaces.srv import KukaReadVariable
from kuka_interfaces.msg import KukaWriteVariable

class KukavarproxyROS2(Node):

    # current_position = KRLPos("$POS_ACT_MES")
    def __init__(self, address = "172.31.1.147"):
        super().__init__('kukavarproxy_ros2')
        # self.kuka = KUKA(address)
        self.read_srv = self.create_service(KukaReadVariable, 'read_kuka_variable', self.read_variable)
        self.write_srv = self.create_subscription(KukaWriteVariable, 'write_kuka_variable', self.write_variable, 10)

    def read_variable(self, request, response):
        response.value = '2'
        if hasattr(self,'kuka'):
            response.value = self.kuka.read(request.name)
        self.get_logger().info(f'read_variable request: {request.name} : {response.value}')
        return response

    def write_variable(self, msg):
        response = self.kuka.write(msg.name, msg.value)
        return response


def main():
    rclpy.init()

    kuka = KukavarproxyROS2(address="172.31.1.147")
    rclpy.spin(kuka)

    rclpy.shutdown()

if __name__ == '__main__':
    main()