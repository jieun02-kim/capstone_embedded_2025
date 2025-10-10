import rclpy
from rclpy.node import Node
from geometry_msgs.msg import Twist
from kobuki_ros2_driver.TMRKobuki2 import CTMRKobuki2

class MotorDriverNode(Node):
    def __init__(self):
        super().__init__('motor_driver_node')
        self.kobuki = CTMRKobuki2()
        self.kobuki.connect('/dev/ttyUSB0')  # 실제 장치 포트로 변경 가능

        self.subscription = self.create_subscription(
            Twist,
            '/cmd_vel',
            self.cmd_vel_callback,
            10
        )
        self.get_logger().info("MotorDriverNode started, waiting for /cmd_vel...")

    def cmd_vel_callback(self, msg):
        linear = msg.linear.x
        angular = msg.angular.z
        self.kobuki.set_velocity_control(linear, angular)

def main(args=None):
    rclpy.init(args=args)
    node = MotorDriverNode()
    try:
        rclpy.spin(node)
    except KeyboardInterrupt:
        pass
    node.destroy_node()
    rclpy.shutdown()
