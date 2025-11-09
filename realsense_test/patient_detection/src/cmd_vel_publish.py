import rclpy
from rclpy.node import Node
from geometry_msgs.msg import Twist
from gown_marker_pipeline import Artificial_Potention_Field, get_apf_inputs
#from pipeline_cmd_vel.py import Artificial_Potention_Field, get_apf_inputs

class CmdVelPublisher(Node):
    def __init__(self):
        super().__init__('cmd_vel_publisher')
        self.publisher_ = self.create_publisher(Twist, '/cmd_vel', 10)
        self.timer = self.create_timer(0.2, self.timer_callback)  # 5Hz (0.2초마다)
        self.get_logger().info('🟢 cmd_vel publisher node started.')
        

    def timer_callback(self):
        # --- 인공 퍼텐셜 필드 계산 ---
        cxy_x, real_dist_m = get_apf_inputs()
        apf_dist, apf_delta = Artificial_Potention_Field(cxy_x, real_dist_m)

        # --- Twist 메시지 생성 ---
        msg = Twist()
        msg.linear.x = float(apf_dist)     # 전진 속도
        msg.angular.z = float(apf_delta)   # 회전 속도

        # --- 퍼블리시 ---
        self.publisher_.publish(msg)
        self.get_logger().info(
            f"📤 cmd_vel → linear.x={msg.linear.x:.3f}, angular.z={msg.angular.z:.3f}"
        )



def main(args=None):
    rclpy.init(args=args)
    node = CmdVelPublisher()
    rclpy.spin(node)
    node.destroy_node()
    rclpy.shutdown()

if __name__ == '__main__':
    main()
