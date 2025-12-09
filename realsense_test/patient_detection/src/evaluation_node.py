# evaluation_node.py
import rclpy
from rclpy.node import Node

from geometry_msgs.msg import Twist
from nav_msgs.msg import Odometry
from std_msgs.msg import Float32

import csv
import time
import math
import matplotlib.pyplot as plt


class EvaluationNode(Node):
    def __init__(self):
        super().__init__("evaluation_node")

        # 로그 저장용 리스트
        self.cmd_log = []          # (t, linear.x, angular.z)
        self.odom_log = []         # (t, x, y, yaw)
        self.dist_log = []         # (t, real_dist)

        # 시작 시각
        self.start_time = time.time()

        # 구독 설정
        self.create_subscription(Twist, "/cmd_vel", self.cmd_callback, 10)
        self.create_subscription(Odometry, "/odom", self.odom_callback, 10)
        self.create_subscription(Float32, "/evaluation/real_dist", self.dist_callback, 10)

        # 타이머 (주기적으로 종료 체크)
        self.create_timer(0.1, self.check_finish)

        self.get_logger().info("EvaluationNode started. Collecting data for 30 seconds...")

    # ================================
    #   콜백 함수
    # ================================
    def cmd_callback(self, msg: Twist):
        t = time.time() - self.start_time
        self.cmd_log.append([t, msg.linear.x, msg.angular.z])

    def odom_callback(self, msg: Odometry):
        t = time.time() - self.start_time
        x = msg.pose.pose.position.x
        y = msg.pose.pose.position.y

        # quaternion → yaw 변환
        q = msg.pose.pose.orientation
        yaw = math.atan2(
            2.0 * (q.w*q.z + q.x*q.y),
            1.0 - 2.0 * (q.y*q.y + q.z*q.z)
        )

        self.odom_log.append([t, x, y, yaw])

    def dist_callback(self, msg: Float32):
        t = time.time() - self.start_time
        self.dist_log.append([t, float(msg.data)])

    # ================================
    #   30초 후 CSV 저장 + 그래프 출력
    # ================================
    def check_finish(self):
        if time.time() - self.start_time < 30.0:
            return

        # --------------------------
        # CSV 저장
        # --------------------------
        csv_path = "evaluation_log.csv"
        with open(csv_path, "w", newline="", encoding="utf-8") as f:
            writer = csv.writer(f)
            writer.writerow(["time", "cmd_linear", "cmd_angular", "odom_x", "odom_y", "odom_yaw", "real_dist"])

            # 시간 기준 정렬을 위해 dict 형태 병합
            max_len = max(len(self.cmd_log), len(self.odom_log), len(self.dist_log))
            for i in range(max_len):
                t = None
                cmd_l = cmd_a = ""
                ox = oy = oyaw = ""
                dist = ""

                if i < len(self.cmd_log):
                    t = self.cmd_log[i][0]
                    cmd_l, cmd_a = self.cmd_log[i][1], self.cmd_log[i][2]

                if i < len(self.odom_log):
                    ox, oy, oyaw = self.odom_log[i][1], self.odom_log[i][2], self.odom_log[i][3]

                if i < len(self.dist_log):
                    dist = self.dist_log[i][1]

                writer.writerow([t, cmd_l, cmd_a, ox, oy, oyaw, dist])

        self.get_logger().info(f"CSV saved: {csv_path}")

        # --------------------------
        # 그래프 출력
        # --------------------------
        self.plot_graph()
        rclpy.shutdown()

    # ================================
    #   Plotting
    # ================================
    def plot_graph(self):
        if len(self.cmd_log) == 0:
            self.get_logger().warning("No data for plotting.")
            return

        # CMD
        t_cmd = [c[0] for c in self.cmd_log]
        lin = [c[1] for c in self.cmd_log]
        ang = [c[2] for c in self.cmd_log]

        # REAL DIST
        t_dist = [d[0] for d in self.dist_log]
        dist = [d[1] for d in self.dist_log]

        # ODOM
        t_odom = [o[0] for o in self.odom_log]
        ox = [o[1] for o in self.odom_log]
        oy = [o[2] for o in self.odom_log]

        # --- Plot ---
        fig, axs = plt.subplots(3, 1, figsize=(10, 12))

        # 1) cmd_vel
        axs[0].plot(t_cmd, lin, label="linear.x")
        axs[0].plot(t_cmd, ang, label="angular.z")
        axs[0].set_title("cmd_vel")
        axs[0].set_xlabel("time (sec)")
        axs[0].legend()

        # 2) odom position
        axs[1].plot(ox, oy, label="Robot path (odom)")
        axs[1].set_title("Odometry trajectory")
        axs[1].set_xlabel("X")
        axs[1].set_ylabel("Y")
        axs[1].legend()
        axs[1].axis("equal")

        # 3) real distance
        axs[2].plot(t_dist, dist, color="green")
        axs[2].set_title("Real Distance (from pipeline)")
        axs[2].set_xlabel("time (sec)")
        axs[2].set_ylabel("cm")

        plt.tight_layout()
        plt.show()


def main(args=None):
    rclpy.init(args=args)
    node = EvaluationNode()
    rclpy.spin(node)


if __name__ == "__main__":
    main()

