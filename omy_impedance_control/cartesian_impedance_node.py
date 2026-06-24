import numpy as np

import rclpy
from rclpy.node import Node

from sensor_msgs.msg import JointState
from std_msgs.msg import Float64MultiArray

from .omy import OMYConfig, OMYRobot
from .cartesian_impedance_control import (
    CartesianImpedanceGains,
    CartesianImpedanceController,
)


class CartesianImpedanceNode(Node):
    def __init__(self):
        super().__init__('cartesian_impedance_node')

        # -----------------------------
        # Parameters
        # -----------------------------
        self.declare_parameter('xml_path', '')
        self.declare_parameter('control_rate', 200.0)

        self.declare_parameter('target_offset', [0.0, 0.0, 0.0])

        self.declare_parameter('k_pos', [10.0, 10.0, 10.0])
        self.declare_parameter('d_pos', [10.0, 10.0, 10.0])

        self.declare_parameter('k_posture', [3.0, 3.0, 2.0, 1.5, 1.0, 0.8])
        self.declare_parameter('d_posture', [1.0, 1.0, 0.8, 0.6, 0.4, 0.3])

        self.declare_parameter('lambda_damping', 1.0e-4)

        # Gazebo에서 큰 토크를 넣어야 움직인다고 했으니 넉넉하게 둠
        # 단, 실물에서는 훨씬 작게 시작해야 함
        self.declare_parameter('torque_limit', [30.0, 30.0, 30.0, 15.0, 15.0, 15.0])

        # 처음에는 True로 두되, 모델 차이 때문에 이상하면 False로 테스트
        self.declare_parameter('use_bias', True)

        # 안전용: 시작 후 처음 몇 cycle은 0 torque publish
        self.declare_parameter('warmup_cycles', 50)

        xml_path = self.get_parameter('xml_path').value

        if xml_path == '':
            raise RuntimeError(
                'xml_path parameter is empty. '
                'Run with --ros-args -p xml_path:=/path/to/omy.xml'
            )

        control_rate = float(self.get_parameter('control_rate').value)
        self.dt = 1.0 / control_rate

        self.target_offset = np.array(
            self.get_parameter('target_offset').value,
            dtype=float,
        ).reshape(3)

        k_pos = np.array(self.get_parameter('k_pos').value, dtype=float).reshape(3)
        d_pos = np.array(self.get_parameter('d_pos').value, dtype=float).reshape(3)

        k_posture = np.array(
            self.get_parameter('k_posture').value,
            dtype=float,
        ).reshape(6)

        d_posture = np.array(
            self.get_parameter('d_posture').value,
            dtype=float,
        ).reshape(6)

        lambda_damping = float(self.get_parameter('lambda_damping').value)

        torque_limit = np.array(
            self.get_parameter('torque_limit').value,
            dtype=float,
        ).reshape(6)

        use_bias = bool(self.get_parameter('use_bias').value)
        self.warmup_cycles = int(self.get_parameter('warmup_cycles').value)
        self.cycle_count = 0

        # -----------------------------
        # MuJoCo model
        # -----------------------------
        cfg = OMYConfig(xml_path=xml_path)
        self.robot = OMYRobot(cfg)

        gains = CartesianImpedanceGains(
            K_pos=np.diag(k_pos),
            D_pos=np.diag(d_pos),
            K_posture=np.diag(k_posture),
            D_posture=np.diag(d_posture),
        )

        self.controller = CartesianImpedanceController(
            robot=self.robot,
            gains=gains,
            lambda_damping=lambda_damping,
            torque_limit=torque_limit,
            use_bias=use_bias,
        )

        # Gazebo / ROS 2 joint names
        self.joint_names = [
            'joint1',
            'joint2',
            'joint3',
            'joint4',
            'joint5',
            'joint6',
        ]

        self.q = np.zeros(6)
        self.qdot = np.zeros(6)
        self.has_joint_state = False
        self.initialized_target = False

        # -----------------------------
        # ROS interfaces
        # -----------------------------
        self.joint_state_sub = self.create_subscription(
            JointState,
            '/joint_states',
            self.joint_state_callback,
            10,
        )

        self.torque_pub = self.create_publisher(
            Float64MultiArray,
            '/arm_controller/commands',
            10,
        )

        self.timer = self.create_timer(self.dt, self.control_loop)

        self.get_logger().info('Cartesian impedance node started.')
        self.get_logger().info(f'xml_path: {xml_path}')
        self.get_logger().info(f'control_rate: {control_rate} Hz')
        self.get_logger().info(f'target_offset: {self.target_offset}')
        self.get_logger().info(f'k_pos: {k_pos}')
        self.get_logger().info(f'd_pos: {d_pos}')
        self.get_logger().info(f'k_posture: {k_posture}')
        self.get_logger().info(f'd_posture: {d_posture}')
        self.get_logger().info(f'lambda_damping: {lambda_damping}')
        self.get_logger().info(f'torque_limit: {torque_limit}')
        self.get_logger().info(f'use_bias: {use_bias}')

    def joint_state_callback(self, msg: JointState):
        name_to_index = {name: i for i, name in enumerate(msg.name)}

        for i, joint_name in enumerate(self.joint_names):
            if joint_name not in name_to_index:
                self.get_logger().warn(
                    f'{joint_name} not found in /joint_states',
                    throttle_duration_sec=2.0,
                )
                return

            idx = name_to_index[joint_name]

            self.q[i] = msg.position[idx]

            if len(msg.velocity) > idx:
                self.qdot[i] = msg.velocity[idx]
            else:
                self.qdot[i] = 0.0

        self.has_joint_state = True

    def publish_torque(self, tau: np.ndarray):
        msg = Float64MultiArray()
        msg.data = np.asarray(tau, dtype=float).reshape(6).tolist()
        self.torque_pub.publish(msg)

    def control_loop(self):
        if not self.has_joint_state:
            return

        q = self.q.copy()
        qdot = self.qdot.copy()

        # Gazebo joint state를 MuJoCo 계산 모델에 반영
        self.robot.set_arm_state(q, qdot)

        # 처음에는 현재 TCP 위치와 현재 자세를 목표로 잡음
        if not self.initialized_target:
            x_now = self.robot.tcp_position_world()

            q_home = self.robot.q_home.copy()

            # q_home에서의 TCP 위치 계산
            self.robot.set_arm_state(q_home, np.zeros(6))
            x_home = self.robot.tcp_position_world()

            # 다시 현재 Gazebo 상태로 복구
            self.robot.set_arm_state(self.q.copy(), self.qdot.copy())

            x_des = x_home + self.target_offset

            self.controller.set_desired_tcp_position(x_des)
            self.controller.set_desired_posture(q_home)

            self.initialized_target = True

            self.get_logger().info(f'Initial TCP position: {x_now}')
            self.get_logger().info(f'Desired TCP position: {x_des}')
            self.get_logger().info(f'Initial q_des: {q_home}')

            self.publish_torque(np.zeros(6))
            return

        # 시작 직후에는 큰 튐 방지를 위해 0 torque 몇 번 publish
        if self.cycle_count < self.warmup_cycles:
            self.cycle_count += 1
            self.publish_torque(np.zeros(6))
            return

        tau = self.controller.compute_torque()

        self.publish_torque(tau)

        # 디버깅 로그: 1초에 한 번 정도만 출력
        if self.cycle_count % 200 == 0:
            x_err = self.controller.last_x_error
            tau_task = self.controller.last_tau_task
            tau_posture = self.controller.last_tau_posture
            tau_bias = self.controller.last_tau_bias

            self.get_logger().info(
                'x_err = '
                f'{np.array2string(x_err, precision=4)}, '
                'tau = '
                f'{np.array2string(tau, precision=3)}, '
                'task = '
                f'{np.array2string(tau_task, precision=3)}, '
                'posture = '
                f'{np.array2string(tau_posture, precision=3)}, '
                'bias = '
                f'{np.array2string(tau_bias, precision=3)}'
            )

        self.cycle_count += 1


def main(args=None):
    rclpy.init(args=args)

    node = CartesianImpedanceNode()

    try:
        rclpy.spin(node)
    except KeyboardInterrupt:
        pass

    # 종료 시 0 torque 한 번 보냄
    zero_msg = Float64MultiArray()
    zero_msg.data = [0.0] * 6
    node.torque_pub.publish(zero_msg)

    node.destroy_node()
    rclpy.shutdown()


if __name__ == '__main__':
    main()