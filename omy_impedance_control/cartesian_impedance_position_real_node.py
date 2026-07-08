import numpy as np

import rclpy
from rclpy.node import Node

from sensor_msgs.msg import JointState
from trajectory_msgs.msg import JointTrajectory
from trajectory_msgs.msg import JointTrajectoryPoint

from .cartesian_impedance_control import (
    CartesianImpedanceController,
    CartesianImpedanceGains,
)
from .omy_pinocchio import OMYConfig, OMYRobot


class CartesianImpedancePositionRealNode(Node):
    def __init__(self):
        super().__init__('cartesian_impedance_position_real_node')

        self.declare_parameter(
            'xml_path',
            '/root/ros2_ws/src/omy_impedance_control/models/omy_f3m_mujoco.urdf',
        )
        self.declare_parameter('control_rate', 400.0)
        self.declare_parameter('joint_states_topic', '/gazebo_joint_states')
        self.declare_parameter(
            'command_topic',
            '/gazebo_arm_controller/joint_trajectory',
        )

        self.declare_parameter('target_offset', [0.0, 0.0, 0.0])

        self.declare_parameter('k_pos', [2.0, 2.0, 2.0])
        self.declare_parameter('d_pos', [3.0, 3.0, 3.0])

        self.declare_parameter('k_posture', [0.5, 0.5, 0.4, 0.3, 0.2, 0.2])
        self.declare_parameter('d_posture', [0.3, 0.3, 0.2, 0.15, 0.1, 0.1])

        self.declare_parameter('lambda_damping', 1.0e-4)
        self.declare_parameter('use_bias', True)
        self.declare_parameter('warmup_cycles', 50)
        self.declare_parameter('torque_limit', [30.0, 30.0, 30.0, 30.0, 30.0, 30.0])

        self.declare_parameter('integrate_bias', False)
        self.declare_parameter('integrator_damping', 8.0)
        self.declare_parameter('mass_regularization', 1.0e-4)
        self.declare_parameter('accel_limit', [8.0, 8.0, 8.0, 10.0, 10.0, 10.0])
        self.declare_parameter('velocity_limit', [0.8, 0.8, 0.8, 1.2, 1.2, 1.2])
        self.declare_parameter('position_step_limit', [0.003, 0.003, 0.003, 0.004, 0.004, 0.004])
        self.declare_parameter('position_feedback_blend', 0.0)
        self.declare_parameter('trajectory_time_from_start', 0.05)

        xml_path = str(self.get_parameter('xml_path').value)
        if xml_path == '':
            raise RuntimeError('xml_path parameter is empty.')

        self.control_rate = float(self.get_parameter('control_rate').value)
        self.dt = 1.0 / self.control_rate
        self.joint_states_topic = str(self.get_parameter('joint_states_topic').value)
        self.command_topic = str(self.get_parameter('command_topic').value)

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

        self.integrate_bias = bool(self.get_parameter('integrate_bias').value)
        self.integrator_damping = float(
            self.get_parameter('integrator_damping').value
        )
        self.mass_regularization = float(
            self.get_parameter('mass_regularization').value
        )
        self.accel_limit = np.array(
            self.get_parameter('accel_limit').value,
            dtype=float,
        ).reshape(6)
        self.velocity_limit = np.array(
            self.get_parameter('velocity_limit').value,
            dtype=float,
        ).reshape(6)
        self.position_step_limit = np.array(
            self.get_parameter('position_step_limit').value,
            dtype=float,
        ).reshape(6)
        self.position_feedback_blend = float(
            self.get_parameter('position_feedback_blend').value
        )
        self.trajectory_time_from_start = float(
            self.get_parameter('trajectory_time_from_start').value
        )

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
        self.q_cmd = np.zeros(6)
        self.qdot_cmd = np.zeros(6)
        self.last_qddot_cmd = np.zeros(6)
        self.last_tau_integrated = np.zeros(6)
        self.has_joint_state = False
        self.initialized_target = False
        self.initialized_integrator = False

        lower = self.robot.model.lowerPositionLimit[self.robot.arm_q_ids].copy()
        upper = self.robot.model.upperPositionLimit[self.robot.arm_q_ids].copy()
        self.q_lower = np.where(np.isfinite(lower), lower, -np.inf)
        self.q_upper = np.where(np.isfinite(upper), upper, np.inf)

        self.joint_state_sub = self.create_subscription(
            JointState,
            self.joint_states_topic,
            self.joint_state_callback,
            10,
        )
        self.command_pub = self.create_publisher(
            JointTrajectory,
            self.command_topic,
            10,
        )
        self.timer = self.create_timer(self.dt, self.control_loop)

        self.get_logger().info('Cartesian impedance POSITION REAL node started.')
        self.get_logger().info('dynamics_backend: pinocchio')
        self.get_logger().info(f'xml_path: {xml_path}')
        self.get_logger().info(f'control_rate: {self.control_rate} Hz')
        self.get_logger().info(f'joint_states_topic: {self.joint_states_topic}')
        self.get_logger().info(f'command_topic: {self.command_topic}')
        self.get_logger().info(f'target_offset: {self.target_offset}')
        self.get_logger().info(f'k_pos: {k_pos}')
        self.get_logger().info(f'd_pos: {d_pos}')
        self.get_logger().info(f'k_posture: {k_posture}')
        self.get_logger().info(f'd_posture: {d_posture}')
        self.get_logger().info(f'lambda_damping: {lambda_damping}')
        self.get_logger().info(f'torque_limit: {torque_limit}')
        self.get_logger().info(f'use_bias: {use_bias}')
        self.get_logger().info(f'integrate_bias: {self.integrate_bias}')
        self.get_logger().info(f'integrator_damping: {self.integrator_damping}')
        self.get_logger().info(f'accel_limit: {self.accel_limit}')
        self.get_logger().info(f'velocity_limit: {self.velocity_limit}')
        self.get_logger().info(f'position_step_limit: {self.position_step_limit}')
        self.get_logger().info(
            f'position_feedback_blend: {self.position_feedback_blend}'
        )
        self.get_logger().info(
            f'trajectory_time_from_start: {self.trajectory_time_from_start}'
        )

    def joint_state_callback(self, msg: JointState):
        name_to_index = {name: i for i, name in enumerate(msg.name)}

        for i, joint_name in enumerate(self.joint_names):
            if joint_name not in name_to_index:
                self.get_logger().warn(
                    f'{joint_name} not found in {self.joint_states_topic}',
                    throttle_duration_sec=2.0,
                )
                return

            idx = name_to_index[joint_name]
            self.q[i] = msg.position[idx]
            self.qdot[i] = msg.velocity[idx] if len(msg.velocity) > idx else 0.0

        self.has_joint_state = True

        if not self.initialized_integrator:
            self.q_cmd = self.q.copy()
            self.qdot_cmd[:] = 0.0
            self.initialized_integrator = True

    def publish_position_command(self, q_cmd: np.ndarray):
        msg = JointTrajectory()
        msg.joint_names = self.joint_names

        point = JointTrajectoryPoint()
        point.positions = np.asarray(q_cmd, dtype=float).reshape(6).tolist()

        total_nsec = int(max(self.trajectory_time_from_start, 0.0) * 1.0e9)
        point.time_from_start.sec = total_nsec // 1_000_000_000
        point.time_from_start.nanosec = total_nsec % 1_000_000_000

        msg.points.append(point)
        self.command_pub.publish(msg)

    def integrate_torque_to_position(self, tau: np.ndarray) -> np.ndarray:
        tau_integrated = np.asarray(tau, dtype=float).reshape(6).copy()
        if not self.integrate_bias:
            tau_integrated -= self.controller.last_tau_bias

        blend = np.clip(self.position_feedback_blend, 0.0, 1.0)
        self.q_cmd = (1.0 - blend) * self.q_cmd + blend * self.q

        M = self.robot.mass_matrix_arm()
        M = M + self.mass_regularization * np.eye(6)
        damped_tau = tau_integrated - self.integrator_damping * self.qdot_cmd

        try:
            qddot_cmd = np.linalg.solve(M, damped_tau)
        except np.linalg.LinAlgError:
            qddot_cmd = self.last_qddot_cmd.copy()

        qddot_cmd = np.clip(qddot_cmd, -self.accel_limit, self.accel_limit)

        self.qdot_cmd = self.qdot_cmd + qddot_cmd * self.dt
        self.qdot_cmd = np.clip(
            self.qdot_cmd,
            -self.velocity_limit,
            self.velocity_limit,
        )

        q_step = self.qdot_cmd * self.dt
        q_step = np.clip(
            q_step,
            -self.position_step_limit,
            self.position_step_limit,
        )

        self.q_cmd = self.q_cmd + q_step
        self.q_cmd = np.clip(self.q_cmd, self.q_lower, self.q_upper)

        self.last_qddot_cmd = qddot_cmd
        self.last_tau_integrated = tau_integrated

        return self.q_cmd.copy()

    def control_loop(self):
        if not self.has_joint_state or not self.initialized_integrator:
            return

        q = self.q.copy()
        qdot = self.qdot.copy()

        self.robot.set_arm_state(q, qdot)

        if not self.initialized_target:
            x_now = self.robot.tcp_position_world()
            q_home = self.robot.q_home.copy()

            self.robot.set_arm_state(q_home, np.zeros(6))
            x_home = self.robot.tcp_position_world()

            self.robot.set_arm_state(self.q.copy(), self.qdot.copy())

            x_des = x_home + self.target_offset
            self.controller.set_desired_tcp_position(x_des)
            self.controller.set_desired_posture(q_home)

            self.q_cmd = self.q.copy()
            self.qdot_cmd[:] = 0.0
            self.initialized_target = True

            self.get_logger().info(f'Initial TCP position: {x_now}')
            self.get_logger().info(f'Home TCP position: {x_home}')
            self.get_logger().info(f'Desired TCP position: {x_des}')
            self.get_logger().info(f'Initial q_des: {q_home}')

            self.publish_position_command(self.q_cmd)
            return

        if self.cycle_count < self.warmup_cycles:
            self.cycle_count += 1
            self.publish_position_command(self.q.copy())
            return

        tau = self.controller.compute_torque()
        q_cmd = self.integrate_torque_to_position(tau)

        self.publish_position_command(q_cmd)

        if self.cycle_count % self.control_rate == 0:
            x_err = self.controller.last_x_error
            tau_task = self.controller.last_tau_task
            tau_posture = self.controller.last_tau_posture
            tau_bias = self.controller.last_tau_bias

            self.get_logger().info(
                '\n'
                '--- Cartesian impedance position debug ---\n'
                f'x_err            : {np.array2string(x_err, precision=4)}\n'
                f'tau [Nm]         : {np.array2string(tau, precision=3)}\n'
                f'tau_integrated   : {np.array2string(self.last_tau_integrated, precision=3)}\n'
                f'qddot_cmd        : {np.array2string(self.last_qddot_cmd, precision=3)}\n'
                f'qdot_cmd         : {np.array2string(self.qdot_cmd, precision=3)}\n'
                f'q_cmd [rad]      : {np.array2string(q_cmd, precision=4)}\n'
                f'q_measured [rad] : {np.array2string(q, precision=4)}\n'
                f'task             : {np.array2string(tau_task, precision=3)}\n'
                f'posture          : {np.array2string(tau_posture, precision=3)}\n'
                f'bias             : {np.array2string(tau_bias, precision=3)}'
            )

        self.cycle_count += 1


def main(args=None):
    rclpy.init(args=args)
    node = CartesianImpedancePositionRealNode()

    try:
        rclpy.spin(node)
    except KeyboardInterrupt:
        pass

    if rclpy.ok() and node.has_joint_state:
        node.publish_position_command(node.q.copy())

    node.destroy_node()
    if rclpy.ok():
        rclpy.shutdown()


if __name__ == '__main__':
    main()
