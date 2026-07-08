import numpy as np

import rclpy
from rclpy.executors import ExternalShutdownException
from rclpy.node import Node

from sensor_msgs.msg import JointState
from std_msgs.msg import Float64MultiArray

from .cartesian_impedance_control import (
    CartesianImpedanceController,
    CartesianImpedanceGains,
    FrictionCompensationConfig,
)
from .omy_pinocchio import OMYConfig, OMYRobot


class CartesianImpedanceRealNode(Node):
    def __init__(self):
        super().__init__('cartesian_impedance_real_node')

        # -----------------------------
        # Parameters
        # -----------------------------
        self.declare_parameter('xml_path', '/root/ros2_ws/src/omy_impedance_control/models/omy_f3m_mujoco.urdf')
        self.declare_parameter('control_rate', 200.0)

        self.declare_parameter('target_offset', [0.0, 0.0, 0.0])

        # 실물 OMY 기준 Cartesian impedance gains
        self.declare_parameter('k_pos', [150.0, 150.0, 230.0])
        self.declare_parameter('d_pos', [1.5, 1.5, 2.0])

        # Joint posture gains. Joint 5/6 barely affect TCP position, so they
        # need posture gains to keep a reasonable posture in effort control.
        self.declare_parameter('k_posture', [4.0, 4.0, 3.0, 2.0, 2.0, 0.2])
        #self.declare_parameter('k_posture', [0.0, 0.0, 0.0, 0.0, 0.0, 0.0])
        self.declare_parameter('d_posture', [0.05, 0.05, 0.05, 0.05, 0.05, 0.05])
        #self.declare_parameter('d_posture', [0.0, 0.0, 0.0, 0.0, 0.0, 0.0])

        self.declare_parameter('lambda_damping', 1.0e-4)
        self.declare_parameter('task_torque_scale', [1.0, 1.0, 1.0, 1.0, 3.0, 1.0])
        self.declare_parameter('friction_compensation_enabled', True)
        self.declare_parameter(
            'kinetic_friction_scalars',
            [0.2, 0.2, 0.2, 0.05, 0.05, 0.05],
        )
        self.declare_parameter(
            'kinetic_friction_torque_scalars',
            [0.0, 0.0, 0.0, 0.0, 0.0, 0.0],
        )
        self.declare_parameter(
            'friction_compensation_velocity_thresholds',
            [1.0, 1.0, 1.0, 1.0, 1.0, 1.0],
        )
        self.declare_parameter(
            'static_friction_scalars',
            [0.0, 0.0, 0.0, 0.0, 0.0, 0.0],
        )
        self.declare_parameter(
            'static_friction_velocity_thresholds',
            [0.0, 0.0, 0.0, 0.0, 0.0, 0.0],
        )
        self.declare_parameter(
            'unloaded_effort_offsets',
            [0.0, 0.0, 0.0, 0.0, 0.0, 0.0],
        )
        self.declare_parameter(
            'unloaded_effort_thresholds',
            [0.0, 0.0, 0.0, 0.0, 0.0, 0.0],
        )

        # 중력/bias 보상 사용 여부
        self.declare_parameter('use_bias', True)

        # Pinocchio 중력 보상 계산에만 추가되는 가상 payload.
        # 실제 URDF를 직접 고치지 않고 엔드이펙터 쪽 질량을 보강한다.
        self.declare_parameter('payload_mass', 1.5)
        self.declare_parameter('payload_com_in_ee', [0.0, -0.109, 0.0])
        self.declare_parameter('payload_inertia_diag', [1.0e-4, 1.0e-4, 1.0e-4])

        # Gazebo 코드와 동일하게 시작 후 몇 cycle 동안 0 command publish
        self.declare_parameter('warmup_cycles', 50)

        self.declare_parameter('torque_limit', [60.0, 70.0, 110.0, 35.0, 30.0, 30.0])

        self.declare_parameter('joint_states_topic', '/joint_states')

        # command topic
        # Effort controller: torque[Nm]
        self.declare_parameter('command_topic', '/arm_controller/commands')

        xml_path = self.get_parameter('xml_path').value

        if xml_path == '':
            raise RuntimeError(
                'xml_path parameter is empty. '
                'Run with --ros-args -p xml_path:=/path/to/omy.xml'
            )

        self.control_rate = float(self.get_parameter('control_rate').value)
        self.dt = 1.0 / self.control_rate

        self.target_offset = np.array(
            self.get_parameter('target_offset').value,
            dtype=float,
        ).reshape(3)

        k_pos = np.array(
            self.get_parameter('k_pos').value,
            dtype=float,
        ).reshape(3)

        d_pos = np.array(
            self.get_parameter('d_pos').value,
            dtype=float,
        ).reshape(3)

        k_posture = np.array(
            self.get_parameter('k_posture').value,
            dtype=float,
        ).reshape(6)

        d_posture = np.array(
            self.get_parameter('d_posture').value,
            dtype=float,
        ).reshape(6)

        lambda_damping = float(self.get_parameter('lambda_damping').value)
        task_torque_scale = np.array(
            self.get_parameter('task_torque_scale').value,
            dtype=float,
        ).reshape(6)
        friction_enabled = bool(
            self.get_parameter('friction_compensation_enabled').value
        )
        friction_config = FrictionCompensationConfig(
            enabled=friction_enabled,
            kinetic_friction_scalars=np.array(
                self.get_parameter('kinetic_friction_scalars').value,
                dtype=float,
            ).reshape(6),
            kinetic_friction_torque_scalars=np.array(
                self.get_parameter('kinetic_friction_torque_scalars').value,
                dtype=float,
            ).reshape(6),
            friction_compensation_velocity_thresholds=np.array(
                self.get_parameter(
                    'friction_compensation_velocity_thresholds'
                ).value,
                dtype=float,
            ).reshape(6),
            static_friction_scalars=np.array(
                self.get_parameter('static_friction_scalars').value,
                dtype=float,
            ).reshape(6),
            static_friction_velocity_thresholds=np.array(
                self.get_parameter(
                    'static_friction_velocity_thresholds'
                ).value,
                dtype=float,
            ).reshape(6),
            unloaded_effort_offsets=np.array(
                self.get_parameter('unloaded_effort_offsets').value,
                dtype=float,
            ).reshape(6),
            unloaded_effort_thresholds=np.array(
                self.get_parameter('unloaded_effort_thresholds').value,
                dtype=float,
            ).reshape(6),
        )

        torque_limit = np.array(
            self.get_parameter('torque_limit').value,
            dtype=float,
        ).reshape(6)

        use_bias = bool(self.get_parameter('use_bias').value)

        payload_mass = max(float(self.get_parameter('payload_mass').value), 0.0)
        payload_com_in_ee = np.array(
            self.get_parameter('payload_com_in_ee').value,
            dtype=float,
        ).reshape(3)
        payload_inertia_diag = np.maximum(
            np.array(
                self.get_parameter('payload_inertia_diag').value,
                dtype=float,
            ).reshape(3),
            1.0e-9,
        )

        self.warmup_cycles = int(self.get_parameter('warmup_cycles').value)
        self.cycle_count = 0

        self.joint_states_topic = str(self.get_parameter('joint_states_topic').value)
        self.command_topic = str(self.get_parameter('command_topic').value)

        # -----------------------------
        # Pinocchio model
        # -----------------------------
        cfg = OMYConfig(
            xml_path=xml_path,
            payload_mass=payload_mass,
            payload_com_in_ee=payload_com_in_ee,
            payload_inertia_diag=payload_inertia_diag,
        )
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
            task_torque_scale=task_torque_scale,
            friction=friction_config,
        )

        # Real OMY / ROS 2 joint names
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
            self.joint_states_topic,
            self.joint_state_callback,
            10,
        )

        self.command_pub = self.create_publisher(
            Float64MultiArray,
            self.command_topic,
            10,
        )

        self.timer = self.create_timer(self.dt, self.control_loop)

        self.get_logger().info('Cartesian impedance REAL node started.')
        self.get_logger().info('dynamics_backend: pinocchio')
        self.get_logger().info(f'xml_path: {xml_path}')
        self.get_logger().info(f'control_rate: {self.control_rate} Hz')
        self.get_logger().info(f'target_offset: {self.target_offset}')
        self.get_logger().info(f'k_pos: {k_pos}')
        self.get_logger().info(f'd_pos: {d_pos}')
        self.get_logger().info(f'k_posture: {k_posture}')
        self.get_logger().info(f'd_posture: {d_posture}')
        self.get_logger().info(f'lambda_damping: {lambda_damping}')
        self.get_logger().info(f'task_torque_scale: {task_torque_scale}')
        self.get_logger().info(
            f'friction_compensation_enabled: {friction_config.enabled}'
        )
        self.get_logger().info(
            'kinetic_friction_scalars: '
            f'{friction_config.kinetic_friction_scalars}'
        )
        self.get_logger().info(
            'kinetic_friction_torque_scalars: '
            f'{friction_config.kinetic_friction_torque_scalars}'
        )
        self.get_logger().info(
            'friction_compensation_velocity_thresholds: '
            f'{friction_config.friction_compensation_velocity_thresholds}'
        )
        self.get_logger().info(
            f'static_friction_scalars: {friction_config.static_friction_scalars}'
        )
        self.get_logger().info(
            'static_friction_velocity_thresholds: '
            f'{friction_config.static_friction_velocity_thresholds}'
        )
        self.get_logger().info(
            f'unloaded_effort_offsets: {friction_config.unloaded_effort_offsets}'
        )
        self.get_logger().info(
            'unloaded_effort_thresholds: '
            f'{friction_config.unloaded_effort_thresholds}'
        )
        self.get_logger().info(f'torque_limit: {torque_limit}')
        self.get_logger().info(f'use_bias: {use_bias}')
        self.get_logger().info(f'payload_mass [kg]: {payload_mass}')
        self.get_logger().info(f'payload_com_in_ee [m]: {payload_com_in_ee}')
        self.get_logger().info(
            f'payload_inertia_diag [kg*m^2]: {payload_inertia_diag}'
        )
        self.get_logger().info(f'warmup_cycles: {self.warmup_cycles}')
        self.get_logger().info(f'joint_states_topic: {self.joint_states_topic}')
        self.get_logger().info(f'command_topic: {self.command_topic}')

    def joint_state_callback(self, msg: JointState):
        name_to_index = {name: i for i, name in enumerate(msg.name)}
        qdot = np.zeros(6)

        for i, joint_name in enumerate(self.joint_names):
            if joint_name not in name_to_index:
                self.get_logger().warn(
                    f'{joint_name} not found in {self.joint_states_topic}',
                    throttle_duration_sec=2.0,
                )
                return

            idx = name_to_index[joint_name]

            self.q[i] = msg.position[idx]

            if len(msg.velocity) > idx:
                qdot[i] = msg.velocity[idx]
            else:
                qdot[i] = 0.0

        self.qdot = qdot
        self.has_joint_state = True

    def publish_command(self, command: np.ndarray):
        """Publish the selected command."""
        msg = Float64MultiArray()
        msg.data = np.asarray(command, dtype=float).reshape(6).tolist()
        self.command_pub.publish(msg)

    def publish_zero_command(self):
        """Publish zero commands."""
        self.publish_command(np.zeros(6))

    def control_loop(self):
        if not self.has_joint_state:
            return

        q = self.q.copy()
        qdot = self.qdot.copy()

        # Real joint state를 Pinocchio 계산 모델에 반영
        self.robot.set_arm_state(q, qdot)

        # 처음에는 q_home 자세를 목표로 잡음
        if not self.initialized_target:
            x_now = self.robot.tcp_position_world()

            q_home = self.robot.q_home.copy()

            # q_home에서의 TCP 위치 계산
            self.robot.set_arm_state(q_home, np.zeros(6))
            x_home = self.robot.tcp_position_world()

            # 다시 현재 실물 상태로 복구
            self.robot.set_arm_state(self.q.copy(), self.qdot.copy())

            x_des = x_home + self.target_offset
            q_des = q_home
            target_source = 'home TCP position + target_offset'

            self.controller.set_desired_tcp_position(x_des)
            self.controller.set_desired_posture(q_des)

            self.initialized_target = True

            self.get_logger().info(f'Initial TCP position: {x_now}')
            self.get_logger().info(f'Initial target source: {target_source}')
            self.get_logger().info(f'Desired TCP position: {x_des}')
            self.get_logger().info(f'Initial q_des: {q_des}')

            self.publish_zero_command()
            return

        # Gazebo 코드와 동일하게 시작 직후 0 command 몇 번 publish
        if self.cycle_count < self.warmup_cycles:
            self.cycle_count += 1
            self.publish_zero_command()
            return

        # impedance controller는 desired torque tau[Nm] 계산
        tau = self.controller.compute_torque()

        if not np.all(np.isfinite(tau)):
            self.get_logger().error('Computed tau contains NaN or inf; publishing zero')
            self.publish_zero_command()
            self.cycle_count += 1
            return

        self.publish_command(tau)

        # 디버깅 로그: 1초에 한 번 정도만 출력
        if self.cycle_count % self.control_rate == 0:
            x_err = self.controller.last_x_error
            tau_task = self.controller.last_tau_task
            tau_posture = self.controller.last_tau_posture
            tau_bias = self.controller.last_tau_bias
            tau_friction = self.controller.last_tau_friction

            self.get_logger().info(
                '\n'
                '--- Cartesian impedance debug ---\n'
                f'x_err        : {np.array2string(x_err, precision=4)}\n'
                f'x_des        : {np.array2string(self.controller.x_des, precision=4)}\n'
                f'tau [Nm]     : {np.array2string(tau, precision=3)}\n'
                f'qdot         : {np.array2string(self.qdot, precision=4)}\n'
                f'task         : {np.array2string(tau_task, precision=3)}\n'
                f'posture      : {np.array2string(tau_posture, precision=3)}\n'
                f'bias         : {np.array2string(tau_bias, precision=3)}\n'
                f'friction     : {np.array2string(tau_friction, precision=3)}'
            )

        self.cycle_count += 1


def main(args=None):
    rclpy.init(args=args)

    node = CartesianImpedanceRealNode()

    try:
        rclpy.spin(node)
    except (KeyboardInterrupt, ExternalShutdownException):
        pass
    finally:
        # 종료 시 context가 살아 있으면 0 command 한 번 보냄
        if rclpy.ok():
            zero_msg = Float64MultiArray()
            zero_msg.data = [0.0] * 6
            node.command_pub.publish(zero_msg)

        node.destroy_node()
        if rclpy.ok():
            rclpy.shutdown()


if __name__ == '__main__':
    main()
