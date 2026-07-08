import numpy as np

import rclpy
from rclpy.executors import ExternalShutdownException
from rclpy.node import Node

from geometry_msgs.msg import PointStamped
from sensor_msgs.msg import JointState
from std_msgs.msg import Float64MultiArray

from .cartesian_impedance_control import (
    CartesianImpedanceController,
    CartesianImpedanceGains,
    FrictionCompensationConfig,
)
from .omy_pinocchio import OMYConfig, OMYRobot


class CartesianImpedanceRealTeleopNode(Node):
    def __init__(self):
        super().__init__('cartesian_impedance_real_teleop_node')

        # -----------------------------
        # Parameters
        # -----------------------------
        self.declare_parameter(
            'xml_path',
            '/root/ros2_ws/src/omy_impedance_control/models/omy_f3m_mujoco.urdf',
        )
        self.declare_parameter('control_rate', 200.0)

        # Teleop target is expected to be the final follower TCP target.
        # If frame calibration is still needed, apply a small extra offset here.
        self.declare_parameter('target_offset', [0.0, 0.0, 0.0])

        # Conservative defaults for leader-driven target motion.
        self.declare_parameter('k_pos', [150.0, 150.0, 200.0])
        self.declare_parameter('d_pos', [1.5, 1.5, 2.0])

        # Teleop posture is a weak leader-joint reference. Keep it softer than
        # Cartesian impedance so external contact does not become joint servoing.
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

        self.declare_parameter('use_bias', True)
        self.declare_parameter('payload_mass', 1.5)
        self.declare_parameter('payload_com_in_ee', [0.0, -0.109, 0.0])
        self.declare_parameter('payload_inertia_diag', [1.0e-4, 1.0e-4, 1.0e-4])

        self.declare_parameter('warmup_cycles', 50)
        self.declare_parameter('torque_limit', [60.0, 70.0, 110.0, 35.0, 30.0, 30.0])

        self.declare_parameter('joint_states_topic', '/joint_states')
        self.declare_parameter('command_topic', '/arm_controller/commands')
        self.declare_parameter(
            'target_position_topic',
            '/cartesian_impedance/target_position',
        )
        self.declare_parameter('target_filter_alpha', 0.15)
        self.declare_parameter('max_target_step', 0.002)
        self.declare_parameter('target_timeout_sec', 0.5)
        self.declare_parameter('use_leader_posture', True)
        self.declare_parameter('leader_joint_states_topic', '/leader/joint_states')
        self.declare_parameter(
            'leader_joint_names',
            ['joint1', 'joint2', 'joint3', 'joint4', 'joint5', 'joint6'],
        )
        self.declare_parameter('leader_joint_sign', [1.0] * 6)
        self.declare_parameter('leader_joint_offset', [0.0] * 6)
        self.declare_parameter('leader_joint_timeout_sec', 0.5)

        xml_path = self.get_parameter('xml_path').value
        if xml_path == '':
            raise RuntimeError(
                'xml_path parameter is empty. '
                'Run with --ros-args -p xml_path:=/path/to/omy.xml'
            )

        self.control_rate = float(self.get_parameter('control_rate').value)
        self.dt = 1.0 / self.control_rate
        self.log_period_cycles = max(1, int(round(self.control_rate)))

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
        self.target_position_topic = str(
            self.get_parameter('target_position_topic').value
        )
        self.target_filter_alpha = float(
            self.get_parameter('target_filter_alpha').value
        )
        self.target_filter_alpha = float(
            np.clip(self.target_filter_alpha, 0.0, 1.0)
        )
        self.max_target_step = max(
            float(self.get_parameter('max_target_step').value),
            0.0,
        )
        self.target_timeout_sec = max(
            float(self.get_parameter('target_timeout_sec').value),
            0.0,
        )
        self.use_leader_posture = bool(
            self.get_parameter('use_leader_posture').value
        )
        self.leader_joint_states_topic = str(
            self.get_parameter('leader_joint_states_topic').value
        )
        self.leader_joint_names = list(
            self.get_parameter('leader_joint_names').value
        )
        if len(self.leader_joint_names) != 6:
            raise RuntimeError(
                'leader_joint_names must contain exactly 6 joint names'
            )

        self.leader_joint_sign = np.array(
            self.get_parameter('leader_joint_sign').value,
            dtype=float,
        ).reshape(6)
        self.leader_joint_offset = np.array(
            self.get_parameter('leader_joint_offset').value,
            dtype=float,
        ).reshape(6)
        self.leader_joint_timeout_sec = max(
            float(self.get_parameter('leader_joint_timeout_sec').value),
            0.0,
        )

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

        self.q_leader = np.zeros(6)
        self.has_leader_joint_state = False
        self.last_leader_joint_time = None

        self.target_position = np.zeros(3)
        self.has_target_position = False
        self.last_target_time = None

        self.initialized_target = False
        self.waiting_log_count = 0

        # -----------------------------
        # ROS interfaces
        # -----------------------------
        self.joint_state_sub = self.create_subscription(
            JointState,
            self.joint_states_topic,
            self.joint_state_callback,
            10,
        )

        self.target_position_sub = self.create_subscription(
            PointStamped,
            self.target_position_topic,
            self.target_position_callback,
            10,
        )

        self.leader_joint_state_sub = self.create_subscription(
            JointState,
            self.leader_joint_states_topic,
            self.leader_joint_state_callback,
            10,
        )

        self.command_pub = self.create_publisher(
            Float64MultiArray,
            self.command_topic,
            10,
        )

        self.timer = self.create_timer(self.dt, self.control_loop)

        self.get_logger().info('Cartesian impedance REAL TELEOP node started.')
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
        self.get_logger().info(f'target_position_topic: {self.target_position_topic}')
        self.get_logger().info(f'target_filter_alpha: {self.target_filter_alpha}')
        self.get_logger().info(f'max_target_step: {self.max_target_step}')
        self.get_logger().info(f'target_timeout_sec: {self.target_timeout_sec}')
        self.get_logger().info(f'use_leader_posture: {self.use_leader_posture}')
        self.get_logger().info(
            f'leader_joint_states_topic: {self.leader_joint_states_topic}'
        )
        self.get_logger().info(f'leader_joint_names: {self.leader_joint_names}')
        self.get_logger().info(f'leader_joint_sign: {self.leader_joint_sign}')
        self.get_logger().info(f'leader_joint_offset: {self.leader_joint_offset}')
        self.get_logger().info(
            f'leader_joint_timeout_sec: {self.leader_joint_timeout_sec}'
        )

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

    def leader_joint_state_callback(self, msg: JointState):
        name_to_index = {name: i for i, name in enumerate(msg.name)}
        q_leader = np.zeros(6)

        for i, joint_name in enumerate(self.leader_joint_names):
            if joint_name not in name_to_index:
                self.get_logger().warn(
                    f'{joint_name} not found in {self.leader_joint_states_topic}',
                    throttle_duration_sec=2.0,
                )
                return

            q_leader[i] = msg.position[name_to_index[joint_name]]

        if not np.all(np.isfinite(q_leader)):
            self.get_logger().warn(
                'Leader joint state contains NaN or inf; ignoring',
                throttle_duration_sec=1.0,
            )
            return

        self.q_leader = q_leader
        self.last_leader_joint_time = self.get_clock().now()
        self.has_leader_joint_state = True

    def target_position_callback(self, msg: PointStamped):
        target_position = np.array(
            [msg.point.x, msg.point.y, msg.point.z],
            dtype=float,
        )
        target_position = target_position + self.target_offset

        if not np.all(np.isfinite(target_position)):
            self.get_logger().warn(
                'Target position contains NaN or inf; ignoring',
                throttle_duration_sec=1.0,
            )
            return

        self.target_position = target_position
        self.last_target_time = self.get_clock().now()
        self.has_target_position = True

    def target_is_fresh(self) -> bool:
        if not self.has_target_position or self.last_target_time is None:
            return False

        if self.target_timeout_sec <= 0.0:
            return True

        age = self.get_clock().now() - self.last_target_time
        return age.nanoseconds <= self.target_timeout_sec * 1.0e9

    def leader_posture_is_ready(self) -> bool:
        if not self.use_leader_posture:
            return True

        if not self.has_leader_joint_state or self.last_leader_joint_time is None:
            return False

        if self.leader_joint_timeout_sec <= 0.0:
            return True

        age = self.get_clock().now() - self.last_leader_joint_time
        return age.nanoseconds <= self.leader_joint_timeout_sec * 1.0e9

    def mapped_leader_q_des(self) -> np.ndarray:
        q_des = self.leader_joint_sign * self.q_leader + self.leader_joint_offset
        return q_des

    def filtered_external_target(self, target_position: np.ndarray) -> np.ndarray:
        previous = self.controller.x_des.copy()
        candidate = (
            self.target_filter_alpha * target_position
            + (1.0 - self.target_filter_alpha) * previous
        )

        delta = candidate - previous
        delta_norm = np.linalg.norm(delta)
        if self.max_target_step > 0.0 and delta_norm > self.max_target_step:
            candidate = previous + delta / delta_norm * self.max_target_step

        return candidate

    def update_external_target(self) -> bool:
        if not self.target_is_fresh():
            self.get_logger().warn(
                'Leader target is missing or stale; publishing zero command',
                throttle_duration_sec=1.0,
            )
            return False

        x_des = self.filtered_external_target(self.target_position)
        if not np.all(np.isfinite(x_des)):
            self.get_logger().warn(
                'Filtered target position contains NaN or inf; ignoring',
                throttle_duration_sec=1.0,
            )
            return False

        self.controller.set_desired_tcp_position(x_des)
        return True

    def update_leader_posture_target(self) -> bool:
        if not self.use_leader_posture:
            return True

        if not self.leader_posture_is_ready():
            self.get_logger().warn(
                'Leader joint state is missing or stale; publishing zero command',
                throttle_duration_sec=1.0,
            )
            return False

        q_des = self.mapped_leader_q_des()
        if not np.all(np.isfinite(q_des)):
            self.get_logger().warn(
                'Mapped leader posture contains NaN or inf; ignoring',
                throttle_duration_sec=1.0,
            )
            return False

        self.controller.set_desired_posture(q_des)
        return True

    def publish_command(self, command: np.ndarray):
        msg = Float64MultiArray()
        msg.data = np.asarray(command, dtype=float).reshape(6).tolist()
        self.command_pub.publish(msg)

    def publish_zero_command(self):
        self.publish_command(np.zeros(6))

    def wait_for_start_inputs(self) -> bool:
        if (
            self.has_joint_state
            and self.target_is_fresh()
            and self.leader_posture_is_ready()
        ):
            return True

        self.waiting_log_count += 1
        if self.waiting_log_count % self.log_period_cycles == 0:
            self.get_logger().info(
                'Waiting for follower joint state, leader target position, '
                'and leader posture...'
            )

        return False

    def initialize_from_leader_target(self) -> None:
        self.robot.set_arm_state(self.q.copy(), self.qdot.copy())
        x_now = self.robot.tcp_position_world()
        x_des = self.target_position.copy()
        if self.use_leader_posture:
            q_des = self.mapped_leader_q_des()
        else:
            q_des = self.q.copy()

        self.controller.set_desired_tcp_position(x_des)
        self.controller.set_desired_posture(q_des)

        self.initialized_target = True

        self.get_logger().info(f'Initial TCP position: {x_now}')
        self.get_logger().info(
            f'Initial target source: {self.target_position_topic}'
        )
        self.get_logger().info(f'Initial leader target position: {x_des}')
        self.get_logger().info(
            f'Initial leader posture enabled: {self.use_leader_posture}'
        )
        self.get_logger().info(f'Initial q_des: {q_des}')

    def control_loop(self):
        if not self.wait_for_start_inputs():
            return

        q = self.q.copy()
        qdot = self.qdot.copy()
        self.robot.set_arm_state(q, qdot)

        if not self.initialized_target:
            self.initialize_from_leader_target()
            self.publish_zero_command()
            return

        if self.cycle_count < self.warmup_cycles:
            self.cycle_count += 1
            self.publish_zero_command()
            return

        if not self.update_external_target():
            self.publish_zero_command()
            self.cycle_count += 1
            return

        if not self.update_leader_posture_target():
            self.publish_zero_command()
            self.cycle_count += 1
            return

        tau = self.controller.compute_torque()

        if not np.all(np.isfinite(tau)):
            self.get_logger().error('Computed tau contains NaN or inf; publishing zero')
            self.publish_zero_command()
            self.cycle_count += 1
            return

        self.publish_command(tau)

        if self.cycle_count % self.log_period_cycles == 0:
            x_err = self.controller.last_x_error
            tau_task = self.controller.last_tau_task
            tau_posture = self.controller.last_tau_posture
            tau_bias = self.controller.last_tau_bias
            tau_friction = self.controller.last_tau_friction

            self.get_logger().info(
                '\n'
                '--- Cartesian impedance teleop debug ---\n'
                f'x_err        : {np.array2string(x_err, precision=4)}\n'
                f'x_des        : {np.array2string(self.controller.x_des, precision=4)}\n'
                f'target_raw   : {np.array2string(self.target_position, precision=4)}\n'
                f'q_des        : {np.array2string(self.controller.q_des, precision=4)}\n'
                f'q_leader     : {np.array2string(self.q_leader, precision=4)}\n'
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

    node = CartesianImpedanceRealTeleopNode()

    try:
        rclpy.spin(node)
    except (KeyboardInterrupt, ExternalShutdownException):
        pass
    finally:
        if rclpy.ok():
            zero_msg = Float64MultiArray()
            zero_msg.data = [0.0] * 6
            node.command_pub.publish(zero_msg)

        node.destroy_node()
        if rclpy.ok():
            rclpy.shutdown()


if __name__ == '__main__':
    main()
