import dataclasses
from typing import Any

import numpy as np


@dataclasses.dataclass
class CartesianImpedanceGains:
    K_pos: np.ndarray = dataclasses.field(
        default_factory=lambda: np.diag([10.0, 10.0, 10.0])
    )
    D_pos: np.ndarray = dataclasses.field(
        default_factory=lambda: np.diag([10.0, 10.0, 10.0])
    )
    K_posture: np.ndarray = dataclasses.field(
        default_factory=lambda: np.diag([3.0, 3.0, 2.0, 1.5, 1.0, 0.8])
    )
    D_posture: np.ndarray = dataclasses.field(
        default_factory=lambda: np.diag([1.0, 1.0, 0.8, 0.6, 0.4, 0.3])
    )


@dataclasses.dataclass
class FrictionCompensationConfig:
    enabled: bool = False
    kinetic_friction_scalars: np.ndarray = dataclasses.field(
        default_factory=lambda: np.zeros(6)
    )
    kinetic_friction_torque_scalars: np.ndarray = dataclasses.field(
        default_factory=lambda: np.zeros(6)
    )
    friction_compensation_velocity_thresholds: np.ndarray = dataclasses.field(
        default_factory=lambda: np.ones(6)
    )
    static_friction_scalars: np.ndarray = dataclasses.field(
        default_factory=lambda: np.zeros(6)
    )
    static_friction_velocity_thresholds: np.ndarray = dataclasses.field(
        default_factory=lambda: np.zeros(6)
    )
    unloaded_effort_offsets: np.ndarray = dataclasses.field(
        default_factory=lambda: np.zeros(6)
    )
    unloaded_effort_thresholds: np.ndarray = dataclasses.field(
        default_factory=lambda: np.zeros(6)
    )


class CartesianImpedanceController:
    def __init__(
        self,
        robot: Any,
        gains: CartesianImpedanceGains | None = None,
        lambda_damping: float = 1.0e-4,
        torque_limit: np.ndarray | None = None,
        use_bias: bool = True,
        task_torque_scale: np.ndarray | None = None,
        friction: FrictionCompensationConfig | None = None,
    ):
        self.robot = robot
        self.gains = gains if gains is not None else CartesianImpedanceGains()
        self.lambda_damping = lambda_damping
        self.use_bias = use_bias
        if task_torque_scale is None:
            self.task_torque_scale = np.ones(6)
        else:
            self.task_torque_scale = np.asarray(
                task_torque_scale,
                dtype=float,
            ).reshape(6)

        self.friction = self._normalize_friction_config(friction)
        self._dither_switch = False

        if torque_limit is None:
            self.torque_lo, self.torque_hi = robot.actuator_force_limits_arm()
        else:
            torque_limit = np.asarray(torque_limit, dtype=float).reshape(6)
            self.torque_lo = -torque_limit
            self.torque_hi = torque_limit

        self.x_des = robot.tcp_position_world()
        self.xdot_des = np.zeros(3)
        self.q_des = robot.arm_q()

        self.last_x = self.x_des.copy()
        self.last_xdot = np.zeros(3)
        self.last_x_error = np.zeros(3)
        self.last_task_accel_cmd = np.zeros(3)
        self.last_lambda = np.eye(3)
        self.last_force_task = np.zeros(3)
        self.last_tau_task = np.zeros(6)
        self.last_tau_posture = np.zeros(6)
        self.last_tau_bias = np.zeros(6)
        self.last_tau_friction = np.zeros(6)
        self.last_tau = np.zeros(6)
        self.last_actuator_force = np.zeros(6)
        self.peak_abs_actuator_force = np.zeros(6)

    def _normalize_friction_config(
        self,
        friction: FrictionCompensationConfig | None,
    ) -> FrictionCompensationConfig:
        if friction is None:
            friction = FrictionCompensationConfig()

        def nonnegative_array(value: np.ndarray) -> np.ndarray:
            return np.maximum(np.asarray(value, dtype=float).reshape(6), 0.0)

        return FrictionCompensationConfig(
            enabled=bool(friction.enabled),
            kinetic_friction_scalars=nonnegative_array(
                friction.kinetic_friction_scalars
            ),
            kinetic_friction_torque_scalars=nonnegative_array(
                friction.kinetic_friction_torque_scalars
            ),
            friction_compensation_velocity_thresholds=nonnegative_array(
                friction.friction_compensation_velocity_thresholds
            ),
            static_friction_scalars=nonnegative_array(
                friction.static_friction_scalars
            ),
            static_friction_velocity_thresholds=nonnegative_array(
                friction.static_friction_velocity_thresholds
            ),
            unloaded_effort_offsets=nonnegative_array(
                friction.unloaded_effort_offsets
            ),
            unloaded_effort_thresholds=nonnegative_array(
                friction.unloaded_effort_thresholds
            ),
        )

    def set_desired_tcp_position(self, x_des: np.ndarray) -> None:
        self.x_des = np.asarray(x_des, dtype=float).reshape(3)

    def set_desired_posture(self, q_des: np.ndarray) -> None:
        self.q_des = np.asarray(q_des, dtype=float).reshape(6)

    def compute_friction_compensation(
        self,
        tau_without_friction: np.ndarray,
        qdot: np.ndarray,
    ) -> np.ndarray:
        if not self.friction.enabled:
            self.last_tau_friction = np.zeros(6)
            return self.last_tau_friction

        tau_without_friction = np.asarray(
            tau_without_friction,
            dtype=float,
        ).reshape(6)
        qdot = np.asarray(qdot, dtype=float).reshape(6)

        speed = np.abs(qdot)
        direction = np.sign(qdot)

        kinetic_scalar = self.friction.kinetic_friction_scalars * (
            1.0
            + np.abs(
                tau_without_friction
                * self.friction.kinetic_friction_torque_scalars
            )
        )
        kinetic_rate = 1.0 - (
            speed * 10.0
            - self.friction.friction_compensation_velocity_thresholds
        )
        kinetic_rate = np.maximum(kinetic_rate, 0.0)
        tau_friction = direction * kinetic_scalar * kinetic_rate * speed

        tau_with_kinetic = tau_without_friction + tau_friction
        unloaded_mask = (
            (direction != 0.0)
            & (
                np.abs(tau_with_kinetic)
                < self.friction.unloaded_effort_thresholds
            )
        )
        tau_friction[unloaded_mask] += (
            direction[unloaded_mask]
            * self.friction.unloaded_effort_offsets[unloaded_mask]
        )

        static_mask = (
            speed < self.friction.static_friction_velocity_thresholds
        )
        if np.any(static_mask):
            dither_direction = 1.0 if self._dither_switch else -1.0
            tau_friction[static_mask] += (
                dither_direction
                * self.friction.static_friction_scalars[static_mask]
                * np.abs(tau_without_friction[static_mask])
            )
            self._dither_switch = not self._dither_switch

        self.last_tau_friction = tau_friction
        return tau_friction

    def compute_torque(self) -> np.ndarray:
        q = self.robot.arm_q()
        qdot = self.robot.arm_qdot()
        x = self.robot.tcp_position_world()

        J = self.robot.space_jacobian_tcp_arm()

        # J = [Jw; Jp]
        # 위치 임피던스만 사용하므로 linear Jacobian만 사용
        Jp = J[3:6, :]

        xdot = Jp @ qdot
        M = self.robot.mass_matrix_arm()

        x_error = self.x_des - x
        xdot_error = self.xdot_des - xdot

        # 원하는 작업공간 가속도 형태의 명령
        task_accel_cmd = (
            self.gains.K_pos @ x_error
            + self.gains.D_pos @ xdot_error
        )

        # Operational space inertia
        # Lambda = (J M^-1 J^T)^-1
        # 특이점 근처 발산 방지를 위해 lambda_damping 사용
        try:
            Minv_JT = np.linalg.solve(M, Jp.T)
            lambda_inv = Jp @ Minv_JT
            lambda_regularized = (
                lambda_inv
                + self.lambda_damping * np.eye(3)
            )
            lambda_task = np.linalg.inv(lambda_regularized)
        except np.linalg.LinAlgError:
            lambda_task = self.last_lambda.copy()

        force_task = lambda_task @ task_accel_cmd
        tau_task = Jp.T @ force_task
        tau_task = self.task_torque_scale * tau_task

        # null-space가 엄밀히 적용된 posture control은 아니고,
        # 기존 코드와 동일하게 joint posture torque를 더하는 방식
        tau_posture = (
            self.gains.K_posture @ (self.q_des - q)
            - self.gains.D_posture @ qdot
        )

        if self.use_bias:
            tau_bias = self.robot.bias_forces_arm()
        else:
            tau_bias = np.zeros(6)

        tau_without_friction = tau_bias + tau_task + tau_posture
        tau_friction = self.compute_friction_compensation(
            tau_without_friction,
            qdot,
        )

        tau = tau_without_friction + tau_friction
        tau = np.clip(tau, self.torque_lo, self.torque_hi)

        self.last_x = x
        self.last_xdot = xdot
        self.last_x_error = x_error
        self.last_task_accel_cmd = task_accel_cmd
        self.last_lambda = lambda_task
        self.last_force_task = force_task
        self.last_tau_task = tau_task
        self.last_tau_posture = tau_posture
        self.last_tau_bias = tau_bias
        self.last_tau_friction = tau_friction
        self.last_tau = tau

        return tau

    def apply(self) -> np.ndarray:
        """Apply one standalone MuJoCo control step."""
        tau = self.compute_torque()

        self.robot.data.ctrl[self.robot.arm_act_ids] = tau

        if self.robot.gripper_act_id >= 0:
            self.robot.data.ctrl[self.robot.gripper_act_id] = 0.0

        return tau

    def update_actuator_force_metrics(self) -> None:
        self.last_actuator_force = self.robot.actuator_forces_arm()
        self.peak_abs_actuator_force = np.maximum(
            self.peak_abs_actuator_force,
            np.abs(self.last_actuator_force),
        )
