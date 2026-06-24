import dataclasses
from typing import Tuple

import mujoco
import numpy as np


def T_from_Rp(R: np.ndarray, p: np.ndarray) -> np.ndarray:
    T = np.eye(4)
    T[:3, :3] = np.asarray(R, dtype=float).reshape(3, 3)
    T[:3, 3] = np.asarray(p, dtype=float).reshape(3)
    return T


@dataclasses.dataclass
class OMYConfig:
    xml_path: str
    ee_body: str = "link6"
    base_body: str = "base_link"
    tcp_offset_in_ee: np.ndarray = dataclasses.field(
        default_factory=lambda: np.array([0.0, -0.109, 0.0], dtype=float)
    )
    arm_joint_names: Tuple[str, ...] = (
        "joint1",
        "joint2",
        "joint3",
        "joint4",
        "joint5",
        "joint6",
    )
    gripper_act_name: str = "Gripper"
    dt: float = 0.002


class OMYRobot:
    def __init__(self, cfg: OMYConfig):
        self.cfg = cfg
        self.model = mujoco.MjModel.from_xml_path(cfg.xml_path)
        self.data = mujoco.MjData(self.model)
        self.cfg.dt = float(self.model.opt.timestep)

        self.ee_body_id = mujoco.mj_name2id(
            self.model,
            mujoco.mjtObj.mjOBJ_BODY,
            cfg.ee_body,
        )
        self.base_body_id = mujoco.mj_name2id(
            self.model,
            mujoco.mjtObj.mjOBJ_BODY,
            cfg.base_body,
        )

        self.arm_joint_ids = [
            mujoco.mj_name2id(self.model, mujoco.mjtObj.mjOBJ_JOINT, name)
            for name in cfg.arm_joint_names
        ]

        self.arm_dof_ids = np.array(
            [self.model.jnt_dofadr[joint_id] for joint_id in self.arm_joint_ids],
            dtype=int,
        )

        self.arm_qpos_adrs = np.array(
            [self.model.jnt_qposadr[joint_id] for joint_id in self.arm_joint_ids],
            dtype=int,
        )

        self.arm_act_ids = np.array(
            [
                mujoco.mj_name2id(self.model, mujoco.mjtObj.mjOBJ_ACTUATOR, name)
                for name in cfg.arm_joint_names
            ],
            dtype=int,
        )

        self.has_arm_actuators = np.all(self.arm_act_ids >= 0)

        try:
            self.gripper_act_id = mujoco.mj_name2id(
                self.model,
                mujoco.mjtObj.mjOBJ_ACTUATOR,
                cfg.gripper_act_name,
            )
        except Exception:
            self.gripper_act_id = -1

        self._jacp = np.zeros((3, self.model.nv), dtype=float)
        self._jacr = np.zeros((3, self.model.nv), dtype=float)
        self._M = np.zeros((self.model.nv, self.model.nv), dtype=float)

        self.q_home = np.deg2rad(
            np.array([0.0, -45.0, 90.0, -45.0, 90.0, 0.0])
        )

    def reset_home_keyframe(self) -> None:
        mujoco.mj_resetData(self.model, self.data)
        self.data.qpos[self.arm_qpos_adrs] = self.q_home
        mujoco.mj_forward(self.model, self.data)

    def set_arm_state(self, q: np.ndarray, qdot: np.ndarray) -> None:
        """
        ROS 2 / Gazebo에서 받은 joint state를 MuJoCo model에 반영한다.
        이후 mj_forward()를 호출해야 FK, Jacobian, M, bias가 현재 q 기준으로 계산된다.
        """
        q = np.asarray(q, dtype=float).reshape(6)
        qdot = np.asarray(qdot, dtype=float).reshape(6)

        self.data.qpos[self.arm_qpos_adrs] = q
        self.data.qvel[self.arm_dof_ids] = qdot

        mujoco.mj_forward(self.model, self.data)

    def arm_q(self) -> np.ndarray:
        return self.data.qpos[self.arm_qpos_adrs].copy()

    def arm_qdot(self) -> np.ndarray:
        return self.data.qvel[self.arm_dof_ids].copy()

    def tcp_pose_world(self) -> np.ndarray:
        p = self.data.xpos[self.ee_body_id].copy()
        R = self.data.xmat[self.ee_body_id].reshape(3, 3).copy()
        p_tcp = p + R @ self.cfg.tcp_offset_in_ee.reshape(3)
        return T_from_Rp(R, p_tcp)

    def tcp_position_world(self) -> np.ndarray:
        return self.tcp_pose_world()[:3, 3].copy()

    def space_jacobian_tcp(self) -> np.ndarray:
        self._jacp[:] = 0.0
        self._jacr[:] = 0.0

        mujoco.mj_jac(
            self.model,
            self.data,
            self._jacp,
            self._jacr,
            self.tcp_position_world(),
            self.ee_body_id,
        )

        # 위 3행: angular Jacobian
        # 아래 3행: linear Jacobian
        return np.vstack([self._jacr, self._jacp])

    def space_jacobian_tcp_arm(self) -> np.ndarray:
        return self.space_jacobian_tcp()[:, self.arm_dof_ids]

    def tcp_twist_world(self) -> np.ndarray:
        return self.space_jacobian_tcp() @ self.data.qvel

    def bias_forces_arm(self) -> np.ndarray:
        return self.data.qfrc_bias[self.arm_dof_ids].copy()

    def mass_matrix_full(self) -> np.ndarray:
        mujoco.mj_fullM(self.model, self.data, self._M)
        return self._M.copy()

    def mass_matrix_arm(self) -> np.ndarray:
        M = self.mass_matrix_full()
        return M[np.ix_(self.arm_dof_ids, self.arm_dof_ids)].copy()

    def actuator_forces_arm(self) -> np.ndarray:
        return self.data.qfrc_actuator[self.arm_dof_ids].copy()

    def actuator_force_limits_arm(self) -> tuple[np.ndarray, np.ndarray]:
        if not getattr(self, "has_arm_actuators", False):
            lows = -np.ones(6) * 30.0
            highs = np.ones(6) * 30.0
            return lows, highs

        lows = []
        highs = []

        for act_id in self.arm_act_ids:
            lows.append(self.model.actuator_ctrlrange[act_id, 0])
            highs.append(self.model.actuator_ctrlrange[act_id, 1])

        return np.asarray(lows, dtype=float), np.asarray(highs, dtype=float)