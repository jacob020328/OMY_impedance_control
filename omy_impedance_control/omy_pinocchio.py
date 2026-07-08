import dataclasses
from typing import Tuple

import numpy as np
import pinocchio as pin


def T_from_Rp(R: np.ndarray, p: np.ndarray) -> np.ndarray:
    T = np.eye(4)
    T[:3, :3] = np.asarray(R, dtype=float).reshape(3, 3)
    T[:3, 3] = np.asarray(p, dtype=float).reshape(3)
    return T


@dataclasses.dataclass
class OMYConfig:
    xml_path: str
    ee_body: str = 'link6'
    base_body: str = 'base_link'
    tcp_offset_in_ee: np.ndarray = dataclasses.field(
        default_factory=lambda: np.array([0.0, -0.109, 0.0], dtype=float)
    )
    payload_mass: float = 0.0
    payload_com_in_ee: np.ndarray = dataclasses.field(
        default_factory=lambda: np.array([0.0, -0.109, 0.0], dtype=float)
    )
    payload_inertia_diag: np.ndarray = dataclasses.field(
        default_factory=lambda: np.ones(3, dtype=float) * 1.0e-4
    )
    arm_joint_names: Tuple[str, ...] = (
        'joint1',
        'joint2',
        'joint3',
        'joint4',
        'joint5',
        'joint6',
    )
    default_torque_limit: np.ndarray = dataclasses.field(
        default_factory=lambda: np.ones(6, dtype=float) * 30.0
    )


class OMYRobot:
    def __init__(self, cfg: OMYConfig):
        self.cfg = cfg
        self.model = pin.buildModelFromUrdf(cfg.xml_path)

        if not self.model.existFrame(cfg.ee_body):
            raise RuntimeError(f'Frame "{cfg.ee_body}" not found in URDF')

        self.ee_frame_id = self.model.getFrameId(cfg.ee_body)
        ee_frame = self.model.frames[self.ee_frame_id]
        self.tcp_frame_id = self.model.addFrame(
            pin.Frame(
                'tcp',
                ee_frame.parentJoint,
                self.ee_frame_id,
                pin.SE3(
                    np.eye(3),
                    np.asarray(cfg.tcp_offset_in_ee, dtype=float).reshape(3),
                ),
                pin.FrameType.OP_FRAME,
            )
        )

        self.payload_mass = max(float(cfg.payload_mass), 0.0)
        self.payload_com_in_ee = np.asarray(
            cfg.payload_com_in_ee,
            dtype=float,
        ).reshape(3)
        self.payload_inertia_diag = np.maximum(
            np.asarray(cfg.payload_inertia_diag, dtype=float).reshape(3),
            1.0e-9,
        )
        if self.payload_mass > 0.0:
            self._append_payload_inertia()

        self.data = self.model.createData()

        self.arm_joint_ids = [
            self.model.getJointId(name)
            for name in cfg.arm_joint_names
        ]
        if any(joint_id == 0 for joint_id in self.arm_joint_ids):
            missing = [
                name
                for name, joint_id in zip(cfg.arm_joint_names, self.arm_joint_ids)
                if joint_id == 0
            ]
            raise RuntimeError(f'Arm joints not found in URDF: {missing}')

        self.arm_q_ids = np.array(
            [self.model.joints[joint_id].idx_q for joint_id in self.arm_joint_ids],
            dtype=int,
        )
        self.arm_dof_ids = np.array(
            [self.model.joints[joint_id].idx_v for joint_id in self.arm_joint_ids],
            dtype=int,
        )

        self.q = np.zeros(self.model.nq)
        self.qdot = np.zeros(self.model.nv)

        self.q_home = np.deg2rad(
            np.array([0.0, -45.0, 90.0, -45.0, 90.0, 0.0])
        )

        self._forward()

    def _append_payload_inertia(self) -> None:
        ee_frame = self.model.frames[self.ee_frame_id]
        payload_placement = ee_frame.placement * pin.SE3(
            np.eye(3),
            self.payload_com_in_ee,
        )
        payload_inertia = pin.Inertia(
            self.payload_mass,
            np.zeros(3),
            np.diag(self.payload_inertia_diag),
        )
        self.model.appendBodyToJoint(
            ee_frame.parentJoint,
            payload_inertia,
            payload_placement,
        )

    def _forward(self) -> None:
        pin.forwardKinematics(self.model, self.data, self.q, self.qdot)
        pin.computeJointJacobians(self.model, self.data, self.q)
        pin.updateFramePlacements(self.model, self.data)

    def reset_home_keyframe(self) -> None:
        self.set_arm_state(self.q_home, np.zeros(6))

    def set_arm_state(self, q: np.ndarray, qdot: np.ndarray) -> None:
        q = np.asarray(q, dtype=float).reshape(6)
        qdot = np.asarray(qdot, dtype=float).reshape(6)

        self.q[:] = 0.0
        self.qdot[:] = 0.0
        self.q[self.arm_q_ids] = q
        self.qdot[self.arm_dof_ids] = qdot

        self._forward()

    def arm_q(self) -> np.ndarray:
        return self.q[self.arm_q_ids].copy()

    def arm_qdot(self) -> np.ndarray:
        return self.qdot[self.arm_dof_ids].copy()

    def tcp_pose_world(self) -> np.ndarray:
        tcp_pose = self.data.oMf[self.tcp_frame_id]
        return T_from_Rp(tcp_pose.rotation, tcp_pose.translation)

    def tcp_position_world(self) -> np.ndarray:
        return self.data.oMf[self.tcp_frame_id].translation.copy()

    def space_jacobian_tcp(self) -> np.ndarray:
        J = pin.getFrameJacobian(
            self.model,
            self.data,
            self.tcp_frame_id,
            pin.ReferenceFrame.LOCAL_WORLD_ALIGNED,
        )

        # Pinocchio returns [linear; angular]. The existing controller expects
        # [angular; linear] to match the MuJoCo wrapper.
        return np.vstack([J[3:6, :], J[0:3, :]])

    def space_jacobian_tcp_arm(self) -> np.ndarray:
        return self.space_jacobian_tcp()[:, self.arm_dof_ids]

    def tcp_twist_world(self) -> np.ndarray:
        return self.space_jacobian_tcp() @ self.qdot

    def bias_forces_arm(self) -> np.ndarray:
        bias = pin.rnea(
            self.model,
            self.data,
            self.q,
            self.qdot,
            np.zeros(self.model.nv),
        )
        return bias[self.arm_dof_ids].copy()

    def mass_matrix_full(self) -> np.ndarray:
        M = pin.crba(self.model, self.data, self.q)
        return 0.5 * (M + M.T)

    def mass_matrix_arm(self) -> np.ndarray:
        M = self.mass_matrix_full()
        return M[np.ix_(self.arm_dof_ids, self.arm_dof_ids)].copy()

    def actuator_forces_arm(self) -> np.ndarray:
        return np.zeros(6)

    def actuator_force_limits_arm(self) -> tuple[np.ndarray, np.ndarray]:
        limit = np.asarray(self.cfg.default_torque_limit, dtype=float).reshape(6)
        return -limit, limit
