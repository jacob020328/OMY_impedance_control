import numpy as np

import rclpy
from rclpy.node import Node

from geometry_msgs.msg import PointStamped
from geometry_msgs.msg import PoseStamped
from sensor_msgs.msg import JointState

from .omy_pinocchio import OMYConfig, OMYRobot


def quaternion_from_rotation_matrix(rotation: np.ndarray) -> np.ndarray:
    """Return quaternion [x, y, z, w] from a 3x3 rotation matrix."""
    r = np.asarray(rotation, dtype=float).reshape(3, 3)
    trace = np.trace(r)

    if trace > 0.0:
        s = 2.0 * np.sqrt(trace + 1.0)
        qw = 0.25 * s
        qx = (r[2, 1] - r[1, 2]) / s
        qy = (r[0, 2] - r[2, 0]) / s
        qz = (r[1, 0] - r[0, 1]) / s
    else:
        axis = int(np.argmax(np.diag(r)))
        if axis == 0:
            s = 2.0 * np.sqrt(1.0 + r[0, 0] - r[1, 1] - r[2, 2])
            qw = (r[2, 1] - r[1, 2]) / s
            qx = 0.25 * s
            qy = (r[0, 1] + r[1, 0]) / s
            qz = (r[0, 2] + r[2, 0]) / s
        elif axis == 1:
            s = 2.0 * np.sqrt(1.0 + r[1, 1] - r[0, 0] - r[2, 2])
            qw = (r[0, 2] - r[2, 0]) / s
            qx = (r[0, 1] + r[1, 0]) / s
            qy = 0.25 * s
            qz = (r[1, 2] + r[2, 1]) / s
        else:
            s = 2.0 * np.sqrt(1.0 + r[2, 2] - r[0, 0] - r[1, 1])
            qw = (r[1, 0] - r[0, 1]) / s
            qx = (r[0, 2] + r[2, 0]) / s
            qy = (r[1, 2] + r[2, 1]) / s
            qz = 0.25 * s

    q = np.array([qx, qy, qz, qw], dtype=float)
    norm = np.linalg.norm(q)
    if norm <= 0.0:
        return np.array([0.0, 0.0, 0.0, 1.0], dtype=float)
    return q / norm


class LeaderTcpTargetNode(Node):
    def __init__(self):
        super().__init__('leader_tcp_target_node')

        self.declare_parameter(
            'xml_path',
            '/root/ros2_ws/src/open_manipulator/open_manipulator_description/urdf/'
            'omy_l100/omy_l100.urdf',
        )
        self.declare_parameter('leader_joint_states_topic', '/leader/joint_states')
        self.declare_parameter('leader_tcp_pose_topic', '/leader/tcp_pose')
        self.declare_parameter('leader_tcp_position_topic', '/leader/tcp_position')
        self.declare_parameter(
            'target_position_topic',
            '/cartesian_impedance/target_position',
        )
        self.declare_parameter(
            'leader_joint_names',
            ['joint1', 'joint2', 'joint3', 'joint4', 'joint5', 'joint6'],
        )
        self.declare_parameter('ee_body', 'link6')
        self.declare_parameter('tcp_offset_in_ee', [0.0, -0.125, -0.0255])
        self.declare_parameter('target_offset', [0.0, 0.0, 0.0])
        self.declare_parameter('target_scale', [1.0, 1.0, 1.0])
        self.declare_parameter('target_mapping_mode', 'absolute')
        self.declare_parameter('leader_reference_position', [0.0, 0.0, 0.0])
        self.declare_parameter('follower_reference_position', [0.0, 0.0, 0.0])
        self.declare_parameter('capture_leader_reference_on_start', False)
        self.declare_parameter('target_filter_alpha', 0.35)
        self.declare_parameter('max_target_step', 0.02)
        self.declare_parameter('publish_rate', 100.0)
        self.declare_parameter('frame_id', 'leader_base_link')
        self.declare_parameter('target_frame_id', 'follower_base_link')

        xml_path = str(self.get_parameter('xml_path').value)
        leader_joint_states_topic = str(
            self.get_parameter('leader_joint_states_topic').value
        )
        self.leader_tcp_pose_topic = str(
            self.get_parameter('leader_tcp_pose_topic').value
        )
        self.leader_tcp_position_topic = str(
            self.get_parameter('leader_tcp_position_topic').value
        )
        self.target_position_topic = str(
            self.get_parameter('target_position_topic').value
        )
        self.joint_names = list(self.get_parameter('leader_joint_names').value)
        self.frame_id = str(self.get_parameter('frame_id').value)
        self.target_frame_id = str(self.get_parameter('target_frame_id').value)

        tcp_offset_in_ee = np.array(
            self.get_parameter('tcp_offset_in_ee').value,
            dtype=float,
        ).reshape(3)
        self.target_offset = np.array(
            self.get_parameter('target_offset').value,
            dtype=float,
        ).reshape(3)
        self.target_scale = np.array(
            self.get_parameter('target_scale').value,
            dtype=float,
        ).reshape(3)
        self.target_mapping_mode = str(
            self.get_parameter('target_mapping_mode').value
        ).strip().lower()
        if self.target_mapping_mode not in ('absolute', 'relative'):
            self.get_logger().warn(
                f'Unknown target_mapping_mode "{self.target_mapping_mode}"; '
                'falling back to absolute'
            )
            self.target_mapping_mode = 'absolute'
        self.leader_reference_position = np.array(
            self.get_parameter('leader_reference_position').value,
            dtype=float,
        ).reshape(3)
        self.follower_reference_position = np.array(
            self.get_parameter('follower_reference_position').value,
            dtype=float,
        ).reshape(3)
        self.capture_leader_reference_on_start = bool(
            self.get_parameter('capture_leader_reference_on_start').value
        )
        self.has_leader_reference = not self.capture_leader_reference_on_start
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
        publish_rate = float(self.get_parameter('publish_rate').value)
        self.dt = 1.0 / max(publish_rate, 1.0)

        cfg = OMYConfig(
            xml_path=xml_path,
            ee_body=str(self.get_parameter('ee_body').value),
            tcp_offset_in_ee=tcp_offset_in_ee,
        )
        self.robot = OMYRobot(cfg)

        self.q = np.zeros(6)
        self.qdot = np.zeros(6)
        self.has_joint_state = False
        self.filtered_target_position: np.ndarray | None = None

        self.joint_state_sub = self.create_subscription(
            JointState,
            leader_joint_states_topic,
            self.joint_state_callback,
            10,
        )
        self.leader_pose_pub = self.create_publisher(
            PoseStamped,
            self.leader_tcp_pose_topic,
            10,
        )
        self.leader_position_pub = self.create_publisher(
            PointStamped,
            self.leader_tcp_position_topic,
            10,
        )
        self.target_position_pub = self.create_publisher(
            PointStamped,
            self.target_position_topic,
            10,
        )
        self.timer = self.create_timer(self.dt, self.publish_tcp_target)

        self.get_logger().info('Leader TCP target node started.')
        self.get_logger().info(f'xml_path: {xml_path}')
        self.get_logger().info(
            f'leader_joint_states_topic: {leader_joint_states_topic}'
        )
        self.get_logger().info(f'leader_tcp_pose_topic: {self.leader_tcp_pose_topic}')
        self.get_logger().info(
            f'leader_tcp_position_topic: {self.leader_tcp_position_topic}'
        )
        self.get_logger().info(f'target_position_topic: {self.target_position_topic}')
        self.get_logger().info(f'joint_names: {self.joint_names}')
        self.get_logger().info(f'tcp_offset_in_ee: {tcp_offset_in_ee}')
        self.get_logger().info(f'target_offset: {self.target_offset}')
        self.get_logger().info(f'target_scale: {self.target_scale}')
        self.get_logger().info(f'target_mapping_mode: {self.target_mapping_mode}')
        self.get_logger().info(
            f'leader_reference_position: {self.leader_reference_position}'
        )
        self.get_logger().info(
            f'follower_reference_position: {self.follower_reference_position}'
        )
        self.get_logger().info(
            'capture_leader_reference_on_start: '
            f'{self.capture_leader_reference_on_start}'
        )
        self.get_logger().info(f'target_filter_alpha: {self.target_filter_alpha}')
        self.get_logger().info(f'max_target_step: {self.max_target_step}')
        self.get_logger().info(f'publish_rate: {publish_rate}')

    def joint_state_callback(self, msg: JointState) -> None:
        name_to_index = {name: i for i, name in enumerate(msg.name)}
        qdot = np.zeros(6)

        for i, joint_name in enumerate(self.joint_names):
            if joint_name not in name_to_index:
                self.get_logger().warn(
                    f'{joint_name} not found in leader joint state',
                    throttle_duration_sec=2.0,
                )
                return

            idx = name_to_index[joint_name]
            self.q[i] = msg.position[idx]
            if len(msg.velocity) > idx:
                qdot[i] = msg.velocity[idx]

        self.qdot = qdot
        self.has_joint_state = True

    def filtered_target(self, target_position: np.ndarray) -> np.ndarray:
        if self.filtered_target_position is None:
            self.filtered_target_position = target_position.copy()
            return self.filtered_target_position

        candidate = (
            self.target_filter_alpha * target_position
            + (1.0 - self.target_filter_alpha) * self.filtered_target_position
        )

        delta = candidate - self.filtered_target_position
        delta_norm = np.linalg.norm(delta)
        if self.max_target_step > 0.0 and delta_norm > self.max_target_step:
            candidate = (
                self.filtered_target_position
                + delta / delta_norm * self.max_target_step
            )

        self.filtered_target_position = candidate
        return self.filtered_target_position

    def mapped_target_position(self, leader_position: np.ndarray) -> np.ndarray:
        if self.target_mapping_mode == 'absolute':
            return self.target_scale * leader_position + self.target_offset

        if not self.has_leader_reference:
            self.leader_reference_position = leader_position.copy()
            self.has_leader_reference = True
            self.get_logger().info(
                'Captured leader reference position: '
                f'{self.leader_reference_position}'
            )

        leader_delta = leader_position - self.leader_reference_position
        return (
            self.follower_reference_position
            + self.target_scale * leader_delta
            + self.target_offset
        )

    def publish_tcp_target(self) -> None:
        if not self.has_joint_state:
            return

        self.robot.set_arm_state(self.q.copy(), self.qdot.copy())
        tcp_pose = self.robot.tcp_pose_world()
        leader_position = tcp_pose[:3, 3].copy()
        target_position = self.mapped_target_position(leader_position)

        if not np.all(np.isfinite(target_position)):
            self.get_logger().warn(
                'Leader target contains NaN or inf; skipping publish',
                throttle_duration_sec=1.0,
            )
            return

        target_position = self.filtered_target(target_position)

        stamp = self.get_clock().now().to_msg()

        pose_msg = PoseStamped()
        pose_msg.header.stamp = stamp
        pose_msg.header.frame_id = self.frame_id
        pose_msg.pose.position.x = float(leader_position[0])
        pose_msg.pose.position.y = float(leader_position[1])
        pose_msg.pose.position.z = float(leader_position[2])
        q = quaternion_from_rotation_matrix(tcp_pose[:3, :3])
        pose_msg.pose.orientation.x = float(q[0])
        pose_msg.pose.orientation.y = float(q[1])
        pose_msg.pose.orientation.z = float(q[2])
        pose_msg.pose.orientation.w = float(q[3])

        leader_pos_msg = PointStamped()
        leader_pos_msg.header.stamp = stamp
        leader_pos_msg.header.frame_id = self.frame_id
        leader_pos_msg.point.x = float(leader_position[0])
        leader_pos_msg.point.y = float(leader_position[1])
        leader_pos_msg.point.z = float(leader_position[2])

        target_msg = PointStamped()
        target_msg.header.stamp = stamp
        target_msg.header.frame_id = self.target_frame_id
        target_msg.point.x = float(target_position[0])
        target_msg.point.y = float(target_position[1])
        target_msg.point.z = float(target_position[2])

        self.leader_pose_pub.publish(pose_msg)
        self.leader_position_pub.publish(leader_pos_msg)
        self.target_position_pub.publish(target_msg)


def main(args=None):
    rclpy.init(args=args)
    node = LeaderTcpTargetNode()

    try:
        rclpy.spin(node)
    except KeyboardInterrupt:
        pass
    finally:
        node.destroy_node()
        if rclpy.ok():
            rclpy.shutdown()


if __name__ == '__main__':
    main()
