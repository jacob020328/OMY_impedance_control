# OMY Real Cartesian Impedance Workspace

이 workspace는 ROBOTIS `open_manipulator` 기반 OMY F3M 실물 로봇에 Cartesian impedance control을 적용하기 위해 구성한 ROS 2 workspace이다.

기존 오픈소스는 주로 position trajectory controller 중심으로 동작한다. 이 작업에서는 실물 OMY F3M을 더 compliant하게 움직이기 위해 follower arm을 current/effort interface로 bringup하고, 노트북에서 Cartesian impedance torque를 계산해 OMY PC의 `/arm_controller/commands`로 보내는 구조로 바꾸었다.

## 전체 구조

주요 패키지 역할은 다음과 같다.

- `open_manipulator`: ROBOTIS OpenMANIPULATOR 오픈소스 패키지. OMY URDF, ros2_control 설정, bringup launch를 수정했다.
- `dynamixel_hardware_interface`, `DynamixelSDK`, `dynamixel_interfaces`: Dynamixel 실물 하드웨어 통신에 필요한 기존 오픈소스 의존성.
- `omy_impedance_control`: 새로 구현한 Cartesian impedance controller 패키지. Pinocchio로 실물 동역학을 계산하고 effort command를 publish한다.
- `rmw_zenoh`, `realsense-ros`: workspace에 같이 있는 통신/센서 관련 패키지. 현재 impedance controller의 핵심 실행 경로에는 직접 들어가지 않는다.

실제 실험 구성은 PC가 둘로 나뉜다.

- OMY PC: follower OMY F3M과 연결되어 있고, `ros2_control` current/effort controller를 실행한다.
- 노트북 또는 로컬 PC: leader arm이 연결되어 있고, impedance controller node를 실행한다.

두 PC는 같은 ROS 2 network에 있어야 한다. 같은 `ROS_DOMAIN_ID`를 사용하고, 네트워크 통신을 막지 않도록 `ROS_LOCALHOST_ONLY`가 `0` 또는 unset 상태인지 확인한다.

## 왜 Current/Effort Control로 바꿨는가

기존 position trajectory controller는 목표 관절 위치를 강하게 추종하는 방식이라, 외력이나 접촉 상황에서 compliance를 만들기 어렵다. Cartesian impedance control은 TCP 위치 오차를 spring-damper처럼 해석해서 필요한 wrench와 joint torque를 계산하므로, 실물에서 손으로 밀거나 leader target이 조금 흔들릴 때 더 부드럽게 반응할 수 있다.

그래서 follower OMY F3M은 position command가 아니라 effort command를 받도록 바꿨다.

```text
노트북 impedance node
  /joint_states 구독
  Pinocchio로 TCP, Jacobian, M(q), bias 계산
  tau 계산
  /arm_controller/commands publish

OMY PC ros2_control
  /joint_states publish
  /arm_controller/commands subscribe
  Dynamixel current/effort command 전송
```

## Cartesian Impedance 구현

`omy_impedance_control/cartesian_impedance_control.py`에 공통 controller를 구현했다. 현재는 TCP orientation이 아니라 TCP 3D position에 대한 translational Cartesian impedance이다.

제어식은 다음 흐름이다.

```text
x      = current TCP position
xdot   = Jp(q) qdot
e_x    = x_des - x
e_v    = xdot_des - xdot
a_task = K_pos e_x + D_pos e_v

Lambda   = (Jp M(q)^-1 Jp^T + lambda I)^-1
F_task   = Lambda a_task
tau_task = Jp^T F_task

tau_posture = K_posture (q_des - q) - D_posture qdot
tau = tau_bias + tau_task + tau_posture + tau_friction
tau = clip(tau, -torque_limit, torque_limit)
```

구현 포인트는 다음과 같다.

- `omy_pinocchio.py`에서 URDF를 Pinocchio model로 읽고 TCP 위치, TCP Jacobian, mass matrix, bias force를 계산한다.
- `Jp`는 TCP linear Jacobian만 사용한다.
- `Lambda` 계산에는 `lambda_damping`을 더해 singularity 근처에서 값이 터지는 것을 줄였다.
- `tau_posture`는 엄밀한 null-space projection은 아니고, TCP 제어에 joint posture 유지 torque를 더하는 방식이다.
- `use_bias:=true`이면 Pinocchio의 RNEA bias force를 더해 중력과 동역학 bias를 보상한다.
- 실물 gripper/payload 영향을 보정하기 위해 `payload_mass`, `payload_com_in_ee`, `payload_inertia_diag`를 Pinocchio 모델에 virtual payload로 추가할 수 있게 했다.
- 실물 Dynamixel 마찰을 줄이기 위해 velocity 기반 kinetic friction compensation과 static friction dither 옵션을 넣었다.
- 시작 직후에는 `warmup_cycles` 동안 zero command를 보내고, 종료 시에도 zero command를 한 번 publish한다.

## 구현한 노드

### `cartesian_impedance_real_node`

단독 follower Cartesian impedance 제어 노드이다.

- `/joint_states`에서 follower OMY F3M의 `joint1`부터 `joint6`까지 읽는다.
- 시작 시 `q_home = [0, -45, 90, -45, 90, 0] deg`의 TCP 위치를 계산해 `x_des`로 둔다.
- `target_offset`을 주면 home TCP 기준 목표 위치를 조금 이동시킬 수 있다.
- 계산한 torque를 `/arm_controller/commands`에 `std_msgs/Float64MultiArray`로 publish한다.

### `cartesian_impedance_real_teleop_node`

leader target을 따라가는 follower Cartesian impedance 제어 노드이다.

- follower `/joint_states`를 읽는다.
- `/cartesian_impedance/target_position`을 TCP 목표 위치로 받는다.
- `/leader/joint_states`를 posture reference로 사용할 수 있다.
- target이 stale되거나 leader joint state가 끊기면 zero command를 publish한다.
- `target_filter_alpha`, `max_target_step`으로 leader target을 한 번 더 부드럽게 제한한다.

### `leader_tcp_target_node`

leader arm의 joint state에서 leader TCP를 계산하고 follower가 따라갈 target position을 만든다.

- leader URDF와 Pinocchio를 사용해 leader TCP 위치를 계산한다.
- `target_mapping_mode:=relative`에서는 leader 기준점과 follower 기준점 사이의 상대 이동량을 target으로 변환한다.
- `omy_l100_leader_ai_current.launch.py` 안에서 함께 실행되도록 넣었다.

## 기존 Open Source 수정 내용

### Follower OMY F3M current/effort control

`open_manipulator_description/urdf/omy_f3m/omy_f3m.urdf.xacro`

- 기본 `ros2_control_type`을 `omy_f3m_current`로 바꿨다.
- `controller_manager_config` argument를 추가했다.
- Gazebo plugin이 항상 `hardware_controller_manager.yaml`만 읽지 않고, 선택한 controller yaml을 읽도록 했다.

이렇게 바꾼 이유는 같은 OMY F3M URDF를 position controller와 current/effort controller 양쪽에서 재사용하기 위해서이다.

`open_manipulator_bringup/config/omy_f3m/hardware_current_controller_manager.yaml`

- `arm_controller`를 `effort_controllers/JointGroupEffortController`로 설정했다.
- command topic은 `/arm_controller/commands`이고, impedance node가 여기에 6축 torque/current command를 보낸다.

이 설정이 없으면 Cartesian impedance controller가 계산한 `tau`를 실물 follower에 직접 보낼 수 없다.

### OMY PC follower current launch

`open_manipulator_bringup/launch/omy_f3m_current.launch.py`

- OMY PC에서 follower OMY F3M을 current/effort mode로 띄우기 위해 사용한 launch이다.
- 기존 `omy_f3m.launch.py`의 기본 bringup 흐름은 유지하되, 단순 topic remapping만 한 파일은 아니다.
- 기존 launch는 `omy_f3m.urdf.xacro`와 `hardware_controller_manager.yaml`을 사용해 position trajectory 기반 `arm_controller`, `gripper_controller`, `joint_state_broadcaster`를 띄운다.
- current launch는 follower arm 제어에 필요한 current/effort 전용 URDF와 controller yaml을 사용하도록 바꿨다.
- `arm_controller`와 `joint_state_broadcaster` 중심으로 띄우고, Cartesian impedance에서 직접 쓰지 않는 gripper trajectory 실행 흐름은 제외했다.
- `port_name` launch argument를 두어 OMY PC에서 follower arm bus가 연결된 serial port를 지정할 수 있게 했다.

이렇게 만든 이유는 OMY PC의 역할을 “목표 위치를 생성하는 controller”가 아니라 “실물 Dynamixel bus를 열고 effort command를 받아 follower에 전달하는 ros2_control endpoint”로 제한하기 위해서이다. Cartesian impedance torque 계산은 노트북의 `cartesian_impedance_real_node` 또는 `cartesian_impedance_real_teleop_node`가 담당한다.

현재 이 repository tree에는 OMY PC에서 사용한 `omy_f3m_current.launch.py` 원본이 남아 있지 않고, 이전 빌드 캐시에는 `omy_f3m_current_pc.launch.py` 형태의 흔적이 남아 있다. 그 캐시 기준으로는 `omy_impedance_control` 패키지의 `omy_f3m_current_arm_only.urdf.xacro`와 `omy_f3m_current_arm_only_controller_manager.yaml`을 사용해 `ros2_control_node`, `arm_controller`, `joint_state_broadcaster`, `robot_state_publisher`를 띄우는 구조였다.

### Gazebo 테스트 분리

`open_manipulator_bringup/launch/omy_f3m_gazebo.launch.py`

- Gazebo에서도 current/effort controller 설정을 쓰도록 바꿨다.
- `/joint_states`, `/tf`, `/arm_controller/commands`를 `/gazebo_*` topic으로 remap했다.

실물 topic과 simulation topic이 섞이면 노트북에서 impedance node를 띄웠을 때 실제 arm과 Gazebo arm이 같은 command를 받을 수 있다. 그래서 Gazebo 테스트용 topic을 분리했다.

`open_manipulator_bringup/launch/omy_f3m_gazebo_position.launch.py`

- position trajectory controller를 사용하는 Gazebo launch를 따로 뒀다.

기본 Gazebo launch를 current/effort 실험용으로 바꿨기 때문에, 기존 position trajectory 방식도 필요할 때 다시 테스트할 수 있도록 분리했다.

### Leader arm current launch

`open_manipulator_bringup/launch/omy_l100_leader_ai_current.launch.py`

- leader arm을 `omy_l100_current` ros2_control type으로 띄운다.
- `gravity_compensation_controller`, `spring_actuator_controller`, `joint_state_broadcaster`를 spawn한다.
- `leader_tcp_target_node`를 같이 실행해 `/cartesian_impedance/target_position`을 publish한다.
- `joint_trajectory_command_broadcaster`는 spawn하지 않도록 했다.

teleop에서 follower가 leader joint trajectory를 직접 따라가면 Cartesian impedance layer가 우회된다. 이 launch에서는 leader를 “직접 follower에 명령하는 arm”이 아니라 “TCP target을 만들어주는 arm”으로 쓰기 위해 joint trajectory command broadcaster를 뺐다.

`open_manipulator_bringup/package.xml`

- `omy_impedance_control` 실행 의존성을 추가했다.

leader current launch 안에서 `leader_tcp_target_node`를 실행하기 때문이다.

## 빌드

두 PC 모두 같은 source를 쓰는 경우 다음처럼 빌드한다.

```bash
cd ~/ros2_ws
source /opt/ros/jazzy/setup.bash
colcon build --symlink-install \
  --packages-select \
  open_manipulator_description \
  open_manipulator_bringup \
  omy_impedance_control
source install/setup.bash
```

실물 노드는 Pinocchio와 NumPy가 필요하다. Gazebo/MuJoCo 테스트 노드는 MuJoCo Python package도 필요하다.

## 실행 방법

아래 명령은 실제로 사용한 분산 실행 구조 기준이다.

`omy_f3m_current.launch.py`는 OMY PC에서 사용한 follower current bringup launch 이름이다. 이 source tree에서 그 launch가 다른 이름으로 관리되는 경우에도 핵심 조건은 동일하다. follower는 `omy_f3m_current` ros2_control type과 `hardware_current_controller_manager.yaml`을 사용해 `/arm_controller/commands`를 받는 effort controller로 떠야 한다.

### 1. 그냥 Cartesian Impedance Control

OMY PC에서 follower current controller를 먼저 띄운다.

```bash
cd ~/ros2_ws
source install/setup.bash
ros2 launch open_manipulator_bringup omy_f3m_current.launch.py
```

노트북 또는 로컬 PC에서 impedance controller를 실행한다.

```bash
cd ~/ros2_ws
source install/setup.bash
ros2 run omy_impedance_control cartesian_impedance_real_node
```

이 모드에서는 follower TCP 목표가 `q_home`에서 계산된 home TCP 위치로 잡힌다. 목표 위치를 조금 바꾸고 싶으면 노트북에서 다음처럼 `target_offset`을 준다.

```bash
ros2 run omy_impedance_control cartesian_impedance_real_node --ros-args \
  -p target_offset:="[0.0, 0.0, 0.02]"
```

### 2. Teleop + Cartesian Impedance Control

OMY PC에서 follower current controller를 띄운다.

```bash
cd ~/ros2_ws
source install/setup.bash
ros2 launch open_manipulator_bringup omy_f3m_current.launch.py
```

노트북에서 leader arm bringup과 target publisher를 띄운다. leader arm은 노트북에 연결되어 있다.

```bash
cd ~/ros2_ws
source install/setup.bash
ros2 launch open_manipulator_bringup omy_l100_leader_ai_current.launch.py
```

같은 노트북의 다른 터미널에서 follower impedance controller를 실행한다.

```bash
cd ~/ros2_ws
source install/setup.bash
ros2 run omy_impedance_control cartesian_impedance_real_teleop_node
```

leader USB port가 다르면 launch argument로 맞춘다.

```bash
ros2 launch open_manipulator_bringup omy_l100_leader_ai_current.launch.py \
  port_name:=/dev/ttyUSB0
```

## 실행 전 확인

OMY PC와 노트북이 같은 ROS graph를 보고 있는지 먼저 확인한다.

```bash
echo $ROS_DOMAIN_ID
echo $ROS_LOCALHOST_ONLY
ros2 topic list
```

노트북에서 follower state와 command topic이 보이는지 확인한다.

```bash
ros2 topic echo /joint_states --once
ros2 topic info /arm_controller/commands
```

teleop 모드에서는 leader topic과 target topic도 확인한다.

```bash
ros2 topic echo /leader/joint_states --once
ros2 topic echo /cartesian_impedance/target_position --once
```

controller 상태는 OMY PC에서 확인한다.

```bash
ros2 control list_controllers
ros2 control list_hardware_interfaces
```

`arm_controller`가 active이고 `/arm_controller/commands`를 받는 effort controller여야 한다.
