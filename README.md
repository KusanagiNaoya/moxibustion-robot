# 艾灸机器人

基于 ROS Noetic 的艾灸机器人研究系统，包含 RGB-D 背部与穴位定位、相机到机器人基座的坐标转换、温度与距离反馈控制，以及 PyQt 整机操作界面。当前机器人后端为 Kinova Gen3 Lite，相机输入面向 Intel RealSense D435i。代码提供统一任务入口和机器人接口；其他机械臂需要实现相应后端。

## 系统组成

| 包 | 功能 |
| --- | --- |
| [`acupoint_detection`](acupoint_detection) | VGG16-UNet 背部分割、基础 HRNet `add_mask` 关键点、OLS 脊柱拟合与穴位规则，输出相机坐标中的穴位 |
| [`system_integration`](system_integration) | 运行期外参 TF、图像选点、基座坐标工作流、PyQt 操作台和整机启动 |
| [`moxibustion_api`](moxibustion_api) | 单穴位 Action、PID/SAC-PID 控制、温度与距离串口输入 |
| [`moxa_robot_interface`](moxa_robot_interface) | 标准位姿状态、同步点目标移动、停止请求及可选能力接口 |
| [`kinova_controller`](kinova_controller) | Kinova/Kortex 后端及统一接口适配 |

```text
RGB + 对齐深度 + CameraInfo
  → 分割 / 关键点 / 穴位计算
  → 相机坐标 → TF → 基座坐标
  → PyQt / Action 客户端 → moxibustion_api
  → moxa_robot_interface → Kinova / Kortex
```

## 环境与资源

源码面向 Ubuntu 20.04、ROS Noetic 和 Python 3。需要外部 ROS1 RealSense 驱动、Kortex 驱动，以及图像处理、PyTorch/Torchvision、Qt 等依赖。

| 资源 | 配置方式 |
| --- | --- |
| 分割权重 `unet_weights.pth` | 下载后通过 `unet_weights` 指定本地路径；7 类 UNet-VGG，当前实现要求 CUDA |
| 关键点权重 `add_mask.pth` | 下载后通过 `hrnet_weights` 指定本地路径；基础 HRNet W32、11 点、RGB 加掩膜输入 |
| 穴位规则 | [`acupoint_info.csv`](acupoint_detection/config/acupoint_info.csv)，可通过 `acupoints_file` 指定 |
| SAC-PID 推理资源 | 直接随仓库提供，位于 [`moxibustion_api/models/`](moxibustion_api/models)：`sac_pid.zip` 与 `config.yaml`，无需另行下载 |
| 相机外参 | `calibration_file` 指定实际 `B_T_C`；[示例文件](system_integration/config/hand_eye.example.yaml) 仅提供结构，没有有效标定值 |
| 任务与串口配置 | [`control.yaml`](moxibustion_api/config/control.yaml)、[`sensors.yaml`](moxibustion_api/config/sensors.yaml) |

## 启动与任务选择

在完成依赖、模型、外参和设备配置的运行环境中，整机入口为：

```bash
roslaunch system_integration system.launch \
  unet_weights:=/path/to/unet_weights.pth \
  hrnet_weights:=/path/to/add_mask.pth \
  calibration_file:=/path/to/hand_eye.yaml
```

入口默认启动相机、视觉、外参、传感器、机器人、任务 API 和 GUI。GUI 在发起任务前选择方法、穴位、目标温度和时长；最小命令行客户端为 `rosrun moxibustion_api send_task.py`。两者均使用同一个 `/moxibustion_task` Action：

| 方法值 | 含义 |
| --- | --- |
| `pid_control` | 固定 PID，默认方法 |
| `sac_pid_control` | SAC 推理调整 PID 增益，不直接输出机器人动作 |

## 机器人接口

默认接口命名空间为 `/moxa_robot`，可通过 `robot_interface_ns` 修改：

| 名称 | 类型 | 含义 |
| --- | --- | --- |
| `state` | `moxa_robot_interface/RobotState` | `PoseStamped`、受控参考点标识与有效性；保留反馈时间 |
| `move_to_pose` | `moxa_robot_interface/MoveToPose` | 同步绝对点目标，返回后端成功标记和说明 |
| `stop` | `std_srvs/Trigger` | 停止请求的处理结果 |
| `home` | `std_srvs/Trigger` | 可选 Home，默认关闭 |
| `set_gripper_position` | `moxa_robot_interface/SetGripperPosition` | 可选归一化夹爪目标：0 开、1 合，默认关闭 |
