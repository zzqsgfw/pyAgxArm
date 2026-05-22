# Piper 机械臂 + LIMO Pro 使用指南

> 面向初学者的快速上手文档

---

## 目录

- [第一部分：Piper 机械臂](#第一部分piper-机械臂)
  - [什么是 Piper？](#什么是-piper)
  - [硬件连接](#硬件连接)
  - [环境安装](#环境安装)
  - [Demo 模式（不需要机械臂）](#demo-模式不需要机械臂)
  - [Hardware 模式（连接真实机械臂）](#hardware-模式连接真实机械臂)
  - [交互操作](#交互操作)
  - [常用工具脚本](#常用工具脚本)
  - [遇到问题？](#遇到问题)
- [第二部分：LIMO Pro 移动机器人](#第二部分limo-pro-移动机器人)
  - [什么是 LIMO Pro？](#什么是-limo-pro)
  - [四种运动模式](#四种运动模式)
  - [开机与连接](#开机与连接)
  - [键盘遥控](#键盘遥控)
  - [建图与导航](#建图与导航)
  - [注意事项](#注意事项)

---

# 第一部分：Piper 机械臂

## 什么是 Piper？

Piper 是松灵机器人（Agilex Robotics）的 **6 轴协作机械臂**，末端带有一个电动夹爪。我们写了一个 3D 可视化工具，可以：

- 在电脑上看到机械臂的实时姿态
- 用鼠标拖动一个"目标坐标系"来控制机械臂移动
- 控制夹爪的开合

## 硬件连接

你需要：

1. **Piper 机械臂**（已通电）
2. **CAN 适配器**（松灵 CANDO USB 适配器，插在电脑 USB 口上）
3. **Windows 电脑**（已安装驱动）

连接方式：`电脑 USB` → `CAN 适配器` → `机械臂 CAN 接口`

## 环境安装

### 方式一：用 conda 从零安装

```bash
# 1. 创建环境
conda env create -f pyAgxArm\demos\piper_viz\environment.yaml

# 2. 激活环境
conda activate pyAgxArm

# 3. 安装 CAN 驱动（Windows 专用）
git clone https://github.com/agilexrobotics/python-can-agx-cando.git
cd python-can-agx-cando
pip install .

# 4. 安装机械臂 SDK
cd pyAgxArm
pip install -e .

# 5. 初始化 URDF 子模块（3D 模型文件）
git submodule update --init --recursive
```

### 方式二：用打包好的环境

如果有人给你一个 `pyAgxArm_env.tar.gz`：

```bash
# 解压
mkdir C:\pyAgxArm_env
tar -xzf pyAgxArm_env.tar.gz -C C:\pyAgxArm_env

# 激活
C:\pyAgxArm_env\Scripts\activate.bat

# 修复路径
conda-unpack

# 还需要单独装 SDK
pip install -e pyAgxArm
```

## Demo 模式（不需要机械臂）

不连硬件也能玩！运行：

```bash
python pyAgxArm\demos\piper_viz\piper_mujoco_viz.py
```

会弹出一个 3D 窗口，里面有：
- 一个 Piper 机械臂模型
- 一个彩色坐标系（**红色=X=前方，绿色=Y=左边，蓝色=Z=上方**）
- 一个黄色的夹爪滑块

用鼠标拖动坐标系，机械臂会跟着动！（通过 IK 逆运动学求解）

## Hardware 模式（连接真实机械臂）

**注意：操作真实机械臂时，手不要靠近机械臂运动范围！**

```bash
# 使用机械臂内置的运动规划（推荐新手）
python pyAgxArm\demos\piper_viz\piper_mujoco_viz.py --mode hardware

# 使用我们自己的 IK 算法控制（进阶）
python pyAgxArm\demos\piper_viz\piper_mujoco_viz.py --mode hardware --control ik
```

启动后程序会：
1. 连接机械臂
2. 重置 → 使能电机 → 移动到初始位置
3. 打开 3D 可视化窗口
4. 你拖动目标坐标系 → 机械臂跟着动

**退出**：按 `ESC` 或 `Ctrl+C`，程序会安全停止机械臂并松开电机。

## 交互操作

### 鼠标操作

| 操作 | 效果 |
|------|------|
| **左键拖红色轴** | 沿 X 轴（前后）平移目标 |
| **左键拖绿色轴** | 沿 Y 轴（左右）平移目标 |
| **左键拖蓝色轴** | 沿 Z 轴（上下）平移目标 |
| **左键拖白色原点** | 在屏幕平面内自由平移 |
| **右键拖任意轴** | 绕该轴旋转目标 |
| **左键拖空白区域** | 旋转相机视角 |
| **右键拖空白区域** | 平移相机 |
| **滚轮** | 缩放 |
| **拖黄色滑块** | 控制夹爪开合 |

### 键盘

| 按键 | 效果 |
|------|------|
| `ESC` | 退出程序 |

### 屏幕信息

左下角会显示：
- **Target**：目标坐标系的位置和朝向
- **EE err**：机械臂末端和目标之间的距离误差（毫米）

## 常用工具脚本

```bash
# 标定关节零点（把机械臂摆到零位后运行）
python pyAgxArm\demos\piper_viz\calibrate.py

# 完整重置（清除错误、恢复默认范围）
python pyAgxArm\demos\piper_viz\reset_arm.py

# 调试使能（逐个关节测试，排查连接问题）
python pyAgxArm\demos\piper_viz\test_enable.py
```

## 遇到问题？

| 问题 | 可能原因 | 解决方法 |
|------|---------|---------|
| "URDF not found" | 子模块没初始化 | `git submodule update --init --recursive` |
| "CAN bus failed" | CAN 适配器没连好 | 检查 USB 连接，重新插拔 |
| 机械臂不动 | 电机没使能 | 运行 `reset_arm.py` |
| 关节卡住 | 软件限位锁了 | 运行 `reset_arm.py`（会关掉限位） |
| 退出后臂还僵硬 | 失能不完全 | 重新运行程序再按 ESC 退出 |

---

# 第二部分：LIMO Pro 移动机器人

## 什么是 LIMO Pro？

LIMO Pro 是松灵机器人的**全地形移动机器人底盘**，特点是**一台车支持四种运动模式**：

- 搭载 NVIDIA Jetson Orin Nano（AI 处理器）
- 自带激光雷达（360° 扫描）和深度相机
- 支持 ROS1/ROS2
- 续航约 2.5 小时

## 四种运动模式

通过更换轮子和调整机械结构来切换：

### 1. 阿克曼模式（像汽车）
- **指示灯**：绿色
- 像开车一样转弯，前轮转向
- 适合走直线和大弯道

### 2. 差速模式（像坦克）
- **指示灯**：黄色
- 左右轮独立控制，可以原地旋转
- 适合狭小空间

### 3. 履带模式（像推土机）
- **指示灯**：黄色
- 换上履带，爬坡能力最强（40°）
- 适合野外和不平地面
- **注意**：侧门要抬起来，不然会磨到履带

### 4. 麦克纳姆轮模式（全向移动）
- **指示灯**：蓝色
- 可以横着走！前后左右斜向都行
- 适合需要精确定位的场景

## 开机与连接

### 硬件开机

1. 安装电池（注意正负极）
2. 按下电源开关
3. 等待 Jetson 启动（约 30 秒）

### SSH 连接

```bash
# 连接 LIMO 的 WiFi 热点（名称在机器背面）
# 然后 SSH 登录
ssh agilex@192.168.12.1
# 密码: agx
```

### 切换控制模式

机器侧面有一个 **三档开关（SWB）**：

| 档位 | 模式 | 用途 |
|------|------|------|
| 下 | 关闭 | 不接受任何控制 |
| 中 | 遥控 | 用手机 App 蓝牙遥控 |
| 上 | 程序控制 | ROS/Python 程序控制 |

**要用程序控制，把开关拨到"上"！**

## 键盘遥控

最简单的控制方式：

```bash
# 先启动底盘驱动
roslaunch limo_bringup limo_start.launch

# 再开键盘控制（另开一个终端）
roslaunch limo_bringup limo_teleop_keyboard.launch
```

按键说明（和打游戏类似）：

| 按键 | 效果 |
|------|------|
| `i` | 前进 |
| `,` | 后退 |
| `j` | 左转 |
| `l` | 右转 |
| `k` | 停止 |
| `q`/`z` | 加速 / 减速 |

## 建图与导航

### 第一步：建地图

让机器人在房间里走一圈，用激光雷达扫描环境：

```bash
# 启动建图（Cartographer 算法）
roslaunch limo_bringup limo_cartographer.launch

# 用键盘控制机器人走遍整个区域
# 建好后保存地图
rosrun map_server map_saver -f ~/my_map
```

### 第二步：用地图导航

有了地图就能自动导航了：

```bash
# 启动导航（差速模式为例）
roslaunch limo_bringup limo_navigation_diff.launch

# 在 RViz 中用 "2D Nav Goal" 工具点击目标位置
# 机器人会自动规划路径并前往
```

## 注意事项

1. **不要推！** 机器人关机时不要用手推着走，会产生反向电流损坏电机
2. **电量** 低电量时 LED 会闪红灯，及时充电（充电器先接电池再插电）
3. **履带模式** 一定要把侧门抬起来
4. **麦克纳姆轮** 安装有方向要求（轮子上的滚轮形成 X 形）
5. **温度** 工作温度 -10°C ~ 40°C，太冷太热都不行
6. **WiFi** LIMO 自带 WiFi 热点，也可以连接外部路由器

---

## 快速参考

### Piper 命令速查

```bash
# Demo 模式
python piper_mujoco_viz.py

# Hardware 模式
python piper_mujoco_viz.py --mode hardware

# Hardware + IK 控制
python piper_mujoco_viz.py --mode hardware --control ik

# 标定
python calibrate.py

# 重置
python reset_arm.py
```

### LIMO 命令速查

```bash
# 启动底盘
roslaunch limo_bringup limo_start.launch

# 键盘控制
roslaunch limo_bringup limo_teleop_keyboard.launch

# 查看雷达
roslaunch limo_bringup lidar_rviz.launch

# 建图
roslaunch limo_bringup limo_cartographer.launch

# 保存地图
rosrun map_server map_saver -f ~/my_map

# 导航
roslaunch limo_bringup limo_navigation_diff.launch
```
