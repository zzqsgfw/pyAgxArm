# Piper 机械臂 + LIMO Pro 使用指南

> 面向初学者的快速上手文档

---

## 仓库地址

- **Piper 机械臂（本仓库）**：https://github.com/zzqsgfw/pyAgxArm
- **LIMO Pro 文档**：https://github.com/agilexrobotics/limo-doc

---

# Piper 机械臂

## 什么是 Piper？

Piper 是松灵机器人的 **6 轴协作机械臂**，末端带电动夹爪。我们写了一个 3D 可视化工具，可以：

- 在电脑上看到机械臂的实时姿态
- 用鼠标拖动"目标坐标系"来控制机械臂
- 控制夹爪开合

## 硬件连接

`电脑 USB` → `CAN 适配器（松灵 CANDO）` → `机械臂 CAN 接口`

## 运行

### 激活环境

```bash
conda activate pyAgxArm
```

### Demo 模式（不需要机械臂）

```bash
python pyAgxArm\demos\piper_viz\piper_mujoco_viz.py
```

弹出 3D 窗口，拖动坐标系，机械臂通过 IK 跟随。

### Hardware 模式（连接真实机械臂）

**操作真实机械臂时，手不要靠近运动范围！**

```bash
# 使用机械臂内置运动规划
python pyAgxArm\demos\piper_viz\piper_mujoco_viz.py --mode hardware

# 使用本地 IK 算法控制
python pyAgxArm\demos\piper_viz\piper_mujoco_viz.py --mode hardware --control ik
```

按 `ESC` 或 `Ctrl+C` 退出，程序会安全停止并松开电机。

## 交互操作

### 鼠标

| 操作 | 效果 |
|------|------|
| 左键拖红色轴 | 沿 X 轴（前后）平移 |
| 左键拖绿色轴 | 沿 Y 轴（左右）平移 |
| 左键拖蓝色轴 | 沿 Z 轴（上下）平移 |
| 左键拖白色原点 | 屏幕平面内自由平移 |
| 右键拖轴 | 绕该轴旋转 |
| 左键拖空白 | 旋转相机 |
| 右键拖空白 | 平移相机 |
| 滚轮 | 缩放 |
| 拖黄色滑块 | 控制夹爪开合 |

### 键盘

| 按键 | 效果 |
|------|------|
| `ESC` | 退出 |

### 屏幕信息

- **Target**：目标位置和朝向
- **EE err**：末端到目标的距离误差（mm）

## 工具脚本

```bash
# 标定关节零点（摆到零位后运行）
python pyAgxArm\demos\piper_viz\calibrate.py

# 完整重置（清除错误、恢复默认）
python pyAgxArm\demos\piper_viz\reset_arm.py

# 调试使能（逐关节测试）
python pyAgxArm\demos\piper_viz\test_enable.py
```

## 常见问题

| 问题 | 解决 |
|------|------|
| "URDF not found" | `git submodule update --init --recursive` |
| CAN 连接失败 | 检查 USB，重新插拔适配器 |
| 机械臂不动 | 运行 `reset_arm.py` |
| 退出后臂僵硬 | 重新运行程序再按 ESC 退出 |

---

# LIMO Pro 移动机器人

LIMO Pro 是松灵的全地形移动底盘，支持四种运动模式（阿克曼/差速/履带/麦克纳姆轮），搭载 Jetson Orin Nano + 激光雷达 + 深度相机。

**详细使用文档请参考官方仓库**：https://github.com/agilexrobotics/limo-doc

包含：开机连接、四种模式切换、ROS 建图导航、传感器使用等完整教程。
