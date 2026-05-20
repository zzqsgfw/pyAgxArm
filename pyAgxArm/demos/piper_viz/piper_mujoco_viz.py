"""
Piper 机械臂 MuJoCo 实时 3D 可视化

通过 pyAgxArm SDK 读取关节编码器角度，驱动 MuJoCo 中的 Piper 模型。
URDF 和 mesh 来自 piper_ros submodule，运行时自动解析 package:// 路径。

Usage:
    # Demo 模式（无硬件，正弦波模拟）
    python piper_mujoco_viz.py

    # 硬件模式（连接真实 Piper 机械臂）
    python piper_mujoco_viz.py --mode hardware --interface agx_cando --channel 0
"""

import argparse
import time
import tempfile
from pathlib import Path

import numpy as np
import mujoco
import mujoco.viewer

SCRIPT_DIR = Path(__file__).resolve().parent
PIPER_DESCRIPTION_DIR = SCRIPT_DIR / "piper_ros" / "src" / "piper_description"
URDF_PATH = PIPER_DESCRIPTION_DIR / "urdf" / "piper_description.urdf"
MESHES_DIR = PIPER_DESCRIPTION_DIR / "meshes"

ARM_JOINT_NAMES = ["joint1", "joint2", "joint3", "joint4", "joint5", "joint6"]


def load_model():
    """加载 URDF 到 MuJoCo，运行时解析 package:// 为本地路径"""
    if not URDF_PATH.exists():
        raise FileNotFoundError(
            f"URDF not found: {URDF_PATH}\n"
            "Run: git submodule update --init --recursive"
        )
    urdf_text = URDF_PATH.read_text(encoding="utf-8")
    meshes_uri = MESHES_DIR.as_posix()
    urdf_text = urdf_text.replace("package://piper_description/meshes", meshes_uri)

    # MuJoCo 从字符串加载需要写临时文件（from_xml_string 不支持 mesh 相对路径）
    with tempfile.NamedTemporaryFile(
        suffix=".urdf", mode="w", encoding="utf-8", delete=False
    ) as f:
        f.write(urdf_text)
        tmp_path = f.name

    model = mujoco.MjModel.from_xml_path(tmp_path)
    Path(tmp_path).unlink(missing_ok=True)
    data = mujoco.MjData(model)
    return model, data


def get_joint_qpos_ids(model):
    """获取 6 个手臂关节在 MuJoCo 中的 qpos index"""
    ids = []
    for name in ARM_JOINT_NAMES:
        jid = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_JOINT, name)
        if jid < 0:
            print(f"Warning: joint '{name}' not found in model")
            continue
        ids.append(model.jnt_qposadr[jid])
    return ids


def get_joint_limits(model):
    """获取关节限位"""
    limits = []
    for name in ARM_JOINT_NAMES:
        jid = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_JOINT, name)
        if jid >= 0 and model.jnt_limited[jid]:
            limits.append(tuple(model.jnt_range[jid]))
        else:
            limits.append((-1.0, 1.0))
    return np.array(limits)


def run_offline_demo(model, data):
    """无硬件模式：正弦波模拟关节运动"""
    print("Demo mode - sine wave simulation. Close window to exit.")
    qpos_ids = get_joint_qpos_ids(model)
    limits = get_joint_limits(model)
    mid = (limits[:, 0] + limits[:, 1]) / 2.0
    amp = (limits[:, 1] - limits[:, 0]) / 4.0
    freqs = np.array([0.3, 0.25, 0.35, 0.5, 0.45, 0.4])

    t0 = time.time()
    with mujoco.viewer.launch_passive(model, data) as viewer:
        while viewer.is_running():
            t = time.time() - t0
            angles = mid + amp * np.sin(2.0 * np.pi * freqs * t)
            for i, qid in enumerate(qpos_ids):
                data.qpos[qid] = angles[i]
            mujoco.mj_forward(model, data)
            viewer.sync()
            time.sleep(1 / 60)


def run_with_hardware(model, data, interface, channel):
    """连接真实 Piper，读取 SDK 关节角度驱动 MuJoCo 模型"""
    from pyAgxArm import create_agx_arm_config, AgxArmFactory, ArmModel, PiperFW

    cfg = create_agx_arm_config(
        robot=ArmModel.PIPER,
        firmeware_version=PiperFW.DEFAULT,
        interface=interface,
        channel=channel,
    )
    robot = AgxArmFactory.create_arm(cfg)
    robot.connect()
    print(f"Connected to Piper via {interface}:{channel}")

    qpos_ids = get_joint_qpos_ids(model)
    num_joints = min(6, len(qpos_ids), robot.joint_nums)
    print(f"Mapping {num_joints} joints from SDK to MuJoCo")

    try:
        with mujoco.viewer.launch_passive(model, data) as viewer:
            while viewer.is_running():
                ja = robot.get_joint_angles()
                if ja is not None:
                    for i in range(num_joints):
                        data.qpos[qpos_ids[i]] = ja.msg[i]
                mujoco.mj_forward(model, data)
                viewer.sync()
                time.sleep(1 / 60)
    except KeyboardInterrupt:
        print("\nShutting down...")
    finally:
        robot.disconnect()


def main():
    parser = argparse.ArgumentParser(description="Piper arm MuJoCo 3D visualizer")
    parser.add_argument(
        "--mode", choices=["hardware", "demo"], default="demo",
        help="'hardware' to connect real arm, 'demo' for sine wave simulation",
    )
    parser.add_argument("--interface", default="agx_cando", help="CAN interface")
    parser.add_argument("--channel", default="0", help="CAN channel")
    args = parser.parse_args()

    print("Loading Piper URDF into MuJoCo...")
    model, data = load_model()
    print(f"Model loaded: nq={model.nq}, nv={model.nv}")

    if args.mode == "hardware":
        run_with_hardware(model, data, args.interface, args.channel)
    else:
        run_offline_demo(model, data)


if __name__ == "__main__":
    main()
