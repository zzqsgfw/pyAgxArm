"""
Piper arm MuJoCo real-time 3D visualizer.

Usage:
    python piper_mujoco_viz.py                # Demo: IK tracking
    python piper_mujoco_viz.py --mode hardware --interface agx_cando --channel 0
"""

import argparse
import time

import mujoco

from model import load_model, get_joint_qpos_ids, get_target_qpos_adr
from viewer import Viewer


# ── Run Modes ────────────────────────────────────────────────────────

def run_offline_demo(model, data):
    from piper_ik import PiperIK

    print("Demo mode - IK tracking + draggable target frame.")
    print("Left-drag on axis=translate, Right-drag=rotate.")
    print("Left/Right-drag elsewhere=camera. Scroll=zoom. ESC=quit.\n")

    ik = PiperIK()
    qpos_ids = get_joint_qpos_ids(model)
    target_qadr = get_target_qpos_adr(model)

    # One-shot IK to initial target pose
    init_pos = data.qpos[target_qadr:target_qadr+3].copy()
    init_pos = ik.clamp_target(init_pos)
    init_quat = data.qpos[target_qadr+3:target_qadr+7].copy()
    q_init = ik.solve(init_pos, target_quat=init_quat)
    for i, qid in enumerate(qpos_ids):
        data.qpos[qid] = q_init[i]

    q_state, optimizer = ik.create_solver_state(q_init=q_init)

    viewer = Viewer(model, data)
    try:
        while viewer.is_running():
            target_pos = data.qpos[target_qadr:target_qadr+3].copy()
            target_pos = ik.clamp_target(target_pos)
            data.qpos[target_qadr:target_qadr+3] = target_pos
            target_quat = data.qpos[target_qadr+3:target_qadr+7].copy()

            q_sol = ik.solve_incremental(
                q_state, optimizer, target_pos, target_quat,
            )
            for i, qid in enumerate(qpos_ids):
                data.qpos[qid] = q_sol[i]

            mujoco.mj_forward(model, data)
            viewer.render()
            time.sleep(1 / 60)
    finally:
        viewer.close()


def run_with_hardware(model, data, interface, channel):
    from pyAgxArm import create_agx_arm_config, AgxArmFactory, ArmModel, PiperFW
    from viewer import quat_to_euler

    cfg = create_agx_arm_config(
        robot=ArmModel.PIPER,
        firmeware_version=PiperFW.DEFAULT,
        interface=interface,
        channel=channel,
    )
    robot = AgxArmFactory.create_arm(cfg)
    robot.connect()
    print(f"Connected to Piper via {interface}:{channel}")
    print("Drag target frame to command arm via move_p.")
    print("Left-drag=translate, Right-drag=rotate. ESC=quit.\n")

    qpos_ids = get_joint_qpos_ids(model)
    target_qadr = get_target_qpos_adr(model)
    num_joints = min(6, len(qpos_ids), robot.joint_nums)

    # Enable arm and set cartesian PTP mode
    robot.enable()
    robot.set_motion_mode('p')
    robot.set_speed_percent(30)

    # Track last sent pose to avoid spamming CAN bus
    last_sent_pose = None
    SEND_THRESHOLD = 0.002  # 2mm / 0.1deg change triggers new command

    viewer = Viewer(model, data)
    try:
        while viewer.is_running():
            # Read joint angles from hardware -> update MuJoCo model
            ja = robot.get_joint_angles()
            if ja is not None:
                for i in range(num_joints):
                    data.qpos[qpos_ids[i]] = ja.msg[i]

            # Read target frame pose
            target_pos = data.qpos[target_qadr:target_qadr+3].copy()
            target_quat = data.qpos[target_qadr+3:target_qadr+7].copy()
            r, p, y = quat_to_euler(target_quat)
            pose = [target_pos[0], target_pos[1], target_pos[2], r, p, y]

            # Send to arm only when target changed significantly
            if last_sent_pose is None:
                last_sent_pose = pose
            else:
                pos_delta = sum((a - b)**2 for a, b in zip(pose[:3], last_sent_pose[:3]))**0.5
                rot_delta = sum((a - b)**2 for a, b in zip(pose[3:], last_sent_pose[3:]))**0.5
                if pos_delta > SEND_THRESHOLD or rot_delta > 0.02:
                    robot.move_p(pose)
                    last_sent_pose = pose

            mujoco.mj_forward(model, data)
            viewer.render()
            time.sleep(1 / 60)
    except KeyboardInterrupt:
        print("\nShutting down...")
    finally:
        robot.disable()
        viewer.close()
        robot.disconnect()


# ── Main ─────────────────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser(description="Piper arm MuJoCo 3D visualizer")
    parser.add_argument(
        "--mode", choices=["hardware", "demo"], default="demo",
        help="'hardware' to connect real arm, 'demo' for IK tracking simulation",
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
