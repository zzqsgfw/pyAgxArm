"""
Piper arm MuJoCo real-time 3D visualizer.

Usage:
    python piper_mujoco_viz.py                # Demo: IK tracking
    python piper_mujoco_viz.py --mode hardware --interface agx_cando --channel 0
"""

import argparse
import time

import mujoco

from model import (
    load_model, get_joint_qpos_ids, get_target_qpos_adr,
    get_gripper_qpos_ids, gripper_width_to_qpos, Q_HOME,
)
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

    # Start at Q_HOME: link6 @ R_CORR == target frame exactly
    for i, qid in enumerate(qpos_ids):
        data.qpos[qid] = Q_HOME[i]
    mujoco.mj_forward(model, data)

    q_state, optimizer = ik.create_solver_state(q_init=Q_HOME)

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


def _wait_motion_done(robot, timeout=10.0):
    """Poll until motion finished or collision. Raises on collision."""
    deadline = time.monotonic() + timeout
    while time.monotonic() < deadline:
        status = robot.get_arm_status()
        if status is not None:
            if status.msg.arm_status == 0x07:  # COLLISION_OCCURRED
                raise RuntimeError("Collision detected — aborting")
            if status.msg.motion_status == 0:  # reached target
                return
        time.sleep(0.05)
    print("  Warning: motion timeout, continuing anyway")


def _init_arm(robot, speed_pct=10):
    """Reset, enable at low speed, with collision detection active."""
    print("  Resetting...")
    robot.reset()
    time.sleep(1.0)

    print("  Disabling crash protection...")
    robot.set_crash_protection_rating(joint_index=255, rating=0)

    print(f"  Setting speed to {speed_pct}%...")
    robot.set_speed_percent(speed_pct)

    print("  Enabling...")
    robot.enable()
    time.sleep(1.0)

    print("  Initializing gripper...")
    effector = robot.init_effector(robot.OPTIONS.EFFECTOR.AGX_GRIPPER)
    time.sleep(0.5)

    print("  Arm + gripper enabled.")
    return effector


def _safe_disable(robot, settle_time=3.0):
    """Disable via electronic emergency stop (damping deceleration).

    Keeps CAN alive during settle_time so the controller can execute damping.
    """
    try:
        robot.electronic_emergency_stop()
        print(f"  Damping stop issued, waiting {settle_time}s for deceleration...")
        time.sleep(settle_time)
    except Exception:
        pass


def run_with_hardware(model, data, interface, channel):
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

    viewer = None
    try:
        qpos_ids = get_joint_qpos_ids(model)
        target_qadr = get_target_qpos_adr(model)
        num_joints = min(6, len(qpos_ids), robot.joint_nums)

        import numpy as _np

        # Fixed 90° rotation: link6 frame → target convention
        # Same as Viewer._R_CORR and piper_ik ee_correction rotation part
        # Ry(-90°): link6 → target (x=front, y=left, z=up)
        R_CORR = _np.array([[0, 0, -1], [0, 1, 0], [1, 0, 0]], dtype=_np.float64)
        R_CORR_INV = R_CORR.T  # target convention → link6 frame

        def _mat_to_rpy(m):
            pitch = _np.arcsin(_np.clip(-m[2, 0], -1, 1))
            if abs(abs(pitch) - _np.pi / 2) < 1e-6:
                return 0.0, pitch, _np.arctan2(-m[0, 1], m[1, 1])
            return _np.arctan2(m[2, 1], m[2, 2]), pitch, _np.arctan2(m[1, 0], m[0, 0])

        ee_bid = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_BODY, "link6")

        # Init arm at low speed → move to Q_HOME with collision guard
        effector = _init_arm(robot, speed_pct=10)
        print("  Moving to home pose (Q_HOME) at low speed...")
        robot.set_motion_mode('j')
        robot.move_j(Q_HOME)
        _wait_motion_done(robot, timeout=15.0)  # collision → exception → safe_disable

        # Ramp up speed after safe arrival
        print("  Ramping speed 10% → 30%...")
        for pct in range(10, 31, 5):
            robot.set_speed_percent(pct)
            time.sleep(0.1)
        robot.set_motion_mode('p')

        # Read actual joint angles → update MuJoCo → snap target to link6
        ja = robot.get_joint_angles()
        if ja is not None:
            for i, qid in enumerate(qpos_ids):
                data.qpos[qid] = ja.msg[i]
        mujoco.mj_forward(model, data)
        ee_pos = data.xpos[ee_bid].copy()
        data.qpos[target_qadr:target_qadr+3] = ee_pos
        mujoco.mj_forward(model, data)
        print(f"  link6 at: {_np.round(ee_pos, 3)}")

        print("Drag target frame to command arm via move_p.")
        print("Keys: O=open gripper, C=close gripper. ESC=quit.\n")

        SEND_INTERVAL = 0.2
        last_send_time = 0.0
        gripper_qpos_ids = get_gripper_qpos_ids(model)
        gripper_cmd_width = 0.0  # commanded width (meters)
        GRIPPER_STEP = 0.005
        GRIPPER_FORCE = 5.0

        # Read initial gripper state
        gs = effector.get_gripper_status()
        if gs is not None and gs.msg.mode == "width":
            gripper_cmd_width = gs.msg.value
            print(f"  Gripper initial width: {gripper_cmd_width*1000:.1f}mm")

        import glfw
        def _on_key(key, action):
            nonlocal gripper_cmd_width
            if action != glfw.PRESS:
                return
            if key == glfw.KEY_O:
                gripper_cmd_width = min(gripper_cmd_width + GRIPPER_STEP, 0.07)
                robot.move_gripper_m(gripper_cmd_width, GRIPPER_FORCE)
                print(f"\n[GRIP] open → {gripper_cmd_width*1000:.1f}mm")
            elif key == glfw.KEY_C:
                gripper_cmd_width = max(gripper_cmd_width - GRIPPER_STEP, 0.0)
                robot.move_gripper_m(gripper_cmd_width, GRIPPER_FORCE)
                print(f"\n[GRIP] close → {gripper_cmd_width*1000:.1f}mm")

        viewer = Viewer(model, data)
        viewer._key_callback = _on_key

        while viewer.is_running():
            # Read joint angles from hardware → update MuJoCo
            ja = robot.get_joint_angles()
            if ja is not None:
                for i in range(num_joints):
                    data.qpos[qpos_ids[i]] = ja.msg[i]

            # Read real gripper width from hardware → update MuJoCo
            gs = effector.get_gripper_status()
            if gs is not None and gs.msg.mode == "width":
                real_width = gs.msg.value
            else:
                real_width = gripper_cmd_width
            q7, q8 = gripper_width_to_qpos(real_width)
            for i, qid in enumerate(gripper_qpos_ids):
                data.qpos[qid] = [q7, q8][i]

            # Read target frame world pose
            target_pos = data.qpos[target_qadr:target_qadr+3].copy()
            target_quat = data.qpos[target_qadr+3:target_qadr+7].copy()
            R_target = _np.zeros(9)
            mujoco.mju_quat2Mat(R_target, target_quat)
            R_target = R_target.reshape(3, 3)

            # Convert: target convention → link6 convention
            R_link6 = R_target @ R_CORR_INV
            roll, pitch, yaw = _mat_to_rpy(R_link6)

            pose = [
                float(target_pos[0]), float(target_pos[1]), float(target_pos[2]),
                float(roll), float(pitch), float(yaw),
            ]

            # Send arm pose at throttled rate
            now = time.monotonic()
            if now - last_send_time >= SEND_INTERVAL:
                robot.move_p(pose)
                last_send_time = now
                print(f"\r[HW] move_p: xyz=[{pose[0]:.3f},{pose[1]:.3f},{pose[2]:.3f}] "
                      f"rpy=[{_np.degrees(roll):.1f},{_np.degrees(pitch):.1f},{_np.degrees(yaw):.1f}]"
                      f"  grip={real_width*1000:.0f}mm",
                      end="", flush=True)

            mujoco.mj_forward(model, data)
            viewer.render()
            time.sleep(1 / 60)

    except KeyboardInterrupt:
        print("\nInterrupted.")
    except Exception as e:
        print(f"\nError: {e}")
    finally:
        # ALWAYS safe-disable, no matter how we exit
        print("Safe shutdown...")
        _safe_disable(robot)
        if viewer is not None:
            viewer.close()
        robot.disconnect()
        print("Done.")


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
