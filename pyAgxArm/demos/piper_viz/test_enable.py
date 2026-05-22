"""Debug enable: try enable step by step, print all status."""
import time
from pyAgxArm import create_agx_arm_config, AgxArmFactory, ArmModel, PiperFW
import numpy as np

cfg = create_agx_arm_config(
    robot=ArmModel.PIPER,
    firmeware_version=PiperFW.DEFAULT,
    interface="agx_cando",
    channel="0",
)
robot = AgxArmFactory.create_arm(cfg)
robot.connect()
print("Connected.\n")

# Read status before anything
ja = robot.get_joint_angles()
if ja:
    print(f"Joint angles: {np.round(np.degrees(ja.msg), 1)} deg")
else:
    print("Joint angles: NONE (no data)")

status = robot.get_arm_status()
if status:
    print(f"Arm status: ctrl_mode={status.msg.ctrl_mode} arm_status={status.msg.arm_status} motion_status={status.msg.motion_status}")
else:
    print("Arm status: NONE")

# Check each motor
for i in range(1, 7):
    ms = robot.get_motor_states(i)
    if ms:
        print(f"  Motor {i}: pos={ms.msg.position:.3f} vel={ms.msg.velocity:.3f} cur={ms.msg.current:.3f}")
    else:
        print(f"  Motor {i}: NONE")

print("\n--- Step 1: reset ---")
robot.reset()
time.sleep(1.0)

print("--- Step 2: clear errors ---")
for i in range(1, 7):
    r = robot.clear_joint_error(i)
    print(f"  Joint {i} clear_error: {r}")
r = robot.clear_joint_error(255)
print(f"  All clear_error: {r}")
time.sleep(0.5)

print("\n--- Step 3: enable one by one ---")
for i in range(1, 7):
    r = robot.enable(i)
    print(f"  Enable joint {i}: {r}")
    time.sleep(0.3)

time.sleep(1.0)

# Check enable status
try:
    enabled = robot.get_joints_enable_status_list()
    print(f"\nEnable status: {enabled}")
except Exception as e:
    print(f"\nget_joints_enable_status_list failed: {e}")

status = robot.get_arm_status()
if status:
    print(f"Arm status: ctrl_mode={status.msg.ctrl_mode} arm_status={status.msg.arm_status} motion_status={status.msg.motion_status}")

ja = robot.get_joint_angles()
if ja:
    print(f"Joint angles: {np.round(np.degrees(ja.msg), 1)} deg")

print("\n--- Step 4: try move_j to current position (hold) ---")
if ja:
    robot.set_motion_mode('j')
    robot.move_j(ja.msg)
    print(f"  Sent move_j to current angles")
    time.sleep(2.0)

    ja2 = robot.get_joint_angles()
    if ja2:
        print(f"  Joint angles after hold: {np.round(np.degrees(ja2.msg), 1)} deg")

print("\n--- Step 5: disable ---")
robot.disable()
robot.disconnect()
print("Done.")
