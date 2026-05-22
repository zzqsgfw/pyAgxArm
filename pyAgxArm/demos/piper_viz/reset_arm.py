"""Full reset: clear errors, disable limits, restore defaults."""
import time
from pyAgxArm import create_agx_arm_config, AgxArmFactory, ArmModel, PiperFW

cfg = create_agx_arm_config(
    robot=ArmModel.PIPER,
    firmeware_version=PiperFW.DEFAULT,
    interface="agx_cando",
    channel="0",
)
robot = AgxArmFactory.create_arm(cfg)
robot.connect()
print("Connected.")

print("1. Reset controller...")
robot.reset()
time.sleep(1.0)

print("2. Clear joint errors (all)...")
print(f"   Result: {robot.clear_joint_error(255)}")
time.sleep(0.5)

print("3. Disable software joint limits...")
robot.set_joint_limits_enabled(False)

print("4. Reset joint angle/vel/acc limits to default...")
print(f"   Result: {robot.set_joint_angle_vel_acc_limits_to_default()}")

print("5. Reset flange vel/acc limits to default...")
print(f"   Result: {robot.set_flange_vel_acc_limits_to_default()}")

print("6. Disable crash protection...")
print(f"   Result: {robot.set_crash_protection_rating(joint_index=255, rating=0)}")

print("7. Enable...")
robot.enable()
time.sleep(1.0)

ja = robot.get_joint_angles()
if ja:
    import numpy as np
    print(f"\nJoint angles: {np.round(np.degrees(ja.msg), 1)} deg")

status = robot.get_arm_status()
if status:
    print(f"Arm status: ctrl_mode={status.msg.ctrl_mode} arm_status={status.msg.arm_status} motion_status={status.msg.motion_status}")

print("\n8. Disable (free-moving)...")
robot.disable()
robot.disconnect()
print("Done. Arm should be free-moving now.")
