"""Calibrate all joint zero points at current position."""
import time
import numpy as np
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

robot.reset()
time.sleep(1.0)

robot.enable()
time.sleep(1.0)

# Before
ja = robot.get_joint_angles()
if ja:
    print(f"BEFORE calibration: {np.round(np.degrees(ja.msg), 1)} deg")

print("\nCalibrating all joints...")
result = robot.calibrate_joint(255)
print(f"Result: {result}")

# Wait longer for flash write
print("Waiting 3s for flash write...")
time.sleep(3.0)

# Read back
ja = robot.get_joint_angles()
if ja:
    print(f"AFTER calibration:  {np.round(np.degrees(ja.msg), 1)} deg")

# If not zero, might need power cycle
if ja and any(abs(a) > 1.0 for a in np.degrees(ja.msg)):
    print("\n*** Angles not zero! Try power-cycling the arm (unplug power, wait 5s, replug). ***")

# Reset limits
print("\nResetting limits...")
robot.set_joint_limits_enabled(False)
robot.set_joint_angle_vel_acc_limits_to_default()
robot.set_flange_vel_acc_limits_to_default()
robot.set_crash_protection_rating(joint_index=255, rating=0)

robot.disable()
robot.disconnect()
print("Done.")
