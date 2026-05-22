"""Calibrate all joint zero points at current position."""
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

robot.enable()
time.sleep(0.5)

print("Calibrating all joints...")
result = robot.calibrate_joint(255)
print(f"Result: {result}")

time.sleep(1.0)

ja = robot.get_joint_angles()
if ja:
    import numpy as np
    print(f"Joint angles after calibration: {np.round(np.degrees(ja.msg), 1)} deg")

robot.disable()
robot.disconnect()
print("Done.")
