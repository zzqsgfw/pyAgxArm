"""
MuJoCo model loading for Piper arm.

Loads URDF from piper_ros submodule, compiles to MJCF, injects target
coordinate frame body and lighting.
"""

import tempfile
import xml.etree.ElementTree as ET
from pathlib import Path

import mujoco

# ── Constants ────────────────────────────────────────────────────────

SCRIPT_DIR = Path(__file__).resolve().parent
PIPER_DESCRIPTION_DIR = SCRIPT_DIR / "piper_ros" / "src" / "piper_description"
URDF_PATH = PIPER_DESCRIPTION_DIR / "urdf" / "piper_description.urdf"
MESHES_DIR = PIPER_DESCRIPTION_DIR / "meshes"

ARM_JOINT_NAMES = ["joint1", "joint2", "joint3", "joint4", "joint5", "joint6"]
TARGET_BODY_NAME = "target_frame"
TARGET_JOINT_NAME = "target_freejoint"

ARROW_INIT_POS = "0.0601 0.0 0.2067"
ARROW_INIT_QUAT = "1 0 0 0"

GRIPPER_SLIDER_BODY_NAME = "gripper_slider"
GRIPPER_SLIDER_JOINT_NAME = "gripper_slider_joint"
GRIPPER_SLIDER_RANGE = 0.07  # full range in meters = max gripper width
# Offset from link6 origin, in link6 local frame
# link6 X points along arm, so put slider along Y (sideways) for easy dragging
GRIPPER_SLIDER_OFFSET = "0 0.06 0.05"  # side + up from EE, avoid overlap with Y axis

# Joint angles where link6 @ R_CORR == I exactly (~2.5° from all-zeros)
Q_HOME = [0.0, 0.0435, 0.0, 0.0, 0.0435, 0.0]

AXIS_LENGTH = 0.08
AXIS_RADIUS = 0.006
AXIS_TIP_RADIUS = 0.015


# ── XML helpers ──────────────────────────────────────────────────────

def _add_light(parent, name, pos, dir_, diffuse):
    el = ET.SubElement(parent, "light")
    el.set("name", name)
    el.set("pos", pos)
    el.set("dir", dir_)
    el.set("diffuse", diffuse)
    el.set("specular", "0.3 0.3 0.3")


def _add_geom(parent, name, gtype, size, pos, rgba, quat=None):
    g = ET.SubElement(parent, "geom")
    g.set("name", name)
    g.set("type", gtype)
    g.set("size", size)
    g.set("pos", pos)
    g.set("rgba", rgba)
    g.set("contype", "0")
    g.set("conaffinity", "0")
    if quat:
        g.set("quat", quat)


# ── Model loading ───────────────────────────────────────────────────

def load_model():
    """Load URDF, compile to MJCF, inject target frame + lights, recompile."""
    if not URDF_PATH.exists():
        raise FileNotFoundError(
            f"URDF not found: {URDF_PATH}\n"
            "Run: git submodule update --init --recursive"
        )
    urdf_text = URDF_PATH.read_text(encoding="utf-8")
    urdf_text = urdf_text.replace(
        "package://piper_description/meshes", MESHES_DIR.as_posix()
    )

    with tempfile.NamedTemporaryFile(
        suffix=".urdf", mode="w", encoding="utf-8", delete=False
    ) as f:
        f.write(urdf_text)
        tmp_urdf = f.name

    model_tmp = mujoco.MjModel.from_xml_path(tmp_urdf)
    Path(tmp_urdf).unlink(missing_ok=True)

    tmp_xml = tmp_urdf.replace(".urdf", ".xml")
    mujoco.mj_saveLastXML(tmp_xml, model_tmp)
    tree = ET.parse(tmp_xml)
    root = tree.getroot()
    Path(tmp_xml).unlink(missing_ok=True)

    # Disable gravity
    option = root.find("option")
    if option is None:
        option = ET.SubElement(root, "option")
    option.set("gravity", "0 0 0")

    # Lights
    wb = root.find("worldbody")
    _add_light(wb, "top_light", "0 0 1.5", "0 0 -1", "0.8 0.8 0.8")
    _add_light(wb, "front_light", "1 -1 1", "-1 1 -0.5", "0.5 0.5 0.5")

    # Target coordinate frame body
    frame = ET.SubElement(wb, "body")
    frame.set("name", TARGET_BODY_NAME)
    frame.set("pos", ARROW_INIT_POS)
    frame.set("quat", ARROW_INIT_QUAT)

    inertial = ET.SubElement(frame, "inertial")
    inertial.set("pos", "0 0 0")
    inertial.set("mass", "0.001")
    inertial.set("diaginertia", "1e-6 1e-6 1e-6")

    ET.SubElement(frame, "freejoint").set("name", TARGET_JOINT_NAME)

    _add_geom(frame, "frame_origin", "sphere",
              f"{AXIS_RADIUS * 2} 0 0", "0 0 0", "0.9 0.9 0.9 1")

    hl = AXIS_LENGTH / 2
    # X (red)
    _add_geom(frame, "frame_x", "capsule",
              f"{AXIS_RADIUS} {hl} 0", f"{hl} 0 0", "1 0 0 1",
              quat="0.707 0 0.707 0")
    _add_geom(frame, "frame_x_tip", "sphere",
              f"{AXIS_TIP_RADIUS} 0 0", f"{AXIS_LENGTH} 0 0", "1 0 0 1")
    # Y (green)
    _add_geom(frame, "frame_y", "capsule",
              f"{AXIS_RADIUS} {hl} 0", f"0 {hl} 0", "0 1 0 1",
              quat="0.707 0.707 0 0")
    _add_geom(frame, "frame_y_tip", "sphere",
              f"{AXIS_TIP_RADIUS} 0 0", f"0 {AXIS_LENGTH} 0", "0 1 0 1")
    # Z (blue)
    _add_geom(frame, "frame_z", "capsule",
              f"{AXIS_RADIUS} {hl} 0", f"0 0 {hl}", "0 0 1 1")
    _add_geom(frame, "frame_z_tip", "sphere",
              f"{AXIS_TIP_RADIUS} 0 0", f"0 0 {AXIS_LENGTH}", "0 0 1 1")

    # Gripper slider: child of link6, slides along link6's local Y axis.
    # Automatically follows EE in world space.
    link6_body = wb.find(".//body[@name='link6']")
    if link6_body is None:
        raise RuntimeError("Cannot find link6 body in MJCF to attach gripper slider")

    slider = ET.SubElement(link6_body, "body")
    slider.set("name", GRIPPER_SLIDER_BODY_NAME)
    slider.set("pos", GRIPPER_SLIDER_OFFSET)

    s_inertial = ET.SubElement(slider, "inertial")
    s_inertial.set("pos", "0 0 0")
    s_inertial.set("mass", "0.001")
    s_inertial.set("diaginertia", "1e-6 1e-6 1e-6")

    s_joint = ET.SubElement(slider, "joint")
    s_joint.set("name", GRIPPER_SLIDER_JOINT_NAME)
    s_joint.set("type", "slide")
    s_joint.set("axis", "0 1 0")  # slide along link6's local Y
    s_joint.set("range", f"0 {GRIPPER_SLIDER_RANGE}")
    s_joint.set("limited", "true")

    # Rail (gray track along Y)
    hr = GRIPPER_SLIDER_RANGE / 2
    _add_geom(slider, "slider_rail", "capsule",
              f"0.003 {hr}",
              f"0 {hr} 0",
              "0.5 0.5 0.5 0.5")

    # Handle (yellow sphere, the draggable part)
    _add_geom(slider, "slider_handle", "sphere",
              "0.015 0 0", "0 0 0", "1 0.8 0 1")

    # End marks: closed (red) / open (green)
    _add_geom(slider, "slider_closed_mark", "sphere",
              "0.006 0 0", f"0 -0.008 0", "1 0.3 0.3 1")
    _add_geom(slider, "slider_open_mark", "sphere",
              "0.006 0 0", f"0 {GRIPPER_SLIDER_RANGE + 0.008} 0", "0.3 1 0.3 1")

    with tempfile.NamedTemporaryFile(
        suffix=".xml", mode="wb", delete=False
    ) as f:
        tree.write(f)
        final_xml = f.name

    model = mujoco.MjModel.from_xml_path(final_xml)
    Path(final_xml).unlink(missing_ok=True)
    data = mujoco.MjData(model)
    return model, data


# ── MuJoCo query helpers ─────────────────────────────────────────────

GRIPPER_JOINT_NAMES = ["joint7", "joint8"]


def get_joint_qpos_ids(model):
    ids = []
    for name in ARM_JOINT_NAMES:
        jid = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_JOINT, name)
        if jid >= 0:
            ids.append(model.jnt_qposadr[jid])
    return ids


def get_target_qpos_adr(model):
    jid = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_JOINT, TARGET_JOINT_NAME)
    return model.jnt_qposadr[jid]


def get_target_body_id(model):
    return mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_BODY, TARGET_BODY_NAME)


def get_gripper_qpos_ids(model):
    """Get qpos addresses for gripper joints (joint7, joint8)."""
    ids = []
    for name in GRIPPER_JOINT_NAMES:
        jid = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_JOINT, name)
        if jid >= 0:
            ids.append(model.jnt_qposadr[jid])
    return ids


def get_gripper_slider_adr(model):
    """Get qpos address for the gripper slider joint."""
    jid = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_JOINT, GRIPPER_SLIDER_JOINT_NAME)
    return model.jnt_qposadr[jid]


def get_gripper_slider_body_id(model):
    return mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_BODY, GRIPPER_SLIDER_BODY_NAME)


def gripper_width_to_qpos(width):
    """Convert gripper width (meters, 0=closed, 0.07=full open) to joint7/joint8 qpos.

    Each finger moves half the total width. joint7 is positive, joint8 is negative.
    """
    half = min(max(width / 2, 0.0), 0.035)
    return half, -half
