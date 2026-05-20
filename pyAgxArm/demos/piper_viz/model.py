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

ARROW_INIT_POS = "0.25 0.0 0.25"
ARROW_INIT_QUAT = "0.5 -0.5 -0.5 -0.5"

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
