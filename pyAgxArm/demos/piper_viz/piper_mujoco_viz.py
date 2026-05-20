"""
Piper 机械臂 MuJoCo 实时 3D 可视化

通过 pyAgxArm SDK 读取关节编码器角度，驱动 MuJoCo 中的 Piper 模型。
包含可拖动的目标坐标系，用于指示目标位姿（后续通过 SDK set_p 发送给机械臂）。
URDF 和 mesh 来自 piper_ros submodule，运行时自动解析 package:// 路径。

交互方式：
    - 左键拖动坐标系：平移
    - 右键拖动坐标系：旋转
    - 中键拖动：相机旋转/平移（Shift+中键=平移）
    - 滚轮：相机缩放

Usage:
    python piper_mujoco_viz.py                # Demo 模式
    python piper_mujoco_viz.py --mode hardware --interface agx_cando --channel 0
"""

import argparse
import time
import tempfile
import xml.etree.ElementTree as ET
from pathlib import Path

import numpy as np
import mujoco
import glfw

SCRIPT_DIR = Path(__file__).resolve().parent
PIPER_DESCRIPTION_DIR = SCRIPT_DIR / "piper_ros" / "src" / "piper_description"
URDF_PATH = PIPER_DESCRIPTION_DIR / "urdf" / "piper_description.urdf"
MESHES_DIR = PIPER_DESCRIPTION_DIR / "meshes"

ARM_JOINT_NAMES = ["joint1", "joint2", "joint3", "joint4", "joint5", "joint6"]
TARGET_BODY_NAME = "target_frame"
TARGET_JOINT_NAME = "target_freejoint"
ARROW_INIT_POS = "0.25 0.0 0.25"

# 坐标轴长度和粗细
AXIS_LENGTH = 0.08
AXIS_RADIUS = 0.006
AXIS_TIP_RADIUS = 0.015


# ── Model Loading ────────────────────────────────────────────────────

def load_model():
    """加载 URDF，编译为 MJCF，注入坐标系 body 和光源后重新编译"""
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

    # 禁用重力
    option = root.find("option")
    if option is None:
        option = ET.SubElement(root, "option")
    option.set("gravity", "0 0 0")

    # 光源
    wb = root.find("worldbody")
    _add_light(wb, "top_light", "0 0 1.5", "0 0 -1", "0.8 0.8 0.8")
    _add_light(wb, "front_light", "1 -1 1", "-1 1 -0.5", "0.5 0.5 0.5")

    # 目标坐标系 body
    frame = ET.SubElement(wb, "body")
    frame.set("name", TARGET_BODY_NAME)
    frame.set("pos", ARROW_INIT_POS)

    inertial = ET.SubElement(frame, "inertial")
    inertial.set("pos", "0 0 0")
    inertial.set("mass", "0.001")
    inertial.set("diaginertia", "1e-6 1e-6 1e-6")

    ET.SubElement(frame, "freejoint").set("name", TARGET_JOINT_NAME)

    # 原点球
    _add_geom(frame, "frame_origin", "sphere",
              f"{AXIS_RADIUS * 2} 0 0", "0 0 0", "0.9 0.9 0.9 1")

    # X 轴 (红)
    hl = AXIS_LENGTH / 2
    _add_geom(frame, "frame_x", "capsule",
              f"{AXIS_RADIUS} {hl} 0", f"{hl} 0 0", "1 0 0 1",
              quat="0.707 0 0.707 0")
    _add_geom(frame, "frame_x_tip", "sphere",
              f"{AXIS_TIP_RADIUS} 0 0", f"{AXIS_LENGTH} 0 0", "1 0 0 1")

    # Y 轴 (绿)
    _add_geom(frame, "frame_y", "capsule",
              f"{AXIS_RADIUS} {hl} 0", f"0 {hl} 0", "0 1 0 1",
              quat="0.707 0.707 0 0")
    _add_geom(frame, "frame_y_tip", "sphere",
              f"{AXIS_TIP_RADIUS} 0 0", f"0 {AXIS_LENGTH} 0", "0 1 0 1")

    # Z 轴 (蓝)
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


# ── Helpers ──────────────────────────────────────────────────────────

def get_joint_qpos_ids(model):
    ids = []
    for name in ARM_JOINT_NAMES:
        jid = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_JOINT, name)
        if jid >= 0:
            ids.append(model.jnt_qposadr[jid])
    return ids


def get_joint_limits(model):
    limits = []
    for name in ARM_JOINT_NAMES:
        jid = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_JOINT, name)
        if jid >= 0 and model.jnt_limited[jid]:
            limits.append(tuple(model.jnt_range[jid]))
        else:
            limits.append((-1.0, 1.0))
    return np.array(limits)


def get_target_qpos_adr(model):
    jid = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_JOINT, TARGET_JOINT_NAME)
    return model.jnt_qposadr[jid]


def get_target_body_id(model):
    return mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_BODY, TARGET_BODY_NAME)


def quat_to_euler(quat):
    qw, qx, qy, qz = quat
    roll = np.arctan2(2 * (qw * qx + qy * qz), 1 - 2 * (qx**2 + qy**2))
    pitch = np.arcsin(np.clip(2 * (qw * qy - qz * qx), -1, 1))
    yaw = np.arctan2(2 * (qw * qz + qx * qy), 1 - 2 * (qy**2 + qz**2))
    return roll, pitch, yaw


def quat_multiply(q1, q2):
    w1, x1, y1, z1 = q1
    w2, x2, y2, z2 = q2
    return np.array([
        w1*w2 - x1*x2 - y1*y2 - z1*z2,
        w1*x2 + x1*w2 + y1*z2 - z1*y2,
        w1*y2 - x1*z2 + y1*w2 + z1*x2,
        w1*z2 + x1*y2 - y1*x2 + z1*w2,
    ])


# ── Custom GLFW Viewer ──────────────────────────────────────────────

class Viewer:
    """自定义 MuJoCo viewer，支持左键拖平移/右键拖旋转目标坐标系"""

    def __init__(self, model, data):
        self.model = model
        self.data = data
        self.target_qadr = get_target_qpos_adr(model)
        self.target_bid = get_target_body_id(model)

        # Mouse state
        self._button_left = False
        self._button_right = False
        self._button_middle = False
        self._last_x = 0.0
        self._last_y = 0.0
        self._dragging_target = False
        self._drag_axis = None  # None, 'x', 'y', 'z', or 'free'

        # MuJoCo rendering
        self.cam = mujoco.MjvCamera()
        self.opt = mujoco.MjvOption()
        self.scene = mujoco.MjvScene(model, maxgeom=2000)
        self.context = None
        self.window = None
        self._running = True

        # 默认相机
        self.cam.azimuth = 135
        self.cam.elevation = -25
        self.cam.distance = 1.2
        self.cam.lookat[:] = [0.1, 0.0, 0.2]

        self._init_glfw()

    def _init_glfw(self):
        if not glfw.init():
            raise RuntimeError("Failed to init GLFW")

        self.window = glfw.create_window(1280, 720, "Piper MuJoCo Viz", None, None)
        if not self.window:
            glfw.terminate()
            raise RuntimeError("Failed to create GLFW window")

        glfw.make_context_current(self.window)
        glfw.swap_interval(1)
        self.context = mujoco.MjrContext(self.model, mujoco.mjtFontScale.mjFONTSCALE_150)

        glfw.set_cursor_pos_callback(self.window, self._on_mouse_move)
        glfw.set_mouse_button_callback(self.window, self._on_mouse_button)
        glfw.set_scroll_callback(self.window, self._on_scroll)
        glfw.set_key_callback(self.window, self._on_key)

    def _pick_body(self, x, y):
        """用 mjv_select 射线检测，返回 body id"""
        win_w, win_h = glfw.get_window_size(self.window)
        fb_w, fb_h = glfw.get_framebuffer_size(self.window)
        aspect = fb_w / fb_h
        # mjv_select uses relx/rely in [0, 1] (not [-1, 1])
        relx = x / win_w
        rely = 1.0 - y / win_h

        mujoco.mjv_updateScene(
            self.model, self.data, self.opt, None, self.cam,
            mujoco.mjtCatBit.mjCAT_ALL, self.scene
        )

        selpnt = np.zeros(3)
        geomid = np.array([-1], dtype=np.int32)
        flexid = np.array([-1], dtype=np.int32)
        skinid = np.array([-1], dtype=np.int32)
        bodyid = mujoco.mjv_select(
            self.model, self.data, self.opt, aspect, relx, rely,
            self.scene, selpnt, geomid, flexid, skinid
        )
        # 判断点击了哪个轴
        axis = None
        if bodyid == self.target_bid and geomid[0] >= 0:
            gname = mujoco.mj_id2name(self.model, mujoco.mjtObj.mjOBJ_GEOM, geomid[0])
            if gname and '_x' in gname:
                axis = 'x'
            elif gname and '_y' in gname:
                axis = 'y'
            elif gname and '_z' in gname:
                axis = 'z'
            else:
                axis = 'free'  # origin sphere
        return bodyid, axis

    def _on_mouse_button(self, window, button, action, mods):
        self._button_left = glfw.get_mouse_button(window, glfw.MOUSE_BUTTON_LEFT) == glfw.PRESS
        self._button_right = glfw.get_mouse_button(window, glfw.MOUSE_BUTTON_RIGHT) == glfw.PRESS
        self._button_middle = glfw.get_mouse_button(window, glfw.MOUSE_BUTTON_MIDDLE) == glfw.PRESS
        x, y = glfw.get_cursor_pos(window)
        self._last_x = x
        self._last_y = y

        if action == glfw.PRESS and (button == glfw.MOUSE_BUTTON_LEFT or button == glfw.MOUSE_BUTTON_RIGHT):
            bodyid, axis = self._pick_body(x, y)
            self._dragging_target = (bodyid == self.target_bid)
            self._drag_axis = axis
        elif action == glfw.RELEASE:
            self._dragging_target = False
            self._drag_axis = None

    def _on_mouse_move(self, window, x, y):
        dx = x - self._last_x
        dy = y - self._last_y
        self._last_x = x
        self._last_y = y

        if not (self._button_left or self._button_right or self._button_middle):
            return

        win_w, win_h = glfw.get_window_size(self.window)

        if self._dragging_target:
            if self._button_left:
                self._translate_target(dx, dy, win_w, win_h)
            elif self._button_right:
                self._rotate_target(dx, dy)
        else:
            # 相机控制
            if self._button_middle:
                shift = glfw.get_key(window, glfw.KEY_LEFT_SHIFT) == glfw.PRESS
                if shift:
                    action = mujoco.mjtMouse.mjMOUSE_MOVE_H
                else:
                    action = mujoco.mjtMouse.mjMOUSE_ROTATE_H
                mujoco.mjv_moveCamera(self.model, action, dx / win_w, dy / win_h, self.scene, self.cam)
            elif self._button_left:
                mujoco.mjv_moveCamera(
                    self.model, mujoco.mjtMouse.mjMOUSE_ROTATE_H,
                    dx / win_w, dy / win_h, self.scene, self.cam
                )
            elif self._button_right:
                mujoco.mjv_moveCamera(
                    self.model, mujoco.mjtMouse.mjMOUSE_MOVE_H,
                    dx / win_w, dy / win_h, self.scene, self.cam
                )

    def _get_local_axes(self):
        """获取目标坐标系在世界中的三个轴方向（从当前四元数）"""
        qadr = self.target_qadr
        quat = self.data.qpos[qadr+3:qadr+7].copy()
        # MuJoCo quat [w,x,y,z] -> rotation matrix
        rot = np.zeros(9)
        mujoco.mju_quat2Mat(rot, quat)
        rot = rot.reshape(3, 3)
        return rot[:, 0], rot[:, 1], rot[:, 2]  # local x, y, z in world

    def _pixel_to_world_scale(self, h):
        """每像素对应的世界距离"""
        qadr = self.target_qadr
        fovy = np.radians(self.model.vis.global_.fovy)
        target_pos = self.data.qpos[qadr:qadr+3]
        cam_pos = np.array(self.scene.camera[0].pos)
        dist = np.linalg.norm(target_pos - cam_pos)
        return 2.0 * dist * np.tan(fovy / 2) / h

    def _translate_target(self, dx, dy, w, h):
        """左键拖动：沿选中的 local 轴平移，或 free 模式在相机平面内平移"""
        qadr = self.target_qadr
        scale = self._pixel_to_world_scale(h)

        if self._drag_axis == 'free':
            # 相机平面内自由平移
            cam = self.scene.camera[0]
            right = np.cross(np.array(cam.forward), np.array(cam.up))
            up = np.array(cam.up)
            self.data.qpos[qadr:qadr+3] += scale * (dx * right - dy * up)
        else:
            # 沿 local 轴平移：将鼠标 dx/dy 投影到该轴在屏幕上的方向
            local_x, local_y, local_z = self._get_local_axes()
            axis_map = {'x': local_x, 'y': local_y, 'z': local_z}
            axis_world = axis_map[self._drag_axis]

            # 轴在屏幕上的投影方向
            cam = self.scene.camera[0]
            right = np.cross(np.array(cam.forward), np.array(cam.up))
            up = np.array(cam.up)
            axis_screen_x = np.dot(axis_world, right)
            axis_screen_y = np.dot(axis_world, -up)
            screen_len = np.sqrt(axis_screen_x**2 + axis_screen_y**2)
            if screen_len < 1e-6:
                return

            # 鼠标移动在轴屏幕投影上的分量
            proj = (dx * axis_screen_x + dy * axis_screen_y) / screen_len
            self.data.qpos[qadr:qadr+3] += scale * proj * axis_world

    def _rotate_target(self, dx, dy):
        """右键拖动：绕选中的 local 轴旋转"""
        qadr = self.target_qadr

        if self._drag_axis == 'free':
            return  # origin 球不支持旋转

        local_x, local_y, local_z = self._get_local_axes()
        axis_map = {'x': local_x, 'y': local_y, 'z': local_z}
        axis_world = axis_map[self._drag_axis]

        # 合并 dx/dy 为旋转量
        angle = (dx + dy) * 0.01

        # 绕该世界轴的旋转四元数
        ha = angle / 2
        dq = np.array([np.cos(ha),
                        axis_world[0] * np.sin(ha),
                        axis_world[1] * np.sin(ha),
                        axis_world[2] * np.sin(ha)])

        cur_quat = self.data.qpos[qadr+3:qadr+7].copy()
        new_quat = quat_multiply(dq, cur_quat)
        new_quat /= np.linalg.norm(new_quat)
        self.data.qpos[qadr+3:qadr+7] = new_quat

    def _on_scroll(self, window, xoffset, yoffset):
        mujoco.mjv_moveCamera(
            self.model, mujoco.mjtMouse.mjMOUSE_ZOOM,
            0, -0.05 * yoffset, self.scene, self.cam
        )

    def _on_key(self, window, key, scancode, action, mods):
        if action == glfw.PRESS and key == glfw.KEY_ESCAPE:
            self._running = False

    def is_running(self):
        return self._running and not glfw.window_should_close(self.window)

    def render(self):
        w, h = glfw.get_framebuffer_size(self.window)
        viewport = mujoco.MjrRect(0, 0, w, h)
        mujoco.mjv_updateScene(
            self.model, self.data, self.opt, None, self.cam,
            mujoco.mjtCatBit.mjCAT_ALL, self.scene
        )
        mujoco.mjr_render(viewport, self.scene, self.context)

        # HUD: 显示 target pose
        pose = self.data.qpos[self.target_qadr:self.target_qadr+7]
        r, p, y = quat_to_euler(pose[3:7])
        text = (
            f"Target: [{pose[0]:+.3f}, {pose[1]:+.3f}, {pose[2]:+.3f}]  "
            f"RPY: [{np.degrees(r):+.1f}, {np.degrees(p):+.1f}, {np.degrees(y):+.1f}]"
        )
        mujoco.mjr_overlay(
            mujoco.mjtFont.mjFONT_NORMAL, mujoco.mjtGridPos.mjGRID_BOTTOMLEFT,
            viewport, text, "", self.context
        )

        glfw.swap_buffers(self.window)
        glfw.poll_events()

    def close(self):
        glfw.terminate()


# ── Run Modes ────────────────────────────────────────────────────────

def run_offline_demo(model, data):
    print("Demo mode - sine wave + draggable target frame.")
    print("Left-drag on frame=translate, Right-drag=rotate.")
    print("Left/Right-drag elsewhere=camera. Scroll=zoom. ESC=quit.\n")

    qpos_ids = get_joint_qpos_ids(model)
    limits = get_joint_limits(model)
    mid = (limits[:, 0] + limits[:, 1]) / 2.0
    amp = (limits[:, 1] - limits[:, 0]) / 4.0
    freqs = np.array([0.3, 0.25, 0.35, 0.5, 0.45, 0.4])

    viewer = Viewer(model, data)
    t0 = time.time()
    try:
        while viewer.is_running():
            t = time.time() - t0
            angles = mid + amp * np.sin(2.0 * np.pi * freqs * t)
            for i, qid in enumerate(qpos_ids):
                data.qpos[qid] = angles[i]
            mujoco.mj_forward(model, data)
            viewer.render()
            time.sleep(1 / 60)
    finally:
        viewer.close()


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
    print("Left-drag on frame=translate, Right-drag=rotate. ESC=quit.\n")

    qpos_ids = get_joint_qpos_ids(model)
    num_joints = min(6, len(qpos_ids), robot.joint_nums)

    viewer = Viewer(model, data)
    try:
        while viewer.is_running():
            ja = robot.get_joint_angles()
            if ja is not None:
                for i in range(num_joints):
                    data.qpos[qpos_ids[i]] = ja.msg[i]
            mujoco.mj_forward(model, data)
            viewer.render()
            time.sleep(1 / 60)
    except KeyboardInterrupt:
        print("\nShutting down...")
    finally:
        viewer.close()
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
