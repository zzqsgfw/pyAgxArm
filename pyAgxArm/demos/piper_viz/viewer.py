"""
Custom GLFW-based MuJoCo viewer with draggable target coordinate frame.

Interaction:
    - Left-drag on axis: translate along that local axis
    - Right-drag on axis: rotate around that local axis
    - Left-drag on origin sphere: free translate in camera plane
    - Left/Right-drag elsewhere: camera rotate/pan
    - Scroll: camera zoom
    - ESC: quit
"""

import numpy as np
import mujoco
import glfw

from model import (
    AXIS_LENGTH, AXIS_RADIUS,
    get_target_qpos_adr, get_target_body_id,
)


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


class Viewer:

    def __init__(self, model, data):
        self.model = model
        self.data = data
        self.target_qadr = get_target_qpos_adr(model)
        self.target_bid = get_target_body_id(model)

        self._button_left = False
        self._button_right = False
        self._button_middle = False
        self._last_x = 0.0
        self._last_y = 0.0
        self._dragging_target = False
        self._drag_axis = None  # 'x' | 'y' | 'z' | 'free' | None

        self.cam = mujoco.MjvCamera()
        self.opt = mujoco.MjvOption()
        self.scene = mujoco.MjvScene(model, maxgeom=2000)
        self.context = None
        self.window = None
        self._running = True

        self.cam.azimuth = 135
        self.cam.elevation = -25
        self.cam.distance = 1.2
        self.cam.lookat[:] = [0.1, 0.0, 0.2]

        self._init_glfw()

    # ── GLFW init ────────────────────────────────────────────────

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

    # ── Pick / hit-test ──────────────────────────────────────────

    def _pick_body(self, x, y):
        win_w, win_h = glfw.get_window_size(self.window)
        fb_w, fb_h = glfw.get_framebuffer_size(self.window)
        aspect = fb_w / fb_h
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
                axis = 'free'
        return bodyid, axis

    # ── Mouse callbacks ──────────────────────────────────────────

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
            if self._button_middle:
                shift = glfw.get_key(window, glfw.KEY_LEFT_SHIFT) == glfw.PRESS
                action = mujoco.mjtMouse.mjMOUSE_MOVE_H if shift else mujoco.mjtMouse.mjMOUSE_ROTATE_H
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

    def _on_scroll(self, window, xoffset, yoffset):
        mujoco.mjv_moveCamera(
            self.model, mujoco.mjtMouse.mjMOUSE_ZOOM,
            0, -0.05 * yoffset, self.scene, self.cam
        )

    def _on_key(self, window, key, scancode, action, mods):
        if action == glfw.PRESS and key == glfw.KEY_ESCAPE:
            self._running = False

    # ── Target manipulation ──────────────────────────────────────

    def _get_local_axes(self):
        qadr = self.target_qadr
        quat = self.data.qpos[qadr+3:qadr+7].copy()
        rot = np.zeros(9)
        mujoco.mju_quat2Mat(rot, quat)
        rot = rot.reshape(3, 3)
        return rot[:, 0], rot[:, 1], rot[:, 2]

    def _pixel_to_world_scale(self, h):
        qadr = self.target_qadr
        fovy = np.radians(self.model.vis.global_.fovy)
        target_pos = self.data.qpos[qadr:qadr+3]
        cam_pos = np.array(self.scene.camera[0].pos)
        dist = np.linalg.norm(target_pos - cam_pos)
        return 2.0 * dist * np.tan(fovy / 2) / h

    def _translate_target(self, dx, dy, w, h):
        qadr = self.target_qadr
        scale = self._pixel_to_world_scale(h)

        if self._drag_axis == 'free':
            cam = self.scene.camera[0]
            right = np.cross(np.array(cam.forward), np.array(cam.up))
            up = np.array(cam.up)
            self.data.qpos[qadr:qadr+3] += scale * (dx * right - dy * up)
        else:
            local_x, local_y, local_z = self._get_local_axes()
            axis_world = {'x': local_x, 'y': local_y, 'z': local_z}[self._drag_axis]

            cam = self.scene.camera[0]
            right = np.cross(np.array(cam.forward), np.array(cam.up))
            up = np.array(cam.up)
            sx = np.dot(axis_world, right)
            sy = np.dot(axis_world, -up)
            slen = np.sqrt(sx**2 + sy**2)
            if slen < 1e-6:
                return
            proj = (dx * sx + dy * sy) / slen
            self.data.qpos[qadr:qadr+3] += scale * proj * axis_world

    def _rotate_target(self, dx, dy):
        qadr = self.target_qadr
        if self._drag_axis == 'free':
            return

        axis_world = {'x': 0, 'y': 1, 'z': 2}
        local_axes = self._get_local_axes()
        aw = local_axes[axis_world[self._drag_axis]]

        angle = (dx + dy) * 0.01
        ha = angle / 2
        dq = np.array([np.cos(ha), aw[0]*np.sin(ha), aw[1]*np.sin(ha), aw[2]*np.sin(ha)])

        cur_quat = self.data.qpos[qadr+3:qadr+7].copy()
        new_quat = quat_multiply(dq, cur_quat)
        new_quat /= np.linalg.norm(new_quat)
        self.data.qpos[qadr+3:qadr+7] = new_quat

    # ── Rendering ────────────────────────────────────────────────

    def _add_axis_geom(self, pos, direction, length, radius, rgba):
        if self.scene.ngeom + 2 > self.scene.maxgeom:
            return

        end = pos + direction * length
        mid = (pos + end) / 2

        z = direction / (np.linalg.norm(direction) + 1e-12)
        tmp = np.array([1, 0, 0]) if abs(z[0]) < 0.9 else np.array([0, 1, 0])
        x = np.cross(tmp, z); x /= np.linalg.norm(x) + 1e-12
        y = np.cross(z, x)
        mat = np.column_stack([x, y, z])

        g = self.scene.geoms[self.scene.ngeom]
        mujoco.mjv_initGeom(
            g, mujoco.mjtGeom.mjGEOM_CAPSULE,
            np.array([radius, radius, length / 2]),
            mid, mat.flatten(), np.array(rgba, dtype=np.float32),
        )
        self.scene.ngeom += 1

        g2 = self.scene.geoms[self.scene.ngeom]
        mujoco.mjv_initGeom(
            g2, mujoco.mjtGeom.mjGEOM_SPHERE,
            np.array([radius * 2.5, 0, 0]),
            end, np.eye(3).flatten(), np.array(rgba, dtype=np.float32),
        )
        self.scene.ngeom += 1

    def _draw_ee_frame(self, ee_pos, ee_mat):
        L, R = AXIS_LENGTH, AXIS_RADIUS
        self._add_axis_geom(ee_pos, ee_mat[:, 0], L, R, [1, 0.3, 0.3, 0.7])
        self._add_axis_geom(ee_pos, ee_mat[:, 1], L, R, [0.3, 1, 0.3, 0.7])
        self._add_axis_geom(ee_pos, ee_mat[:, 2], L, R, [0.3, 0.3, 1, 0.7])

    def render(self, ee_body_name="gripper_base"):
        w, h = glfw.get_framebuffer_size(self.window)
        viewport = mujoco.MjrRect(0, 0, w, h)
        mujoco.mjv_updateScene(
            self.model, self.data, self.opt, None, self.cam,
            mujoco.mjtCatBit.mjCAT_ALL, self.scene
        )

        ee_bid = mujoco.mj_name2id(self.model, mujoco.mjtObj.mjOBJ_BODY, ee_body_name)
        if ee_bid >= 0:
            ee_pos = self.data.xpos[ee_bid].copy()
            ee_mat = self.data.xmat[ee_bid].reshape(3, 3).copy()
            self._draw_ee_frame(ee_pos, ee_mat)

        mujoco.mjr_render(viewport, self.scene, self.context)

        # HUD
        pose = self.data.qpos[self.target_qadr:self.target_qadr+7]
        r, p, y = quat_to_euler(pose[3:7])
        hud_top = (
            f"Target: [{pose[0]:+.3f}, {pose[1]:+.3f}, {pose[2]:+.3f}]  "
            f"RPY: [{np.degrees(r):+.1f}, {np.degrees(p):+.1f}, {np.degrees(y):+.1f}]"
        )
        hud_bot = ""
        if ee_bid >= 0:
            ep = self.data.xpos[ee_bid]
            err = np.linalg.norm(ep - pose[:3]) * 1000
            hud_bot = f"EE err: {err:.1f}mm"

        mujoco.mjr_overlay(
            mujoco.mjtFont.mjFONT_NORMAL, mujoco.mjtGridPos.mjGRID_BOTTOMLEFT,
            viewport, hud_top, hud_bot, self.context
        )

        glfw.swap_buffers(self.window)
        glfw.poll_events()

    # ── Lifecycle ────────────────────────────────────────────────

    def is_running(self):
        return self._running and not glfw.window_should_close(self.window)

    def close(self):
        glfw.terminate()
