"""
Differentiable FK / IK solver for the Piper arm.

Parses the joint chain from URDF, builds a differentiable FK with PyTorch,
and solves IK via Adam optimisation.

Usage:
    from piper_ik import PiperIK
    ik = PiperIK()
    q = ik.solve(target_pos, target_quat, q_init=None)
"""

import xml.etree.ElementTree as ET
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import torch
import yaml

# ═══════════════════════════════════════════════════════════════════════
#  URDF Parsing
# ═══════════════════════════════════════════════════════════════════════

SCRIPT_DIR = Path(__file__).resolve().parent
URDF_PATH = (
    SCRIPT_DIR / "piper_ros" / "src" / "piper_description"
    / "urdf" / "piper_description.urdf"
)

ARM_JOINT_NAMES = ["joint1", "joint2", "joint3", "joint4", "joint5", "joint6"]


@dataclass
class JointDef:
    name: str
    origin_xyz: np.ndarray
    origin_rpy: np.ndarray
    axis: np.ndarray
    lower: float
    upper: float


def parse_urdf_joints(urdf_path: str = None) -> list[JointDef]:
    """Parse the 6 arm revolute joints from a URDF file."""
    path = urdf_path or str(URDF_PATH)
    tree = ET.parse(path)
    root = tree.getroot()

    joints = []
    for jname in ARM_JOINT_NAMES:
        for joint_el in root.findall("joint"):
            if joint_el.get("name") != jname:
                continue
            origin = joint_el.find("origin")
            xyz = np.array([float(v) for v in origin.get("xyz", "0 0 0").split()])
            rpy = np.array([float(v) for v in origin.get("rpy", "0 0 0").split()])
            axis_el = joint_el.find("axis")
            axis = np.array([float(v) for v in axis_el.get("xyz", "0 0 1").split()])
            limit = joint_el.find("limit")
            lo = float(limit.get("lower", "-3.14"))
            hi = float(limit.get("upper", "3.14"))
            joints.append(JointDef(jname, xyz, rpy, axis, lo, hi))
            break
    return joints


# ═══════════════════════════════════════════════════════════════════════
#  Differentiable Transforms
# ═══════════════════════════════════════════════════════════════════════

def _rpy_to_matrix(r, p, y):
    """Fixed RPY angles (float) -> 3x3 rotation matrix tensor."""
    cr, sr = np.cos(r), np.sin(r)
    cp, sp = np.cos(p), np.sin(p)
    cy, sy = np.cos(y), np.sin(y)
    return torch.tensor([
        [cy*cp, cy*sp*sr - sy*cr, cy*sp*cr + sy*sr],
        [sy*cp, sy*sp*sr + cy*cr, sy*sp*cr - cy*sr],
        [-sp,   cp*sr,            cp*cr            ],
    ], dtype=torch.float64)


def _axis_angle_matrix(axis: torch.Tensor, angle: torch.Tensor):
    """Axis-angle -> 3x3 rotation matrix (differentiable)."""
    ax = axis / (torch.norm(axis) + 1e-12)
    c = torch.cos(angle)
    s = torch.sin(angle)
    t = 1 - c
    x, y, z = ax[0], ax[1], ax[2]
    return torch.stack([
        torch.stack([t*x*x + c,   t*x*y - s*z, t*x*z + s*y]),
        torch.stack([t*x*y + s*z, t*y*y + c,   t*y*z - s*x]),
        torch.stack([t*x*z - s*y, t*y*z + s*x, t*z*z + c  ]),
    ])


def _make_transform(R: torch.Tensor, t: torch.Tensor):
    """3x3 rotation + 3-vec translation -> 4x4 homogeneous matrix."""
    T = torch.eye(4, dtype=torch.float64)
    T[:3, :3] = R
    T[:3, 3] = t
    return T


def _rot_to_quat(R: torch.Tensor):
    """3x3 rotation matrix -> quaternion [w, x, y, z] (differentiable)."""
    tr = R[0, 0] + R[1, 1] + R[2, 2]
    w = torch.sqrt(torch.clamp(1 + tr, min=1e-8)) / 2
    x = (R[2, 1] - R[1, 2]) / (4 * w + 1e-12)
    y = (R[0, 2] - R[2, 0]) / (4 * w + 1e-12)
    z = (R[1, 0] - R[0, 1]) / (4 * w + 1e-12)
    return torch.stack([w, x, y, z])


# ═══════════════════════════════════════════════════════════════════════
#  FK Chain
# ═══════════════════════════════════════════════════════════════════════

class PiperFK(torch.nn.Module):
    """Differentiable FK: joint angles -> end-effector pose."""

    def __init__(self, joints: list[JointDef]):
        super().__init__()
        self.n_joints = len(joints)

        fixed_transforms = []
        axes = []
        for j in joints:
            R = _rpy_to_matrix(*j.origin_rpy)
            t = torch.tensor(j.origin_xyz, dtype=torch.float64)
            fixed_transforms.append(_make_transform(R, t))
            axes.append(torch.tensor(j.axis, dtype=torch.float64))

        self.register_buffer("_dummy", torch.zeros(1))
        self.fixed_transforms = fixed_transforms
        self.axes = axes

        # Ry(-90°): link6 frame → target convention (x=front, y=left, z=up)
        self.ee_correction = torch.tensor([
            [ 0.0,  0.0, -1.0, 0.0],
            [ 0.0,  1.0,  0.0, 0.0],
            [ 1.0,  0.0,  0.0, 0.0],
            [ 0.0,  0.0,  0.0, 1.0],
        ], dtype=torch.float64)

    def forward(self, q: torch.Tensor):
        """q: (6,) joint angles -> (pos(3), quat(4))"""
        T = torch.eye(4, dtype=torch.float64)
        for i in range(self.n_joints):
            T = T @ self.fixed_transforms[i]
            R_joint = _axis_angle_matrix(self.axes[i], q[i])
            T = T @ _make_transform(R_joint, torch.zeros(3, dtype=torch.float64))

        T = T @ self.ee_correction
        pos = T[:3, 3]
        quat = _rot_to_quat(T[:3, :3])
        return pos, quat


# ═══════════════════════════════════════════════════════════════════════
#  IK Solver
# ═══════════════════════════════════════════════════════════════════════

@dataclass
class _OptimConfig:
    """Groups the weights / options passed to _run_optim."""
    n_steps: int = 100
    pos_weight: float = 10.0
    rot_weight: float = 1.0
    smooth_weight: float = 0.0
    limit_weight: float = 10.0
    debug: bool = False


IK_CONFIG_PATH = SCRIPT_DIR / "ik_config.yaml"


def load_ik_config(path: str = None) -> dict:
    """Load IK config from YAML. Returns empty dict if file not found."""
    p = Path(path) if path else IK_CONFIG_PATH
    if p.exists():
        with open(p, encoding="utf-8") as f:
            return yaml.safe_load(f) or {}
    return {}


class PiperIK:
    """Adam-based IK solver with optional warm-start for incremental use."""

    def __init__(self, urdf_path: str = None, config_path: str = None):
        joints = parse_urdf_joints(urdf_path)
        self.joints = joints
        self.fk = PiperFK(joints)
        self.n_joints = len(joints)

        self.lower = torch.tensor([j.lower for j in joints], dtype=torch.float64)
        self.upper = torch.tensor([j.upper for j in joints], dtype=torch.float64)
        self.mid = (self.lower + self.upper) / 2

        # Load config
        self.cfg = load_ik_config(config_path)
        ws = self.cfg.get("workspace", {})
        self.r_min = ws.get("r_min", 0.08)
        self.r_max = ws.get("r_max", 0.55)

        self._oneshot_cfg = self.cfg.get("oneshot", {})
        self._inc_cfg = self.cfg.get("incremental", {})

    # ── public helpers ────────────────────────────────────────────────

    def clamp_target(self, pos: np.ndarray) -> np.ndarray:
        """Clamp a target position to the reachable workspace shell."""
        d = np.linalg.norm(pos)
        if d < 1e-6:
            return np.array([self.r_min, 0, 0])
        if d < self.r_min:
            return pos / d * self.r_min
        if d > self.r_max:
            return pos / d * self.r_max
        return pos

    def fk_np(self, q: np.ndarray):
        """NumPy FK wrapper: (6,) -> (pos(3), quat(4))."""
        with torch.no_grad():
            pos, quat = self.fk(torch.tensor(q, dtype=torch.float64))
        return pos.numpy(), quat.numpy()

    # ── one-shot solve ────────────────────────────────────────────────

    def solve(
        self,
        target_pos: np.ndarray,
        target_quat: np.ndarray = None,
        q_init: np.ndarray = None,
        **overrides,
    ) -> np.ndarray:
        """
        One-shot IK solve. Params from ik_config.yaml[oneshot], overridable via kwargs.

        Returns:
            (6,) joint angles in radians.
        """
        oc = self._oneshot_cfg
        lr = overrides.get("lr", oc.get("lr", 0.05))
        q = self._make_q(q_init)
        optimizer = torch.optim.Adam([q], lr=lr)
        target_p, target_q = self._prepare_targets(target_pos, target_quat)

        cfg = _OptimConfig(
            n_steps=overrides.get("n_steps", oc.get("n_steps", 100)),
            pos_weight=overrides.get("pos_weight", oc.get("pos_weight", 10.0)),
            rot_weight=overrides.get("rot_weight", oc.get("rot_weight", 1.0)),
            smooth_weight=0.0,
            limit_weight=overrides.get("limit_weight", oc.get("limit_weight", 10.0)),
            debug=overrides.get("debug", True),
        )
        self._run_optim(q, optimizer, target_p, target_q, q_prev=None, cfg=cfg)
        return q.detach().numpy()

    # ── incremental solve (warm start) ────────────────────────────────

    def create_solver_state(self, q_init: np.ndarray = None, lr: float = None):
        """Create persistent (q, optimizer) pair for incremental IK."""
        if lr is None:
            lr = self._inc_cfg.get("lr", 0.05)
        q = self._make_q(q_init)
        optimizer = torch.optim.Adam([q], lr=lr)
        return q, optimizer

    def solve_incremental(
        self,
        q: torch.Tensor,
        optimizer: torch.optim.Optimizer,
        target_pos: np.ndarray,
        target_quat: np.ndarray = None,
        **overrides,
    ) -> np.ndarray:
        """
        Incremental IK: a few Adam steps per frame with warm start.
        Params from ik_config.yaml[incremental], overridable via kwargs.
        """
        ic = self._inc_cfg
        target_p, target_q = self._prepare_targets(target_pos, target_quat)
        q_prev = q.detach().clone()

        # Reset optimizer momentum on large target jumps
        threshold = ic.get("momentum_reset_threshold", 0.05)
        if not hasattr(self, '_last_target_p'):
            self._last_target_p = target_p.clone()
        if torch.norm(target_p - self._last_target_p).item() > threshold:
            optimizer.state.clear()
        self._last_target_p = target_p.clone()

        cfg = _OptimConfig(
            n_steps=overrides.get("n_steps", ic.get("n_steps", 5)),
            pos_weight=overrides.get("pos_weight", ic.get("pos_weight", 10.0)),
            rot_weight=overrides.get("rot_weight", ic.get("rot_weight", 1.0)),
            smooth_weight=overrides.get("smooth_weight", ic.get("smooth_weight", 5.0)),
            limit_weight=overrides.get("limit_weight", ic.get("limit_weight", 10.0)),
        )
        self._run_optim(q, optimizer, target_p, target_q, q_prev=q_prev, cfg=cfg)

        # Debug: print every 30 frames
        if not hasattr(self, '_inc_count'):
            self._inc_count = 0
        self._inc_count += 1
        if self._inc_count % 30 == 0:
            with torch.no_grad():
                pos, quat = self.fk(q)
                pe = float(torch.sqrt(torch.sum((pos - target_p)**2))) * 1000
                re = 0.0
                if target_q is not None:
                    dot = torch.clamp(torch.abs(torch.sum(quat * target_q)), 0, 1)
                    re = float(2 * torch.acos(dot) * 180 / 3.14159)
                dq = float(torch.norm(q.detach() - q_prev)) * 180 / 3.14159
                print(f"\r[IK inc] pos={pe:.1f}mm rot={re:.1f}deg dq={dq:.1f}deg",
                      end="", flush=True)

        return q.detach().numpy()

    # ── private ───────────────────────────────────────────────────────

    def _make_q(self, q_init: np.ndarray = None) -> torch.Tensor:
        """Build a requires_grad joint-angle tensor from init or midpoint."""
        if q_init is not None:
            return torch.tensor(q_init, dtype=torch.float64, requires_grad=True)
        return self.mid.clone().detach().requires_grad_(True)

    @staticmethod
    def _prepare_targets(target_pos, target_quat):
        """Convert numpy targets to tensors; normalise quaternion if given."""
        target_p = torch.tensor(target_pos, dtype=torch.float64)
        target_q = None
        if target_quat is not None:
            target_q = torch.tensor(target_quat, dtype=torch.float64)
            target_q = target_q / (torch.norm(target_q) + 1e-12)
        return target_p, target_q

    def _run_optim(self, q, optimizer, target_p, target_q, *, q_prev, cfg):
        """Core optimisation loop shared by solve() and solve_incremental()."""
        for step in range(cfg.n_steps):
            optimizer.zero_grad()
            pos, quat = self.fk(q)

            # Position loss
            pos_err_sq = torch.sum((pos - target_p) ** 2)
            loss = cfg.pos_weight * pos_err_sq

            # Rotation loss: SO(3) distance  1 - (q1.q2)^2
            rot_err_deg = 0.0
            if target_q is not None:
                dot = torch.clamp(torch.sum(quat * target_q), -1.0, 1.0)
                loss = loss + cfg.rot_weight * (1 - dot ** 2)
                with torch.no_grad():
                    rot_err_deg = float(
                        2 * torch.acos(torch.clamp(torch.abs(dot), 0, 1)) * 180 / 3.14159
                    )

            # Joint-limit penalty
            below = torch.clamp(self.lower - q, min=0)
            above = torch.clamp(q - self.upper, min=0)
            loss = loss + cfg.limit_weight * torch.sum(below ** 2 + above ** 2)

            # Smoothness loss (incremental mode only)
            if q_prev is not None and cfg.smooth_weight > 0:
                loss = loss + cfg.smooth_weight * torch.sum((q - q_prev) ** 2)

            # Debug print (one-shot mode)
            if cfg.debug and step % max(1, cfg.n_steps // 10) == 0:
                pos_err_mm = float(torch.sqrt(pos_err_sq.detach())) * 1000
                print(f"  [IK] step={step:3d} loss={float(loss.detach()):.6f} "
                      f"pos_err={pos_err_mm:.1f}mm rot_err={rot_err_deg:.1f}deg "
                      f"q={[f'{v:.2f}' for v in q.detach().tolist()]}")

            loss.backward()
            optimizer.step()

            with torch.no_grad():
                q.clamp_(self.lower, self.upper)

        if cfg.debug:
            with torch.no_grad():
                pos, quat = self.fk(q)
                pos_err_mm = float(torch.sqrt(torch.sum((pos - target_p)**2))) * 1000
                rot_err_deg = 0.0
                if target_q is not None:
                    dot = torch.clamp(torch.abs(torch.sum(quat * target_q)), 0, 1)
                    rot_err_deg = float(2 * torch.acos(dot) * 180 / 3.14159)
                print(f"  [IK] FINAL pos_err={pos_err_mm:.1f}mm rot_err={rot_err_deg:.1f}deg")
