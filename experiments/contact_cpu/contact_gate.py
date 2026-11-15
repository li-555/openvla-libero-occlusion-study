"""CPU-only causal pose hypotheses. No simulator, policy, files, or truth access."""
from dataclasses import dataclass, asdict
import numpy as np


def array(x, shape):
    a = np.array(x, dtype=float, copy=True)
    if a.shape != shape or not np.isfinite(a).all():
        raise ValueError(f"Expected finite {shape}, got {a.shape}")
    return a


def pose(pos, quat, order):
    """World-from-local transform; explicit xyzw or wxyz input, unit-normalized."""
    p, q = array(pos, (3,)), array(quat, (4,))
    if order == "wxyz":
        q = q[[1, 2, 3, 0]]
    elif order != "xyzw":
        raise ValueError("Quaternion order must be xyzw or wxyz")
    norm = np.linalg.norm(q)
    if norm < 1e-12:
        raise ValueError("Zero quaternion")
    x, y, z, w = q / norm
    T = np.eye(4)
    T[:3, :3] = [[1-2*(y*y+z*z), 2*(x*y-z*w), 2*(x*z+y*w)],
                 [2*(x*y+z*w), 1-2*(x*x+z*z), 2*(y*z-x*w)],
                 [2*(x*z-y*w), 2*(y*z+x*w), 1-2*(x*x+y*y)]]
    T[:3, 3] = p
    return T


def transform(x):
    T = array(x, (4, 4))
    R = T[:3, :3]
    if (not np.allclose(T[3], [0, 0, 0, 1], atol=1e-7)
            or not np.allclose(R.T @ R, np.eye(3), atol=1e-6)
            or not np.isclose(np.linalg.det(R), 1., atol=1e-6)):
        raise ValueError("Invalid SE(3) transform")
    return T


def inv(T):
    out = np.eye(4)
    out[:3, :3] = T[:3, :3].T
    out[:3, 3] = -out[:3, :3] @ T[:3, 3]
    return out


def angle(R):
    return float(np.arccos(np.clip((np.trace(R) - 1) / 2, -1, 1)))


def errors(pred, truth):
    return (float(np.linalg.norm(pred[:3, 3] - truth[:3, 3])),
            angle(pred[:3, :3].T @ truth[:3, :3]))


@dataclass(frozen=True)
class Config:
    # Illustrative engineering thresholds, NOT fitted or validated on HKU data.
    history: int = 5
    max_gap_steps: int = 1
    proximity_m: float = .12
    rel_translation_m: float = .008
    rel_rotation_rad: float = .12
    min_motion_m: float = .015
    min_motion_rad: float = .15
    static_translation_m: float = .003
    static_rotation_rad: float = .05
    # Explicit calibration: scalar aperture = dot(weights, gripper_qpos).
    # No calibration => positive contact gate disabled; uncertainty is preserved.
    grip_weights: tuple = (1., -1.)
    grip_closed: float | None = None
    grip_open: float | None = None
    closed_fraction_max: float = .35
    open_fraction_min: float = .8
    grip_valid_margin: float = .1

    def __post_init__(self):
        if (not isinstance(self.history, (int, np.integer))
                or not isinstance(self.max_gap_steps, (int, np.integer))
                or self.history < 3 or self.max_gap_steps < 1):
            raise ValueError("history >=3 and max_gap_steps >=1 must be integers")
        for k in ("proximity_m", "rel_translation_m", "rel_rotation_rad",
                  "min_motion_m", "min_motion_rad", "static_translation_m",
                  "static_rotation_rad"):
            if not np.isfinite(getattr(self, k)) or getattr(self, k) <= 0:
                raise ValueError(f"Invalid {k}")
        array(self.grip_weights, (2,))
        if (self.grip_closed is None) != (self.grip_open is None):
            raise ValueError("Supply BOTH gripper calibration endpoints")
        if self.grip_closed is not None:
            if not np.isfinite([self.grip_closed, self.grip_open]).all() or abs(self.grip_open-self.grip_closed) < 1e-8:
                raise ValueError("Distinct finite gripper endpoints required")
        if not 0 <= self.closed_fraction_max < self.open_fraction_min <= 1:
            raise ValueError("Invalid gripper thresholds")
        if not np.isfinite(self.grip_valid_margin) or self.grip_valid_margin < 0:
            raise ValueError("Invalid gripper margin")

    def grip(self, qpos):
        q = array(qpos, (2,))
        if self.grip_closed is None:
            return "unknown"
        f = (np.dot(self.grip_weights, q)-self.grip_closed)/(self.grip_open-self.grip_closed)
        if f < -self.grip_valid_margin or f > 1+self.grip_valid_margin:
            return "unknown"
        return "closed" if f <= self.closed_fraction_max else "open" if f >= self.open_fraction_min else "unknown"


@dataclass(frozen=True)
class Visible:
    step: int
    eef: np.ndarray
    obj: np.ndarray
    gripper: np.ndarray


@dataclass(frozen=True)
class Robot:
    step: int
    eef: np.ndarray
    gripper: np.ndarray


@dataclass
class Prediction:
    step: int
    memory: np.ndarray
    physics: np.ndarray
    mode: str
    reason: str

    @property
    def selected(self):
        return {"memory": self.memory, "physics": self.physics}.get(self.mode)


class ContactGate:
    """One occlusion episode. Construct ONLY from actually visible observations.

    Positive evidence is a heuristic, never a calibrated contact probability.
    During occlusion, loss of closed-gripper evidence latches uncertainty until
    re-observation; it does not imply the object returned to the memory pose.
    """
    def __init__(self, visible, config=Config()):
        if not visible:
            raise ValueError("At least one visible anchor required")
        self.cfg = config
        hs = []
        for h in visible:
            if not isinstance(h.step, (int, np.integer)):
                raise ValueError("Integer steps required")
            hs.append(Visible(int(h.step), transform(h.eef), transform(h.obj), array(h.gripper, (2,))))
        if any(b.step <= a.step for a, b in zip(hs, hs[1:])):
            raise ValueError("Visible history must be chronological")
        hs = hs[-config.history:]
        self.anchor = hs[-1]
        self.last_step = self.anchor.step
        self.relative = inv(self.anchor.eef) @ self.anchor.obj
        self.anchor_gripper_state = self.cfg.grip(self.anchor.gripper)
        self.features = {}
        self.mode, self.reason = self._gate(hs)
        self.initial_mode, self.initial_reason = self.mode, self.reason

    def _gate(self, hs):
        c = self.cfg
        if len(hs) < c.history:
            return "uncertain", "insufficient_visible_history"
        if any(b.step-a.step > c.max_gap_steps for a, b in zip(hs, hs[1:])):
            return "uncertain", "gapped_visible_history"
        rel = [inv(h.eef) @ h.obj for h in hs]
        # All-pairs diameter avoids depending on a lucky anchor and detects drift.
        rp = max(np.linalg.norm(a[:3, 3]-b[:3, 3]) for a in rel for b in rel)
        rr = max(angle(a[:3, :3].T @ b[:3, :3]) for a in rel for b in rel)
        def motion(poses):
            return (max(np.linalg.norm(a[:3, 3]-b[:3, 3]) for a in poses for b in poses),
                    max(angle(a[:3, :3].T @ b[:3, :3]) for a in poses for b in poses))
        ep, er = motion([h.eef for h in hs])
        op, ori = motion([h.obj for h in hs])
        dist = max(np.linalg.norm(h.obj[:3, 3]-h.eef[:3, 3]) for h in hs)
        grip = [c.grip(h.gripper) for h in hs]
        self.features = dict(relative_translation_m=float(rp), relative_rotation_rad=float(rr),
                             eef_motion_m=float(ep), eef_motion_rad=float(er),
                             obj_motion_m=float(op), obj_motion_rad=float(ori),
                             max_distance_m=float(dist), gripper_states=grip)
        excited = ep >= c.min_motion_m or er >= c.min_motion_rad
        comoving = op >= c.min_motion_m or ori >= c.min_motion_rad
        stable = rp <= c.rel_translation_m and rr <= c.rel_rotation_rad
        static = op <= c.static_translation_m and ori <= c.static_rotation_rad
        if (excited and comoving and stable and dist <= c.proximity_m
                and all(g == "closed" for g in grip)):
            return "physics", "visible_rigid_comotion_and_closed_gripper"
        # Memory is also only a working hypothesis (external motion can invalidate it).
        if static and excited and not stable:
            return "memory", "visible_static_object_despite_robot_motion"
        return "uncertain", "ambiguous_contact_evidence"

    def predict(self, robot):
        if not isinstance(robot, Robot):
            raise TypeError("Only Robot observations accepted during occlusion")
        if not isinstance(robot.step, (int, np.integer)) or robot.step <= self.last_step:
            raise ValueError("Prediction steps must strictly increase after anchor")
        eef = transform(robot.eef)
        state = self.cfg.grip(robot.gripper)
        # Telemetry gaps invalidate BOTH a rigid-grasp assertion and a static-object
        # assertion: something relevant may have happened while no robot state was
        # available. Conservatively latch uncertainty until a new visible prefix.
        if robot.step - self.last_step > self.cfg.max_gap_steps:
            self.mode, self.reason = "uncertain", "gapped_robot_telemetry_while_hidden"
        elif self.mode == "physics" and state != "closed":
            self.mode, self.reason = "uncertain", "lost_closed_gripper_evidence_while_hidden"
        elif self.mode == "memory" and self.anchor_gripper_state == "open" and state != "open":
            # A closing/ambiguous gripper while the object is hidden can change
            # contact state. Do NOT silently continue asserting static memory.
            self.mode, self.reason = "uncertain", "contact_may_change_while_hidden"
        self.last_step = int(robot.step)
        return Prediction(int(robot.step), self.anchor.obj.copy(), eef @ self.relative,
                          self.mode, self.reason)

    def metadata(self):
        return dict(config=asdict(self.cfg), features=self.features,
                    initial_mode=self.initial_mode, initial_reason=self.initial_reason)
