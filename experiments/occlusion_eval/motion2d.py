"""Experimental pixel-space M1/M2 object-memory propagation.

Inputs are *only* visible-RGB object pixel estimates and measured robot
end-effector/gripper state. No simulator object truth, segmentation oracle, or
hidden observation is accepted. M1/M2 are hypotheses, not contact detectors.

Conventions: image (u,v) pixels after get_libero_image(..., 224), and
camera.project(eef_pos) produces pixels in the same coordinate frame.
"""
from collections import deque
from dataclasses import dataclass
from typing import Optional

import numpy as np

from experiments.contact_cpu.contact_gate import Config
from experiments.physics_occlusion.pipeline import CameraModel


@dataclass(frozen=True)
class Pixel2DConfig:
    history: int = 5
    min_robot_motion_px: float = 2.0
    min_object_motion_px: float = 1.5
    static_tolerance_px: float = 1.5
    relative_tolerance_px: float = 2.0
    max_gap_steps: int = 1
    min_dark_pixels: int = 20
    max_dark_fraction: float = 0.75
    dark_threshold: int = 75

    def __post_init__(self):
        if self.history < 3 or self.max_gap_steps < 1:
            raise ValueError('history >= 3, max_gap_steps >= 1 required')
        for key in ('min_robot_motion_px','min_object_motion_px',
                    'static_tolerance_px','relative_tolerance_px'):
            if not np.isfinite(getattr(self, key)) or getattr(self, key) <= 0:
                raise ValueError(f'invalid {key}')
        if not (0 <= self.dark_threshold <= 255 and self.min_dark_pixels > 0 and
                0 < self.max_dark_fraction < 1):
            raise ValueError('invalid detector settings')


class DarkPixel2D:
    """A transparent, intentionally weak RGB proxy; NOT robust grasp detection.

    The fixed ROI is specified in visible image space. It must be checked
    against actual target image annotations, particularly after grasp.
    """
    def __init__(self, roi, config: Pixel2DConfig = Pixel2DConfig()):
        self.roi = tuple(float(v) for v in roi)
        if len(self.roi) != 4 or not (0 <= self.roi[0] < self.roi[2] <= 1 and
                                     0 <= self.roi[1] < self.roi[3] <= 1):
            raise ValueError('ROI must be normalized x0,y0,x1,y1')
        self.config = config

    def estimate(self, rgb):
        a = np.asarray(rgb)
        if a.dtype != np.uint8 or a.ndim != 3 or a.shape[2] != 3:
            raise ValueError('RGB must be uint8 HxWx3')
        h, w = a.shape[:2]
        x0, y0, x1, y1 = self.roi
        xa, xb = int(np.floor(x0*w)), int(np.ceil(x1*w))
        ya, yb = int(np.floor(y0*h)), int(np.ceil(y1*h))
        sub = a[ya:yb,xa:xb]
        m = sub.max(axis=2) <= self.config.dark_threshold
        n = int(m.sum())
        if n < self.config.min_dark_pixels or n > m.size*self.config.max_dark_fraction:
            return None
        yy, xx = np.nonzero(m)
        return np.array([xa+xx.mean(),ya+yy.mean()],dtype=float)


@dataclass(frozen=True)
class Visible2D:
    step: int
    pixel: np.ndarray
    eef_pixel: np.ndarray
    gripper_state: str


class PixelMotion2D:
    """Frozen visible anchor, then pixel-memory or projected EEF displacement.

    M2 only chooses physics if the *visible prefix* exhibits co-motion and
    independently calibrated gripper is closed. Ambiguity yields no cue.
    The gate is frozen at the start of hidden window except for conservative
    loss-of-contact evidence from measured robot joints.
    """
    def __init__(self, mode, camera: CameraModel, gripper_config: Config,
                 config: Pixel2DConfig = Pixel2DConfig()):
        if mode not in ('m0_2d','m1_2d','m2_2d'):
            raise ValueError('mode must be m1_2d or m2_2d')
        self.mode, self.camera, self.gripper_config, self.cfg = mode, camera, gripper_config, config
        self.visible = deque(maxlen=config.history)
        self.anchor = None
        self.gate_mode = None
        self.gate_reason = None
        self.last_hidden_step = None
        self.anchor_grip = None

    def observe_visible(self, step, pixel, eef_pos, gripper_qpos):
        if self.anchor is not None:
            raise RuntimeError('Cannot reanchor while hidden')
        if pixel is None:
            return False
        p = np.asarray(pixel, dtype=float)
        if p.shape != (2,) or not np.isfinite(p).all():
            raise ValueError('pixel must be finite 2-vector')
        e = self.camera.project(eef_pos)
        if e is None or not np.isfinite(e).all():
            return False
        if self.visible and step <= self.visible[-1].step:
            raise ValueError('visible steps must increase')
        self.visible.append(Visible2D(int(step),p.copy(),np.array(e),
                                      self.gripper_config.grip(gripper_qpos)))
        return True

    @staticmethod
    def _diameter(values):
        x = np.array(values)
        return float(np.linalg.norm(x[:,None,:]-x[None,:,:],axis=2).max())

    def _decide(self):
        v = list(self.visible)
        if len(v) < self.cfg.history:
            return 'uncertain','insufficient_visible_pixel_history'
        if any(b.step-a.step > self.cfg.max_gap_steps for a,b in zip(v,v[1:])):
            return 'uncertain','noncontiguous_visible_pixel_history'
        obj = [x.pixel for x in v]
        eef = [x.eef_pixel for x in v]
        rel = [x.pixel-x.eef_pixel for x in v]
        motion_o = self._diameter(obj)
        motion_e = self._diameter(eef)
        motion_rel = self._diameter(rel)
        grip = [x.gripper_state for x in v]
        if (motion_e >= self.cfg.min_robot_motion_px and
            motion_o >= self.cfg.min_object_motion_px and
            motion_rel <= self.cfg.relative_tolerance_px and
            all(x=='closed' for x in grip)):
            return 'physics','visible_2d_comotion_with_calibrated_closed_gripper'
        if (motion_e >= self.cfg.min_robot_motion_px and
            motion_o <= self.cfg.static_tolerance_px and
            all(x=='open' for x in grip)):
            return 'memory','visible_static_object_with_open_gripper'
        return 'uncertain','insufficient_2d_motion_or_grip_evidence'

    def predict_hidden(self, step, eef_pos, gripper_qpos):
        if self.anchor is None:
            if not self.visible:
                return None, 'uncertain', 'no_visible_anchor'
            self.anchor = self.visible[-1]
            self.anchor_grip = self.anchor.gripper_state
            self.gate_mode,self.gate_reason = self._decide()
        if int(step) <= self.anchor.step or (self.last_hidden_step is not None and
                                            step <= self.last_hidden_step):
            raise ValueError('hidden steps must increase after visible anchor')
        if (self.last_hidden_step is not None and
            step-self.last_hidden_step > self.cfg.max_gap_steps) or (self.last_hidden_step is None and
            step-self.anchor.step > self.cfg.max_gap_steps):
            self.gate_mode,self.gate_reason='uncertain','gapped_hidden_robot_telemetry'
        self.last_hidden_step = int(step)
        eef_px = self.camera.project(eef_pos)
        if eef_px is None:
            return None,'uncertain','eef_not_projectable'
        phys = self.anchor.pixel + np.array(eef_px)-self.anchor.eef_pixel
        if self.mode == 'm0_2d':
            if self.gate_reason == 'gapped_hidden_robot_telemetry':
                return None, 'uncertain', self.gate_reason
            return self.anchor.pixel.copy(), 'memory', 'shared_pixel_anchor_memory'
        if self.mode == 'm1_2d':
            # Even an intentionally unconditional physics baseline must not use
            # an anchor made stale by missing visible detections/telemetry.
            if self.gate_reason == 'gapped_hidden_robot_telemetry':
                return None,'uncertain','gapped_hidden_robot_telemetry'
            return phys,'physics','assumed_2d_rigid_translation'
        if self.gate_mode == 'physics' and self.gripper_config.grip(gripper_qpos) != 'closed':
            self.gate_mode,self.gate_reason='uncertain','closed_gripper_evidence_lost'
        if (self.gate_mode == 'memory' and self.anchor_grip == 'open' and
                self.gripper_config.grip(gripper_qpos) != 'open'):
            self.gate_mode,self.gate_reason='uncertain','grasp_state_changed_during_occlusion'
        if self.gate_mode=='physics':
            return phys,'physics',self.gate_reason
        if self.gate_mode=='memory':
            return self.anchor.pixel.copy(),'memory',self.gate_reason
        return None,'uncertain',self.gate_reason
