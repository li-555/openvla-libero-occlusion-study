"""Causal observation pipeline for M0/M1/M2 visual compensation.

Prediction accepts only an RGB image, measured end-effector state and gripper
joints. The privileged object-state adapter is a separate, explicitly labeled
upper-bound development mode: it is invoked ONLY while the object is visible.
"""
from dataclasses import dataclass
from collections import deque
from typing import Optional
import numpy as np

from experiments.contact_cpu.contact_gate import Config, ContactGate, Robot, Visible, pose
from experiments.physics_occlusion.pipeline import (
    CameraModel, CompensationPipeline, OcclusionSchedule, PosePrediction,
)


@dataclass(frozen=True)
class RGBPlaneConfig:
    """Task-specific RGB centroid detector for *visible* frames only.

    The ROI is a fixed box, selected on development images. Not a general
    6-DoF pose estimator; assumes a known object-center height and orientation.
    """
    roi: tuple
    center_height_m: float
    dark_threshold: int = 75
    min_dark_pixels: int = 30
    max_dark_pixels_fraction: float = 0.6

    def __post_init__(self):
        x0, y0, x1, y1 = self.roi
        if not (0 <= x0 < x1 <= 1 and 0 <= y0 < y1 <= 1):
            raise ValueError('RGB ROI must be normalized and ordered')
        if not (np.isfinite(self.center_height_m) and -5 < self.center_height_m < 5):
            raise ValueError('Specify a finite, independently calibrated object-center height')
        if not (0 <= self.dark_threshold <= 255 and 0 < self.min_dark_pixels and
                0 < self.max_dark_pixels_fraction < 1):
            raise ValueError('Invalid RGB detector thresholds')


class RGBPlaneEstimator:
    """Deliberately simple visible-RGB-only proxy; NOT FoundationPose."""
    def __init__(self, camera: CameraModel, cfg: RGBPlaneConfig):
        self.camera, self.cfg = camera, cfg

    def estimate(self, image: np.ndarray) -> Optional[np.ndarray]:
        img = np.asarray(image)
        if img.dtype != np.uint8 or img.ndim != 3 or img.shape[-1] != 3:
            raise ValueError('Expected uint8 RGB')
        h, w = img.shape[:2]
        x0, y0, x1, y1 = self.cfg.roi
        xa, xb = int(x0*w), int(np.ceil(x1*w))
        ya, yb = int(y0*h), int(np.ceil(y1*h))
        region = img[ya:yb, xa:xb]
        if region.size == 0:
            return None
        # Basic darkness cue; may confuse dark gripper/background. Thus report
        # its failure rate and inspect calibration frames, never assert 6D pose.
        mask = np.max(region, axis=2) <= self.cfg.dark_threshold
        count = int(mask.sum())
        if count < self.cfg.min_dark_pixels or count > mask.size*self.cfg.max_dark_pixels_fraction:
            return None
        ys, xs = np.nonzero(mask)
        xy = (float(xa+xs.mean()), float(ya+ys.mean()))
        point = backproject_to_horizontal_plane(self.camera, xy, self.cfg.center_height_m)
        if point is None:
            return None
        T = np.eye(4)  # orientation UNKNOWN; fixed proxy, not a 6DoF estimate
        T[:3, 3] = point
        return T


def backproject_to_horizontal_plane(camera: CameraModel, xy, world_height: float):
    """Ray-plane intersection from public camera calibration, no object truth."""
    K = camera.intrinsic
    C = camera.camera_from_world
    ray_c = np.linalg.solve(K, np.array([xy[0], xy[1], 1.0]))
    world_from_cam = np.linalg.inv(C)
    origin = world_from_cam[:3, 3]
    ray_w = world_from_cam[:3, :3] @ ray_c
    if abs(ray_w[2]) < 1e-9:
        return None
    scale = (world_height-origin[2])/ray_w[2]
    if scale <= 0 or not np.isfinite(scale):
        return None
    return origin + scale*ray_w


def calibrated_camera_from_env(sim, size: int, native_size=256, image_convention=None):
    """Read *camera calibration only*; match 180-degree rotation + resize.

    robosuite's camera helper returns world-from-camera with OpenCV axes.
    Does not read object/site/body position or simulator contact data.
    Projection MUST be visually checked with saved calibration QA images.
    """
    from robosuite.utils.camera_utils import (
        get_camera_extrinsic_matrix, get_camera_intrinsic_matrix
    )
    K = np.asarray(get_camera_intrinsic_matrix(sim, 'agentview', native_size, native_size), dtype=float)
    E = np.asarray(get_camera_extrinsic_matrix(sim, 'agentview'), dtype=float)
    if image_convention is None:
        from robosuite import macros
        image_convention = macros.IMAGE_CONVENTION
    if image_convention not in ('opengl', 'opencv'):
        raise ValueError(f'Unknown observation image convention: {image_convention}')
    # E uses OpenCV axes. Native OpenGL observations are vertically flipped
    # relative to these projected pixels. LIBERO subsequently rotates both axes.
    to_observation = np.eye(3)
    if image_convention == 'opengl':
        to_observation[1, 1] = -1
        to_observation[1, 2] = native_size - 1
    rotate = np.array([[-1, 0, native_size-1],
                       [0, -1, native_size-1], [0, 0, 1]], dtype=float)
    scale = float(size)/native_size
    resize = np.array([[scale, 0, (scale-1)/2],
                       [0, scale, (scale-1)/2], [0, 0, 1]])
    return CameraModel(resize @ rotate @ to_observation @ K, np.linalg.inv(E))



def pose_from_visible_oracle(visible_obs, target_name, object_quat_order):
    """PRIVILEGED upper-bound only; caller must enforce pre-occlusion access.

    Do not pass this function or obs to an online deployable method once hidden.
    """
    p = np.asarray(visible_obs[f'{target_name}_pos'], dtype=float)
    q = np.asarray(visible_obs[f'{target_name}_quat'], dtype=float)
    return pose(p, q, object_quat_order)


@dataclass
class StepOutput:
    raw: np.ndarray
    occluded: np.ndarray
    action_image: np.ndarray
    metadata: dict


class TrialController:
    """Isolated per-trial state. step=0 is first policy inference after warmup."""
    MODES = ('clean', 'occluded', 'm0', 'm1', 'm2', 'cue_control', 'm1_2d', 'm2_2d', 'm0_2d', 'occluded_hold', 'm2_2d_hold')

    def __init__(self, method, camera, schedule, gate_cfg=Config(),
                 estimator=None, *, object_quat_order='xyzw',
                 allow_oracle=False, target_name='akita_black_bowl_1',
                 pixel_estimator=None, motion2d_cfg=None):
        if method not in self.MODES:
            raise ValueError(f'Invalid method {method}')
        self.method = method
        self.camera = camera
        self.pipeline = CompensationPipeline(camera, schedule)
        self.gate_cfg = gate_cfg
        self.estimator = estimator
        self.allow_oracle = allow_oracle
        self.target_name = target_name
        self.quat_order = object_quat_order
        self.visible = deque(maxlen=max(gate_cfg.history, 5))
        self.gate = None
        self.last_visible_step = None
        self.n_visible_success = 0
        self.n_visible_fail = 0
        self.history_methods = []
        # Optional 2D experiment; no object GT or simulator object pose is exposed.
        self.pixel_estimator = pixel_estimator
        self.motion2d = None
        if method in ('m0_2d', 'm1_2d', 'm2_2d'):
            if pixel_estimator is None:
                raise ValueError('Motion2D requires visible-RGB pixel estimator')
            from experiments.occlusion_eval.motion2d import PixelMotion2D, Pixel2DConfig
            self.motion2d = PixelMotion2D(method, camera, gate_cfg,
                                         motion2d_cfg or Pixel2DConfig())

    def _visible_pose(self, image, privileged_visible_observation):
        if self.allow_oracle:
            if privileged_visible_observation is None:
                raise ValueError('Development oracle prefix requires visible observation')
            return pose_from_visible_oracle(privileged_visible_observation,
                                            self.target_name, self.quat_order)
        if self.estimator is None:
            return None
        return self.estimator.estimate(image)

    def process(self, *, step, raw, eef_pos, eef_quat, gripper_qpos,
                privileged_visible_observation=None):
        """Privileged prefix used only if schedule INACTIVE, never while hidden.

        In deployable mode privileged_visible_observation should always be None.
        The action_image can go to the existing original get_action path.
        """
        if not isinstance(step, int):
            raise ValueError('step must be int')
        rgb = np.asarray(raw)
        if rgb.ndim != 3 or rgb.shape[-1] != 3 or rgb.dtype != np.uint8:
            raise ValueError('raw RGB must be uint8 HxWx3')
        eef = pose(eef_pos, eef_quat, 'xyzw')
        gripper = np.asarray(gripper_qpos, dtype=float).copy()
        active = self.pipeline.schedule.active(step)
        if active and privileged_visible_observation is not None:
            # Fail closed rather than accept hidden truth even if unused.
            raise ValueError('Hidden privileged obs forbidden')
        if not active:
            if self.motion2d is not None and step < self.pipeline.schedule.start_step:
                visible_px = self.pixel_estimator.estimate(rgb)
                if self.motion2d.observe_visible(step, visible_px, eef_pos, gripper_qpos):
                    self.last_visible_step = step
                    self.n_visible_success += 1
                else:
                    self.n_visible_fail += 1
            self.gate = None
            if step < self.pipeline.schedule.start_step and self.method in ('m0','m1','m2'):
                observed = self._visible_pose(rgb, privileged_visible_observation)
                if observed is not None:
                    self.visible.append(Visible(step, eef, observed, gripper))
                    self.last_visible_step = step
                    self.n_visible_success += 1
                else:
                    self.n_visible_fail += 1
            return StepOutput(rgb.copy(), rgb.copy(), rgb.copy(),
                              {'active':False, 'method':self.method,
                               'pose_observed':self.last_visible_step == step})
        occluded = self.pipeline.occlude(rgb, step)
        if self.method == 'clean':
            return StepOutput(rgb.copy(), occluded, rgb.copy(), {'active':True,'method':'clean'})
        if self.method == 'occluded':
            return StepOutput(rgb.copy(), occluded, occluded.copy(), {'active':True,'method':'occluded'})
        if self.method == 'cue_control':
            # Cue without position information, same renderer, fixed center of ROI.
            # Independent control for visual token/annotation response.
            xx = (self.pipeline.schedule.x0+self.pipeline.schedule.x1)/2*rgb.shape[1]
            yy = (self.pipeline.schedule.y0+self.pipeline.schedule.y1)/2*rgb.shape[0]
            from experiments.physics_occlusion.pipeline import _draw_disk
            x, y = int(xx), int(yy)
            output = occluded.copy()
            _draw_disk(output, (x,y), self.pipeline.config.cue_radius_px,
                       self.pipeline.config.cue_color, self.pipeline.config.cue_alpha)
            return StepOutput(rgb.copy(), occluded, output,
                              {'active':True,'method':'cue_control', 'cue_position': 'fixed_roi_center', 'compensation_applied': True})
        if self.motion2d is not None:
            pixel, gate_mode, reason = self.motion2d.predict_hidden(step, eef_pos, gripper_qpos)
            result_image = occluded.copy()
            applied = False
            if pixel is not None:
                uu, vv = (float(pixel[0]), float(pixel[1]))
                if np.isfinite([uu,vv]).all() and 0 <= uu < rgb.shape[1] and 0 <= vv < rgb.shape[0]:
                    from experiments.physics_occlusion.pipeline import _draw_disk
                    _draw_disk(result_image, (uu,vv), self.pipeline.config.cue_radius_px,
                               self.pipeline.config.cue_color, self.pipeline.config.cue_alpha)
                    applied = True
            return StepOutput(rgb.copy(), occluded, result_image,
                  {'active':True, 'method':self.method, 'gate':gate_mode,
                   'gate_reason':reason, 'compensation_applied':applied,
                   'projected_center_px': pixel.tolist() if pixel is not None else None,
                   'last_visible_step': self.motion2d.anchor.step if self.motion2d.anchor else None,
                   'study_label':'EXPERIMENTAL_PIXEL_2D_NO_OBJECT_GT'})
        if self.gate is None:
            if self.visible:
                self.gate = ContactGate(list(self.visible), self.gate_cfg)
            else:
                return StepOutput(rgb.copy(), occluded, occluded.copy(),
                                  {'active':True,'method':self.method,'gate':'no_visible_anchor', 'compensation_applied':False})
        p = self.gate.predict(Robot(step, eef, gripper))
        if self.method == 'm0':
            p_pred = PosePrediction('M0', 'memory','fixed_last_visible_pose',p.memory)
        elif self.method == 'm1':
            p_pred = PosePrediction('M1','physics','rigid_grasp_assumed',p.physics)
        else:
            p_pred = PosePrediction('M2',p.mode,p.reason,p.selected)
        out, meta = self.pipeline.compensate(occluded, p_pred)
        return StepOutput(rgb.copy(), occluded, out,
                          {'active':True,'method':self.method,'gate':p_pred.mode, 'contact_gate':p.mode,
                           'gate_reason':p_pred.reason, 'contact_gate_reason':p.reason, 'compensation_applied':meta['applied'],
                           'projected_center_px':meta['projected_center_px'],
                           'last_visible_step':self.last_visible_step})
