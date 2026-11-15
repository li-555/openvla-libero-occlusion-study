"""Dependency-light causal occlusion and visual compensation pipeline.

The predictor-facing API accepts only a predicted pose and a robot-visible
image. Ground-truth object state, segmentation, and simulator handles are
intentionally absent from these interfaces.
"""
from dataclasses import dataclass
import json
from pathlib import Path
from typing import Optional

import numpy as np


def _finite_array(value, shape, name):
    array = np.asarray(value, dtype=float)
    if array.shape != shape or not np.isfinite(array).all():
        raise ValueError(f"{name} must be finite with shape {shape}")
    return array.copy()


@dataclass(frozen=True)
class OcclusionSchedule:
    """A fixed image-space rectangle and fixed inclusive step interval."""

    start_step: int
    end_step: int
    x0: float = 0.32
    y0: float = 0.32
    x1: float = 0.70
    y1: float = 0.70
    fill: tuple = (0, 0, 0)

    def __post_init__(self):
        if self.start_step > self.end_step:
            raise ValueError("start_step must be <= end_step")
        if not (0 <= self.x0 < self.x1 <= 1 and 0 <= self.y0 < self.y1 <= 1):
            raise ValueError("Occlusion rectangle must be normalized and ordered")
        if len(self.fill) != 3 or any(int(v) != v or not 0 <= int(v) <= 255 for v in self.fill):
            raise ValueError("fill must contain three uint8 values")

    def active(self, step: int) -> bool:
        return self.start_step <= step <= self.end_step

    def apply(self, image, step: int):
        image = _image(image)
        if not self.active(step):
            return image
        h, w = image.shape[:2]
        out = image.copy()
        xa, xb = int(np.floor(self.x0 * w)), int(np.ceil(self.x1 * w))
        ya, yb = int(np.floor(self.y0 * h)), int(np.ceil(self.y1 * h))
        out[ya:yb, xa:xb] = np.asarray(self.fill, dtype=np.uint8)
        return out


@dataclass(frozen=True)
class CameraModel:
    """Pinhole camera model with an explicit camera-from-world transform."""

    intrinsic: np.ndarray
    camera_from_world: np.ndarray

    def __post_init__(self):
        object.__setattr__(self, "intrinsic", _finite_array(self.intrinsic, (3, 3), "intrinsic"))
        object.__setattr__(
            self, "camera_from_world", _finite_array(self.camera_from_world, (4, 4), "camera_from_world")
        )
        if not np.allclose(self.camera_from_world[3], [0, 0, 0, 1]):
            raise ValueError("camera_from_world must be homogeneous")
        if abs(np.linalg.det(self.intrinsic[:2, :2])) < 1e-12:
            raise ValueError("intrinsic focal block is singular")

    def project(self, world_point):
        point = np.r_[ _finite_array(world_point, (3,), "world_point"), 1.0]
        camera_point = self.camera_from_world @ point
        if camera_point[2] <= 1e-9:
            return None
        pixels = self.intrinsic @ camera_point[:3]
        return float(pixels[0] / pixels[2]), float(pixels[1] / pixels[2])


@dataclass(frozen=True)
class CompensationConfig:
    object_half_extent_m: tuple = (0.035, 0.035, 0.035)
    cue_color: tuple = (40, 220, 40)
    cue_alpha: int = 170
    cue_radius_px: int = 5

    def __post_init__(self):
        if len(self.object_half_extent_m) != 3 or any(v <= 0 for v in self.object_half_extent_m):
            raise ValueError("object_half_extent_m must contain positive values")
        if len(self.cue_color) != 3 or any(not 0 <= int(v) <= 255 for v in self.cue_color):
            raise ValueError("cue_color must contain uint8 values")
        if not 0 <= self.cue_alpha <= 255 or self.cue_radius_px < 1:
            raise ValueError("invalid cue rendering parameters")


@dataclass(frozen=True)
class PosePrediction:
    method: str
    mode: str
    reason: str
    pose_world: Optional[np.ndarray]

    def __post_init__(self):
        if self.method not in {"M0", "M1", "M2"}:
            raise ValueError("method must be M0, M1, or M2")
        if self.mode not in {"memory", "physics", "uncertain"}:
            raise ValueError("invalid prediction mode")
        if self.pose_world is not None:
            object.__setattr__(self, "pose_world", _finite_array(self.pose_world, (4, 4), "pose_world"))


def _image(image):
    array = np.asarray(image)
    if array.ndim != 3 or array.shape[2] != 3 or array.dtype != np.uint8:
        raise ValueError("image must be uint8 HxWx3")
    return array


def _draw_disk(image, center, radius, color, alpha):
    x, y = center
    h, w = image.shape[:2]
    xa, xb = max(0, int(x - radius)), min(w, int(x + radius + 1))
    ya, yb = max(0, int(y - radius)), min(h, int(y + radius + 1))
    yy, xx = np.ogrid[ya:yb, xa:xb]
    mask = (xx - x) ** 2 + (yy - y) ** 2 <= radius**2
    color = np.asarray(color, dtype=float)
    image_region = image[ya:yb, xa:xb].astype(float)
    image_region[mask] = (1 - alpha / 255) * image_region[mask] + alpha / 255 * color
    image[ya:yb, xa:xb] = np.clip(np.round(image_region), 0, 255).astype(np.uint8)


class CompensationPipeline:
    """Apply one fixed occlusion and one shared pose-cue rendering rule."""

    def __init__(self, camera, schedule, config=CompensationConfig()):
        if not isinstance(camera, CameraModel):
            raise TypeError("camera must be CameraModel")
        if not isinstance(schedule, OcclusionSchedule):
            raise TypeError("schedule must be OcclusionSchedule")
        self.camera = camera
        self.schedule = schedule
        self.config = config

    def occlude(self, raw_image, step):
        return self.schedule.apply(raw_image, step)

    def compensate(self, occluded_image, prediction):
        image = _image(occluded_image)
        if not isinstance(prediction, PosePrediction):
            raise TypeError("prediction must be PosePrediction")
        projected = None
        applied = False
        output = image.copy()
        if prediction.mode != "uncertain" and prediction.pose_world is not None:
            projected = self.camera.project(prediction.pose_world[:3, 3])
            if projected is not None:
                h, w = output.shape[:2]
                if 0 <= projected[0] < w and 0 <= projected[1] < h:
                    _draw_disk(
                        output,
                        projected,
                        self.config.cue_radius_px,
                        self.config.cue_color,
                        self.config.cue_alpha,
                    )
                    applied = True
        return output, {
            "method": prediction.method,
            "mode": prediction.mode,
            "reason": prediction.reason,
            "applied": applied,
            "projected_center_px": list(projected) if projected is not None else None,
        }

    def process(self, raw_image, step, prediction):
        raw = _image(raw_image)
        occluded = self.occlude(raw, step)
        compensated, record = self.compensate(occluded, prediction)
        return raw, occluded, compensated, record

    def save_frame(self, directory, step, raw, occluded, compensated, prediction, record):
        """Persist all image stages and metadata; no simulator state is accepted."""
        directory = Path(directory)
        directory.mkdir(parents=True, exist_ok=True)
        for name, image in (("raw", raw), ("occluded", occluded), ("compensated", compensated)):
            _save_png(directory / f"step_{step:06d}_{name}.png", image)
        payload = dict(step=int(step), prediction=dict(
            method=prediction.method,
            mode=prediction.mode,
            reason=prediction.reason,
            pose_world=prediction.pose_world.tolist() if prediction.pose_world is not None else None,
        ), compensation=record)
        (directory / f"step_{step:06d}.json").write_text(
            json.dumps(payload, indent=2, allow_nan=False), encoding="utf-8"
        )


def _save_png(path, image):
    try:
        from PIL import Image
    except ImportError as exc:
        raise RuntimeError("Pillow is required to save image artifacts") from exc
    Image.fromarray(_image(image)).save(path)
