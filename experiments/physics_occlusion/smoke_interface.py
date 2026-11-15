"""Generate CPU-only sample image artifacts for the compensation interface."""
import argparse
from pathlib import Path

import numpy as np

from pipeline import CameraModel, CompensationPipeline, OcclusionSchedule, PosePrediction


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--out", type=Path, required=True)
    args = parser.parse_args()
    camera = CameraModel(
        intrinsic=np.array([[280.0, 0.0, 112.0], [0.0, 280.0, 112.0], [0.0, 0.0, 1.0]]),
        camera_from_world=np.eye(4),
    )
    pipeline = CompensationPipeline(camera, OcclusionSchedule(5, 5))
    raw = np.full((224, 224, 3), 128, dtype=np.uint8)
    pose = np.eye(4)
    pose[2, 3] = 1.0
    for prediction in (
        PosePrediction("M0", "memory", "hold_last_pose", pose),
        PosePrediction("M1", "physics", "rigid_motion", pose),
        PosePrediction("M2", "uncertain", "calibration_missing", None),
    ):
        raw_image, occluded, compensated, record = pipeline.process(raw, 5, prediction)
        pipeline.save_frame(args.out / prediction.method, 5, raw_image, occluded, compensated, prediction, record)


if __name__ == "__main__":
    main()
