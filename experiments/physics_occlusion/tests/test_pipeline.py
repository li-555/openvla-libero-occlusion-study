import tempfile
import unittest
from pathlib import Path

import numpy as np

from pipeline import CameraModel, CompensationConfig, CompensationPipeline, OcclusionSchedule, PosePrediction


def camera():
    return CameraModel(
        intrinsic=np.array([[80.0, 0.0, 32.0], [0.0, 80.0, 32.0], [0.0, 0.0, 1.0]]),
        camera_from_world=np.eye(4),
    )


def image():
    return np.full((64, 64, 3), 120, dtype=np.uint8)


class PipelineTests(unittest.TestCase):
    def setUp(self):
        self.pipeline = CompensationPipeline(
            camera(),
            OcclusionSchedule(5, 7, x0=0.25, y0=0.25, x1=0.75, y1=0.75),
            CompensationConfig(cue_radius_px=3),
        )

    def test_schedule_is_fixed_and_not_target_tracked(self):
        raw = image()
        outside = self.pipeline.occlude(raw, 4)
        inside = self.pipeline.occlude(raw, 5)
        self.assertTrue(np.array_equal(outside, raw))
        self.assertEqual(tuple(inside[20, 20]), (0, 0, 0))
        self.assertEqual(tuple(inside[5, 5]), (120, 120, 120))

    def test_same_rendering_form_for_methods_and_uncertain_is_noop(self):
        pose = np.eye(4)
        pose[2, 3] = 1.0
        occluded = self.pipeline.occlude(image(), 5)
        outputs = []
        for method, mode in (("M0", "memory"), ("M1", "physics"), ("M2", "physics")):
            output, record = self.pipeline.compensate(
                occluded, PosePrediction(method, mode, "test", pose)
            )
            outputs.append(output)
            self.assertTrue(record["applied"])
        self.assertTrue(np.array_equal(outputs[0], outputs[1]))
        self.assertTrue(np.array_equal(outputs[1], outputs[2]))
        unchanged, record = self.pipeline.compensate(
            occluded, PosePrediction("M2", "uncertain", "ambiguous", None)
        )
        self.assertTrue(np.array_equal(unchanged, occluded))
        self.assertFalse(record["applied"])

    def test_truth_is_not_an_input_and_pose_changes_compensation(self):
        raw = image()
        pose_a = np.eye(4)
        pose_a[2, 3] = 1.0
        pose_b = pose_a.copy()
        pose_b[0, 3] = 0.2
        _, _, out_a, _ = self.pipeline.process(
            raw, 5, PosePrediction("M1", "physics", "test", pose_a)
        )
        _, _, out_b, _ = self.pipeline.process(
            raw, 5, PosePrediction("M1", "physics", "test", pose_b)
        )
        self.assertFalse(np.array_equal(out_a, out_b))

    def test_artifacts_include_all_image_stages(self):
        with tempfile.TemporaryDirectory() as tmp:
            raw, occluded, compensated, record = self.pipeline.process(
                image(), 5, PosePrediction("M0", "memory", "test", np.diag([1, 1, 1, 1]))
            )
            prediction = PosePrediction("M0", "memory", "test", np.diag([1, 1, 1, 1]))
            self.pipeline.save_frame(tmp, 5, raw, occluded, compensated, prediction, record)
            names = {path.name for path in Path(tmp).iterdir()}
            self.assertIn("step_000005_raw.png", names)
            self.assertIn("step_000005_occluded.png", names)
            self.assertIn("step_000005_compensated.png", names)
            self.assertIn("step_000005.json", names)


if __name__ == "__main__":
    unittest.main()
