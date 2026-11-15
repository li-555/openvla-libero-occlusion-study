import json
import tempfile
import unittest
from pathlib import Path

import numpy as np

from experiments.occlusion_eval.eval_state_trace import make_eval_state_row, write_eval_state_trace


class SimpleCamera:
    def project(self, xyz):
        return (xyz[0] * 100, xyz[1] * 100)


class TestEvalStateTrace(unittest.TestCase):
    def setUp(self):
        self.obs = {
            'robot0_eef_pos': np.array([0.1, 0.2, 0.3]),
            'robot0_eef_quat': np.array([0, 0, 0, 1]),
            'robot0_gripper_qpos': np.array([0.020, -0.021]),
            'akita_black_bowl_1_pos': np.array([0.2, 0.3, 0.4]),
            'akita_black_bowl_1_quat': np.array([0, 0, 0, 1]),
        }
        self.args = dict(step=58, target_name='akita_black_bowl_1',
                         camera=SimpleCamera(), occlusion_active=True,
                         executed_action=np.array([0, 0, 0, 0, 0, 0, 1]),
                         action_raw=np.array([0, 0, 0, 0, 0, 0, 0]),
                         mask=(0.1,0.2,0.3,0.4))

    def test_snapshot_has_real_robot_and_gt_keys(self):
        out = make_eval_state_row(obs=self.obs, **self.args)
        self.assertEqual(out['robot0_gripper_qpos'], [0.02, -0.021])
        self.assertEqual(out['target_center_px_gt_EVAL_ONLY'], [20.0, 30.0])
        self.assertAlmostEqual(out['eef_to_target_center_m_EVAL_ONLY'], np.sqrt(0.03))
        self.assertIsNone(out['contact_or_attachment_confirmed'])

    def test_sim_truth_is_not_sent_to_action_dict_or_modified(self):
        original = {k:v.copy() for k,v in self.obs.items()}
        out = make_eval_state_row(obs=self.obs, **self.args)
        self.assertNotIn('full_image', out)
        for k,v in original.items():
            np.testing.assert_array_equal(self.obs[k], v)
        # Altering hidden truth changes ONLY diagnostic output.
        self.obs['akita_black_bowl_1_pos'] = np.array([0.4,0.5,0.6])
        other = make_eval_state_row(obs=self.obs, **self.args)
        self.assertNotEqual(out['target_pos_gt_EVAL_ONLY'], other['target_pos_gt_EVAL_ONLY'])
        self.assertEqual(out['robot0_eef_pos'],other['robot0_eef_pos'])
        self.assertEqual(out['action_executed'],other['action_executed'])

    def test_missing_object_or_gripper_is_not_fabricated(self):
        row = make_eval_state_row(obs={}, **self.args)
        self.assertIsNone(row['robot0_gripper_qpos'])
        self.assertIsNone(row['target_pos_gt_EVAL_ONLY'])
        self.assertIsNone(row['target_center_px_gt_EVAL_ONLY'])
        self.assertIsNone(row['eef_to_target_center_m_EVAL_ONLY'])

    def test_writer_separate_and_parseable(self):
        rows = [make_eval_state_row(obs=self.obs, **self.args)]
        with tempfile.TemporaryDirectory() as root:
            rel = write_eval_state_trace(root,0,0,'occluded',rows)
            self.assertTrue(rel.startswith('eval_only/'))
            content = (Path(root)/rel).read_text(encoding='utf-8').splitlines()
            self.assertEqual(len(content),1)
            self.assertEqual(json.loads(content[0])['step'],58)


if __name__ == '__main__':
    unittest.main()
