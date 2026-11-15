import json
import tempfile
import unittest
from pathlib import Path
from experiments.occlusion_eval.analyze import analyze, paired_bootstrap_diff
from experiments.occlusion_eval.run_gpu import parse_args, parse_rect
from experiments.occlusion_eval.local_occlusion_support import rgb_hash
import numpy as np


def _plan(path):
    rgb = np.full((64, 64, 3), 130, dtype=np.uint8)
    path.write_text(json.dumps({
        "schema": "local_occlusion_plan_v1", "task_id": 0, "episode_id": 0,
        "anchor_step": 2, "start_step": 3, "end_step": 10,
        "image_shape": list(rgb.shape), "anchor_rgb_sha256": rgb_hash(rgb),
        "roi": [.40, .40, .65, .65], "fill": [128, 128, 128],
    }), encoding="utf-8")



class AnalysisTests(unittest.TestCase):
    def test_paired_analysis_not_frames(self):
        with tempfile.TemporaryDirectory() as temp:
            path=Path(temp)
            rec=[]
            for ep,good in [(0,True),(1,False),(2,True)]:
                rec.append({'task_id':0,'episode_id':ep,'method':'m2','success':good})
                rec.append({'task_id':0,'episode_id':ep,'method':'occluded','success':False})
            (path/'episodes.json').write_text(json.dumps(rec))
            s=analyze(path)
            self.assertEqual(s['paired']['m2_minus_occluded']['n'],3)
            self.assertAlmostEqual(s['paired']['m2_minus_occluded']['difference'],2/3)
            self.assertTrue((path/'summary.csv').exists())

    def test_no_confidence_interval_for_single_pair(self):
        self.assertEqual(paired_bootstrap_diff([1],[0])['ci95'],[None,None])

    def test_requires_explicit_oracle_upper_bound(self):
        with tempfile.TemporaryDirectory() as d:
            plan = Path(d) / 'local_plan.json'
            _plan(plan)
            common = ['--episode-ids', '0', '--local-mask-plan', str(plan)]
            # With a valid plan, this must still reject unauthorized oracle visibility.
            with self.assertRaises(SystemExit):
                parse_args(['--pose-source','oracle_visible','--methods','clean,occluded'] + common)
            opts=parse_args(['--pose-source','oracle_visible','--allow-oracle-upper-bound',
                             '--methods','clean,occluded'] + common)
            self.assertEqual(opts.pose_source,'oracle_visible')

    def test_required_rgb_config(self):
        with tempfile.TemporaryDirectory() as d:
            plan = Path(d) / 'local_plan.json'
            _plan(plan)
            with self.assertRaises(SystemExit):
                parse_args(['--pose-source','rgb_plane','--methods','occluded,m0',
                            '--episode-ids','0','--local-mask-plan',str(plan)])

    def test_rect_validation(self):
        with self.assertRaises(ValueError): parse_rect('0,0,0.1')
        with self.assertRaises(ValueError): parse_rect('0.7,0,0.2,1')

if __name__=='__main__':unittest.main()
