"""Fake-policy CPU integration tests, NOT a substitute for GPU smoke tests."""
import tempfile
import unittest
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import patch

import numpy as np
from PIL import Image

from experiments.contact_cpu.contact_gate import Config
from experiments.occlusion_eval.run_gpu import ProcessorAudit, run_one_episode
from experiments.physics_occlusion.pipeline import CameraModel
from experiments.occlusion_eval.local_occlusion_support import rgb_hash


class FakeEnv:
    def __init__(self):
        self.step_no=0
    def reset(self):
        self.step_no=0
    def _obs(self):
        return {'robot0_eef_pos':np.array([0.1,0.,0.]),
                'robot0_eef_quat':np.array([0.,0.,0.,1.]),
                'robot0_gripper_qpos':np.array([0.,0.]),
                'cup_pos':np.array([0.1,0.,0.]),
                'cup_quat':np.array([0.,0.,0.,1.])}
    def set_init_state(self, state):
        return self._obs()
    def step(self, action):
        self.step_no+=1
        return self._obs(), 0., self.step_no>=9, {}
    def check_success(self):
        return self.step_no>=9


class FakeProcessor:
    def __call__(self, prompt, image):
        return {'image':np.asarray(image).copy()}


class MockRunner(unittest.TestCase):
    def test_closed_loop_image_changes_and_true_state_never_needed_when_hidden(self):
        cam=np.eye(4);cam[2,3]=1
        camera=CameraModel(np.array([[100,0,32],[0,100,32],[0,0,1]]),cam)
        args=SimpleNamespace(pose_source='oracle_visible',rgb_dark_threshold=75,
                             warmup=0,max_steps=10,object_quat_order='xyzw',mask=(.2,.2,.8,.8),suite='libero_spatial',
                             save_every=0,max_reserved_gib=15, no_eval_state_trace=True,
                             local_plan={'anchor_step':0, 'image_shape':[64,64,3],
                                         'anchor_rgb_sha256':rgb_hash(np.full((64,64,3),130,dtype=np.uint8)),
                                         'fill':[128,128,128]})
        spec=dict(mask=(.40,.40,.65,.65),start_step=5,end_step=12,
                  target_object='cup',rgb_roi=None,object_center_height=None)
        image_hashes={}
        def fake_action(cfg,model,obs,task,processor):
            # ProcessorAudit wraps the real call and confirms final input.
            processor('prompt',Image.fromarray(obs['full_image']))
            image_hashes.setdefault(cfg.method,[]).append(np.asarray(obs['full_image']).copy())
            return np.zeros(7)
        from unittest.mock import patch
        with tempfile.TemporaryDirectory() as tmp:
            with patch('torch.cuda.reset_peak_memory_stats'),\
                 patch('torch.cuda.max_memory_reserved',return_value=0),\
                 patch('torch.cuda.max_memory_allocated',return_value=0):
                results={}
                for m in ('clean','occluded','m0'):
                    cfg=SimpleNamespace(method=m)
                    env=FakeEnv()
                    results[m]=run_one_episode(
                        env=env, initial_state=[], task_description='pick up cup',
                        task_id=0, episode_idx=0, method=m, model=None,
                        processor=ProcessorAudit(FakeProcessor()), cfg=cfg,args=args,
                        camera=camera,gate_cfg=Config(grip_closed=0,grip_open=.08),
                        run_dir=Path(tmp),get_img=lambda obs,size:np.full((64,64,3),130,dtype=np.uint8),
                        get_action=fake_action,normalize=lambda a,binarize=True:a,
                        invert=lambda a:a,dummy=lambda name:[0]*7, spec=spec)
                self.assertTrue(all(r['success'] for r in results.values()))
                np.testing.assert_array_equal(image_hashes['clean'][5],
                                              np.full((64,64,3),130,dtype=np.uint8))
                self.assertFalse(np.array_equal(image_hashes['clean'][5],image_hashes['occluded'][5]))
                self.assertTrue(results['m0']['masked_frames']>0)
                self.assertGreater(results['m0']['compensated_frames'],0)
                self.assertTrue(list(Path(tmp).rglob('*_processor.png')))

if __name__=='__main__':unittest.main()
