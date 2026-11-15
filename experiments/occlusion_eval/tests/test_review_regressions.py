"""Regression tests. Fake CUDA is explicit and validates orchestration only."""
import contextlib
import json
import sys
import tempfile
import types
import unittest
from pathlib import Path
from unittest.mock import patch
import numpy as np
from experiments.contact_cpu.contact_gate import Config
from experiments.occlusion_eval.core import TrialController, calibrated_camera_from_env
from experiments.occlusion_eval.run_gpu import parse_args, run_one_episode, ProcessorAudit
from experiments.occlusion_eval.runtime_checks import resolve_sim, success_check, method_order, pair_seed
from experiments.occlusion_eval.analyze import analyze, paired_bootstrap_diff
from experiments.physics_occlusion.pipeline import CameraModel, OcclusionSchedule
from experiments.occlusion_eval.local_occlusion_support import rgb_hash


def fake_torch():
    # No real tensors, model, CUDA, processor preprocessing, or inference tested.
    return types.SimpleNamespace(inference_mode=contextlib.nullcontext,
        cuda=types.SimpleNamespace(reset_peak_memory_stats=lambda:None,
            max_memory_reserved=lambda:0, max_memory_allocated=lambda:0))


class Env:
    def __init__(self, bad_warmup=False):
        self.resets=0; self.n=0; self.bad_warmup=bad_warmup; self.states=[]
    def reset(self): self.resets+=1; self.n=0
    def obs(self):
        return dict(robot0_eef_pos=np.zeros(3),robot0_eef_quat=np.array([0,0,0,1]),
                    robot0_gripper_qpos=np.zeros(2), agentview_image=np.full((64,64,3),130,np.uint8))
    def set_init_state(self, state): self.states.append(list(state)); return self.obs()
    def step(self, action): self.n+=1; return self.obs(),0,self.bad_warmup,{}
    def check_success(self): return self.bad_warmup


class ReviewRegressions(unittest.TestCase):
    def setUp(self):
        C=np.eye(4); C[2,3]=1
        self.camera=CameraModel(np.array([[100.,0,32],[0,100,32],[0,0,1]]),C)
        self.img=np.full((64,64,3),130,np.uint8)
    def process(self,c,step,truth=None,img=None):
        return c.process(step=step,raw=self.img if img is None else img,eef_pos=[0,0,0],
            eef_quat=[0,0,0,1],gripper_qpos=[0,0],privileged_visible_observation=truth)
    def test_oracle_cue_control_needs_no_pose(self):
        c=TrialController('cue_control',self.camera,OcclusionSchedule(3,5),allow_oracle=True)
        self.process(c,0)
        self.assertTrue(self.process(c,3).metadata['compensation_applied'])
    def test_oracle_not_read_after_window(self):
        c=TrialController('m0',self.camera,OcclusionSchedule(3,5),allow_oracle=True)
        self.process(c,6)  # no truth allowed/needed after window
        self.assertEqual(c.n_visible_success,0)
    def test_hidden_pixels_inside_mask_do_not_change_policy_input(self):
        a=TrialController('m0',self.camera,OcclusionSchedule(3,5),allow_oracle=True,target_name='cup')
        b=TrialController('m0',self.camera,OcclusionSchedule(3,5),allow_oracle=True,target_name='cup')
        truth=dict(cup_pos=np.zeros(3),cup_quat=[0,0,0,1])
        for i in range(3): self.process(a,i,truth);self.process(b,i,truth)
        changed=self.img.copy();changed[21:44,21:44]=255
        np.testing.assert_array_equal(self.process(a,3).action_image,self.process(b,3,img=changed).action_image)
    def test_hidden_truth_is_rejected(self):
        c=TrialController('m2',self.camera,OcclusionSchedule(3,5))
        with self.assertRaises(ValueError):self.process(c,3,{'cup_pos':[99,99,99]})
    def test_effective_m0_mode_is_memory_not_contact_gate_uncertain(self):
        c=TrialController('m0',self.camera,OcclusionSchedule(3,5),allow_oracle=True,target_name='cup')
        self.process(c,2,dict(cup_pos=np.zeros(3),cup_quat=[0,0,0,1]))
        meta=self.process(c,3).metadata
        self.assertEqual(meta['gate'],'memory');self.assertEqual(meta['contact_gate'],'uncertain')
    def test_rgb_roi_string_in_task_spec(self):
        with tempfile.TemporaryDirectory() as d:
            p=Path(d)/'spec.json';p.write_text(json.dumps({'0':{'rgb_roi':'0.1,0.2,0.4,0.5','object_center_height':0.2}}))
            mask_plan = Path(d)/'mask_plan.json'
            mask_plan.write_text(json.dumps({
                'schema':'local_occlusion_plan_v1','task_id':0,'episode_id':0,
                'anchor_step':2,'start_step':3,'end_step':10,
                'image_shape':[64,64,3],
                'anchor_rgb_sha256':rgb_hash(self.img),
                'roi':[.40,.40,.65,.65],'fill':[128,128,128]}))
            a=parse_args(['--pose-source','rgb_plane','--methods','m0','--task-specs',str(p),
                          '--episode-ids','0','--local-mask-plan',str(mask_plan)])
            self.assertEqual(a.specs['0']['rgb_roi'],(.1,.2,.4,.5))
    @patch("experiments.occlusion_eval.diagnostic_probes.record_probes", new=lambda *a, **k: None)
    def run_mock(self,method='clean',env=None,warmup=0,pose_source='rgb_plane'):
        env=env or Env()
        args=types.SimpleNamespace(pose_source=pose_source,rgb_dark_threshold=75,warmup=warmup,
              max_steps=7,object_quat_order='xyzw',suite='libero_spatial',save_every=0,max_reserved_gib=15,
              no_eval_state_trace=True, local_plan={
                  'anchor_step':0, 'image_shape':list(self.img.shape),
                  'anchor_rgb_sha256':rgb_hash(self.img), 'fill':[128,128,128]})
        spec=dict(rgb_roi=None,object_center_height=None,start_step=3,end_step=10,
                  mask=(.40,.40,.65,.65),target_object='cup')
        with tempfile.TemporaryDirectory() as d,patch.dict(sys.modules,{'torch':fake_torch()}):
            r=run_one_episode(env=env,initial_state=[1,2,3],task_description='pick cup',task_id=0,episode_idx=0,
                method=method,model=None,processor=ProcessorAudit(lambda *a:None),cfg=None,args=args,
                camera=self.camera,gate_cfg=Config(),run_dir=Path(d),get_img=lambda o,n:o['agentview_image'],
                get_action=lambda *a,**kw:np.zeros(7),normalize=lambda a,**kw:a,invert=lambda a:a,
                dummy=lambda n:[0]*7,spec=spec)
        return r,env
    def test_rgb_clean_does_not_construct_estimator(self):
        r,e=self.run_mock('clean');self.assertFalse(r['success']);self.assertEqual(e.resets,1)
    def test_rgb_occluded_does_not_construct_estimator(self):self.run_mock('occluded')
    def test_rgb_cue_control_does_not_construct_estimator(self):self.run_mock('cue_control')
    def test_oracle_cue_control_full_episode(self):self.run_mock('cue_control',pose_source='oracle_visible')
    def test_warmup_terminal_invalidates_trial(self):
        with self.assertRaisesRegex(RuntimeError,'warmup'):self.run_mock(env=Env(True),warmup=1)
    def test_reset_and_state_each_method(self):
        env=Env()
        for m in ('clean','occluded'):self.run_mock(m,env)
        self.assertEqual(env.resets,2);self.assertEqual(env.states,[[1,2,3]]*2)
    def test_missing_success_api_does_not_mean_done_success(self):
        with self.assertRaises(RuntimeError):success_check(object())
    def test_sim_wrapper_paths_and_cycles(self):
        sim=object();self.assertIs(resolve_sim(types.SimpleNamespace(sim=sim)),sim)
        self.assertIs(resolve_sim(types.SimpleNamespace(env=types.SimpleNamespace(sim=sim))),sim)
        cyc=types.SimpleNamespace();cyc.env=cyc
        with self.assertRaises(RuntimeError):resolve_sim(cyc)
    def test_order_reproducible_and_pair_seed_method_independent(self):
        ms=list(TrialController.MODES)
        self.assertEqual(method_order(ms,7,0,0),method_order(ms,7,0,0))
        self.assertEqual(set(method_order(ms,7,0,0)),set(ms))
        self.assertNotEqual(pair_seed(7,0,0),pair_seed(7,0,1))
    def camera_case(self,convention):
        K=np.array([[100.,0,128],[0,100,128],[0,0,1]])
        module=types.ModuleType('robosuite.utils.camera_utils')
        module.get_camera_intrinsic_matrix=lambda *a:K
        module.get_camera_extrinsic_matrix=lambda *a:np.eye(4)
        with patch.dict(sys.modules,{'robosuite':types.ModuleType('robosuite'),
            'robosuite.utils':types.ModuleType('robosuite.utils'),'robosuite.utils.camera_utils':module}):
            cam=calibrated_camera_from_env(None,224,256,convention)
        # Independent reference: project CV pixel -> observation -> LIBERO rotation -> resize.
        u,v=148.,158.
        ov=255-v if convention=='opengl' else v
        expected=np.array([255-u,255-ov]);expected=(expected+.5)*224/256-.5
        np.testing.assert_allclose(cam.project([.2,.3,1]),expected)
    def test_opengl_camera_projection(self):self.camera_case('opengl')
    def test_opencv_camera_projection(self):self.camera_case('opencv')
    def test_tied_pairs_do_not_imply_zero_uncertainty(self):
        s=paired_bootstrap_diff([0,0],[0,0]);self.assertLess(s['ci95'][0],0);self.assertGreater(s['ci95'][1],0)
        self.assertEqual(s['exact_mcnemar_p'],1.)
    def test_discordant_binary_pairs(self):
        s=paired_bootstrap_diff([1]*6,[0]*6);self.assertEqual(s['exact_mcnemar_p'],.03125)
    def test_nonbinary_pairs_rejected(self):
        with self.assertRaises(ValueError):paired_bootstrap_diff([.2],[0])
    def test_duplicate_records_rejected(self):
        r=dict(task_id=0,episode_id=0,method='m2',success=True)
        with tempfile.TemporaryDirectory() as d:
            p=Path(d);(p/'episodes.json').write_text(json.dumps([r,r]))
            with self.assertRaises(ValueError):analyze(p)
    def test_errors_and_missing_planned_runs_flagged(self):
        with tempfile.TemporaryDirectory() as d:
            p=Path(d);(p/'episodes.json').write_text(json.dumps([dict(task_id=0,episode_id=0,method='m2',success=None)]))
            (p/'manifest.json').write_text(json.dumps({'settings':{'task_ids':[0],'episode_ids':[0],'methods':['m2','m0']}}))
            s=analyze(p);self.assertEqual(s['missing_planned_runs'],[[0,0,'m0']]);self.assertEqual(s['method']['m2']['n'],0)
            self.assertEqual(s['evidence_status'],'INCOMPLETE_DO_NOT_CLAIM_GAIN')

if __name__=='__main__':unittest.main()
