import copy
import unittest
import numpy as np
from experiments.learned_contact.stream import (
    public_row,validate_public,post_action_row,eval_only_row,PHASE_PRE,PHASE_POST,source_version)
from experiments.learned_contact.dataset import sequence_features,assemble_trace,group_split,FEATURE_NAMES
from experiments.learned_contact.labels import derive_attachment_labels
from experiments.learned_contact.mixture import combine
from experiments.learned_contact.adapter import RunConfig,EstimationInterface


def row(step=0,px=(100,120),group='state0',traj='t0',last_action=None):
    return public_row(step=step,episode_id=0,initial_state_id=group,
        method='data_collection',seed=17,config={'suite':'libero_spatial'},
        code_version=source_version(),trajectory_id=traj,
        task_id=0,intervention='natural_policy',eef_xyz=[0.01*step,.01,.3],
        eef_pixel=[50+step,60],gripper_qpos=[.04,-.04],
        previous_action=last_action,visible_pixel=px,confidence=1. if px is not None else 0.)


class TestLearnedContact(unittest.TestCase):
    def test_public_rejects_oracle_and_future(self):
        r=row();r['target_pos_gt_EVAL_ONLY']=[0,0,0]
        with self.assertRaisesRegex(ValueError,'invalid fields'):validate_public(r)
        del r['target_pos_gt_EVAL_ONLY'];r['future_action']=[0]*7
        with self.assertRaisesRegex(ValueError,'invalid fields'):validate_public(r)

    def test_hidden_no_detection_and_causal_history(self):
        rows=[row(0,px=(100,120)),row(1,px=None,last_action=[0]*7)]
        x,h=sequence_features(rows,history=3)
        self.assertEqual(x.shape,(2,3,len(FEATURE_NAMES)))
        self.assertTrue(np.all(x[0,0]==0))
        self.assertEqual(x[1,-1,FEATURE_NAMES.index('visible_flag')],0)
        self.assertEqual(h[1]['anchor_step'],0)
        np.testing.assert_allclose(h[1]['memory'],[100,120])
        np.testing.assert_allclose(h[1]['rigid'],[101,120])
        r2=copy.deepcopy(rows);r2[1]['previous_action']=[1]*7
        x2,_=sequence_features(r2,history=3)
        np.testing.assert_array_equal(x[0],x2[0]) # a future action never affects t=0

    def test_hidden_leak_flag(self):
        r=row(0,px=None)
        r['target_pixel_confidence']=.9
        with self.assertRaises(ValueError):validate_public(r)

    def test_mixture_preserves_components_and_abstains(self):
        m=combine([4,4],[9,9],.5)
        self.assertEqual(m.weights,(.5,.5))
        self.assertIsNone(m.cue_pixel)
        self.assertEqual(m.decision,'uncertain')
        self.assertEqual(combine([4,4],[9,9],.9).cue_pixel,(9.,9.))

    def test_label_requires_contact_and_motion(self):
        rows=[]
        for i in range(4):
            rows.append({'step':i,'gripper_target_contact_EVAL_ONLY':True,
                'target_pos_gt_EVAL_ONLY':[i*.01+.01,0,0],
                'robot0_eef_pos_EVAL_ONLY':[i*.01,0,0],
                'robot0_gripper_qpos_EVAL_ONLY':[0,0]})
        res=derive_attachment_labels(rows,grip_open_aperture=.08)
        self.assertEqual(res[0]['attached_label_EVAL_ONLY'],None)
        self.assertEqual(res[2]['attached_label_EVAL_ONLY'],1)
        rows[-1]['gripper_target_contact_EVAL_ONLY']=False
        rows[-1]['robot0_gripper_qpos_EVAL_ONLY']=[.04,-.04]
        res=derive_attachment_labels(rows,grip_open_aperture=.08)
        self.assertEqual(res[-1]['attached_label_EVAL_ONLY'],0)

    def test_absent_contact_is_unknown(self):
        r={'step':0,'gripper_target_contact_EVAL_ONLY':None,
            'target_pos_gt_EVAL_ONLY':[0,0,0],
            'robot0_eef_pos_EVAL_ONLY':[0,0,0],'robot0_gripper_qpos_EVAL_ONLY':[.04,-.04]}
        self.assertIsNone(derive_attachment_labels([r],grip_open_aperture=.08)[0]['attached_label_EVAL_ONLY'])

    def test_group_variants_never_split(self):
        traces=[]
        for i in range(10):
            for intervention in ('natural','release'):
                traces.append({'group':f'g{i}','intervention':intervention})
        split=group_split(traces)
        owners={}
        for name,ts in split.items():
            for t in ts:
                self.assertNotIn(t['group'],owners) if t['group'] in owners and owners[t['group']]!=name else None
                owners[t['group']]=name
        self.assertEqual(len(owners),10)

    def test_online_interface_no_control_mutation(self):
        model=EstimationInterface(RunConfig(estimator='rigid',intervention='none'))
        a=model.step(row(0))
        b=model.step(row(1,px=None,last_action=[0]*7))
        self.assertIsNone(b['action_override'])
        self.assertEqual(b['status'],'rigid')
        self.assertEqual(b['prediction']['rigid_pixel'],(101.,120.))
        self.assertIsNone(a['action_override'])
        with self.assertRaises(ValueError): model.step(row(3))

    def test_assemble_never_uses_truth_as_model_feature(self):
        rows=[row(0),row(1,px=None,last_action=[0]*7)]
        evalrows=[eval_only_row(public=rows[0],attached_label_EVAL_ONLY=1,
                 target_pixel_reference_rgb_EVAL_ONLY=[100,120]),
                  eval_only_row(public=rows[1],attached_label_EVAL_ONLY=0,
                 target_pixel_reference_rgb_EVAL_ONLY=[500,300])]
        t=assemble_trace(rows,evalrows)
        evalrows[1]['target_pixel_reference_rgb_EVAL_ONLY']=[-1000,-1000]
        t2=assemble_trace(rows,evalrows)
        np.testing.assert_array_equal(t['X'],t2['X'])

    def test_required_provenance(self):
        r=row(0)
        for k in ('episode_id','initial_state_id','step','method','seed','config',
                  'code_version','record_phase'):
            self.assertIn(k,r)
        self.assertEqual(r['record_phase'],PHASE_PRE)
        self.assertEqual(r['previous_action'],None)

    def test_post_action_phase_and_identity(self):
        pre=row(0)
        post=post_action_row(episode_id=pre['episode_id'],initial_state_id=pre['initial_state_id'],
            step=pre['step'],method=pre['method'],seed=pre['seed'],config=pre['config'],
            code_version=pre['code_version'],trajectory_id=pre['trajectory_id'],task_id=pre['task_id'],
            policy_action=[0]*7,executed_action=[0]*7,post_eef_xyz=[.1,.2,.3],
            post_gripper_qpos=[0,0],done=False,action_source='frozen_openvla')
        self.assertEqual(post['record_phase'],PHASE_POST)
        with self.assertRaisesRegex(ValueError,'invalid fields'):
            validate_public(post)

    def test_privileged_truth_never_enters_model(self):
        pre=row(0)
        ev=eval_only_row(public=pre,attached_label_EVAL_ONLY=1,
                         target_pixel_reference_rgb_EVAL_ONLY=[10,20])
        with self.assertRaisesRegex(ValueError,'invalid fields'):
            EstimationInterface(RunConfig(estimator='memory')).step(ev)
        with self.assertRaises(ValueError):
            eval_only_row(public=pre,robot0_eef_pos=[0,0,0])

    def test_eval_alignment_mismatch_is_hard_error(self):
        r0=row(0);r1=row(1,px=None,last_action=[0]*7)
        ev=[eval_only_row(public=r0,attached_label_EVAL_ONLY=None),
            eval_only_row(public=r1,attached_label_EVAL_ONLY=None)]
        ev[1]['code_version']='wrong-version'
        with self.assertRaisesRegex(ValueError,'identity'):
            assemble_trace([r0,r1],ev)

    def test_env_is_rejected(self):
        with self.assertRaises(TypeError):
            EstimationInterface(RunConfig()).step(object())

    def test_initial_state_can_not_leak_via_features(self):
        r0=row(0);r1=row(1,px=None,last_action=[0]*7)
        x,_=sequence_features([r0,r1],history=3)
        r0b=copy.deepcopy(r0);r1b=copy.deepcopy(r1)
        r0b['initial_state_id']='other';r1b['initial_state_id']='other'
        r0b['config']={'suite':'other'};r1b['config']={'suite':'other'}
        xb,_=sequence_features([r0b,r1b],history=3)
        np.testing.assert_array_equal(x,xb)

if __name__=='__main__':unittest.main()
