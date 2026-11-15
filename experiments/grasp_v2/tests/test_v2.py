import json
from pathlib import Path
import numpy as np
import pytest
from experiments.grasp_v2.features import PreAction, FEATURE_NAMES, obs_from_dict
from experiments.grasp_v2.controller import AdaptiveGraspController
from experiments.grasp_v2.train import fit,load_rows
from experiments.grasp_v2.install_runner import patch_text
from experiments.grasp_v2.controller import CANDIDATE_PROTOCOL
from experiments.grasp_v2.aperture import aperture_from_qpos


def obs(i=0):
    return {'target_px':[54.+i,139.], 'eef_px':[66.,128.],
      'eef_xyz_m':[.1,.2,.3],'eef_vel_m':[0.,0.,0.],
      'gripper_aperture_m':.065,'last_seen_age':1,'target_confidence':1.,
      'previous_action_xyz':[0.,0.,0.],'proposed_action_xyz':[0.,0.,0.]}


def make_rows(tmp_path):
    p=tmp_path/'probes.jsonl'
    rows=[]
    for g in range(12):
        for i in range(8):
            y=int(i%2==0)
            data={'schema':'grasp_probe_outcome_v1','event_timing':'post_probe_outcome',
                'label_source':'executed_grasp_and_lift',
                'candidate_protocol':CANDIDATE_PROTOCOL,
                'split':'development','initial_state_id':f'initial_{g}',
                'checkpoint_id':f'cp_{g}','seed':g,
                'pre_action':obs(i%3),
                'candidate':{'xyz_action':[0.,float(y)*.1,0.],
                             'close_now':bool(y)},
                'post_action_label':{'stable_grasp_after_lift':y}}
            rows.append(json.dumps(data))
    p.write_text('\n'.join(rows),encoding='utf-8')
    return p


def test_features_no_ground_truth():
    x=obs_from_dict(obs()).vector([0.,0.,0.],True)
    assert x.shape==(len(FEATURE_NAMES),)
    with pytest.raises(ValueError):obs_from_dict({**obs(),'object_pos_EVAL_ONLY':[1,2,3]})


def test_training_requires_real_label_and_groups(tmp_path):
    p=make_rows(tmp_path)
    X,y,g,c=load_rows(p)
    assert len(X)==96 and len(set(g))==12
    corrupt=tmp_path/'corrupt.jsonl'
    line=json.loads(p.read_text().splitlines()[0]);line['label_source']='imputed_from_clean'
    corrupt.write_text(json.dumps(line))
    with pytest.raises(ValueError,match='label'):load_rows(corrupt)


def test_fit_and_controller_does_not_invent_success(tmp_path):
    p=make_rows(tmp_path)
    report=fit(p,tmp_path/'model')
    assert 0<=report['heldout_brier']<=1
    ctrl=AdaptiveGraspController(tmp_path/'model'/'candidate_model.joblib')
    cmd=np.array([.1,.1,.0,0.,0.,0.,1.])
    ctrl.observe(step=0,occluded=False,memory_pixel=[54.,139.],
      eef_pixel=[66.,128.],eef_xyz=[.1,.2,.3],gripper_aperture_m=.07)
    initial_open=cmd.copy();initial_open[-1]=-1.
    a,e=ctrl.override(initial_open,step=0)
    assert np.array_equal(a,initial_open) and not e['modified']
    ctrl.observe(step=1,occluded=False,memory_pixel=[54.,139.],
      eef_pixel=[66.,128.],eef_xyz=[.1,.2,.3],gripper_aperture_m=.07,previous_action=cmd)
    ctrl.observe(step=2,occluded=True,memory_pixel=None,
      eef_pixel=[66.,128.],eef_xyz=[.1,.2,.3],gripper_aperture_m=.07,previous_action=cmd)
    a,e=ctrl.override(cmd,step=2)
    assert a.shape==(7,) and np.isfinite(a).all() and e['model_used']
    assert a[6] in (-1.,1.)


def test_no_visible_anchor_abstains(tmp_path):
    p=make_rows(tmp_path);fit(p,tmp_path/'model')
    ctrl=AdaptiveGraspController(tmp_path/'model'/'candidate_model.joblib')
    ctrl.observe(step=0,occluded=True,memory_pixel=None,eef_pixel=[66.,128.],
        eef_xyz=[.1,.2,.3],gripper_aperture_m=.07)
    cmd=np.array([.1,.0,.0,0,0,0,1.])
    a,e=ctrl.override(cmd,step=0)
    assert np.array_equal(a,cmd) and e['reason'].startswith('abstain_')


def test_disallow_out_of_order_step(tmp_path):
    p=make_rows(tmp_path);fit(p,tmp_path/'model')
    ctrl=AdaptiveGraspController(tmp_path/'model'/'candidate_model.joblib')
    params=dict(occluded=True,memory_pixel=None,eef_pixel=[66.,128.],eef_xyz=[.1,.2,.3],gripper_aperture_m=.07)
    ctrl.observe(step=0,**params)
    with pytest.raises(ValueError):ctrl.observe(step=0,**params)


def test_installer_ast_and_read_only():
    path=Path('<local-temp-path>
    if not path.exists():pytest.skip('local reference unavailable')
    before=path.read_text()
    after=patch_text(before)
    assert 'grasp_v2.override' in after and 'grasp_recovery.override' in after
    assert path.read_text()==before
    with pytest.raises(ValueError):patch_text(after)


def test_real_probe_adapter_contract_mock_does_not_use_gt_in_features():
    from experiments.grasp_v2.collect_probes import collect_checkpoint
    class MockAdapter:
        def __init__(self):self.nstep=0;self.calls=[]
        def capture_verified(self):return {'checkpoint_id':'test_verified_checkpoint','hash':'hash'}
        def restore_verified(self,snapshot):
            assert snapshot['hash']=='hash';self.nstep=0
        def public_preaction(self):return obs()
        def gripper_joint_qpos(self):return [.04,-.025]
        def openvla_action(self):return [.1,.2,.3,0.,0.,0.,1.]
        def step(self,cmd):self.nstep+=1;self.calls.append(cmd)
        def stable_grasp_after_lift(self):
            assert self.nstep in (1,2)
            return self.nstep==2
    rows=collect_checkpoint(MockAdapter(),initial_state_id='dev_a',
             episode_id=5,seed=7,split='development')
    assert len(rows)==8 and rows[0]['candidate']['close_now'] is True
    assert all(x['label_source']=='executed_grasp_and_lift' for x in rows)
    assert all(set(x['pre_action'])==set(obs()) for x in rows)
    assert all(abs(x['pre_action']['gripper_aperture_m']-.065)<1e-10 for x in rows)


def test_no_fake_probe_outcome():
    from experiments.grasp_v2.collect_probes import score_candidate
    class Adapter:
        def restore_verified(self,s):pass
        def openvla_action(self):return [0]*7
        def step(self,a):pass
        def stable_grasp_after_lift(self):return None
    with pytest.raises(ValueError,match='ACTUAL'):
        score_candidate(Adapter(),{'checkpoint_id':'s'},candidate_xyz=[0,0,0],close_now=True)


def test_learned_action_correction_changes_xyz_not_just_cue(tmp_path):
    p=make_rows(tmp_path);fit(p,tmp_path/'m')
    ctrl=AdaptiveGraspController(tmp_path/'m'/'candidate_model.joblib')
    class Model:
        def predict_proba(self,X):
            # First option closes now. All others move then close.
            score=np.full(len(X),.2)
            score[2]=.92  # best learned offset, NOT a stored CLEAN pose
            return np.stack([1-score,score],axis=-1)
    ctrl.models=[Model(),Model(),Model()]
    prop=np.array([.1,.2,.0,0.,0.,0.,1.])
    ctrl.observe(step=3,occluded=False,memory_pixel=[54.,139.],eef_pixel=[66.,128.],
       eef_xyz=[.1,.2,.3],gripper_aperture_m=.08)
    ctrl.observe(step=4,occluded=False,memory_pixel=[54.,139.],eef_pixel=[66.,128.],
       eef_xyz=[.1,.2,.3],gripper_aperture_m=.08)
    ctrl.observe(step=5,occluded=True,memory_pixel=None,eef_pixel=[66.,128.],
       eef_xyz=[.1,.2,.3],gripper_aperture_m=.08)
    executed,event=ctrl.override(prop,step=5)
    assert event['modified'] and event['rejected_close']
    assert executed[-1]==-1. and not np.array_equal(executed[:3],prop[:3])
    assert np.array_equal(executed[3:6],prop[3:6])


def test_installer_supports_baseline_without_demo_guided():
    source=Path('<local-temp-path>
    if not source.exists():pytest.skip('full baseline unavailable')
    patched=patch_text(source.read_text())
    assert 'grasp_v2.override' in patched
    assert "aperture_from_qpos(obs['robot0_gripper_qpos'])" in patched
    assert "np.mean(obs['robot0_gripper_qpos'])" not in patched
    assert 'grasp_recovery=None' not in patched


def _ready_controller(tmp_path):
    p = make_rows(tmp_path)
    fit(p, tmp_path/'model')
    ctrl = AdaptiveGraspController(tmp_path/'model'/'candidate_model.joblib')
    ctrl.observe(step=0,occluded=False,memory_pixel=[54,139],eef_pixel=[66,128],
                 eef_xyz=[.1,.2,.3],gripper_aperture_m=.065)
    ctrl.observe(step=1,occluded=False,memory_pixel=[54,139],eef_pixel=[66,128],
                 eef_xyz=[.1,.2,.3],gripper_aperture_m=.065)
    return ctrl


class _FixedScores:
    def __init__(self, probs):
        self.probs=np.array(probs,dtype=float)
    def predict_proba(self, X):
        assert len(X)==len(self.probs)
        return np.column_stack([1-self.probs,self.probs])


def test_gripper_aperture_uses_difference_not_mean():
    assert aperture_from_qpos([.04,-.025]) == pytest.approx(.065)
    assert np.mean([.04,-.025]) == pytest.approx(.0075)
    with pytest.raises(ValueError):aperture_from_qpos([.04])
    with pytest.raises(ValueError):aperture_from_qpos([float('nan'),-.02])


def test_two_step_candidate_executes_forced_next_close_then_stops(tmp_path):
    ctrl=_ready_controller(tmp_path)
    ctrl.models=[_FixedScores([.2,.25,.91,.3,.3,.3,.3,.3])]*3
    cmd=np.array([.1,.2,.0,0.,0.,0.,1.])
    ctrl.observe(step=2,occluded=True,memory_pixel=None,eef_pixel=[66,128],
                 eef_xyz=[.1,.2,.3],gripper_aperture_m=.065)
    a,e=ctrl.override(cmd,step=2)
    assert e['pending_close_next_step'] and a[-1]==-1.0
    assert a[0]==pytest.approx(.22)  # chosen +X offset
    # Model is deliberately not called on the next step; protocol forces one close.
    ctrl.models=[_FixedScores([float('nan')]*8)]*3
    ctrl.observe(step=3,occluded=True,memory_pixel=None,eef_pixel=[66,128],
                 eef_xyz=[.1,.2,.3],gripper_aperture_m=.065,previous_action=a)
    second=np.array([.4,.2,.0,0.,0.,0.,-1.])
    x,e=ctrl.override(second,step=3)
    assert x[-1] == 1. and np.array_equal(x[:6],second[:6])
    assert e['reason']=='complete_precommitted_two_step_candidate'
    # Following OPEN is untouched, even though still occluded.
    ctrl.observe(step=4,occluded=True,memory_pixel=None,eef_pixel=[66,128],
                 eef_xyz=[.1,.2,.3],gripper_aperture_m=.065)
    open_cmd=np.array([.1,0,0,0,0,0,-1.])
    x,e=ctrl.override(open_cmd,step=4)
    assert np.array_equal(x,open_cmd) and e['reason'].startswith('first_grasp')


def test_low_scores_bounded_hold_never_extra_close(tmp_path):
    ctrl=_ready_controller(tmp_path)
    ctrl.models=[_FixedScores([.05]*8)]*3
    cmd=np.array([.1,.2,.0,0.,0.,0.,1.])
    for step in range(2,6):
        ctrl.observe(step=step,occluded=True,memory_pixel=None,eef_pixel=[66,128],
                     eef_xyz=[.1,.2,.3],gripper_aperture_m=.065)
        out,e=ctrl.override(cmd,step=step)
        if step in (2,3):
            assert out[-1]==-1. and e['reason']=='bounded_open_fallback'
            assert np.array_equal(out[:6],cmd[:6])
        else:
            assert out[-1]==1. and e['reason'] in (
                'fallback_budget_exhausted_passthrough',
                'first_grasp_attempt_finished_passthrough')


def test_open_policy_action_during_fallback_is_untouched(tmp_path):
    ctrl=_ready_controller(tmp_path)
    ctrl.models=[_FixedScores([.05]*8)]*3
    ctrl.observe(step=2,occluded=True,memory_pixel=None,eef_pixel=[66,128],
                 eef_xyz=[.1,.2,.3],gripper_aperture_m=.065)
    ctrl.override([.1,0,0,0,0,0,1],step=2)
    ctrl.observe(step=3,occluded=True,memory_pixel=None,eef_pixel=[66,128],
                 eef_xyz=[.1,.2,.3],gripper_aperture_m=.065)
    x,e=ctrl.override([.1,0,0,0,0,0,-1],step=3)
    assert x[-1]==-1. and not e['modified']


def test_probe_aperture_and_proposed_action_use_same_preaction_contract():
    from experiments.grasp_v2.collect_probes import collect_checkpoint
    class Adapter:
        def __init__(self):self.steps=0
        def capture_verified(self):return {'checkpoint_id':'verified_1'}
        def restore_verified(self,s):self.steps=0
        def public_preaction(self):return obs()
        def gripper_joint_qpos(self):return [.04,-.025]
        def openvla_action(self):return [.3,.2,.1,0,0,0,1]
        def step(self,a):self.steps+=1
        def stable_grasp_after_lift(self):return self.steps==2
    rows=collect_checkpoint(Adapter(),initial_state_id='a',episode_id=5,seed=1,split='development')
    assert len(rows)==8
    assert all(r['candidate_protocol']==CANDIDATE_PROTOCOL for r in rows)
    assert all(r['pre_action']['proposed_action_xyz']==[.3,.2,.1] for r in rows)
    assert all(r['pre_action']['gripper_aperture_m']==pytest.approx(.065) for r in rows)
    assert rows[0]['post_action_label']['stable_grasp_after_lift']==0
    assert rows[1]['post_action_label']['stable_grasp_after_lift']==1
