import csv
import json
from pathlib import Path
import numpy as np
import pytest
from experiments.study_ready.core import inspect_episode,inspect_dataset,paired_binary_summary
from experiments.study_ready.sam2_offline import frame2record,validate_box,forward_from_predictor


def test_mask_candidate_empty_and_valid():
    m=np.zeros((32,32),bool)
    assert frame2record(2,m,m.shape)['candidate_pixel_xy'] is None
    m[8:13,15:19]=True
    o=frame2record(3,m,m.shape)
    assert o['candidate_pixel_xy']==[16.5,10.0]
    assert o['identity_verified'] is False
    with pytest.raises(ValueError):frame2record(3,m,(224,224))


def test_bbox_cannot_cover_invalid_region():
    assert validate_box((2,3,20,25),(32,32,3))==(2,3,20,25)
    with pytest.raises(ValueError):validate_box((0,0,1,1),(32,32,3))
    with pytest.raises(ValueError):validate_box((0,0,33,32),(32,32,3))


class FakeTensor:
    def __init__(self,a):self.a=np.asarray(a)
    def detach(self):return self
    def cpu(self):return self
    def numpy(self):return self.a

class FakePredictor:
    def init_state(self,**kwargs):return kwargs
    def add_new_points_or_box(self,**kwargs):return 0,[1],FakeTensor(np.zeros((1,1,32,32)))
    def propagate_in_video(self,**kwargs):
        for i in range(4):
            m=np.zeros((1,1,32,32));m[0,0,12:17,10+i:17+i]=10
            yield i,[1],[FakeTensor(m[0])]

def test_official_sam2_contract_mock_is_not_gpu_claim():
    vals=list(forward_from_predictor(FakePredictor(),'.',(5,5,20,20),(32,32,3),4))
    assert len(vals)==4
    assert all(x['candidate_pixel_xy'] and not x['identity_verified'] for x in vals)


def test_predictor_index_gap_rejected():
    class Broken(FakePredictor):
        def propagate_in_video(self,**kwargs):
            for i in (0,2):
                yield i,[1],[FakeTensor(np.ones((1,1,32,32)))]
    with pytest.raises(RuntimeError,match='missing'):
        list(forward_from_predictor(Broken(),'.',(5,5,20,20),(32,32,3),4))


def make_episode(root,ep,positive=False,missing_but_not_mask=False):
    child=root/'raw'/f'task0_ep{ep}';child.mkdir(parents=True)
    n=10;metadata={'episode_id':ep,'initial_state_id':f'group{ep}', 'trajectory_id':f'trace{ep}', 'task_id':0}
    p=[];e=[];a=[]
    for i in range(n):
        hidden=i in (4,5) or missing_but_not_mask and i>6
        p.append({**metadata,'step':i,'record_phase':'pre_action_observation',
                  'target_pixel_visible':None if hidden else [30.,50.],
                  'target_pixel_missing':bool(hidden),'previous_action':None if i==0 else [0,0,0,0,0,0,1]})
        e.append({**metadata,'step':i,'gripper_target_contact_EVAL_ONLY':bool(positive and i==4),
                  'attached_label_EVAL_ONLY':1 if positive and i==4 else 0,
                  'policy_clean_rgb_sha256_EVAL_ONLY':'a'*64,
                  'observation_rgb_mask_applied_EVAL_ONLY':False})
        a.append({**metadata,'step':i,'record_phase':'post_action_state','executed_action':[0,0,0,0,0,0,1],
                  'action_source':'frozen_openvla'})
    for sub,rows in (('public',p),('eval_only',e),('execution',a)):
        folder=child/sub;folder.mkdir()
        with (folder/f'{sub}.jsonl').open('w') as f:
            for row in rows:f.write(json.dumps(row)+'\n')
    for vid in ('policy.mp4','diagnostic.mp4'):(child/vid).write_bytes(b'fakevideo')
    (child/'manifest.json').write_text(json.dumps([{'public_path':str(child/'public/public.jsonl'),
                  'eval_only_path':str(child/'eval_only/eval_only.jsonl'),
                  'execution_path':str(child/'execution/execution.jsonl'),
                  'observation_dropout':[4,5],'video_policy_rgb':str(child/'policy.mp4'),
                  'video_diagnostic_eval_only':str(child/'diagnostic.mp4')}]))
    return child


def test_natural_missing_and_no_positive_are_warnings(tmp_path):
    child=make_episode(tmp_path,1,missing_but_not_mask=True)
    r=inspect_episode(child,tmp_path)
    assert not r['integrity_errors']
    assert 'NO_ATTACHED_POSITIVE' in r['coverage_warnings']


def test_ground_truth_leak_is_hard_error(tmp_path):
    child=make_episode(tmp_path,1)
    p=child/'public/public.jsonl'
    s=p.read_text().replace('"step": 0','"step": 0, "cheat_EVAL_ONLY": 1',1);p.write_text(s)
    r=inspect_episode(child,tmp_path)
    assert any('leak' in x.lower() for x in r['integrity_errors'])


def test_group_coverage_prevents_false_training_readiness(tmp_path):
    for ep in range(1,11):make_episode(tmp_path,ep,positive=False)
    s=inspect_dataset(tmp_path,tmp_path)
    assert s['episodes']==10
    assert not s['train_validation_class_coverage_precheck']
    assert s['attached_positive_frames']==0


def test_paired_bootstrap_uses_initial_states_not_frames(tmp_path):
    data=[]
    for i in range(12):
        data.extend([{'initial_state_id':str(i),'method':'clean','success':i%2==0},
                     {'initial_state_id':str(i),'method':'occluded','success':False}])
    p=tmp_path/'paired.json';p.write_text(json.dumps(data))
    s=paired_binary_summary(p,seed=7,replicates=1000)
    assert s['complete_pairs']==12
    assert s['success_rate_by_method']['clean']==.5
    assert s['comparisons'][0]['mean_paired_difference']==-.5
    assert s['comparisons'][0]['harmed']==6


def test_reject_unpaired_not_zero_fill(tmp_path):
    p=tmp_path/'paired.csv'
    p.write_text('initial_state_id,method,success\na,clean,1\nb,occluded,0\n')
    with pytest.raises(ValueError,match='No fully paired states'):
        paired_binary_summary(p)


def test_training_gate_fails_early_for_all_negative_labels():
    from experiments.study_ready.training_gate import group_class_report
    result=group_class_report([{'group':str(i),'y':np.array([0,0,-1])} for i in range(10)])
    assert not result['ready']
    assert 'positive' in result['reason']

def test_training_gate_succeeds_eligible_group_split():
    from experiments.study_ready.training_gate import group_class_report
    result=group_class_report([{'group':str(i),'y':np.array([0,1,-1])} for i in range(10)])
    assert result['ready']
    assert all(x['positive']>0 and x['negative']>0 for x in result['classes'].values())


def test_paired_protocol_mismatch_rejected(tmp_path):
    common={'task_id':'0','policy_checkpoint':'frozen','mask_plan_hash':'same',
            'max_steps':'220','seed':'7'}
    rows=[{'initial_state_id':'a','method':'clean','success':True,**common},
          {'initial_state_id':'a','method':'occluded','success':False,**{**common,'max_steps':'100'}}]
    p=tmp_path/'p.json';p.write_text(json.dumps(rows))
    with pytest.raises(ValueError,match='different'):
        paired_binary_summary(p)

def test_paired_protocol_verified_if_logged(tmp_path):
    common={'task_id':'0','policy_checkpoint':'frozen','mask_plan_hash':'same',
            'max_steps':'220','seed':'7'}
    rows=[{'initial_state_id':'a','method':'clean','success':True,**common},
          {'initial_state_id':'a','method':'occluded','success':False,**common}]
    p=tmp_path/'p.json';p.write_text(json.dumps(rows))
    assert paired_binary_summary(p,replicates=50)['protocol_verified']
