import json
import pytest
from experiments.learned_contact.audit import audit_trajectory,audit_directory
from experiments.learned_contact.make_synthetic_fixture import generate
from experiments.learned_contact.stream import load_jsonl
from experiments.learned_contact.adapter import RunConfig,EstimationInterface


def test_full_three_stream_clock_and_no_env(tmp_path):
    d=generate(tmp_path,groups=5,steps=14)
    assert len(audit_directory(d))==5
    pub=load_jsonl(d/'public'/'fake_000.jsonl')
    model=EstimationInterface(RunConfig(estimator='rigid',intervention='none'))
    predictions=[model.step(row) for row in pub]
    for row,pred in zip(pub,predictions):
        assert pred['episode_id']==row['episode_id']
        assert pred['initial_state_id']==row['initial_state_id']
        assert pred['step']==row['step']
        assert pred['method']=='rigid'
        assert isinstance(pred['seed'],int)
        assert pred['code_version'].startswith('ai2-src-sha256:')
        assert pred['record_phase']=='pre_action_observation'
        assert pred['action_override'] is None


def test_misaligned_previous_action_caught(tmp_path):
    d=generate(tmp_path,groups=1,steps=14)
    pub=load_jsonl(d/'public'/'fake_000.jsonl')
    ev=load_jsonl(d/'eval_only'/'fake_000_state_trace.jsonl')
    post=load_jsonl(d/'execution'/'fake_000_actions.jsonl')
    pub[2]['previous_action']=[123]*7
    with pytest.raises(ValueError,match='previous_action'):
        audit_trajectory(pub,ev,post)


def test_post_action_does_not_look_like_pre_action(tmp_path):
    d=generate(tmp_path,groups=1,steps=14)
    post=load_jsonl(d/'execution'/'fake_000_actions.jsonl')
    with pytest.raises(ValueError):
        EstimationInterface(RunConfig()).step(post[0])
