"""Regression tests for four review-requested fixes; no model or GPU required."""
import hashlib
import json
from pathlib import Path

import numpy as np
import pytest

from experiments.study_ready.cli import main
from experiments.study_ready.core import inspect_dataset, paired_binary_summary
from experiments.study_ready.sam2_offline import sha256_file
from experiments.study_ready.training_gate import group_class_report


def test_empty_dataset_is_integrity_failure(tmp_path):
    result = inspect_dataset(tmp_path, tmp_path)
    assert result['episodes'] == 0
    assert result['all_integrity_pass'] is False
    assert 'No episode trajectories' in result['integrity_errors'][0]
    assert result['train_validation_class_coverage_precheck'] is False


def test_missing_raw_directory_cannot_make_cli_pass(tmp_path, capsys):
    rc = main(['dataset-audit', '--repo', str(tmp_path), '--root', str(tmp_path)])
    assert rc == 2
    assert 'No episode trajectories' in capsys.readouterr().out


def test_python310_chunked_sha256_hashes_binary_file(tmp_path):
    path = tmp_path / 'video.mp4'
    payload = bytes(range(256)) * 8193  # crosses the 1 MiB chunk boundary
    path.write_bytes(payload)
    assert sha256_file(path) == hashlib.sha256(payload).hexdigest()


def test_binary_fit_ready_when_val_test_each_one_class(monkeypatch):
    from experiments.learned_contact import dataset
    def fake_split(traces, seed):
        return {'train': [traces[0]], 'val': [traces[1]], 'test': [traces[2]]}
    monkeypatch.setattr(dataset, 'group_split', fake_split)
    traces = [{'group': str(i), 'y':np.asarray([0,1] if i == 0 else [i % 2])} for i in range(5)]
    r = group_class_report(traces)
    assert r['ready'] and r['can_fit_binary_classifier']
    assert not r['ready_for_model_selection']
    assert not r['ready_for_existing_train_eval']
    assert not r['test_two_class_metrics_available']
    assert 'VALIDATION_ONE_CLASS_OR_EMPTY' in r['warnings'][0]
    assert any('TEST_ONE_CLASS_OR_EMPTY' in w for w in r['warnings'])


def test_validation_fit_but_test_single_class_not_block_model_selection(monkeypatch):
    from experiments.learned_contact import dataset
    monkeypatch.setattr(dataset, 'group_split', lambda traces,seed:
                        {'train':[traces[0]],'val':[traces[1]],'test':[traces[2]]})
    traces = [{'group':str(i), 'y':np.asarray([0,1] if i<=1 else [0])} for i in range(5)]
    r=group_class_report(traces)
    assert r['ready'] and r['ready_for_model_selection']
    assert r['ready_for_existing_train_eval']
    assert not r['test_two_class_metrics_available']
    assert any('TEST_ONE_CLASS' in w for w in r['warnings'])


def test_missing_train_positive_does_block_fit(monkeypatch):
    from experiments.learned_contact import dataset
    monkeypatch.setattr(dataset, 'group_split', lambda traces,seed:
                        {'train':[traces[0]],'val':[traces[1]],'test':[traces[2]]})
    traces = [{'group':str(i), 'y':np.asarray([0] if i==0 else [0,1])} for i in range(5)]
    r=group_class_report(traces)
    assert not r['ready'] and not r['ready_for_existing_train_eval']
    assert 'Training split lacks' in r['reason']


def test_explicit_hold_minus_occluded_with_dropped_states(tmp_path):
    rows=[
        {'initial_state_id':'s1','method':'clean','success':1},
        {'initial_state_id':'s1','method':'occluded','success':0},
        {'initial_state_id':'s1','method':'hold','success':1},
        {'initial_state_id':'s1','method':'learned','success':0},
        {'initial_state_id':'s2','method':'occluded','success':1},
        {'initial_state_id':'s2','method':'hold','success':0},
        {'initial_state_id':'s3','method':'occluded','success':0},
        {'initial_state_id':'s3','method':'hold','success':'','missing_reason':'gpu_job_interrupted'},
        {'initial_state_id':'s4','method':'hold','success':0},
        {'initial_state_id':'s5','method':'learned','success':0},
        {'initial_state_id':'s5','method':'rule','success':1},
    ]
    path=tmp_path/'paired.json';path.write_text(json.dumps(rows))
    result=paired_binary_summary(path,comparisons=['hold:occluded','learned:rule'],seed=0,replicates=150)
    a,b=result['comparisons']
    assert a['comparison']=='hold - occluded' and a['matched_states']==2
    assert a['mean_paired_difference']==0.0
    assert a['rescued']==1 and a['harmed']==1
    assert [x['initial_state_id'] for x in a['excluded_states']]==['s3','s4','s5']
    assert a['excluded_states'][0]['reasons']['hold']=='gpu_job_interrupted'
    assert a['excluded_states'][1]['reasons']['occluded']=='no_run_record'
    assert b['matched_states']==1 and b['mean_paired_difference']==-1.0
    assert result['complete_pairs']==0  # per-contrast pairing works even without all-method pairs


def test_all_pairs_default_includes_hold_vs_occluded(tmp_path):
    rows=[{'initial_state_id':'s1','method':m,'success':int(m=='hold')}
          for m in ('clean','occluded','hold','learned','rule')]
    path=tmp_path/'result.json';path.write_text(json.dumps(rows))
    r=paired_binary_summary(path,replicates=10)
    assert len(r['comparisons'])==10
    assert set(frozenset([c['candidate'],c['reference']]) for c in r['comparisons'])=={
        frozenset(pair) for i,p in enumerate(r['methods']) for pair in [[p,q] for q in r['methods'][i+1:]]}


def test_cli_explicit_comparison_and_missing_state_reasons(tmp_path,capsys):
    path=tmp_path/'outcomes.csv'
    path.write_text('initial_state_id,method,success,missing_reason\n'
                    '1,occluded,0,\n1,hold,1,\n2,occluded,1,\n2,hold,,timeout\n')
    rc=main(['paired-report','--outcomes',str(path),'--compare','hold:occluded','--replicates','20'])
    assert rc==0
    result=json.loads(capsys.readouterr().out)
    assert result['comparisons'][0]['matched_states']==1
    assert result['comparisons'][0]['excluded_states']==[
        {'initial_state_id':'2','missing_methods':['hold'],'reasons':{'hold':'timeout'}}]


def test_invalid_or_unpaired_comparisons_fail_closed(tmp_path):
    path=tmp_path/'outcomes.csv'
    path.write_text('initial_state_id,method,success\na,occluded,0\nb,hold,1\n')
    with pytest.raises(ValueError,match='No fully paired states'):
        paired_binary_summary(path,comparisons=['hold:occluded'])
    with pytest.raises(ValueError,match='Invalid comparison'):
        paired_binary_summary(path,comparisons=['unknown:occluded'])
    with pytest.raises(ValueError,match='Use --compare'):
        paired_binary_summary(path,comparisons=['hold-occluded'])


def test_cli_train_preflight_reports_fit_without_claiming_pipeline_ready(monkeypatch, tmp_path, capsys):
    import experiments.study_ready.training_gate as gate
    monkeypatch.setattr(gate, 'preflight_data', lambda *args, **kwargs: {
        'ready':True,'can_fit_binary_classifier':True,
        'ready_for_model_selection':False,'ready_for_existing_train_eval':False,
        'test_two_class_metrics_available':False,
        'reason':'train is viable but validation has one class',
        'warnings':['VALIDATION_ONE_CLASS_OR_EMPTY']})
    rc=main(['train-preflight','--data',str(tmp_path)])
    assert rc == 2  # running existing train_eval.py would still fail model selection
    result=json.loads(capsys.readouterr().out)
    assert result['can_fit_binary_classifier'] and not result['ready_for_existing_train_eval']
