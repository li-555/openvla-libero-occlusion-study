"""Audits public PRE-action, privileged PRE-action and execution POST-action clocks.

This is a data-only checker; it never imports LIBERO or opens the environment.
"""
import argparse
import json
from pathlib import Path
import numpy as np
from .stream import load_jsonl,validate_public,PHASE_PRE,PHASE_POST,META_FIELDS


def _matched_identity(pre,other):
    keys=META_FIELDS-{'record_phase'}
    if any(pre.get(k)!=other.get(k) for k in keys):
        raise ValueError('Metadata/clock identity mismatch for PRE vs POST/EVAL')


def audit_trajectory(public,ev,execution):
    if len(public)!=len(ev) or len(public)!=len(execution):
        raise ValueError('Each PRE-action t must have matched PRE-eval and POST-action t')
    if not public:raise ValueError('Empty trajectory')
    for i,(r,e,p) in enumerate(zip(public,ev,execution)):
        validate_public(r)
        if r['step']!=i or e['step']!=i or p['step']!=i:
            raise ValueError('Missing or reordered step')
        _matched_identity(r,e);_matched_identity(r,p)
        if r['record_phase']!=PHASE_PRE or e['record_phase']!=PHASE_PRE or p['record_phase']!=PHASE_POST:
            raise ValueError('Wrong pre/post phase')
        if any(not (k in META_FIELDS or k.endswith('_EVAL_ONLY')) for k in e):
            raise ValueError('EVAL_ONLY has unsuffixed privileged field')
        for action in ('executed_action','policy_action'):
            if not isinstance(p.get(action),list) or len(p[action])!=7:
                raise ValueError(f'Missing {action} in POST record')
        if i>0:
            prior=execution[i-1]
            if not np.allclose(r['previous_action'],prior['executed_action'],atol=1e-7,rtol=0):
                raise ValueError('previous_action at PRE t must equal executed_action at POST t-1')
            if prior.get('post_eef_xyz') is None: raise ValueError('Missing post EEF state')
            if p_prev_eef:=prior.get('post_eef_xyz'):
                if not np.allclose(r['eef_xyz'],p_prev_eef,atol=1e-6):
                    raise ValueError('EEF at PRE t differs from POST t-1')
            if prior.get('post_gripper_qpos') is None: raise ValueError('Missing post gripper state')
            if p_prev_gripper:=prior.get('post_gripper_qpos'):
                if not np.allclose(r['gripper_qpos'],p_prev_gripper,atol=1e-6):
                    raise ValueError('Gripper at PRE t differs from POST t-1')
    return dict(trajectory_id=public[0]['trajectory_id'],episode_id=public[0]['episode_id'],
                initial_state_id=public[0]['initial_state_id'],seed=public[0]['seed'],
                code_version=public[0]['code_version'],steps=len(public),status='aligned')


def audit_directory(directory):
    directory=Path(directory);results=[]
    for path in sorted((directory/'public').glob('*.jsonl')):
        name=path.stem
        ev=directory/'eval_only'/f'{name}_state_trace.jsonl'
        post=directory/'execution'/f'{name}_actions.jsonl'
        if not ev.exists() or not post.exists():
            raise FileNotFoundError(f'Missing paired EVAL_ONLY/POST data for {name}')
        results.append(audit_trajectory(load_jsonl(path),load_jsonl(ev),load_jsonl(post)))
    if not results:raise ValueError('No data found')
    return results


def main():
    ap=argparse.ArgumentParser();ap.add_argument('--data',required=True)
    args=ap.parse_args()
    print(json.dumps(audit_directory(args.data),indent=2))

if __name__=='__main__':main()
