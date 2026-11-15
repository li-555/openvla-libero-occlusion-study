"""Leak-resistant trajectory builder. Ground truth is used ONLY for supervision/evaluation.

No evaluation target ever becomes a feature. Test grouping is by initial-state hash,
not by overlapping windows or intervention variants.
"""
from collections import defaultdict
from pathlib import Path
import numpy as np
from .stream import load_jsonl, validate_public, META_FIELDS, PHASE_PRE

FEATURE_NAMES = ('visible_u','visible_v','visible_flag','pixel_confidence',
    'eef_u','eef_v','eef_x','eef_y','eef_z','eef_dx','eef_dy','eef_dz',
    'grip_q0','grip_q1','grip_delta','prev_action_0','prev_action_1',
    'prev_action_2','prev_action_3','prev_action_4','prev_action_5',
    'prev_action_6','time_since_visible','last_rel_u','last_rel_v',
    'last_object_du','last_object_dv','last_eef_du','last_eef_dv')


def sequence_features(rows, history=6):
    """Return [N,history,F], online-causal per-row features, and anchor hypotheses."""
    rows=sorted(rows,key=lambda r:r['step'])
    if len({r['step'] for r in rows}) != len(rows):
        raise ValueError('Duplicate time index')
    if any(b['step'] != a['step']+1 for a,b in zip(rows,rows[1:])):
        raise ValueError('Missing robot telemetry; do not interpolate secretly')
    prev=None; last_visible=None; last_rel=np.zeros(2); last_du=np.zeros(2); last_eef_du=np.zeros(2)
    last_pixel=None; last_px_step=None; last_eef_px=None; anchor=None
    state=[]; hypotheses=[]
    for r in rows:
        validate_public(r)
        eef=np.array(r['eef_xyz'],float)
        ep=np.asarray(r['eef_pixel'],float) if r['eef_pixel'] is not None else None
        visible=np.asarray(r['target_pixel_visible'],float) if not r['target_pixel_missing'] else None
        # Last reliable pixel anchor is an RGB proxy, not a projected object 3D center.
        if visible is not None and ep is not None:
            if last_pixel is not None and last_px_step == r['step']-1 and last_eef_px is not None:
                last_du=(visible-last_pixel).copy()
                last_eef_du=(ep-last_eef_px).copy()
            last_visible=(visible.copy(), ep.copy(),r['step'])
            last_rel=visible-ep
            last_pixel=visible.copy();last_eef_px=ep.copy();last_px_step=r['step']
        age = 1000. if last_visible is None else float(r['step']-last_visible[2])
        # The two hypotheses are distinct, no misleading mean-only outputs.
        if last_visible is None or ep is None:
            memory=rigid=None
        else:
            memory=last_visible[0].copy()
            rigid=last_visible[0]+ep-last_visible[1]
        hypotheses.append(dict(memory=memory,rigid=rigid,anchor_step=None if last_visible is None else last_visible[2]))
        eef_delta=np.zeros(3) if prev is None else eef-np.asarray(prev['eef_xyz'],float)
        qp=np.asarray(r['gripper_qpos'],float)
        prev_qp=np.asarray(prev['gripper_qpos'],float) if prev is not None else qp
        action=np.zeros(7) if r['previous_action'] is None else np.asarray(r['previous_action'],float)
        pix=np.zeros(2) if visible is None else visible/224.
        epx=np.zeros(2) if ep is None else ep/224.
        x=np.r_[pix,float(visible is not None),r['target_pixel_confidence'],
            epx,eef,eef_delta,qp,float(np.linalg.norm(qp-prev_qp)),action,
            min(age,100.)/100.,last_rel/224.,last_du/224.,last_eef_du/224.]
        if x.size != len(FEATURE_NAMES):
            raise AssertionError(f'feature mismatch: {x.size}')
        state.append(x)
        prev=r
    # Pad earliest history with zeros, not future frames.
    X=[]
    for i in range(len(rows)):
        prefix=np.asarray(state[max(0,i-history+1):i+1],float)
        padded=np.zeros((history,len(FEATURE_NAMES)))
        padded[-len(prefix):]=prefix
        X.append(padded)
    return np.stack(X),hypotheses


def assemble_trace(public_rows, eval_rows, history=6):
    if len(public_rows) != len(eval_rows):
        raise ValueError('Public and eval traces must cover exactly the same timestamps')
    if not public_rows:
        raise ValueError('empty trace')
    public_rows=sorted(public_rows,key=lambda x:x['step'])
    eval_rows=sorted(eval_rows,key=lambda x:x['step'])
    if len({r['initial_state_id'] for r in public_rows}) != 1 or len({r['trajectory_id'] for r in public_rows}) != 1:
        raise ValueError('Mixed initial states or trajectories')
    if any(a['step'] != b['step'] for a,b in zip(public_rows,eval_rows)):
        raise ValueError('Trace time mismatch')
    for a,b in zip(public_rows,eval_rows):
        validate_public(a)
        if {k:a[k] for k in META_FIELDS} != {k:b.get(k) for k in META_FIELDS}:
            raise ValueError('EVAL_ONLY/public identity or PRE-action alignment mismatch')
        if b['record_phase'] != PHASE_PRE:
            raise ValueError('EVAL_ONLY labels must belong to the PRE-action state')
        if any(not (k in META_FIELDS or k.endswith('_EVAL_ONLY')) for k in b):
            raise ValueError('EVAL_ONLY contains unsuffixed privileged field')
    X,hyp=sequence_features(public_rows,history)
    y=[];target=[];status=[]
    for r in eval_rows:
        # Labels only when contact AND relative-motion evidence confirm the event.
        label=r.get('attached_label_EVAL_ONLY')
        if label not in (0,1,None):
            raise ValueError('Attachment labels must be 0,1,null')
        y.append(-1 if label is None else int(label))
        gt=r.get('target_pixel_reference_rgb_EVAL_ONLY')
        if gt is not None:
            gt=np.asarray(gt,float)
            if gt.shape!=(2,) or not np.isfinite(gt).all():
                raise ValueError('Invalid projected GT label')
        target.append(gt)
        status.append(r.get('attachment_reason_EVAL_ONLY','unverified'))
    return dict(X=X,y=np.asarray(y,int),target=target,hyp=hyp,steps=np.asarray([r['step'] for r in public_rows]),
        group=public_rows[0]['initial_state_id'],trajectory=public_rows[0]['trajectory_id'],
        intervention=public_rows[0]['intervention'],reason=status,
        episode_id=public_rows[0]['episode_id'], task_id=public_rows[0]['task_id'],
        collection_seed=public_rows[0]['seed'],config=public_rows[0]['config'],
        code_version=public_rows[0]['code_version'],
        public=public_rows)


def load_pairs(public_dir, eval_dir, history=6):
    traces=[]
    for path in sorted(Path(public_dir).glob('*.jsonl')):
        name=path.stem
        eval_path=Path(eval_dir)/f'{name}_state_trace.jsonl'
        if not eval_path.exists():
            raise FileNotFoundError(f'Missing truth-isolated eval trace: {eval_path}')
        traces.append(assemble_trace(load_jsonl(path),load_jsonl(eval_path),history))
    if not traces:
        raise ValueError('No matching real traces found. Do NOT train on fabricated traces')
    return traces


def group_split(traces,seed=2026):
    groups=sorted(set(t['group'] for t in traces))
    if len(groups)<5:
        raise ValueError('At least 5 distinct initial-state groups needed for train/val/test')
    rng=np.random.default_rng(seed); rng.shuffle(groups)
    ntest=max(1,int(np.ceil(len(groups)*.2)))
    nval=max(1,int(np.ceil(len(groups)*.2)))
    split={'test':set(groups[:ntest]),'val':set(groups[ntest:ntest+nval]),
           'train':set(groups[ntest+nval:])}
    if not split['train']:
        raise ValueError('Too few groups to split')
    if len(set().union(*split.values())) != len(groups) or any(split[a]&split[b] for a in split for b in split if a!=b):
        raise AssertionError('Group leakage')
    return {name:[t for t in traces if t['group'] in ids] for name,ids in split.items()}


def labeled_data(traces):
    xx=[]; yy=[]; ids=[]
    for t in traces:
        mask=t['y']>=0
        xx.extend(t['X'][mask]);yy.extend(t['y'][mask]);ids.extend([t['trajectory']]*int(mask.sum()))
    if not xx:
        raise ValueError('No independently labeled contact rows. Acquire/contact-verify data first.')
    return np.stack(xx),np.asarray(yy,int),ids
