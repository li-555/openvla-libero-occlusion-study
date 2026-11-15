"""AI-2 strict collection schemas: PRE-action public observations, POST-action logs,
privileged EVAL_ONLY supervision. Estimator accepts *public dicts only*, never envs.

Clock: public(step=t) is obs before action_t; previous_action is executed action_(t-1).
execution(step=t) is the action actually applied at t plus state after action_t.
Eval labels for step t refer to the same PRE-action state as public(step=t).
"""
import hashlib
import json
from pathlib import Path
import numpy as np

PHASE_PRE = 'pre_action_observation'
PHASE_POST = 'post_action_state'
META_FIELDS = frozenset({'episode_id','initial_state_id','step','method','seed',
                         'config','code_version','record_phase','trajectory_id','task_id'})
PUBLIC_FIELDS = META_FIELDS | frozenset({'intervention','target_pixel_visible',
    'target_pixel_confidence','target_pixel_missing','eef_pixel','eef_xyz',
    'gripper_qpos','previous_action','observation_type'})
ALLOWED_CONFIG = frozenset({'suite','checkpoint','roi','dark_threshold',
                            'camera_resolution','collection_mode','history'})

def source_version():
    """Content digest (works for source archives with no git metadata)."""
    d=Path(__file__).parent
    h=hashlib.sha256()
    for p in sorted(d.glob('*.py')):
        h.update(p.name.encode());h.update(p.read_bytes())
    return 'ai2-src-sha256:'+h.hexdigest()[:20]


def _values(v,n):
    if v is None:return None
    a=np.asarray(v,dtype=float).reshape(-1)
    if a.size!=n or not np.isfinite(a).all():
        raise ValueError(f'Expected {n} finite values')
    return a.tolist()


def _metadata(*,episode_id,initial_state_id,step,method,seed,config,code_version,
              record_phase,trajectory_id,task_id):
    if type(config) is not dict or set(config)-ALLOWED_CONFIG:
        raise ValueError('Only safe, declared collection config keys allowed')
    # Metadata is never a model feature. Keep controlled release / occlusion
    # schedules and oracle IDs out of public config entirely.
    if not initial_state_id or not trajectory_id or not method or not code_version:
        raise ValueError('Missing mandatory record identity')
    if int(step)<0 or int(episode_id)<0 or not isinstance(seed,(int,np.integer)):
        raise ValueError('Invalid step/episode/seed')
    if record_phase not in (PHASE_PRE,PHASE_POST):
        raise ValueError('Unknown record phase')
    meta=dict(episode_id=int(episode_id),initial_state_id=str(initial_state_id),
        step=int(step),method=str(method),seed=int(seed),config=config.copy(),
        code_version=str(code_version),record_phase=record_phase,
        trajectory_id=str(trajectory_id),task_id=int(task_id))
    json.dumps(meta,allow_nan=False)  # reject non-serializable metadata
    return meta


def public_row(*,episode_id,initial_state_id,step,method,seed,config,code_version,
               trajectory_id,task_id,intervention,eef_xyz,eef_pixel,gripper_qpos,
               previous_action,visible_pixel=None,confidence=0.,observation_type='rgb_proxy_px'):
    m=_metadata(episode_id=episode_id,initial_state_id=initial_state_id,step=step,
        method=method,seed=seed,config=config,code_version=code_version,
        record_phase=PHASE_PRE,trajectory_id=trajectory_id,task_id=task_id)
    pixel=_values(visible_pixel,2)
    result={**m,'intervention':str(intervention),'target_pixel_visible':pixel,
        'target_pixel_confidence':float(confidence if pixel is not None else 0.),
        'target_pixel_missing':pixel is None,'eef_pixel':_values(eef_pixel,2),
        'eef_xyz':_values(eef_xyz,3),'gripper_qpos':_values(gripper_qpos,2),
        'previous_action':_values(previous_action,7),'observation_type':observation_type}
    validate_public(result)
    return result


def validate_public(row):
    if type(row) is not dict:
        raise TypeError('Estimator accepts a public dictionary, NEVER an environment object')
    unknown=set(row)-PUBLIC_FIELDS
    missing=PUBLIC_FIELDS-set(row)
    if unknown or missing:
        raise ValueError(f'Public stream invalid fields: extra={sorted(unknown)}, missing={sorted(missing)}')
    _metadata(**{k:row[k] for k in META_FIELDS})
    if row['record_phase']!=PHASE_PRE:
        raise ValueError('Estimator accepts PRE-action observations only')
    if row['target_pixel_missing'] != (row['target_pixel_visible'] is None):
        raise ValueError('Missing observation flag inconsistent with pixel evidence')
    if row['target_pixel_missing'] and row['target_pixel_confidence']!=0:
        raise ValueError('Hidden object confidence must be zero')
    if row['observation_type']!='rgb_proxy_px':
        raise ValueError('RGB proxy pixels cannot be interpreted as simulator 3D origin')
    for key,n in (('eef_xyz',3),('gripper_qpos',2),('eef_pixel',2),
                  ('target_pixel_visible',2),('previous_action',7)):
        _values(row[key],n)
    if not 0<=float(row['target_pixel_confidence'])<=1:
        raise ValueError('Confidence outside [0,1]')
    if row['step']==0 and row['previous_action'] is not None:
        raise ValueError('Action at t-1 must be null at t=0')
    if row['step']>0 and row['previous_action'] is None:
        raise ValueError('Missing executed action from t-1')
    return True


def post_action_row(*,episode_id,initial_state_id,step,method,seed,config,code_version,
                    trajectory_id,task_id,policy_action,executed_action,post_eef_xyz,
                    post_gripper_qpos,done,action_source):
    m=_metadata(episode_id=episode_id,initial_state_id=initial_state_id,step=step,
        method=method,seed=seed,config=config,code_version=code_version,
        record_phase=PHASE_POST,trajectory_id=trajectory_id,task_id=task_id)
    return {**m,'policy_action':_values(policy_action,7),
        'executed_action':_values(executed_action,7),
        'post_eef_xyz':_values(post_eef_xyz,3),
        'post_gripper_qpos':_values(post_gripper_qpos,2),
        'done':bool(done),'action_source':str(action_source)}


def eval_only_row(*,public,**truth):
    """Create privileged row with matched PRE-action identity, NOT for model.step()."""
    validate_public(public)
    if any(not k.endswith('_EVAL_ONLY') for k in truth):
        raise ValueError('Privileged fields must use _EVAL_ONLY suffix')
    return {**{k:public[k] for k in META_FIELDS},**truth}


def dump_jsonl(path,rows):
    p=Path(path);p.parent.mkdir(parents=True,exist_ok=True)
    with p.open('w',encoding='utf-8') as f:
        for r in rows:f.write(json.dumps(r,ensure_ascii=False,allow_nan=False)+'\n')
    return p


def load_jsonl(path):
    with Path(path).open(encoding='utf-8') as f:
        return [json.loads(l) for l in f if l.strip()]


def sim_gripper_target_contact(sim,target_tokens,finger_tokens):
    """ONLY label/evaluation stream; unknown if geom mapping cannot be established.

    Substring geom tokens must be verified against a real MuJoCo geom inventory.
    """
    if not target_tokens or not finger_tokens:return None,[]
    try:
        model,data=sim.model,sim.data
        names=[model.geom_id2name(i) or '' for i in range(model.ngeom)]
        targets={i for i,n in enumerate(names) if any(s in n for s in target_tokens)}
        fingers={i for i,n in enumerate(names) if any(s in n for s in finger_tokens)}
        if not targets or not fingers or targets & fingers:return None,[]
        pairs=[]
        for i in range(data.ncon):
            c=data.contact[i];a,b=int(c.geom1),int(c.geom2)
            if (a in targets and b in fingers) or (b in targets and a in fingers):
                pairs.append([names[a],names[b]])
        return bool(pairs),pairs
    except (AttributeError,ValueError,TypeError,IndexError):return None,[]
