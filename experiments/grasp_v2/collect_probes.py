"""Actual counterfactual grasp-probe driver with a verified simulator adapter.

This is a *caller-owned adapter contract*, not a fake simulator. Plug the
existing paired_decision_eval verified checkpoint / restore functions into
these methods. No online GT is ever read. An unverified restore aborts.

Required adapter methods:
  capture_verified() -> snapshot + integrity hash
  restore_verified(snapshot) -> None, raises unless hash/state matches
  public_preaction() -> dict EXACT features.PreAction fields (no object GT)
  gripper_joint_qpos() -> actual two PUBLIC finger joint coordinates
  openvla_action() -> executed 7D action based on policy observation only
  step(executed_action_7d) -> None, actually advances simulator
  stable_grasp_after_lift() -> bool, EXECUTES probe lift and reads post-action
       outcome; object/contact GT may be used ONLY inside this scorer.

Candidate move_then_close: execute candidate XYZ with open gripper for one
step, then issue next OpenVLA XYZ with forced close; both branches share the
same stable-lift outcome protocol. Candidate close_now: immediately execute
candidate XYZ with close command. This is a controlled action intervention.
"""
import hashlib
import json
from pathlib import Path
import numpy as np
from .features import obs_from_dict
from .aperture import aperture_from_qpos
from .controller import CANDIDATE_PROTOCOL
from .train import SCHEMA


def score_candidate(adapter, snapshot, *, candidate_xyz, close_now):
    adapter.restore_verified(snapshot)
    a=np.asarray(adapter.openvla_action(),dtype=float).copy()
    if a.shape!=(7,) or not np.isfinite(a).all():
        raise ValueError('policy must yield executed 7-vector')
    c=np.asarray(candidate_xyz,dtype=float)
    if c.shape!=(3,) or not np.isfinite(c).all() or (abs(c)>1).any():
        raise ValueError('invalid bounded XYZ candidate')
    a[:3]=c
    a[-1]=1. if close_now else -1.
    adapter.step(a.tolist())
    if not close_now:
        a2=np.asarray(adapter.openvla_action(),dtype=float).copy()
        if a2.shape!=(7,) or not np.isfinite(a2).all():
            raise ValueError('invalid follow-up OpenVLA action')
        a2[-1]=1.
        adapter.step(a2.tolist())
    result=adapter.stable_grasp_after_lift()
    if not isinstance(result,(bool,np.bool_)):
        raise ValueError('lift evaluator must return ACTUAL post-action bool, no imputation')
    return bool(result)


def collect_checkpoint(adapter, *, initial_state_id, episode_id, seed, split,
                       xyz_step=.12, max_candidates=8):
    """One verified SAME-STATE candidate set; train/confirmation split frozen."""
    if split not in ('development','confirmation'):
        raise ValueError('predeclare development vs confirmation BEFORE outcomes')
    if not 0<xyz_step<=.25 or max_candidates!=8:
        raise ValueError('fixed 8-candidate protocol')
    snapshot=adapter.capture_verified()
    # Only whitelist exact PRE-ACTION public fields. Object GT is never included.
    row=dict(adapter.public_preaction())
    obs_from_dict(row)
    # The same calibrated aperture definition is used at train and deployment.
    # Aborts if the verified public joint observation is not available.
    row['gripper_aperture_m']=aperture_from_qpos(adapter.gripper_joint_qpos())
    proposal=np.asarray(adapter.openvla_action(),dtype=float)
    if proposal.shape!=(7,) or not np.isfinite(proposal).all():
        raise ValueError('invalid original OpenVLA action')
    # The proposed motion feature must refer to the action actually probed.
    row['proposed_action_xyz']=proposal[:3].tolist()
    obs_from_dict(row)
    base=proposal[:3]
    directions=[np.zeros(3),
                np.array([xyz_step,0,0]),np.array([-xyz_step,0,0]),
                np.array([0,xyz_step,0]),np.array([0,-xyz_step,0]),
                np.array([0,0,xyz_step]),np.array([0,0,-xyz_step])]
    spec=[(np.clip(base,-1,1),True)]+[(np.clip(base+d,-1,1),False) for d in directions]
    checkpoint_id=str(snapshot['checkpoint_id'])
    result=[]
    for candidate_xyz,now in spec:
        stable=score_candidate(adapter,snapshot,
                               candidate_xyz=candidate_xyz,close_now=now)
        result.append({'schema':SCHEMA,'event_timing':'post_probe_outcome',
            'label_source':'executed_grasp_and_lift',
            'candidate_protocol':CANDIDATE_PROTOCOL,'split':split,
            'initial_state_id':str(initial_state_id), 'checkpoint_id':checkpoint_id,
            'episode_id':str(episode_id),'seed':int(seed),
            'pre_action':row,
            'candidate':{'xyz_action':candidate_xyz.tolist(),'close_now':bool(now)},
            'post_action_label':{'stable_grasp_after_lift':int(stable)}})
    return result


def append_jsonl(rows,path):
    path=Path(path);path.parent.mkdir(parents=True,exist_ok=True)
    if path.exists():
        raise FileExistsError('Use unique files per checkpoint and preserve existing evidence')
    path.write_text(''.join(json.dumps(r,sort_keys=True)+'\n' for r in rows),encoding='utf-8')
    return {'path':str(path),'count':len(rows),
        'sha256':hashlib.sha256(path.read_bytes()).hexdigest()}
