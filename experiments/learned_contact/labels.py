"""Conservative privileged attachment labels; UNKNOWN boundaries are not negatives.

All simulator state fields end in _EVAL_ONLY. These rows are never estimator inputs.
"""
import numpy as np


def derive_attachment_labels(eval_rows, *, min_motion_m=.008,
                             stable_relative_m=.012,separation_m=.16,
                             grip_open_aperture=None,grip_weights=(1.,-1.),history=3):
    if grip_open_aperture is None:
        raise ValueError('Attachment labeling requires independently calibrated aperture')
    rows=sorted(eval_rows,key=lambda r:r['step'])
    if len({r['step'] for r in rows}) != len(rows):raise ValueError('Duplicate step')
    out=[]
    for i,r in enumerate(rows):
        contact=r.get('gripper_target_contact_EVAL_ONLY')
        if contact not in (True,False,None):raise ValueError('Contact evidence invalid')
        label=None;reason='contact_unverified'
        p=r.get('target_pos_gt_EVAL_ONLY');eef=r.get('robot0_eef_pos_EVAL_ONLY')
        q=r.get('robot0_gripper_qpos_EVAL_ONLY')
        if p is None or eef is None or q is None:
            reason='missing_privileged_kinematics'
        elif contact is not None:
            p=np.asarray(p,float);eef=np.asarray(eef,float);q=np.asarray(q,float)
            if p.shape!=(3,) or eef.shape!=(3,) or q.shape!=(2,):
                raise ValueError('Unexpected truth geometry')
            aperture=float(np.dot(np.asarray(grip_weights,float),q))
            distance=float(np.linalg.norm(p-eef))
            if contact is False and (aperture>=grip_open_aperture or distance>separation_m):
                label=0;reason='no_contact_and_open_or_separated'
            elif contact is True and i>=history-1:
                window=rows[i-history+1:i+1]
                if all(z.get('gripper_target_contact_EVAL_ONLY') is True and
                       z.get('target_pos_gt_EVAL_ONLY') is not None and
                       z.get('robot0_eef_pos_EVAL_ONLY') is not None for z in window):
                    obj=np.array([z['target_pos_gt_EVAL_ONLY'] for z in window],float)
                    robot=np.array([z['robot0_eef_pos_EVAL_ONLY'] for z in window],float)
                    rel=obj-robot
                    rel_drift=float(np.linalg.norm(rel-rel[0],axis=1).max())
                    eef_motion=float(np.linalg.norm(robot-robot[0],axis=1).max())
                    if rel_drift<=stable_relative_m and eef_motion>=min_motion_m:
                        label=1;reason='verified_contact_stable_relative_motion'
                    else:reason='contact_not_sufficiently_stable_or_excited'
                else:reason='insufficient_persistent_contact_evidence'
            else:reason='contact_boundary_ambiguous'
        out.append({**r,'attached_label_EVAL_ONLY':label,
                    'attachment_reason_EVAL_ONLY':reason})
    return out
