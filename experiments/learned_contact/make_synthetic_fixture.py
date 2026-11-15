"""Generate SCHEMA-ONLY fake streams for CPU integration testing.

NEVER use these trajectories as robot research evidence or report model accuracy.
"""
import argparse
from pathlib import Path
import numpy as np
from .stream import public_row,eval_only_row,post_action_row,dump_jsonl,source_version


def generate(root,groups=12,steps=18):
    root=Path(root)
    for group in range(groups):
        seed=group+100;gid=f'SYNTHETIC_STATE_{group:03d}';tid=f'fake_{group:03d}'
        identity=dict(episode_id=group,initial_state_id=gid,task_id=0,
            trajectory_id=tid,method='synthetic_schema_test',seed=seed,
            code_version=source_version(),config=dict(suite='synthetic-only',
                camera_resolution=224,collection_mode='synthetic_test'))
        previous_action=None;public=[];evals=[];actions=[]
        for t in range(steps):
            eef=np.array([.02*t+group*.005,.1,.3]);ep=np.array([50+t*2.,60.])
            attached=(t<12);ob=np.array([ep[0]+10,70.]) if attached else np.array([76.+group*.2,70.])
            hidden=5<=t<=14
            p=public_row(step=t,**identity,intervention='fake_scripted_release',
                eef_xyz=eef,eef_pixel=ep,gripper_qpos=[.04,-.04] if t>=12 else [.01,-.01],
                previous_action=previous_action,visible_pixel=None if hidden else ob,
                confidence=0. if hidden else 1.)
            public.append(p)
            evals.append(eval_only_row(public=p,attached_label_EVAL_ONLY=int(attached),
                attachment_reason_EVAL_ONLY='synthetic_label_NOT_verified',
                target_pixel_reference_rgb_EVAL_ONLY=ob.tolist()))
            action=np.zeros(7);action[0]=.02
            if t>=12:action[-1]=-1
            post=post_action_row(step=t,**identity,policy_action=action,
                executed_action=action,post_eef_xyz=([.02*(t+1)+group*.005,.1,.3]),
                post_gripper_qpos=[.04,-.04] if t+1>=12 else [.01,-.01],
                done=t==steps-1,action_source='fake_scripted')
            actions.append(post);previous_action=action.tolist()
        dump_jsonl(root/'public'/f'{tid}.jsonl',public)
        dump_jsonl(root/'eval_only'/f'{tid}_state_trace.jsonl',evals)
        dump_jsonl(root/'execution'/f'{tid}_actions.jsonl',actions)
    return root


def main():
    ap=argparse.ArgumentParser();ap.add_argument('--output',required=True)
    args=ap.parse_args();print('SYNTHETIC TEST DATA ONLY:',generate(args.output))

if __name__=='__main__':main()
