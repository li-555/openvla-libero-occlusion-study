"""Independent clean LIBERO collection path for learned contact research.

Runs frozen OpenVLA in the original, unoccluded policy path. Offline object
observation dropout is applied to THE PUBLIC RECORD ONLY, not to policy RGB.
This is NOT an RGB-occluded closed-loop experiment. Uses an explicit optional
controlled release for diagnostic data; it is never a natural failure.

Must run on an allocated CUDA node with LIBERO installed and checkpoint available.
"""
import argparse
import hashlib
import json
from pathlib import Path
from types import SimpleNamespace
import numpy as np
from experiments.learned_contact.stream import (public_row,post_action_row,eval_only_row,
    source_version,dump_jsonl,sim_gripper_target_contact)
from experiments.learned_contact.labels import derive_attachment_labels
from experiments.occlusion_eval.core import calibrated_camera_from_env
from experiments.occlusion_eval.runtime_checks import resolve_sim,success_check,configure_tensorflow_cpu
from experiments.occlusion_eval.motion2d import DarkPixel2D,Pixel2DConfig
from experiments.occlusion_eval.run_gpu import use_gpu_allocation_or_die,parse_ids


def collect_one(*, env, initial_state, task_description,task_id,episode_id,
                model,processor,cfg,run_dir,args):
    from experiments.robot.libero.libero_utils import get_libero_image,get_libero_dummy_action
    from experiments.robot.robot_utils import get_action,normalize_gripper_action,invert_gripper_action
    import torch
    from experiments.robot.robot_utils import set_seed_everywhere
    seed=args.seed+task_id*1009+episode_id*7
    set_seed_everywhere(seed);env.seed(seed)
    env.reset();obs=env.set_init_state(initial_state)
    for _ in range(args.warmup):
        obs,_,done,_=env.step(get_libero_dummy_action('openvla'))
        if done: raise RuntimeError('warmup terminated')
    sim=resolve_sim(env)
    camera=calibrated_camera_from_env(sim,224,native_size=256)
    detector=DarkPixel2D(tuple(float(x) for x in args.roi.split(',')),Pixel2DConfig(
        dark_threshold=args.dark_threshold))
    gid=hashlib.sha256((args.suite+':'+str(task_id)).encode()+np.asarray(initial_state).tobytes()).hexdigest()
    intervention='controlled_early_release' if args.early_release_step is not None else 'natural_policy'
    runid=(f'task{task_id}_ep{episode_id}_seed{seed}_{intervention}'
           f'_dropout{args.dropout_start}-{args.dropout_end}'
           f'_release{args.early_release_step if args.early_release_step is not None else "none"}')
    public_rows=[];eval_rows=[];execution_rows=[];previous_action=None
    version=source_version()
    safe_config=dict(suite=args.suite,checkpoint=args.checkpoint,
                     roi=[float(x) for x in args.roi.split(',')],
                     dark_threshold=args.dark_threshold,camera_resolution=224,
                     collection_mode='clean_rgb_offline_observation_dropout')
    record_identity=dict(episode_id=episode_id,initial_state_id=gid,
                         method='data_collection',seed=seed,config=safe_config,
                         code_version=version,trajectory_id=runid,task_id=task_id)
    success_check_fn,_=success_check(env)
    for t in range(args.steps):
        rgb=np.asarray(get_libero_image(obs,224),dtype=np.uint8)
        # IMPORTANT: missingness affects only estimator inputs, not OpenVLA RGB.
        hidden=(args.dropout_start<=t<=args.dropout_end)
        # Detector on clean RGB saved in eval-only, including artificially missing t.
        # It is NOT projected simulator truth and is never available to training features.
        reference_px=detector.estimate(rgb)
        px=None if hidden else reference_px
        eef=obs.get('robot0_eef_pos');q=obs.get('robot0_gripper_qpos')
        ep=camera.project(eef) if eef is not None else None
        pub=public_row(step=t,**record_identity,intervention=intervention,
            eef_xyz=eef,eef_pixel=ep,gripper_qpos=q,
            previous_action=previous_action,visible_pixel=px,
            confidence=float(px is not None))
        public_rows.append(pub)
        # Isolated eval-only labels may NEVER pass into model/get_action/public row.
        contact,pairs=sim_gripper_target_contact(sim,args.target_geom_tokens,args.finger_geom_tokens)
        gt=obs.get(args.target_object+'_pos')
        gtpx=camera.project(gt) if gt is not None else None
        eval_rows.append(eval_only_row(public=pub,
            target_pos_gt_EVAL_ONLY=np.asarray(gt).tolist() if gt is not None else None,
            target_center_px_gt_EVAL_ONLY=list(gtpx) if gtpx is not None else None,
            target_pixel_reference_rgb_EVAL_ONLY=(list(reference_px) if reference_px is not None else None),
            robot0_eef_pos_EVAL_ONLY=np.asarray(eef).tolist() if eef is not None else None,
            robot0_gripper_qpos_EVAL_ONLY=np.asarray(q).tolist() if q is not None else None,
            gripper_target_contact_EVAL_ONLY=contact,
            verified_contact_geom_pairs_EVAL_ONLY=pairs))
        observation={'full_image':rgb,'state':np.zeros(8,dtype=float)}
        with torch.inference_mode():
            a=np.asarray(get_action(cfg,model,observation,task_description,processor=processor),float)
        if a.shape!=(7,) or not np.isfinite(a).all():
            raise ValueError('invalid policy action')
        executed=np.asarray(invert_gripper_action(normalize_gripper_action(a.copy(),binarize=True)),float)
        # In LIBERO's executed action convention, -1 opens, +1 closes.
        if args.early_release_step is not None and args.early_release_step<=t<args.early_release_step+args.release_duration:
            executed[-1]=-1.
            eval_rows[-1]['release_override_applied_EVAL_ONLY']=True
        else:
            eval_rows[-1]['release_override_applied_EVAL_ONLY']=False
        previous_action=executed.tolist()
        obs,_,done,_=env.step(executed.tolist())
        execution_rows.append(post_action_row(step=t,**record_identity,
            policy_action=a,executed_action=executed,
            post_eef_xyz=obs.get('robot0_eef_pos'),
            post_gripper_qpos=obs.get('robot0_gripper_qpos'),done=done,
            action_source=('controlled_release' if eval_rows[-1]['release_override_applied_EVAL_ONLY'] else 'frozen_openvla')))
        if done or success_check_fn(): break
    if args.calibrated_open_aperture is None:
        eval_rows=[dict(row,attached_label_EVAL_ONLY=None,
                        attachment_reason_EVAL_ONLY='no_calibrated_open_aperture') for row in eval_rows]
    else:
        eval_rows=derive_attachment_labels(eval_rows,grip_open_aperture=args.calibrated_open_aperture)
    path_pub=dump_jsonl(run_dir/'public'/f'{runid}.jsonl',public_rows)
    path_eval=dump_jsonl(run_dir/'eval_only'/f'{runid}_state_trace.jsonl',eval_rows)
    path_execution=dump_jsonl(run_dir/'execution'/f'{runid}_actions.jsonl',execution_rows)
    return dict(initial_state_id=gid,episode_id=episode_id,task_id=task_id,seed=seed,
        method='data_collection',code_version=version,config=safe_config,trajectory_id=runid,
        public_path=str(path_pub),
        execution_path=str(path_execution),eval_only_path=str(path_eval),rows=len(public_rows),labeled=sum(r['attached_label_EVAL_ONLY'] is not None for r in eval_rows),
        intervention=intervention,observation_dropout=[args.dropout_start,args.dropout_end],
        note='PRE-action RGB proxy reference is same geometry as estimator; EVAL_ONLY 3D projection is separate diagnostic')


def main(argv=None):
    ap=argparse.ArgumentParser(description=__doc__)
    ap.add_argument('--suite',default='libero_spatial')
    ap.add_argument('--checkpoint',default='<OPENVLA_CHECKPOINT>')
    ap.add_argument('--task-ids',default='0')
    ap.add_argument('--episode-ids',default='0,1,2,3,4,5,6,7,8,9')
    ap.add_argument('--target-object',required=True)
    ap.add_argument('--roi',required=True,help='Fixed RGB target ROI x0,y0,x1,y1; visually audit!')
    ap.add_argument('--target-geom-tokens',nargs='+',required=True)
    ap.add_argument('--finger-geom-tokens',nargs='+',required=True)
    ap.add_argument('--calibrated-open-aperture',type=float,default=None,
                    help='Actual calibrated aperture in qpos[0]-qpos[1] convention; otherwise NO attachment labels')
    ap.add_argument('--dropout-start',type=int,default=80)
    ap.add_argument('--dropout-end',type=int,default=87)
    ap.add_argument('--early-release-step',type=int,default=None)
    ap.add_argument('--release-duration',type=int,default=10)
    ap.add_argument('--steps',type=int,default=220)
    ap.add_argument('--warmup',type=int,default=10)
    ap.add_argument('--seed',type=int,default=7)
    ap.add_argument('--dark-threshold',type=int,default=75)
    ap.add_argument('--output',default='experiments/learned_contact/data_real')
    args=ap.parse_args(argv)
    if args.dropout_start>args.dropout_end or args.steps<1 or args.release_duration<1:
        ap.error('Invalid window/steps')
    from experiments.occlusion_eval.run_gpu import parse_rect
    parse_rect(args.roi)
    configure_tensorflow_cpu();use_gpu_allocation_or_die()
    from libero.libero import benchmark
    from experiments.robot.libero.libero_utils import get_libero_env
    from experiments.robot.robot_utils import get_model,set_seed_everywhere
    task_ids,episode_ids=parse_ids(args.task_ids),parse_ids(args.episode_ids)
    cfg=SimpleNamespace(model_family='openvla',pretrained_checkpoint=args.checkpoint,
        load_in_4bit=True,load_in_8bit=False,center_crop=True,unnorm_key=args.suite)
    set_seed_everywhere(args.seed)
    model=get_model(cfg);model.eval()
    if not getattr(model,'is_loaded_in_4bit',False):
        raise RuntimeError('Model was not loaded in 4-bit')
    if cfg.unnorm_key not in model.norm_stats and cfg.unnorm_key+'_no_noops' in model.norm_stats:
        cfg.unnorm_key+='_no_noops'
    if cfg.unnorm_key not in model.norm_stats: raise ValueError('No LIBERO action normalization')
    from experiments.robot.openvla_utils import get_processor
    processor=get_processor(cfg)
    suite=benchmark.get_benchmark_dict()[args.suite]()
    out=Path(args.output);out.mkdir(parents=True,exist_ok=True)
    manifest=[]
    for task_id in task_ids:
        task=suite.get_task(task_id)
        initial_states=suite.get_task_init_states(task_id)
        env,description=get_libero_env(task,'openvla',resolution=256)
        try:
            for ep in episode_ids:
                if ep>=len(initial_states):raise ValueError('invalid episode id')
                row=collect_one(env=env,initial_state=initial_states[ep],task_description=description,
                    task_id=task_id,episode_id=ep,model=model,processor=processor,
                    cfg=cfg,run_dir=out,args=args)
                manifest.append(row)
                (out/'manifest.json').write_text(json.dumps(manifest,indent=2),encoding='utf-8')
                print(row['trajectory_id'],'labels=',row['labeled'],flush=True)
        finally:
            env.close()
    print('Data stored:',out)

if __name__=='__main__': main()
