"""One REAL LIBERO same-prefix replay grasp-probe collector, development only.

Reaches one pre-action decision state by restarting the same LIBERO initial state
and replaying the EXACT previously executed Occluded prefix. No opaque restore
hooks, Clean targets, object GT features, or scripted successful paths.

Fail-closed: matching warmup observation, original unoccluded anchor RGB,
original step-36 raw RGB, full MuJoCo flattened state and public robot state
are required on EVERY branch. This verifies replay consistency, not all unknown
hidden task/controller internals; see README before citing results.
"""
from __future__ import annotations
import argparse
import hashlib
import json
import os
from pathlib import Path
import socket
from types import SimpleNamespace

import numpy as np
from PIL import Image

from experiments.grasp_v2.aperture import aperture_from_qpos
from experiments.grasp_v2.collect_probes import collect_checkpoint, append_jsonl
from experiments.occlusion_eval.local_occlusion_support import read_plan, rectangle, rgb_hash



FROZEN_EP5_ANCHOR = 'd28d0d464b4e02a5aefd05b3f35302e1a26ffe163bf0a64d2f9ebcb4a79209e9'
FROZEN_EP5_INITIAL = 'c9c2f8022034e9a0705daab34a7f5bf9947ac939dfc5cf5a0ab493a377028a02'
FROZEN_EP5_ROI = [0.1, 0.48, 0.38, 0.64]


def discover_ep5_inputs(root):
    """Select EXACT frozen scenario, never silently choose a different mask."""
    root = Path(root)
    if not root.exists():
        raise FileNotFoundError(f'Experiment root missing: {root}')
    plans=[]
    for f in root.rglob('*.json'):
        if ('plan' not in f.name.lower() and 'occlusion' not in f.name.lower()
                and 'mask' not in f.name.lower()):
            continue
        try:
            d=json.loads(f.read_text(encoding='utf8'))
        except (ValueError, UnicodeError, OSError):
            continue
        if isinstance(d,dict) and (d.get('schema')=='local_occlusion_plan_v1'
                and d.get('anchor_rgb_sha256')==FROZEN_EP5_ANCHOR
                and d.get('roi')==FROZEN_EP5_ROI
                and d.get('start_step')==36 and d.get('end_step')==51):
            plans.append(f)
    plans=sorted(set(plans))
    if len(plans)!=1:
        raise RuntimeError(f'Expected exactly ONE original frozen mask plan under {root}; found {plans}. Supply explicit --frozen-plan if duplicates.')
    records=[]
    for f in root.rglob('task0_ep5_occluded.json'):
        try:
            d=json.loads(f.read_text(encoding='utf8'))
        except (ValueError,UnicodeError,OSError):
            continue
        if (d.get('initial_state_sha256')==FROZEN_EP5_INITIAL
            and d.get('masked_frames')==16
            and d.get('target_center_masked_frames')==16
            and len(d.get('frames',[]))>36
            and d['frames'][36]['active'] is True):
            records.append(f)
    records=sorted(set(records))
    if len(records)!=1:
        raise RuntimeError(f'Expected exactly ONE verified original Occluded run under {root}; found {records}. Supply explicit --baseline-json if duplicates.')
    return records[0], plans[0]

def _sha(a):
    return hashlib.sha256(np.asarray(a, dtype=np.float64).tobytes()).hexdigest()


def _require_real_baseline(path, plan):
    path = Path(path).resolve()
    d = json.loads(path.read_text(encoding='utf8'))
    if (d.get('task_id'), d.get('episode_id'), d.get('method')) != (
        plan['task_id'], plan['episode_id'], 'occluded'):
        raise ValueError('Expected the EXACT real Occluded Episode 5 record')
    if d.get('masked_frames') != 16 or d.get('target_center_masked_frames') != 16:
        raise ValueError('Baseline mask audit does not match frozen 16-step trial')
    if d.get('termination') != 'horizon' or not d.get('frames'):
        raise ValueError('Expected a finished real Occluded baseline with action trace')
    first = plan['start_step']
    frames = d['frames']
    if len(frames) <= first or [r['step'] for r in frames[:first+1]] != list(range(first+1)):
        raise ValueError('Baseline action steps are not continuous')
    prefix = []
    for frame in frames[:first]:
        action = np.asarray(frame['action_executed'], dtype=float)
        if action.shape != (7,) or not np.isfinite(action).all():
            raise ValueError('Invalid recorded pre-mask action')
        if frame.get('active'):
            raise ValueError('Unexpected occlusion in pre-action replay prefix')
        prefix.append(action)
    if frames[first].get('active') is not True:
        raise ValueError('Probe decision step not under occlusion')
    rawfile = path.parent/'images'/f"task{plan['task_id']}_ep{plan['episode_id']}_occluded"/f'{first:04d}_raw.png'
    if not rawfile.is_file():
        raise FileNotFoundError(f'Need actual ORIGINAL step-{first} raw image: {rawfile}')
    raw_ref = np.asarray(Image.open(rawfile).convert('RGB'), dtype=np.uint8)
    return d, prefix, rgb_hash(raw_ref), rawfile


class SamePrefixReplayAdapter:
    """Implements collect_probes adapter using real replay, never fabricated GT.

    Values derived from EEF poses are public, object GT is held solely for
    verifying restore consistency and POST-ACTION lift outcome scoring.
    """
    def __init__(self, *, env, initial_state, task_description, cfg, model, processor,
                 get_img, get_action, normalize, invert, dummy, seed,
                 plan, baseline, prefix, step_raw_hash, detector_roi,
                 lift_steps=12, lift_action_z=0.35, min_object_lift_m=0.035,
                 max_relative_drift_m=0.035, evidence_dir=None):
        self.env = env
        self.init = np.asarray(initial_state)
        self.description = task_description
        self.cfg, self.model, self.processor = cfg, model, processor
        self.get_img, self.get_action = get_img, get_action
        self.normalize, self.invert, self.dummy = normalize, invert, dummy
        self.seed = seed
        self.plan = plan
        self.baseline = baseline
        self.prefix = prefix
        self.step_raw_hash = step_raw_hash
        self.detector_roi = tuple(detector_roi)
        self.lift_steps = lift_steps
        self.lift_action_z = lift_action_z
        self.min_object_lift_m = min_object_lift_m
        self.max_relative_drift_m = max_relative_drift_m
        self.obs = None
        self.raw = None
        self.anchor = None
        self.previous_eef = None
        self.previous_action = np.asarray(prefix[-1])
        self._checkpoint = None
        self.label_details = []
        self.restore_count = 0
        self._reference_state = None
        self._reference_public = None
        self._eval_start_object = None
        self._eval_start_rel = None
        self.evidence_dir = None if evidence_dir is None else Path(evidence_dir)
        self.evidence_index = []
        self._branch_actions = []
        self._policy_images = []

    def _fingerprint(self):
        # Simulator truth used ONLY for integrity of restore, never as model input.
        from experiments.occlusion_eval.runtime_checks import resolve_sim
        sim = resolve_sim(self.env).get_state().flatten()
        eef = np.asarray(self.obs['robot0_eef_pos'], dtype=float)
        fingers = np.asarray(self.obs['robot0_gripper_qpos'], dtype=float)
        target = np.asarray(self.obs[f"{self.plan['target_object']}_pos"], dtype=float)
        return np.concatenate([np.asarray(sim,dtype=float), eef, fingers, target])

    def _replay_prefix(self):
        from experiments.robot.robot_utils import set_seed_everywhere
        from experiments.occlusion_eval.runtime_checks import observation_digest
        from experiments.occlusion_eval.motion2d import DarkPixel2D, Pixel2DConfig
        set_seed_everywhere(self.seed)
        self.env.seed(self.seed)
        self.env.reset()
        obs = self.env.set_init_state(self.init)
        for _ in range(self.baseline['probe_warmup']):
            obs, _, done, _ = self.env.step(self.dummy('openvla'))
            if done:
                raise RuntimeError('Terminated during warmup')
        if observation_digest(obs) != self.baseline['post_warmup_observation_sha256']:
            raise RuntimeError('Post-warmup state differs from recorded Occluded baseline')
        detector = DarkPixel2D(self.detector_roi, Pixel2DConfig())
        seen=[]; prior_eef=None
        for step, a in enumerate(self.prefix):
            raw = np.asarray(self.get_img(obs,224),dtype=np.uint8)
            if step == self.plan['anchor_step']:
                if rgb_hash(raw) != self.plan['anchor_rgb_sha256']:
                    raise RuntimeError('Step-35 RGB does NOT match frozen anchor: abort')
            if step in (self.plan['start_step']-2, self.plan['start_step']-1):
                target = detector.estimate(raw)
                if target is not None:
                    seen.append(target)
            if step == self.plan['start_step']-1:
                prior_eef=np.asarray(obs['robot0_eef_pos'],dtype=float).copy()
            obs, _, done, _ = self.env.step(a.tolist())
            if done:
                raise RuntimeError(f'Baseline replay terminated at step {step}')
        raw=np.asarray(self.get_img(obs,224),dtype=np.uint8)
        if rgb_hash(raw) != self.step_raw_hash:
            raise RuntimeError('Step-36 raw RGB differs from historical Occluded frame: abort')
        if len(seen) != 2 or np.linalg.norm(seen[0]-seen[1]) > 12.0:
            raise RuntimeError('No reliable two-frame visible target anchor: ABSTAIN; no training sample')
        self.obs=obs; self.raw=raw; self.anchor=seen[-1]
        self.previous_eef=prior_eef
        if prior_eef is None:
            raise RuntimeError('Missing prior EEF sample')
        return self._fingerprint()

    def capture_verified(self):
        fp = self._replay_prefix()
        self._reference_state = fp.copy()
        self._reference_public = self.public_preaction()
        checkpoint_id = hashlib.sha256(fp.tobytes()+self.raw.tobytes()).hexdigest()[:24]
        self._checkpoint={'checkpoint_id':checkpoint_id,'state_sha256':_sha(fp)}
        # Read only for POST-ACTION evaluation baseline, never in pre_action dict.
        self._eval_start_object=np.asarray(self.obs[f"{self.plan['target_object']}_pos"],dtype=float).copy()
        return dict(self._checkpoint)

    def restore_verified(self, snapshot):
        if self._checkpoint is None or snapshot != self._checkpoint:
            raise RuntimeError('Wrong or unverified checkpoint')
        fp = self._replay_prefix()
        if fp.shape != self._reference_state.shape or not np.allclose(
                fp,self._reference_state,atol=1e-8,rtol=0):
            raise RuntimeError('State replay mismatch; all branches must start identical')
        # Verify public input identity as well.
        if self.public_preaction() != self._reference_public:
            raise RuntimeError('Public preaction mismatch after replay')
        self.restore_count += 1
        self._branch_actions = []
        self._policy_images = []

    def public_preaction(self):
        from experiments.occlusion_eval.core import calibrated_camera_from_env
        from experiments.occlusion_eval.runtime_checks import resolve_sim
        camera=calibrated_camera_from_env(resolve_sim(self.env),224,native_size=256)
        eef=np.asarray(self.obs['robot0_eef_pos'],dtype=float)
        pixel=camera.project(eef)
        if pixel is None or not np.isfinite(pixel).all():
            raise RuntimeError('EEF projection invalid, cannot create public features')
        return {
            'target_px':list(map(float,self.anchor)),
            'eef_px':list(map(float,pixel)),
            'eef_xyz_m':eef.tolist(),
            'eef_vel_m':(eef-self.previous_eef).tolist(),
            'gripper_aperture_m':aperture_from_qpos(self.obs['robot0_gripper_qpos']),
            'last_seen_age':1,
            'target_confidence':1.0,  # two-frame verified gate, not calibrated probability
            'previous_action_xyz':self.previous_action[:3].tolist(),
            'proposed_action_xyz':[0.,0.,0.],  # filled by collect_checkpoint
        }

    def gripper_joint_qpos(self):
        return np.asarray(self.obs['robot0_gripper_qpos'],dtype=float).copy()

    def openvla_action(self):
        import torch
        raw=np.asarray(self.get_img(self.obs,224),dtype=np.uint8)
        r=rectangle(raw.shape,self.plan['roi'])
        if self.plan['start_step'] <= self.plan['end_step']:
            raw=raw.copy();raw[r]=np.array(self.plan['fill'],dtype=np.uint8)
        self._policy_images.append(raw.copy())
        with torch.inference_mode():
            original=np.asarray(self.get_action(
                self.cfg,self.model,{'full_image':raw,'state':np.zeros(8,dtype=float)},
                self.description,processor=self.processor),dtype=float)
        a=np.asarray(self.invert(self.normalize(original.copy(),binarize=True)),dtype=float)
        if a.shape!=(7,) or not np.isfinite(a).all():
            raise RuntimeError('OpenVLA returned invalid action')
        if torch.cuda.max_memory_reserved()/1024**3>15.0:
            raise RuntimeError('CUDA reserved memory exceeded 15 GiB')
        return a

    def step(self, executed_action_7d):
        a=np.asarray(executed_action_7d,dtype=float)
        if a.shape!=(7,) or not np.isfinite(a).all():
            raise ValueError('invalid executed action')
        self._branch_actions.append(a.tolist())
        self.obs,_,done,_=self.env.step(a.tolist())
        if done:
            raise RuntimeError('Probe terminated before evaluation')

    def stable_grasp_after_lift(self):
        """POST-EXECUTION evaluation ONLY; no object truth returned to scorer."""
        target_key=f"{self.plan['target_object']}_pos"
        end_records=[]
        for i in range(self.lift_steps):
            action=[0.,0.,self.lift_action_z,0.,0.,0.,1.]
            self.obs,_,done,_=self.env.step(action)
            bowl=np.asarray(self.obs[target_key],dtype=float).copy()
            eef=np.asarray(self.obs['robot0_eef_pos'],dtype=float).copy()
            end_records.append((bowl,eef))
            if done:
                break
        if len(end_records)<3:
            self.label_details.append({'status':'terminated_before_lift_completed',
                'success':False,'criterion':'measured_lift_proxy_not_full_task_success'})
            self.evidence_index.append({'branch':self.restore_count,
                'executed_pre_lift_actions':self._branch_actions,
                'terminated_before_lift_completed':True})
            return False
        obj_end,eef_end=end_records[-1]
        obj_at_lift_start=np.asarray(end_records[0][0],dtype=float)
        eef_at_lift_start=np.asarray(end_records[0][1],dtype=float)
        # Conservative stable-grasp PROXY: target actually lifted with the wrist,
        # relative position remains consistent over late part of the fixed lift.
        object_up=float(obj_end[2]-self._eval_start_object[2])
        eef_up=float(eef_end[2]-eef_at_lift_start[2])
        rel=np.stack([b-e for b,e in end_records[-3:]])
        late_relative_drift=float(max(np.linalg.norm(x-rel[0]) for x in rel[1:]))
        success=bool(object_up>=self.min_object_lift_m and eef_up>=0.02
                     and late_relative_drift<=self.max_relative_drift_m)
        self.label_details.append({'success':success,'object_lift_m':object_up,
            'eef_lift_m':eef_up,'late_relative_drift_m':late_relative_drift,
            'criterion':'measured_lift_proxy_not_full_task_success'})
        evidence = {'branch':self.restore_count,'executed_pre_lift_actions':self._branch_actions,
            'policy_image_sha256':[rgb_hash(a) for a in self._policy_images]}
        if self.evidence_dir is not None:
            self.evidence_dir.mkdir(parents=True,exist_ok=True)
            stem=f'candidate_{self.restore_count:02d}'
            paths=[]
            for i,arr in enumerate(self._policy_images):
                path=self.evidence_dir/f'{stem}_policy_input_{i}.png'
                Image.fromarray(arr).save(path)
                paths.append(str(path))
            final=self.evidence_dir/f'{stem}_after_lift_raw.png'
            Image.fromarray(np.asarray(self.get_img(self.obs,224),dtype=np.uint8)).save(final)
            evidence['image_paths']=paths+[str(final)]
            report=self.evidence_dir/f'{stem}_actions.json'
            report.write_text(json.dumps(evidence,indent=2,ensure_ascii=False),encoding='utf8')
        self.evidence_index.append(evidence)
        return success


def main(argv=None):
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--baseline-json',default=None)
    p.add_argument('--frozen-plan',default=None)
    p.add_argument('--ep5-root',default='experiments/real_collection_runs/dev10/closed_loop_dev15',
        help='Auto discover exact frozen plan + real Occluded run if explicit paths omitted')
    p.add_argument('--output',required=True)
    p.add_argument('--checkpoint',default='<OPENVLA_CHECKPOINT>')
    p.add_argument('--suite',default='libero_spatial')
    p.add_argument('--seed',type=int,default=7)
    p.add_argument('--warmup',type=int,default=10)
    p.add_argument('--xyz-step',type=float,default=.12)
    p.add_argument('--lift-steps',type=int,default=12)
    p.add_argument('--max-reserved-gib',type=float,default=15.)
    p.add_argument('--split',choices=['development','confirmation'],default='development')
    p.add_argument('--dry-run',action='store_true',help='Checks files ONLY; no GPU/model/env')
    a=p.parse_args(argv)
    if a.warmup!=10 or a.seed!=7 or a.suite!='libero_spatial' or not 0<a.xyz_step<=.25:
        p.error('This first adapter is deliberately scoped to the frozen Episode-5 baseline')
    if not 0<a.max_reserved_gib<=16 or a.lift_steps<3:
        p.error('Invalid GPU/label settings')
    if not a.frozen_plan or not a.baseline_json:
        found_baseline,found_plan=discover_ep5_inputs(a.ep5_root)
        a.baseline_json = a.baseline_json or str(found_baseline)
        a.frozen_plan = a.frozen_plan or str(found_plan)
    plan=read_plan(a.frozen_plan)
    if (plan['task_id'],plan['episode_id'],plan['start_step'],plan['end_step']) != (0,5,36,51):
        p.error('Frozen Episode-5 plan required; do not change occlusion')
    if (plan['roi']!=FROZEN_EP5_ROI or plan['anchor_rgb_sha256']!=FROZEN_EP5_ANCHOR
         or plan['fill']!=[128,128,128]):
        p.error('Original Episode 5 mask has changed: refusing')
    plan=dict(plan,target_object='akita_black_bowl_1')
    base,prefix,raw36_hash,rawfile=_require_real_baseline(a.baseline_json,plan)
    if base.get('initial_state_sha256') != FROZEN_EP5_INITIAL:
        raise RuntimeError('Original Episode 5 initial state hash changed')
    evidence_dir=Path(a.output).parent/(Path(a.output).stem+'_evidence')
    if Path(a.output).exists() or evidence_dir.exists():
        raise FileExistsError('Never overwrite existing probe output or evidence')
    print(json.dumps({'status':'inputs_verified','probe_step':36,
        'prefix_steps':len(prefix),'raw36_path':str(rawfile),
        'output':a.output,'dry_run':a.dry_run},ensure_ascii=False,indent=2),flush=True)
    if a.dry_run:
        return
    if any(x in socket.gethostname().lower() for x in ('login','gate','head')):
        raise RuntimeError('No GPU on login/gateway node')
    import torch
    if not torch.cuda.is_available():
        raise RuntimeError('GPU allocation required')
    if os.environ.get('VLA_OCCLUSION','clean')!='clean':
        raise RuntimeError('Double masking prohibited')
    os.environ['VLA_OCCLUSION']='clean'
    from libero.libero import benchmark
    from experiments.robot.libero.libero_utils import get_libero_env,get_libero_image,get_libero_dummy_action
    from experiments.robot.openvla_utils import get_processor
    from experiments.robot.robot_utils import (get_model,get_action,normalize_gripper_action,
       invert_gripper_action,set_seed_everywhere)
    from experiments.occlusion_eval.runtime_checks import configure_tensorflow_cpu,pair_seed
    configure_tensorflow_cpu()
    seed=pair_seed(a.seed,0,5)
    set_seed_everywhere(seed)
    cfg=SimpleNamespace(model_family='openvla',pretrained_checkpoint=a.checkpoint,
       load_in_4bit=True,load_in_8bit=False,center_crop=True,unnorm_key=a.suite)
    model=get_model(cfg);model.eval()
    if not getattr(model,'is_loaded_in_4bit',False):
        raise RuntimeError('Expected original 4-bit OpenVLA')
    if cfg.unnorm_key not in model.norm_stats and cfg.unnorm_key+'_no_noops' in model.norm_stats:
        cfg.unnorm_key+='_no_noops'
    if cfg.unnorm_key not in model.norm_stats:
        raise RuntimeError('Missing unnorm stats for LIBERO')
    if torch.cuda.memory_reserved()/1024**3>a.max_reserved_gib:
        raise RuntimeError('4bit model exceeds VRAM limit')
    processor=get_processor(cfg)
    suite=benchmark.get_benchmark_dict()[a.suite]()
    initial_states=suite.get_task_init_states(0)
    state=initial_states[5]
    if hashlib.sha256(np.asarray(state).tobytes()).hexdigest()!=base['initial_state_sha256']:
        raise RuntimeError('LIBERO initial state differs from recorded baseline')
    task=suite.get_task(0)
    env,description=get_libero_env(task,'openvla',resolution=256)
    try:
        adapter=SamePrefixReplayAdapter(env=env,initial_state=state,
            task_description=description,cfg=cfg,model=model,processor=processor,
            get_img=get_libero_image,get_action=get_action,
            normalize=normalize_gripper_action,invert=invert_gripper_action,
            dummy=get_libero_dummy_action,seed=seed,plan=plan,
            baseline={**base,'probe_warmup':a.warmup},
            prefix=prefix,step_raw_hash=raw36_hash,detector_roi=plan['roi'],
            lift_steps=a.lift_steps,evidence_dir=evidence_dir)
        rows=collect_checkpoint(adapter,initial_state_id=base['initial_state_sha256'],
            episode_id=5,seed=seed,split=a.split,xyz_step=a.xyz_step)
        for i,row in enumerate(rows):
            row['label_diagnostic']=adapter.label_details[i]
            row['probe_evidence']=adapter.evidence_index[i]
            row['method']='same_prefix_reset_replay_v1'
            row['probe_step']=36
            row['reference_baseline_sha256']=hashlib.sha256(Path(a.baseline_json).read_bytes()).hexdigest()
            row['mask_plan_sha256']=hashlib.sha256(Path(a.frozen_plan).read_bytes()).hexdigest()
        report=append_jsonl(rows,a.output)
        report.update({'restore_verified_count':adapter.restore_count,
            'labels_positive':sum(x['post_action_label']['stable_grasp_after_lift'] for x in rows),
            'labels_negative':len(rows)-sum(x['post_action_label']['stable_grasp_after_lift'] for x in rows),
            'warning':'Development only; sample-label lift proxy; no proof of full task success or generalization'})
        Path(a.output).with_suffix('.report.json').write_text(
            json.dumps(report,indent=2,ensure_ascii=False),encoding='utf8')
        print(json.dumps(report,indent=2,ensure_ascii=False),flush=True)
    finally:
        env.close()

if __name__=='__main__':
    main()
