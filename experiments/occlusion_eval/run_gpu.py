"""Research-oriented *evaluation harness* for frozen OpenVLA/LIBERO.

This script has NOT been run against the user's HKU GPU environment by its author.
Deploy inside ~/openvla_eval and run only in an allocated compute node.

It deliberately calls the original robot_utils.get_action() path, preserving
center-crop, 4-bit quantization, attention-mask fix, gripper normalization,
unnorm key, and model preprocessing. It does not modify upstream source.
"""
import argparse
import csv
from dataclasses import asdict
from datetime import datetime, timezone
import hashlib
import json
import os
from pathlib import Path
import platform
import subprocess
import sys
import time
from types import SimpleNamespace

import numpy as np

# python experiments/occlusion_eval/run_gpu.py from repo root
ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from experiments.contact_cpu.contact_gate import Config
from experiments.occlusion_eval.core import (
    RGBPlaneConfig, RGBPlaneEstimator, TrialController,
    calibrated_camera_from_env,
)
from experiments.physics_occlusion.pipeline import OcclusionSchedule
# LOCAL_OCCLUSION_PATCH_V1
from experiments.occlusion_eval.local_occlusion_support import apply_plan_args, validate_anchor, PixelAudit
from experiments.occlusion_eval.eval_state_trace import make_eval_state_row, write_eval_state_trace
from experiments.ai1_protocol.bridge import AI1_METHODS, policy_method, load_ai1_config, FixedHold, EpisodeRecorder
from experiments.occlusion_eval.runtime_checks import (resolve_sim, success_check,
    pair_seed, method_order, observation_digest, runtime_metadata, configure_tensorflow_cpu)


def json_safe(value):
    if isinstance(value, np.ndarray):
        return value.tolist()
    if isinstance(value, np.generic):
        return value.item()
    if isinstance(value, Path):
        return str(value)
    raise TypeError(type(value).__name__)


def parse_ids(raw):
    ids = [int(v.strip()) for v in raw.split(',') if v.strip()]
    if not ids or any(i < 0 for i in ids) or len(ids) != len(set(ids)):
        raise ValueError('Provide unique, nonnegative comma-separated ids')
    return ids


def parse_args(argv=None):
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--checkpoint', default='<OPENVLA_CHECKPOINT>')
    p.add_argument('--suite', default='libero_spatial')
    p.add_argument('--task-ids', default='0')
    p.add_argument('--episode-ids', default='0,1')
    p.add_argument('--methods', default='clean,occluded,m0,m1,m2,cue_control')
    p.add_argument('--pose-source', choices=['oracle_visible','rgb_plane'], required=True)
    p.add_argument('--allow-oracle-upper-bound', action='store_true',
                   help='Required for privileged, non-deployable development run')
    p.add_argument('--target-object', default='akita_black_bowl_1')
    p.add_argument('--task-specs', default=None,
                   help='JSON mapping task id to target_object/mask/start_step/end_step/optional RGB calibration')
    p.add_argument('--object-quat-order', choices=['xyzw','wxyz'], default='xyzw')
    p.add_argument('--gate-config', default=None)
    p.add_argument('--gate-status', choices=['xml_reference','independently_calibrated'],
                   default='xml_reference')
    p.add_argument('--start-step', type=int, default=35)
    p.add_argument('--end-step', type=int, default=85)
    p.add_argument('--mask', default='0.32,0.32,0.70,0.70',
                   help='Fixed normalized x0,y0,x1,y1; NOT target-tracked')
    p.add_argument('--rgb-roi', default=None,
                   help='Required for rgb_plane. Visible-frame fixed ROI x0,y0,x1,y1.')
    p.add_argument('--object-center-height', type=float, default=None,
                   help='Required independently measured center height for rgb_plane')
    p.add_argument('--rgb-dark-threshold', type=int, default=75)
    p.add_argument('--motion-2d-roi', default=None,
                   help='Visible RGB ROI (normalized) for opt-in m1_2d/m2_2d; must be audited')
    p.add_argument('--max-steps', type=int, default=0, help='0 = official per-suite horizon')
    p.add_argument('--warmup', type=int, default=10)
    p.add_argument('--seed', type=int, default=7)
    p.add_argument('--max-reserved-gib', type=float, default=15.0,
                   help='Fail fast if peak torch CUDA reserved memory exceeds this budget')
    p.add_argument('--save-every', type=int, default=20, help='PNG stride; 0 disables except first mask')
    p.add_argument('--output-root', default='experiments/occlusion_eval/results')
    p.add_argument('--center-crop', action=argparse.BooleanOptionalAction, default=True)
    p.add_argument('--fail-fast', action='store_true')
    p.add_argument('--local-mask-plan', default=None, help='Frozen pre-occlusion local mask JSON')
    p.add_argument('--ai1-config', default=None, help='Optional AI1 baseline logging and bounded Hold')
    p.add_argument('--ai1-capture-step', type=int, default=None, help='Clean-only development RGB anchor capture')
    p.add_argument('--paired-capture-dev', action='store_true', help='Development-only public checkpoint capture; no additional policy calls')
    p.add_argument('--unified-estimator', choices=['memory','rigid','rule_contact','learned_contact'], default=None)
    p.add_argument('--unified-intervention', choices=['none','visual_cue','bounded_gripper_confirmation'], default='none')
    p.add_argument('--unified-weights', default=None)
    p.add_argument('--unified-history', type=int, default=6)
    p.add_argument('--unified-rule-closed', type=float, default=None)
    p.add_argument('--unified-rule-open', type=float, default=None)
    p.add_argument('--unified-contact-threshold', type=float, default=0.5)
    p.add_argument('--progress-every', type=int, default=20)
    p.add_argument('--no-eval-state-trace', action='store_true',
                   help='Disable writing the separate eval-only state JSONL (enabled by default)')
    a = p.parse_args(argv)
    a.ai1_config_data = load_ai1_config(a.ai1_config)
    a.task_ids = parse_ids(a.task_ids)
    a.episode_ids = parse_ids(a.episode_ids)
    a.methods = [m.strip() for m in a.methods.split(',')]
    a.motion_2d_roi = parse_rect(a.motion_2d_roi) if a.motion_2d_roi else None
    if 'integrated' in a.methods and not a.ai1_config_data: p.error('integrated requires --ai1-config')
    if a.unified_estimator and 'integrated' not in a.methods: p.error('unified estimator configured without integrated method')
    if a.unified_estimator == 'learned_contact' and not a.unified_weights: p.error('learned requires --unified-weights')
    if a.unified_estimator == 'rule_contact' and (a.unified_rule_closed is None or a.unified_rule_open is None): p.error('rule_contact requires calibrated thresholds')
    if a.paired_capture_dev and a.ai1_config_data and a.ai1_config_data['study_label']!='DEVELOPMENT_ONLY': p.error('paired checkpoints forbidden on confirmation set')
    if a.paired_capture_dev and a.pose_source!='rgb_plane': p.error('public capture requires rgb_plane')
    if a.progress_every<1: p.error('progress-every must be positive')
    if any(m in ('m0_2d','m1_2d','m2_2d','m2_2d_hold') for m in a.methods) and a.motion_2d_roi is None:
        p.error('m1_2d/m2_2d require explicit --motion-2d-roi')
    if any(m in ('m0_2d','m1_2d','m2_2d','m2_2d_hold') for m in a.methods) and a.pose_source != 'rgb_plane':
        p.error('m1_2d/m2_2d are visible RGB-only experiments, not oracle-visible')
    allowed = TrialController.MODES + (AI1_METHODS if a.ai1_config_data else ()) + (('integrated',) if a.unified_estimator else ())
    if len(a.methods) != len(set(a.methods)) or any(m not in allowed for m in a.methods):
        p.error(f'Expected unique methods from {allowed}')
    if a.ai1_capture_step is not None:
        if (a.ai1_capture_step < 0 or a.methods != ['clean']
                or a.local_mask_plan is not None or a.ai1_config):
            p.error('Anchor capture requires one clean development run without local-mask-plan / ai1-config')
        if len(a.task_ids) != 1 or len(a.episode_ids) != 1:
            p.error('Anchor capture permits exactly one task and one episode per run')
    a.mask = parse_rect(a.mask)
    a.specs = {}
    if a.task_specs:
        with open(a.task_specs,encoding='utf-8') as f:
            a.specs = json.load(f)
        if not isinstance(a.specs,dict):
            p.error('task-specs must be a JSON object keyed by task id')
        for tid in a.task_ids:
            if str(tid) not in a.specs:
                p.error(f'task-specs missing task {tid}')
    elif len(a.task_ids)>1 and any(policy_method(m) in ('m0','m1','m2') for m in a.methods):
        p.error('Multiple tasks with compensation require explicit --task-specs')
    if a.start_step < 1 or a.end_step < a.start_step or a.warmup < 0 or a.max_steps < 0:
        p.error('Invalid step range or horizon')
    if a.pose_source == 'oracle_visible' and not a.allow_oracle_upper_bound:
        p.error('Explicit --allow-oracle-upper-bound required for privileged visible prefix')
    if a.pose_source == 'rgb_plane':
        if a.rgb_roi is not None: a.rgb_roi=parse_rect(a.rgb_roi)
    if any(policy_method(m) in ('m2','m2_2d','m2_2d_hold') for m in a.methods) and not a.gate_config:
        p.error('M2 requires explicit --gate-config; no silent guessed contact calibration')
    if not (0 < a.max_reserved_gib <= 16):
        p.error('max-reserved-gib must be in (0,16]')
    if a.save_every < 0:
        p.error('save-every must be >=0')
    for tid in a.task_ids:
        spec = a.specs.get(str(tid),{})
        if not isinstance(spec,dict):
            p.error(f'task {tid} spec must be an object')
        mask = spec.get('mask',a.mask)
        if isinstance(mask,str): mask=parse_rect(mask)
        else: mask=parse_rect(','.join(map(str,mask)))
        start=int(spec.get('start_step',a.start_step))
        end=int(spec.get('end_step',a.end_step))
        if start<1 or end<start: p.error(f'Invalid occlusion window for task {tid}')
        roi = spec.get('rgb_roi',a.rgb_roi)
        if isinstance(roi,(tuple,list)): roi=parse_rect(','.join(map(str,roi)))
        elif isinstance(roi, str): roi=parse_rect(roi)
        if a.pose_source=='rgb_plane' and roi is None and any(policy_method(m) in ('m0','m1','m2') for m in a.methods):
            p.error(f'rgb_plane task {tid} needs rgb_roi')
        height=spec.get('object_center_height',a.object_center_height)
        if a.pose_source=='rgb_plane' and height is None and any(policy_method(m) in ('m0','m1','m2') for m in a.methods):
            p.error(f'rgb_plane task {tid} needs object_center_height')
        a.specs[str(tid)]={'target_object':spec.get('target_object',a.target_object),
                           'mask':mask,'start_step':start,'end_step':end,
                           'rgb_roi':roi,'object_center_height':height}
    if a.ai1_capture_step is not None:
        # Separate development-only capture; never included in controlled experiment pairs.
        a.local_plan = {'fill':[128,128,128], 'anchor_step':-1}
    else:
        apply_plan_args(a)
    return a


def parse_rect(raw):
    vals = tuple(float(x) for x in raw.split(','))
    if len(vals) != 4 or not (0 <= vals[0] < vals[2] <= 1 and 0 <= vals[1] < vals[3] <= 1):
        raise ValueError('ROI requires x0,y0,x1,y1 normalized, nonempty')
    return vals


def load_gate_config(file):
    if file is None:
        return Config()
    with open(file, encoding='utf-8') as f:
        data = json.load(f)
    return Config(**data)


def manifest(run_dir, args, cfg, model=None, policy_cfg=None):
    try:
        commit = subprocess.check_output(['git','rev-parse','HEAD'], cwd=ROOT, stderr=subprocess.DEVNULL,
                                         text=True, timeout=3).strip()
    except (OSError, subprocess.SubprocessError):
        commit = 'unavailable'
    result = {
        'created_utc': datetime.now(timezone.utc).isoformat(),
        'code_version': 'occlusion_eval_review_patch_1', 'git_commit': commit,
        'settings': vars(args), 'gate_config': asdict(cfg),
        'study_type': ('DEVELOPMENT_UPPER_BOUND_PRIVILEGED_VISIBLE_PREFIX'
                       if args.pose_source == 'oracle_visible' else
                       'EXPERIMENTAL_VISIBLE_RGB_PLANE_PROXY_NOT_6DOF'),
        'gate_calibration_status': args.gate_status,
        'warnings': [
            'No GPU results were created by code delivery. Results exist only if this script was executed.',
            'Fixed ROI occlusion may not cover target: audit target coverage offline.',
            'Do not use oracle_visible results as deployment evidence.',
            'XML joint limits are not independent contact calibration.',
            'RGB plane detector is not 6-DoF and needs independent visual validation.',
            'Task success requires environment check_success(); no done fallback.',
        ],
        'environment': runtime_metadata(ROOT),
    }
    if model is not None:
        result['loaded_model'] = {
            'class':type(model).__module__+'.'+type(model).__name__,
            'config_commit_hash':getattr(model.config, '_commit_hash', None),
            'attention_implementation':getattr(model.config, '_attn_implementation', None),
            'is_loaded_in_4bit':bool(getattr(model, 'is_loaded_in_4bit', False)),
            'unnorm_key':policy_cfg.unnorm_key,
            'action_norm_stats_sha256':hashlib.sha256(json.dumps(
                model.norm_stats[policy_cfg.unnorm_key], sort_keys=True, default=json_safe).encode()).hexdigest(),
        }
    (run_dir/'manifest.json').write_text(json.dumps(result, indent=2, default=json_safe), encoding='utf-8')


def max_steps_for_suite(suite):
    return dict(libero_spatial=220, libero_object=280, libero_goal=300,
                libero_10=520, libero_90=400)[suite]


def should_save(step, active, schedule, args):
    return (step == schedule.start_step or (active and args.save_every>0 and
                                            (step-schedule.start_step)%args.save_every == 0))


def save_stages(base, step, output):
    from PIL import Image
    base.mkdir(parents=True, exist_ok=True)
    for label, image in [('raw',output.raw),('occluded',output.occluded),
                         ('action_input',output.action_image)]:
        Image.fromarray(image).save(base / f'{step:04d}_{label}.png')
    (base/f'{step:04d}_metadata.json').write_text(
        json.dumps(output.metadata, indent=2, default=json_safe), encoding='utf-8')


def save_episode_video(run_dir, task_id, episode_idx, method, success, frames, fps=30):
    """Write raw RGB and actual policy input as a side-by-side MP4."""
    import imageio.v2 as imageio
    from PIL import Image

    if not frames:
        raise ValueError('Cannot write an episode video without frames')
    path = (run_dir /
            f'task{task_id}_ep{episode_idx}_{method}_success-{str(bool(success)).lower()}.mp4')
    first_raw, first_input = frames[0]
    height, width = first_raw.shape[:2]
    writer = imageio.get_writer(path, fps=fps, codec='libx264', macro_block_size=1)
    try:
        for raw, policy_input in frames:
            raw = np.asarray(raw, dtype=np.uint8)
            policy_input = np.asarray(policy_input, dtype=np.uint8)
            if raw.shape[:2] != (height, width):
                raise ValueError('Episode raw frames have inconsistent dimensions')
            if policy_input.shape[:2] != (height, width):
                policy_input = np.asarray(
                    Image.fromarray(policy_input).resize((width, height)), dtype=np.uint8)
            writer.append_data(np.concatenate((raw, policy_input), axis=1))
    finally:
        writer.close()
    return path


class ProcessorAudit:
    """A forwarding proxy: records actual PIL image seen by processor once.

    Original OpenVLA center-crop occurs inside get_vla_action before processor.
    This spy therefore measures the *real* processor image, not a guessed image.
    """
    def __init__(self, original):
        self.original = original
        self.audit_path = None
        self.last_sha256 = None
    def __call__(self, prompt, image, *args, **kwargs):
        if self.audit_path is not None:
            from PIL import Image
            arr = np.asarray(image.convert('RGB'), dtype=np.uint8)
            self.last_sha256 = hashlib.sha256(arr.tobytes()).hexdigest()
            self.audit_path.parent.mkdir(parents=True, exist_ok=True)
            Image.fromarray(arr).save(self.audit_path)
            self.audit_path = None
        return self.original(prompt, image, *args, **kwargs)
    def __getattr__(self, name):
        return getattr(self.original, name)


def use_gpu_allocation_or_die():
    """Fail fast on gateway; GPU allocation policy still determined by HKU."""
    import torch
    import socket
    host = socket.gethostname().lower()
    if any(s in host for s in ('gate','login','head')):
        raise RuntimeError(f'Refusing GPU evaluation on login/gateway host: {host}')
    if not torch.cuda.is_available():
        raise RuntimeError('CUDA unavailable; connect to an HKU-allocated compute node')
    props = torch.cuda.get_device_properties(0)
    print(f'GPU: {props.name}, {props.total_memory / 1024**3:.2f} GiB')
    # A visible device alone is NOT proof of an authorized allocation.
    return torch


def run_one_episode(*, env, initial_state, task_description, task_id, episode_idx,
                    method, model, processor, cfg, args, camera, gate_cfg, run_dir,
                    get_img, get_action, normalize, invert, dummy, spec):
    """Reuse original LIBERO action path. One separate environment reset per method."""
    env.reset()
    obs = env.set_init_state(initial_state)
    check_success, success_source = success_check(env)
    if camera is None:
        camera = calibrated_camera_from_env(resolve_sim(env), 224, native_size=256)
    ai1_data = getattr(args, 'ai1_config_data', None)
    underlying_method = policy_method(method)
    ai1_hold = FixedHold(**ai1_data['hold']) if ai1_data and method in AI1_METHODS else None
    ai1_recorder = None
    if ai1_data is not None:
        log_config = {**ai1_data, 'checkpoint':args.checkpoint,
                      'pose_source':args.pose_source, 'max_steps':args.max_steps,
                      'warmup':args.warmup, 'center_crop':args.center_crop,
                      'local_mask_roi':list(spec['mask']),
                      'mask_steps':[spec['start_step'],spec['end_step']]}
        ai1_recorder = EpisodeRecorder(
            run_dir,task_id=task_id,episode_id=episode_idx,method=method,
            seed=pair_seed(args.seed,task_id,episode_idx),
            initial_state_id=hashlib.sha256(np.asarray(initial_state).tobytes()).hexdigest(),
            config=log_config, code_version='run_gpu_sha256:'+hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
            plan_sha256=hashlib.sha256(Path(args.local_mask_plan).read_bytes()).hexdigest())
    estimator = None
    base_method = {'occluded_hold':'occluded','m2_2d_hold':'m2_2d','integrated':'occluded'}.get(underlying_method,underlying_method)
    pixel_estimator = None
    if underlying_method in ('m0_2d','m1_2d','m2_2d','m2_2d_hold'):
        from experiments.occlusion_eval.motion2d import DarkPixel2D, Pixel2DConfig
        pixel_estimator = DarkPixel2D(args.motion_2d_roi, Pixel2DConfig(
            dark_threshold=args.rgb_dark_threshold))
    if args.pose_source == 'rgb_plane' and underlying_method in ('m0','m1','m2'):
        estimator = RGBPlaneEstimator(camera, RGBPlaneConfig(
            spec['rgb_roi'], spec['object_center_height'],
            dark_threshold=args.rgb_dark_threshold))
    schedule = OcclusionSchedule(spec['start_step'], spec['end_step'], *spec['mask'], fill=tuple(args.local_plan['fill']))
    controller = TrialController(base_method, camera, schedule, gate_cfg, estimator,
                                 object_quat_order=args.object_quat_order,
                                 allow_oracle=(args.pose_source == 'oracle_visible'),
                                 target_name=spec['target_object'],
                                  pixel_estimator=pixel_estimator)
    unified=None
    if method=='integrated':
        from experiments.unified_eval.online import OnlineEstimator
        roi=spec.get('rgb_roi') or args.motion_2d_roi
        if roi is None: raise ValueError('integrated requires calibrated --rgb-roi')
        unified=OnlineEstimator(method=args.unified_estimator,intervention=args.unified_intervention,
            history=args.unified_history,weights=args.unified_weights,roi=roi,camera=camera,
            threshold=args.unified_contact_threshold,closed_aperture=args.unified_rule_closed,
            open_aperture=args.unified_rule_open,hold_config=ai1_data['hold'],
            task_id=task_id,episode_id=episode_idx,
            initial_state_id=hashlib.sha256(np.asarray(initial_state).tobytes()).hexdigest(),
            seed=pair_seed(args.seed,task_id,episode_idx),
            trajectory_id=f'integrated_task{task_id}_ep{episode_idx}_{args.unified_estimator}_{args.unified_intervention}',
            code_version='run_gpu_sha256:'+hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
            suite=args.suite,checkpoint=args.checkpoint,dark_threshold=args.rgb_dark_threshold)
    for _ in range(args.warmup):
        obs, _, warm_done, _ = env.step(dummy('openvla'))
        if warm_done or bool(check_success()):
            raise RuntimeError('Episode terminated/succeeded during warmup; invalid policy trial')
    warmup_digest = observation_digest(obs)
    horizon = args.max_steps or max_steps_for_suite(args.suite)
    frames = []
    paired_public_history=[]
    paired_capture_dir=run_dir/'paired_public_preaction'
    executed_previous=None
    held_gripper_command = None
    video_frames = []
    eval_state_rows = []  # Separate diagnostics; NEVER passed to model or controller
    success = False
    terminal = 'horizon'
    max_mem_mb = None
    import torch
    torch.cuda.reset_peak_memory_stats()
    t0 = time.monotonic()
    for step in range(horizon):
        raw = np.asarray(get_img(obs, 224), dtype=np.uint8)
        if getattr(args,'ai1_capture_step',None) is not None:
            if step == args.ai1_capture_step:
                from PIL import Image
                capdir=run_dir/'ai1_anchor_capture'
                capdir.mkdir(parents=True,exist_ok=True)
                capfile=capdir/f'task{task_id}_ep{episode_idx}_step{step:04d}_raw.png'
                Image.fromarray(raw).save(capfile)
                capfile.with_suffix('.json').write_text(json.dumps({
                    'record_phase':'PRE_ACTION_OBSERVATION','task_id':task_id,
                    'episode_id':episode_idx,'step':step,
                    'initial_state_id':hashlib.sha256(np.asarray(initial_state).tobytes()).hexdigest(),
                    'seed':pair_seed(args.seed,task_id,episode_idx),
                    'raw_rgb_sha256':hashlib.sha256(raw.tobytes()).hexdigest(),
                    'study_label':'DEVELOPMENT_ONLY_ANCHOR_CAPTURE',
                    'not_final_test':True},indent=2),encoding='utf-8')
        else:
            validate_anchor(raw, step, args.local_plan)
        # The *one and only* place where privileged visible prefix can be read.
        # During occlusion there is never an object pose passed to controller.
        upper_obs = None
        if (args.pose_source == 'oracle_visible' and step < schedule.start_step
                and method in ('m0','m1','m2')):
            upper_obs = {f"{spec['target_object']}_{suffix}": obs[f"{spec['target_object']}_{suffix}"]
                         for suffix in ('pos', 'quat')}
        output = controller.process(
            step=int(step), raw=raw,
            eef_pos=obs['robot0_eef_pos'], eef_quat=obs['robot0_eef_quat'],
            gripper_qpos=obs['robot0_gripper_qpos'],
            privileged_visible_observation=upper_obs)
        # Adapter only: A2 receives PRE action public RGB + measured robot state; no GT/env.
        unified_est=None
        if unified is not None:
            from experiments.unified_eval.contracts import Observation
            pub_obs=Observation(raw.copy(),tuple(map(float,obs['robot0_eef_pos'])),
                tuple(map(float,obs['robot0_gripper_qpos'])),
                None if executed_previous is None else tuple(map(float,executed_previous)),int(step))
            unified_est=unified.update(pub_obs,occluded=schedule.active(step))
            modified,applied=unified.render(output.action_image,unified_est,active=schedule.active(step))
            output.action_image=modified
            output.metadata.update({'unified_estimator':args.unified_estimator,
                'unified_intervention':args.unified_intervention,
                'unified_estimate_pixel_px':unified_est.projected_xy_px,
                'unified_contact_probability':unified_est.contact_prob,
                'unified_valid':unified_est.valid,'unified_cue_applied':applied})
        if getattr(args,'paired_capture_dev',False):
            from experiments.paired_decision_eval.capture import capture_preaction
            from experiments.paired_decision_eval.io_utils import image_digest
            from experiments.occlusion_eval.motion2d import DarkPixel2D, Pixel2DConfig
            capture_roi=spec.get('rgb_roi') or args.motion_2d_roi
            if capture_roi is None:
                raise ValueError('A3 public checkpoints require calibrated fixed RGB ROI')
            visible_pixel=(None if schedule.active(step) else
                DarkPixel2D(tuple(capture_roi),Pixel2DConfig(
                    dark_threshold=args.rgb_dark_threshold)).estimate(raw))
            ep_px=camera.project(np.asarray(obs['robot0_eef_pos']))
            rb={'eef_pos':np.asarray(obs['robot0_eef_pos']).tolist(),
                'eef_quat':np.asarray(obs['robot0_eef_quat']).tolist(),
                'gripper_qpos':np.asarray(obs['robot0_gripper_qpos']).tolist(),
                'eef_pixel':None if ep_px is None else np.asarray(ep_px).tolist(),
                'target_pixel_visible':None if visible_pixel is None else np.asarray(visible_pixel).tolist(),
                'previous_action':None if executed_previous is None else np.asarray(executed_previous).tolist(),
                'occlusion_active':bool(schedule.active(step))}
            cap=capture_preaction(image=output.occluded,robot_state=rb,
                history=paired_public_history[-64:],task_text=task_description,
                task_id=task_id,episode_id=str(episode_idx),
                initial_state_id=hashlib.sha256(np.asarray(initial_state).tobytes()).hexdigest(),
                step=int(step),seed=pair_seed(args.seed,task_id,episode_idx),
                method=method,config={'suite':args.suite,'checkpoint':args.checkpoint,
                    'pose_source':args.pose_source,'gate_config':args.gate_config},
                code_version='run_gpu_sha256:'+hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
                destination=paired_capture_dir)
            paired_public_history.append({'step':int(step),'robot_state':rb,
                'rgb_sha256':image_digest(output.occluded),
                'rgb_path':cap['public']['rgb_path']})
        # Use ORIGINAL get_action -> get_vla_action (correct center crop,
        # 4-bit model, original prompt, attention mask and unnorm key).
        output.metadata['method'] = method
        selected_img = output.action_image
        video_frames.append((raw.copy(), np.asarray(selected_img, dtype=np.uint8).copy()))
        frame_dir = (run_dir/'images'/f'task{task_id}_ep{episode_idx}_{method}')
        save = should_save(step, schedule.active(step), schedule, args)
        processor.audit_path = (frame_dir/f'{step:04d}_processor.png') if save else None
        if save:
            save_stages(frame_dir, step, output)
        observation = {'full_image':selected_img,
                       'state':np.zeros(8, dtype=float)}  # Original OpenVLA only uses full_image
        with torch.inference_mode():
            action_raw = np.asarray(get_action(cfg, model, observation, task_description,
                                               processor=processor), dtype=float)
        if action_raw.shape != (7,) or not np.isfinite(action_raw).all():
            raise ValueError(f'Invalid OpenVLA action {action_raw}')
        executed = np.asarray(invert(normalize(action_raw.copy(), binarize=True)), dtype=float)
        if executed.shape != (7,) or not np.isfinite(executed).all():
            raise ValueError('Invalid executed action')
        proposed_executed = executed.copy()
        gripper_override = False
        if step == schedule.start_step - 1:
            held_gripper_command = float(executed[-1])
        if method in ('occluded_hold', 'm2_2d_hold') and schedule.active(step):
            if held_gripper_command is None:
                raise RuntimeError('Diagnostic hold lacks a pre-occlusion command')
            executed[-1] = held_gripper_command
            gripper_override = bool(executed[-1] != proposed_executed[-1])
        # Legacy in-loop probes disabled: run A3 probes independently on saved checkpoints.
        if unified is not None and args.unified_intervention=='bounded_gripper_confirmation':
            executed,evt=unified.action(executed,unified_est,step=step,active=schedule.active(step))
            output.metadata['unified_action_event']=evt
        ai1_hold_event = None
        if ai1_hold is not None:
            executed,ai1_hold_event=ai1_hold.apply(
                step=step,active=schedule.active(step),requested=executed)
        if ai1_recorder is not None and save:
            pixelpath=frame_dir/f'{step:04d}_processor.pixel_values.npz'
            if not pixelpath.exists():
                raise RuntimeError('AI1: actual processor pixel_values missing (no PNG-only success)')
        if ai1_recorder is not None:
            ai1_recorder.before(
                step=step,occlusion_active=schedule.active(step),
                eef_pos=obs['robot0_eef_pos'],eef_quat=obs['robot0_eef_quat'],
                gripper_qpos=obs['robot0_gripper_qpos'],policy_image=selected_img,
                raw_policy_action=action_raw,proposed_executed_action=proposed_executed,
                executed_action=executed,hold_event=ai1_hold_event,
                processor_npz_path=(str((frame_dir/f'{step:04d}_processor.pixel_values.npz').relative_to(run_dir)) if save else None))
        # Evaluation-only center coverage: after policy action has been chosen.
        # This metadata is NOT passed to controller, processor, or policy.
        center_covered = None
        if schedule.active(step) and f"{spec['target_object']}_pos" in obs:
            # A simulator-truth audit label, not a feature: keep only boolean.
            xy = camera.project(np.asarray(obs[f"{spec['target_object']}_pos"], dtype=float))
            if xy is not None:
                x0,y0,x1,y1 = spec['mask']
                center_covered = bool(np.floor(x0*raw.shape[1]) <= xy[0] < np.ceil(x1*raw.shape[1])
                                      and np.floor(y0*raw.shape[0]) <= xy[1] < np.ceil(y1*raw.shape[0]))
        if torch.cuda.max_memory_reserved()/1024**3 > args.max_reserved_gib:
            raise RuntimeError('Exceeded configured CUDA reserved-memory budget')
        # Evaluation-only pre-action state snapshot, AFTER policy action is fixed.
        # Simulator truth remains isolated and never touches model/controller.
        if not getattr(args, 'no_eval_state_trace', False):
            eval_state_rows.append(make_eval_state_row(
                obs=obs, step=step, target_name=spec['target_object'],
                camera=camera, occlusion_active=schedule.active(step),
                executed_action=executed, action_raw=action_raw, mask=spec['mask']))
        executed_previous=np.asarray(executed).copy()
        obs, reward, done, info = env.step(executed.tolist())
        success = bool(check_success())
        if ai1_recorder is not None:
            ai1_recorder.after(
                step=step,eef_pos=obs['robot0_eef_pos'],eef_quat=obs['robot0_eef_quat'],
                gripper_qpos=obs['robot0_gripper_qpos'])
            ai1_recorder.evaluator_after(
                step=step,target_pos_gt=obs.get(f"{spec['target_object']}_pos"),
                target_quat_gt=obs.get(f"{spec['target_object']}_quat"),done=done,success=success)
        source = success_source
        frames.append({'step':step, 'active':schedule.active(step),
                       'method':method, 'action_raw':action_raw.tolist(),
                       'action_policy_before_override':proposed_executed.tolist(),
                       'diagnostic_gripper_override':gripper_override,
                       'diagnostic_hold_arm':method.endswith('_hold'),
                       'action_executed':executed.tolist(),
                       'input_sha256':hashlib.sha256(selected_img.tobytes()).hexdigest(),
                       'eval_only_target_center_masked':center_covered,
                       'processor_image_sha256':processor.last_sha256 if save else None,
                       **output.metadata})
        if (step+1)%getattr(args,'progress_every',20)==0:
            print(f'[PROGRESS] task={task_id} episode={episode_idx} method={method} '
                  f'step={step+1}/{horizon} elapsed={time.monotonic()-t0:.1f}s',flush=True)
        if success:
            terminal = 'success'
            break
        if done:
            terminal = 'environment_done_without_confirmed_success'
            break
    elapsed = time.monotonic() - t0
    max_mem_mb = float(torch.cuda.max_memory_allocated() / 1024**2)
    max_reserved_mb = float(torch.cuda.max_memory_reserved() / 1024**2)
    result = {
        'task_id':task_id, 'episode_id':episode_idx, 'method':method,
        'initial_state_sha256':hashlib.sha256(np.asarray(initial_state).tobytes()).hexdigest(),
        'post_warmup_observation_sha256':warmup_digest,
        'camera_intrinsic':camera.intrinsic.tolist(),
        'camera_from_world':camera.camera_from_world.tolist(),
        'success':bool(success), 'success_source':source if frames else None,
        'termination':terminal, 'steps':len(frames),
        'task_success':bool(success),
        'terminated':bool(done) if frames else False,
        'truncated':bool((not done) and (not success) and len(frames) >= horizon),
        'termination_reason':terminal,
        'episode_length':len(frames),
        'runtime_s':elapsed, 'peak_gpu_allocated_mib':max_mem_mb,
        'peak_gpu_reserved_mib':max_reserved_mb,
        'pose_observation_success':controller.n_visible_success,
        'pose_observation_failed':controller.n_visible_fail,
        'gate_counts':{k:sum(f.get('gate') == k for f in frames if f.get('active'))
                       for k in ('memory','physics','uncertain')},
        'compensated_frames':sum(bool(f.get('compensation_applied')) for f in frames),
        'masked_frames':sum(bool(f.get('active')) for f in frames),
        'target_center_masked_frames':sum(f.get('eval_only_target_center_masked') is True for f in frames),
        'mask_coverage_label_available':sum(f.get('eval_only_target_center_masked') is not None for f in frames),
        'frames':frames,
        'ai1_hold_intervened_steps':ai1_hold.total_held if ai1_hold else 0,
        'ai1_diagnostic_hold_uses_mask_timing':bool(ai1_hold or method in ('occluded_hold','m2_2d_hold')),
        'ai1_study_label':ai1_data.get('study_label') if ai1_data else None,
        'ai1_code_version':ai1_recorder.identity['code_version'] if ai1_recorder else None,
        'ai1_config_sha256':ai1_recorder.identity['config_sha256'] if ai1_recorder else None,
        'ai1_plan_sha256':ai1_recorder.identity['plan_sha256'] if ai1_recorder else None,
        'ai1_pre_action_trace':str(ai1_recorder.pre_file.relative_to(run_dir)) if ai1_recorder else None,
        'ai1_post_action_trace':str(ai1_recorder.post_file.relative_to(run_dir)) if ai1_recorder else None,
        'ai1_eval_only_post_trace':str(ai1_recorder.eval_file.relative_to(run_dir)) if ai1_recorder else None,
    }
    if not getattr(args, 'no_eval_state_trace', False):
        rel = write_eval_state_trace(run_dir, task_id, episode_idx, method, eval_state_rows)
        result['eval_only_state_trace_path'] = rel
        result['eval_only_state_trace_rows'] = len(eval_state_rows)
    if getattr(args, 'save_video', False):
        video_path = save_episode_video(
            run_dir, task_id, episode_idx, method, success, video_frames)
        result['video_path'] = str(video_path.relative_to(run_dir))
    return result


def main(argv=None):
    args = parse_args(argv)
    args.save_video = True
    if os.environ.get('VLA_OCCLUSION', 'clean') != 'clean':
        raise RuntimeError('Remove existing VLA_OCCLUSION=center; double masking is forbidden')
    os.environ['VLA_OCCLUSION'] = 'clean'
    configure_tensorflow_cpu()
    torch = use_gpu_allocation_or_die()
    from libero.libero import benchmark
    from experiments.robot.libero.libero_utils import (
        get_libero_env, get_libero_image, get_libero_dummy_action,
    )
    from experiments.robot.openvla_utils import get_processor
    from experiments.robot.robot_utils import (
        get_model, get_action, normalize_gripper_action,
        invert_gripper_action, set_seed_everywhere,
    )
    gate_cfg = load_gate_config(args.gate_config)
    cfg = SimpleNamespace(model_family='openvla', pretrained_checkpoint=args.checkpoint,
                          load_in_4bit=True, load_in_8bit=False,
                          center_crop=args.center_crop, unnorm_key=args.suite)
    set_seed_everywhere(args.seed)
    model = get_model(cfg)  # only ONE quantized model is loaded
    model.eval()
    if not getattr(model, 'is_loaded_in_4bit', False):
        raise RuntimeError('Loaded model does not confirm 4-bit quantization')
    if cfg.unnorm_key not in model.norm_stats and cfg.unnorm_key+'_no_noops' in model.norm_stats:
        cfg.unnorm_key = cfg.unnorm_key+'_no_noops'
    if cfg.unnorm_key not in model.norm_stats:
        raise ValueError(f'Checkpoint lacks action unnorm stats for {cfg.unnorm_key}')
    processor = PixelAudit(ProcessorAudit(get_processor(cfg)))
    if torch.cuda.memory_reserved()/1024**3 > args.max_reserved_gib:
        raise RuntimeError('Model load exceeds configured GPU reserved-memory budget')
    stamp = datetime.now(timezone.utc).strftime('%Y%m%dT%H%M%S%fZ')
    run_dir = Path(args.output_root)/('eval_'+stamp)
    run_dir.mkdir(parents=True, exist_ok=False)
    manifest(run_dir, args, gate_cfg, model, cfg)
    suite = benchmark.get_benchmark_dict()[args.suite]()
    rows = []
    for task_id in args.task_ids:
        if task_id >= suite.n_tasks:
            raise ValueError(f'task_id {task_id} outside suite size {suite.n_tasks}')
        task = suite.get_task(task_id)
        initial_states = suite.get_task_init_states(task_id)
        if max(args.episode_ids) >= len(initial_states):
            raise ValueError('episode_ids exceed available LIBERO initial states')
        env, description = get_libero_env(task, 'openvla', resolution=256)
        try:
            for ep in args.episode_ids:
                reference_warmup = None
                for method in method_order(args.methods, args.seed, task_id, ep):
                    seed = pair_seed(args.seed, task_id, ep)
                    set_seed_everywhere(seed)
                    env.seed(seed)
                    camera = None  # calibrated after the actual trial reset
                    print(f'Running task={task_id} episode={ep} method={method}', flush=True)
                    try:
                        result = run_one_episode(
                            env=env, initial_state=initial_states[ep],
                            task_description=description, task_id=task_id,
                            episode_idx=ep, method=method, model=model,
                            processor=processor, cfg=cfg, args=args, camera=camera,
                            gate_cfg=gate_cfg, run_dir=run_dir,
                            get_img=get_libero_image, get_action=get_action,
                            normalize=normalize_gripper_action,
                            invert=invert_gripper_action, dummy=get_libero_dummy_action,
                            spec=args.specs[str(task_id)])
                        result['pair_seed'] = seed
                        if reference_warmup is None:
                            reference_warmup = result['post_warmup_observation_sha256']
                        elif reference_warmup != result['post_warmup_observation_sha256']:
                            raise RuntimeError('Paired reset mismatch: post-warmup observations differ')
                        entry = {k:v for k,v in result.items() if k != 'frames'}
                        (run_dir/f'task{task_id}_ep{ep}_{method}.json').write_text(
                            json.dumps(result, indent=2, default=json_safe), encoding='utf-8')
                    except Exception as exc:
                        entry = {'task_id':task_id, 'episode_id':ep, 'method':method,
                                 'success':None, 'termination':'exception',
                                 'error':f'{type(exc).__name__}: {exc}'}
                        (run_dir/f'task{task_id}_ep{ep}_{method}_ERROR.json').write_text(
                            json.dumps(entry, indent=2), encoding='utf-8')
                        rows.append(entry)
                        (run_dir/'episodes.json').write_text(
                            json.dumps(rows, indent=2, default=json_safe), encoding='utf-8')
                        if (args.fail_fast or
                            ('memory' in str(exc).lower() and 'budget' in str(exc).lower()) or
                            isinstance(exc, torch.cuda.OutOfMemoryError)):
                            raise
                        continue
                    rows.append(entry)
                    (run_dir/'episodes.json').write_text(
                        json.dumps(rows, indent=2, default=json_safe), encoding='utf-8')
                    print(f'Result: {entry.get("success")} / {entry.get("termination")}', flush=True)
        finally:
            env.close()
    from experiments.occlusion_eval.analyze import analyze
    analyze(run_dir)
    print('Completed output:', run_dir, flush=True)


if __name__ == '__main__':
    main()
