"""Evaluation-only per-step robot/object snapshot logger for LIBERO/OpenVLA.

This code must never provide simulator-truth fields to TrialController or
get_action. It is called *after* the action has been selected for the current
observation and before env.step, and saved in a separate eval_only/ directory.
"""
import json
from pathlib import Path

import numpy as np


def _vector(mapping, key, size=None):
    if key not in mapping:
        return None
    try:
        x = np.asarray(mapping[key], dtype=float).reshape(-1)
    except (ValueError, TypeError):
        return None
    if x.size == 0 or (size is not None and x.size != size) or not np.isfinite(x).all():
        return None
    return x.copy()


def _project(camera, xyz):
    if camera is None or xyz is None:
        return None
    try:
        xy = camera.project(xyz.copy())
        if xy is None:
            return None
        xy = np.asarray(xy, dtype=float).reshape(-1)
        if xy.shape != (2,) or not np.isfinite(xy).all():
            return None
        return xy.tolist()
    except (ValueError, TypeError, AttributeError, IndexError, ZeroDivisionError):
        return None


def make_eval_state_row(*, obs, step, target_name, camera, occlusion_active,
                        executed_action, mask, action_raw=None):
    """Capture PRE-action observation; action was already predicted by policy.

    'target_*_gt' are privileged simulator labels only for posthoc analysis.
    The output goes to an isolated file; none of it is passed to the policy.
    Missing keys are explicitly null, not estimated or invented.
    """
    eef_pos = _vector(obs, 'robot0_eef_pos', 3)
    eef_quat = _vector(obs, 'robot0_eef_quat', 4)
    gripper = _vector(obs, 'robot0_gripper_qpos')
    target_pos = _vector(obs, f'{target_name}_pos', 3)
    target_quat = _vector(obs, f'{target_name}_quat', 4)
    action = _vector({'a': executed_action}, 'a', 7)
    raw = _vector({'a': action_raw}, 'a', 7) if action_raw is not None else None
    # Finite coordinates only; projection is an audit label, not an input.
    uv = _project(camera, target_pos)
    dist = (float(np.linalg.norm(target_pos - eef_pos))
            if eef_pos is not None and target_pos is not None else None)
    return {
        'step': int(step),
        'observation_timing': 'pre_env_step; policy_action_already_selected',
        'occlusion_active': bool(occlusion_active),
        'target_object': str(target_name),
        'mask_normalized': [float(v) for v in mask],
        'robot0_eef_pos': eef_pos.tolist() if eef_pos is not None else None,
        'robot0_eef_quat': eef_quat.tolist() if eef_quat is not None else None,
        'robot0_gripper_qpos': gripper.tolist() if gripper is not None else None,
        'target_pos_gt_EVAL_ONLY': target_pos.tolist() if target_pos is not None else None,
        'target_quat_gt_EVAL_ONLY': target_quat.tolist() if target_quat is not None else None,
        'target_center_px_gt_EVAL_ONLY': uv,
        'eef_to_target_center_m_EVAL_ONLY': dist,
        'action_executed': action.tolist() if action is not None else None,
        'action_raw': raw.tolist() if raw is not None else None,
        'contact_or_attachment_confirmed': None,
        'contact_status': 'NOT_RECORDED; gripper command and qpos alone do not prove grasp',
    }


def write_eval_state_trace(run_dir, task_id, episode_idx, method, rows):
    """Write a completely separate analysis-only JSONL. No policy access."""
    relative = Path('eval_only') / f'task{int(task_id)}_ep{int(episode_idx)}_{method}_state_trace.jsonl'
    path = Path(run_dir) / relative
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix(path.suffix + '.tmp')
    with tmp.open('w', encoding='utf-8') as f:
        for row in rows:
            f.write(json.dumps(row, ensure_ascii=False, allow_nan=False, separators=(',', ':')) + '\n')
    tmp.replace(path)
    return relative.as_posix()
