"""First-grasp-attempt, model-ranked two-step candidate controller.

No oracle / clean-demo coordinates. Learning labels and deployment execute the
SAME candidate protocol:
  close_now: OpenVLA XYZ + close this step
  move_then_close: candidate XYZ + open now; then OpenVLA XYZ + forced close
                   on exactly the next step, WITHOUT re-ranking that next step.
The one explicitly planned forced close is the only later gripper override.
Subsequent opening/placing actions are never changed by this controller.
If reliable scores support neither option, a bounded open-gripper fallback
(its own diagnostic arm, NOT evidence of learned scoring) is allowed.
"""
from collections import deque
from pathlib import Path
import joblib
import numpy as np
from .features import FEATURE_NAMES, PreAction

CANDIDATE_PROTOCOL = 'one_step_move_then_forced_next_close_v1'
MODEL_SCHEMA = 'grasp_v2_candidate_success_v2'


class AdaptiveGraspController:
    def __init__(self, model_file, *, correction_step=0.12,
                 min_close_score=0.65, min_move_score=0.55,
                 min_predicted_gain=0.08, max_uncertainty=0.20,
                 max_last_seen_age=30, max_fallback_wait_steps=2):
        obj = joblib.load(Path(model_file))  # ONLY trusted, user-trained artifacts
        if (obj.get('schema') != MODEL_SCHEMA or
                obj.get('candidate_protocol') != CANDIDATE_PROTOCOL or
                tuple(obj.get('feature_names', ())) != FEATURE_NAMES):
            raise ValueError('incompatible grasp_v2 protocol/model; retrain on v2 probes')
        self.models = obj['models']
        if not self.models:
            raise ValueError('empty model ensemble')
        if not (0 < correction_step <= .25 and 0 < min_close_score < 1
                and 0 < min_move_score < 1 and 0 <= min_predicted_gain < 1
                and 0 <= max_uncertainty <= .5 and 1 <= max_last_seen_age <= 100
                and 0 <= max_fallback_wait_steps <= 10):
            raise ValueError('invalid controller bounds')
        self.step = float(correction_step)
        self.close_threshold = float(min_close_score)
        self.move_threshold = float(min_move_score)
        self.gain = float(min_predicted_gain)
        self.uncertainty = float(max_uncertainty)
        self.max_age = int(max_last_seen_age)
        self.max_wait = int(max_fallback_wait_steps)
        self.reset()

    def reset(self):
        self.last_seen = None
        self.last_seen_step = None
        self.seen = deque(maxlen=3)
        self.prev_eef = None
        self.prev_step = None
        self.prev_action_xyz = np.zeros(3)
        self.step_context = None
        self.pending_next_close = False
        self.fallback_started = False
        self.fallback_frames = 0
        self.first_attempt_finished = False

    def observe(self, *, step, occluded, memory_pixel, eef_pixel, eef_xyz,
                gripper_aperture_m, previous_action=None):
        if self.prev_step is not None and step <= self.prev_step:
            raise ValueError('steps out of order; reset at each rollout')
        eef = np.asarray(eef_xyz, dtype=float)
        ep = np.asarray(eef_pixel, dtype=float) if eef_pixel is not None else None
        if eef.shape != (3,) or not np.isfinite(eef).all():
            raise ValueError('EEF XYZ must be finite three-vector')
        if ep is not None and (ep.shape != (2,) or not np.isfinite(ep).all()):
            ep = None
        if not np.isfinite(float(gripper_aperture_m)):
            raise ValueError('invalid observed gripper aperture')
        if previous_action is not None:
            pa = np.asarray(previous_action, dtype=float)
            if pa.shape != (7,) or not np.isfinite(pa).all():
                raise ValueError('prior executed action must be finite seven-vector')
            self.prev_action_xyz = pa[:3].copy()
        velocity = (np.zeros(3) if self.prev_eef is None else
                    (eef - self.prev_eef) / max(1, step-self.prev_step))
        self.prev_eef = eef.copy()
        self.prev_step = int(step)
        if not occluded:
            if memory_pixel is None:
                self.seen.clear()
            else:
                p = np.asarray(memory_pixel, dtype=float)
                if p.shape != (2,) or not np.isfinite(p).all():
                    self.seen.clear()
                else:
                    if self.seen and np.linalg.norm(self.seen[-1]-p) > 12:
                        self.seen.clear()
                    self.seen.append(p)
                    if len(self.seen) >= 2:
                        self.last_seen = p.copy()
                        self.last_seen_step = int(step)
        age = None if self.last_seen_step is None else int(step-self.last_seen_step)
        self.step_context = {
            'step': int(step), 'occluded': bool(occluded),
            'target_px': None if self.last_seen is None else self.last_seen.tolist(),
            'eef_px': None if ep is None else ep.tolist(),
            'eef_xyz': eef.tolist(), 'eef_vel': velocity.tolist(),
            'age': age, 'gripper_aperture': float(gripper_aperture_m),
            'previous_action_xyz': self.prev_action_xyz.tolist(),
            'anchor_confident': bool(len(self.seen) >= 2 and age is not None
                                     and age <= self.max_age),
        }

    def _scores(self, context, proposed_xyz, candidates):
        obs = PreAction(target_px=tuple(context['target_px']),
                        eef_px=tuple(context['eef_px']),
                        eef_xyz_m=tuple(context['eef_xyz']),
                        eef_vel_m=tuple(context['eef_vel']),
                        gripper_aperture_m=float(context['gripper_aperture']),
                        last_seen_age=int(context['age']),
                        target_confidence=1.,  # gate flag, not a calibrated probability
                        previous_action_xyz=tuple(context['previous_action_xyz']),
                        proposed_action_xyz=tuple(proposed_xyz))
        X = np.stack([obs.vector(candidate, now) for candidate, now in candidates])
        p = np.stack([m.predict_proba(X)[:, 1] for m in self.models])
        if not np.isfinite(p).all():
            raise ValueError('model emitted nonfinite scores')
        return p.mean(axis=0), p.std(axis=0)

    def override(self, proposed_action, *, step):
        action = np.asarray(proposed_action, dtype=float).copy()
        if action.shape != (7,) or not np.isfinite(action).all():
            raise ValueError('proposed action must be finite 7D vector')
        if self.step_context is None or self.step_context['step'] != step:
            raise ValueError('observe must precede override at the same step')
        c = self.step_context
        event = dict(method='learned_grasp_v2', step=int(step),
                     protocol=CANDIDATE_PROTOCOL, model_used=False,
                     modified=False, rejected_close=False, uses_gt=False,
                     reason='no_first_grasp_close_proposal')

        # Exactly the one forced close that the executed probe used as its label.
        # Do not run the scorer again on this step or later in this episode.
        if self.pending_next_close:
            self.pending_next_close = False
            self.first_attempt_finished = True
            event['reason'] = 'complete_precommitted_two_step_candidate'
            event['scheduled_next_step_close'] = True
            event['policy_requested_open'] = bool(action[-1] < .5)
            action[-1] = 1.
            event['modified'] = event['policy_requested_open']
            return action, event

        if self.first_attempt_finished:
            event['reason'] = 'first_grasp_attempt_finished_passthrough'
            return action, event

        # No learned interventions outside the first occluded close request.
        if not c['occluded']:
            if action[-1] >= .5:
                self.first_attempt_finished = True
            event['reason'] = 'not_occluded'
            return action, event

        if self.fallback_started:
            # This is a separate bounded non-learned fallback. It never
            # invents an extra close or changes later post-grasp opening.
            if self.fallback_frames >= self.max_wait:
                self.first_attempt_finished = True
                event['reason'] = 'fallback_budget_exhausted_passthrough'
                return action, event
            self.fallback_frames += 1
            event['reason'] = 'bounded_open_fallback'
            event['fallback_count'] = self.fallback_frames
            if action[-1] >= .5:
                action[-1] = -1.
                event['modified'] = True
                event['rejected_close'] = True
            return action, event

        if action[-1] < .5:
            return action, event
        if not c['anchor_confident'] or c['eef_px'] is None:
            self.first_attempt_finished = True
            event['reason'] = 'abstain_missing_or_stale_visual_anchor'
            return action, event

        base = action[:3].copy()
        offsets = [(0., 0., 0.), (self.step, 0., 0.), (-self.step, 0., 0.),
                   (0., self.step, 0.), (0., -self.step, 0.),
                   (0., 0., self.step), (0., 0., -self.step)]
        options = [(base, True)] + [
            (np.clip(base + np.asarray(offset), -1., 1.), False)
            for offset in offsets]
        try:
            p, unc = self._scores(c, base, options)
        except (ValueError, OverflowError, IndexError) as exc:
            self.first_attempt_finished = True
            event['reason'] = 'abstain_invalid_model_output:' + type(exc).__name__
            return action, event

        event.update(model_used=True, prob_close_now=float(p[0]),
                     uncertainty_close_now=float(unc[0]),
                     prob_best_move=float(np.max(p[1:])),
                     anchor_age=int(c['age']))
        if p[0] >= self.close_threshold and unc[0] <= self.uncertainty:
            self.first_attempt_finished = True
            event['reason'] = 'allow_close_model_ready'
            return action, event

        # Alternative motion evaluated by model only when it clears gain,
        # absolute-score, and ensemble-uncertainty bounds.
        best = 1 + int(np.argmax(p[1:]))
        if (p[best] >= self.move_threshold and p[best]-p[0] >= self.gain
                and unc[best] <= self.uncertainty):
            action[:3] = options[best][0]
            action[-1] = -1.
            self.pending_next_close = True
            event.update(modified=True, rejected_close=True,
                         reason='move_then_forced_next_close_protocol',
                         chosen_action_xyz=action[:3].tolist(),
                         expected_gain=float(p[best]-p[0]),
                         pending_close_next_step=True)
            return action, event

        # A bounded conservative open-gripper fallback: unlike a learned
        # move decision, report separately; do not count as model gain.
        if self.max_wait > 0:
            self.fallback_started = True
            self.fallback_frames = 1
            action[-1] = -1.
            event.update(reason='bounded_open_fallback',
                         modified=True, rejected_close=True,
                         fallback_count=1,
                         fallback_budget=self.max_wait)
            return action, event
        self.first_attempt_finished = True
        event['reason'] = 'abstain_no_reliable_candidate'
        return action, event
