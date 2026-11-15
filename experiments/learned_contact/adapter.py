"""Prediction-only integration API with independent intervention switch.

The API deliberately returns hypotheses and confidence without choosing any
OpenVLA action override. Bounded confirmation must be separately implemented,
verified, and assigned the same delay budget as fixed hold controls.
"""
from dataclasses import dataclass
from pathlib import Path
import json
import numpy as np
from .dataset import sequence_features,FEATURE_NAMES
from .stream import validate_public,source_version,PHASE_PRE
from .mixture import combine

@dataclass(frozen=True)
class RunConfig:
    estimator: str='memory'
    intervention: str='none'
    history: int=6
    def __post_init__(self):
        if self.estimator not in ('memory','rigid','rule_contact','learned_contact'):
            raise ValueError('Unknown estimator')
        if self.intervention not in ('none','visual_cue','bounded_gripper_confirmation'):
            raise ValueError('Unknown independent intervention')
        if self.history<1: raise ValueError('history must be positive')


class EstimationInterface:
    """State estimator only; no simulation handle, controller or object GT."""
    def __init__(self,config,*,artifact_directory=None,rule_fn=None):
        self.config=config
        self.rows=[];self.rule_fn=rule_fn
        self._fit=None;self._model_type=None;self._model_seed=0
        if config.estimator=='learned_contact':
            if artifact_directory is None:
                raise ValueError('learned_contact requires pre-trained, real-data artifacts')
            import joblib
            out=Path(artifact_directory)
            selection=json.loads((out/'selection.json').read_text())
            if selection.get('evaluation_code_version')!=source_version():
                raise ValueError('Estimator code version differs from trained checkpoint')
            if tuple(selection.get('feature_names',()))!=FEATURE_NAMES or selection.get('history')!=config.history:
                raise ValueError('Feature schema/history mismatch with training artifact')
            self._model_type=selection['selected_model']
            self._model_seed=selection['seed']
            if self._model_type=='gru':
                import torch
                from .models import TinyGRU
                meta=joblib.load(out/'selected_gru_scaler.joblib')
                model=TinyGRU(meta['input_dim'],meta['hidden_dim'])
                model.load_state_dict(torch.load(out/'selected_gru_weights.pt',map_location='cpu',weights_only=True))
                model.eval()
                self._fit=dict(model=model,scaler=meta['scaler'])
            else:
                self._fit=joblib.load(out/'selected_model.joblib')['fit']

    def step(self,public_observation):
        validate_public(public_observation)
        if self.rows and public_observation['step']!=self.rows[-1]['step']+1:
            raise ValueError('Online telemetry must be contiguous; reset estimator after gaps')
        if self.rows and any(public_observation[k]!=self.rows[-1][k] for k in (
                'trajectory_id','initial_state_id','episode_id','code_version')):
            raise ValueError('Cannot mix episodes/sources; instantiate a new estimator')
        self.rows.append(public_observation)
        seq,hyp=sequence_features(self.rows,self.config.history)
        if self.config.estimator=='memory':p=0.
        elif self.config.estimator=='rigid':p=1.
        elif self.config.estimator=='rule_contact':
            if self.rule_fn is None:
                raise ValueError('Explicit calibrated rule callback required')
            r=self.rule_fn(self.rows)
            p=np.nan if r is None else float(r)
        else:
            from .train_eval import probabilities
            p=float(probabilities(self._fit,seq[-1:],self._model_type)[0])
        last=hyp[-1]
        common=dict(episode_id=public_observation['episode_id'],
            initial_state_id=public_observation['initial_state_id'],
            trajectory_id=public_observation['trajectory_id'],step=public_observation['step'],
            method=self.config.estimator,seed=self._model_seed,
            collection_seed=public_observation['seed'],code_version=source_version(),
            collection_code_version=public_observation['code_version'],
            record_phase=PHASE_PRE,
            config=dict(estimator=self.config.estimator,
                intervention=self.config.intervention,history=self.config.history))
        if not np.isfinite(p):
            return dict(**common,status='uncertain',prediction=None,
                intervention=self.config.intervention,reason='rule_abstained',action_override=None)
        age=0 if last['anchor_step'] is None else public_observation['step']-last['anchor_step']
        pred=combine(last['memory'],last['rigid'],p,max_age=100,age=age)
        return dict(**common,status='no_anchor' if pred is None else pred.decision,
            prediction=None if pred is None else {
                'attached_probability':pred.probability_attached,
                'memory_pixel':pred.memory_pixel,'rigid_pixel':pred.rigid_pixel,
                'mixture_weights':pred.weights,'cue_pixel':pred.cue_pixel,
                'geometry':pred.proxy_geometry},
            intervention=self.config.intervention,
            # No control override should happen implicitly in this object.
            action_override=None)
