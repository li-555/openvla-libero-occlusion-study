"""Train a small candidate-action success classifier on actual grasp probes.

Input JSONL schema: grasp_probe_outcome_v1, with pre_action (public) and
post_action_label.stable_grasp_after_lift (from executed rollout ONLY).
Paired probes from one state MUST have same checkpoint_id and initial_state_id.
No successful CLEAN goal positions, episodes or mask step indices in features.
"""
import argparse
import hashlib
import json
from pathlib import Path
import joblib
import numpy as np
from sklearn.ensemble import ExtraTreesClassifier
from sklearn.metrics import balanced_accuracy_score, brier_score_loss, roc_auc_score
from sklearn.model_selection import GroupShuffleSplit
from .features import FEATURE_NAMES,obs_from_dict
from .controller import CANDIDATE_PROTOCOL, MODEL_SCHEMA

SCHEMA='grasp_probe_outcome_v1'


def load_rows(path):
    rows=[]
    for number,line in enumerate(Path(path).read_text(encoding='utf-8').splitlines(),1):
        if not line.strip(): continue
        data=json.loads(line)
        if data.get('schema')!=SCHEMA or data.get('event_timing')!='post_probe_outcome':
            raise ValueError(f'line {number}: not an executed grasp probe outcome')
        if data.get('candidate_protocol') != CANDIDATE_PROTOCOL:
            raise ValueError(f'line {number}: incompatible two-step candidate protocol')
        if data.get('label_source')!='executed_grasp_and_lift':
            raise ValueError(f'line {number}: simulated/unexecuted/imputed label forbidden')
        if data.get('split') not in ('development','confirmation'):
            raise ValueError(f'line {number}: missing split')
        if data['split']!='development':
            continue  # confirmation set is sealed; do not train or choose threshold on it
        for key in ('initial_state_id','checkpoint_id','seed','pre_action','candidate','post_action_label'):
            if key not in data: raise ValueError(f'line {number}: missing {key}')
        label=data['post_action_label'].get('stable_grasp_after_lift')
        if not isinstance(label,(bool,int)) or label not in (0,1):
            raise ValueError(f'line {number}: stable grasp label must be 0/1')
        cand=data['candidate']
        if set(cand)!={'xyz_action','close_now'}:
            raise ValueError(f'line {number}: unexpected candidate keys')
        features=obs_from_dict(data['pre_action']).vector(cand['xyz_action'],cand['close_now'])
        group=str(data['initial_state_id'])  # no leakage across same initial condition
        rows.append((features,int(label),group,str(data['checkpoint_id'])))
    if not rows: raise ValueError('no eligible actual development grasp probes')
    X=np.stack([r[0] for r in rows]);y=np.array([r[1] for r in rows],dtype=int)
    groups=np.array([r[2] for r in rows]); checkpoints=np.array([r[3] for r in rows])
    if len(set(groups))<5:
        raise ValueError('need >=5 independent initial-state groups for meaningful group validation')
    if len(set(y))<2: raise ValueError('need both stable and failed executed grasp outcomes')
    if len(set(checkpoints))<5: raise ValueError('need multiple independent checkpoints')
    return X,y,groups,checkpoints


def fit(path, dest, *, seed=7, min_validation_groups=2):
    src=Path(path)
    X,y,groups,_=load_rows(src)
    split=GroupShuffleSplit(n_splits=40,test_size=.3,random_state=seed)
    candidate=[]
    for train_idx,val_idx in split.split(X,y,groups):
        if len(set(y[train_idx]))==2 and len(set(y[val_idx]))==2 and len(set(groups[val_idx]))>=min_validation_groups:
            candidate.append((train_idx,val_idx))
    if not candidate: raise ValueError('could not form class-balanced heldout validation groups')
    train_idx,val_idx=candidate[0]
    Xtrain,ytrain=X[train_idx],y[train_idx]
    Xval,yval=X[val_idx],y[val_idx]
    # Explicit ensemble variance is used as an uncertainty gate online.
    seeds=[seed,seed+101,seed+202]
    models=[]
    for s in seeds:
        m=ExtraTreesClassifier(n_estimators=125,min_samples_leaf=3,max_features=.85,
                               class_weight='balanced',random_state=s,n_jobs=1)
        m.fit(Xtrain,ytrain)
        models.append(m)
    val_probs=np.mean([m.predict_proba(Xval)[:,1] for m in models],axis=0)
    report={
        'schema':'grasp_v2_training_report_v1',
        'development_samples':len(y),
        'positive_samples':int(y.sum()),'negative_samples':int(len(y)-y.sum()),
        'train_initial_states':sorted(set(groups[train_idx].tolist())),
        'validation_initial_states':sorted(set(groups[val_idx].tolist())),
        'heldout_brier':float(brier_score_loss(yval,val_probs)),
        'heldout_roc_auc':float(roc_auc_score(yval,val_probs)),
        'heldout_balanced_accuracy_at_0_5':float(balanced_accuracy_score(yval,val_probs>=.5)),
        'source_sha256':hashlib.sha256(src.read_bytes()).hexdigest(),
        'feature_names':list(FEATURE_NAMES),
        'note':'These are development validation metrics, NOT closed-loop success rates.'}
    dest=Path(dest);dest.mkdir(parents=True,exist_ok=True)
    joblib.dump({'schema':MODEL_SCHEMA,'candidate_protocol':CANDIDATE_PROTOCOL,'models':models,
                 'feature_names':FEATURE_NAMES,'source_sha256':report['source_sha256'],
                 'report':report},dest/'candidate_model.joblib')
    (dest/'training_report.json').write_text(json.dumps(report,indent=2,ensure_ascii=False),encoding='utf-8')
    return report


def main(argv=None):
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--probes',required=True)
    p.add_argument('--output',required=True)
    a=p.parse_args(argv)
    print(json.dumps(fit(a.probes,a.output),indent=2))

if __name__=='__main__':main()
