"""Early, read-only guard for the EXISTING sklearn/GRU training script.

Never constructs synthetic labels, tunes a threshold, or modifies any source file.
"""
from __future__ import annotations
from collections import Counter
from pathlib import Path


def group_class_report(traces,seed=2026):
    """Separate ability to fit, to select on validation, and to evaluate test.

    Never change split seed to chase positive classes. The existing train_eval
    script still needs both labels in train/val and at least test labels; that
    stricter execution readiness is reported separately from fit readiness.
    """
    from experiments.learned_contact.dataset import group_split
    out={'ready':False,'can_fit_binary_classifier':False,
         'ready_for_model_selection':False,'ready_for_existing_train_eval':False,
         'test_two_class_metrics_available':False,'seed':seed,'classes':{},
         'reason':None,'warnings':[]}
    groups={t['group'] for t in traces}
    if len(groups)<5:
        out['reason']='At least five different initial-state groups required'
        return out
    try:splits=group_split(traces,seed)
    except ValueError as exc:
        out['reason']=str(exc)
        return out
    for name,ts in splits.items():
        counted=Counter()
        for t in ts:
            for val in t['y']:
                if val in (0,1):counted[int(val)]+=1
                else:counted['unknown']+=1
        out['classes'][name]={'initial_states':len({t['group'] for t in ts}),
            'negative':counted[0],'positive':counted[1],
            'unknown':counted['unknown']}
    has_both=lambda name:all(out['classes'][name][k]>0 for k in ('positive','negative'))
    train=has_both('train');val=has_both('val');test=has_both('test')
    out['ready']=out['can_fit_binary_classifier']=train
    out['ready_for_model_selection']=train and val
    out['test_two_class_metrics_available']=test
    out['ready_for_existing_train_eval']=train and val and (
        out['classes']['test']['positive']+out['classes']['test']['negative']>0)
    if not train:
        out['reason']='Training split lacks verified positive or negative labels: cannot fit binary classifier'
    else:
        out['reason']='Training split supports binary fitting; see validation/test readiness separately'
    if not val:
        out['warnings'].append('VALIDATION_ONE_CLASS_OR_EMPTY: two-class model selection/threshold evaluation unavailable')
    if not test:
        out['warnings'].append('TEST_ONE_CLASS_OR_EMPTY: two-class test metrics unavailable; do not resplit to manufacture classes')
    if not out['ready_for_existing_train_eval']:
        out['warnings'].append('EXISTING_TRAIN_EVAL_NOT_READY: original script requires train+val two-class labels and nonempty labeled test')
    return out


def preflight_data(data_path,seed=2026,history=6):
    from experiments.learned_contact.dataset import load_pairs
    p=Path(data_path)
    if not (p/'public').is_dir() or not (p/'eval_only').is_dir():
        return {'ready':False,'reason':f'Missing merged public/ and eval_only/ under {p}'}
    try:
        traces=load_pairs(p/'public',p/'eval_only',history=history)
        return group_class_report(traces,seed=seed)
    except (FileNotFoundError,ValueError,KeyError,IndexError,TypeError) as e:
        return {'ready':False,'reason':f'Data schema or label failure: {type(e).__name__}: {e}'}
