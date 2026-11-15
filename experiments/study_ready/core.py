"""Read-only scientific quality gates for frozen OpenVLA/LIBERO experiments.

This module never imports OpenVLA, MuJoCo, SAM2, or environment objects.
"""
from __future__ import annotations
import csv
import hashlib
import json
from collections import Counter
from pathlib import Path


def read_jsonl(p):
    with Path(p).open(encoding='utf-8') as f:
        return [json.loads(s) for s in f if s.strip()]


def _resolve(root, p):
    p=Path(p)
    if p.is_absolute(): return p
    # Prefer path relative to project root rather than CWD.
    return (root/p).resolve()


def inspect_episode(child: Path, repo_root: Path):
    """Audit data provenance, timing, and class coverage WITHOUT filtering failures."""
    child=Path(child);repo_root=Path(repo_root)
    results={'path':str(child),'integrity_errors':[],'perception_warnings':[],
             'coverage_warnings':[], 'data_origin':'frozen_clean_policy_collection'}
    manifest=child/'manifest.json'
    if not manifest.is_file():
        results['integrity_errors'].append('Missing manifest.json')
        return results
    m=json.loads(manifest.read_text(encoding='utf-8'))
    if not isinstance(m,list) or len(m)!=1:
        results['integrity_errors'].append('Expected one trace per episode manifest')
        return results
    entry=m[0]
    streams={}
    for key in ('public_path','eval_only_path','execution_path'):
        name={'public_path':'public','eval_only_path':'eval','execution_path':'execution'}[key]
        if key not in entry:
            results['integrity_errors'].append(f'Missing {key}')
            continue
        path=_resolve(repo_root,entry[key]);
        if not path.is_file():
            results['integrity_errors'].append(f'Missing {name} file {path}')
            continue
        try: streams[name]=read_jsonl(path)
        except (json.JSONDecodeError,UnicodeError) as exc:
            results['integrity_errors'].append(f'Invalid {name}: {exc}')
    if len(streams)!=3:return results
    p,e,a=streams['public'],streams['eval'],streams['execution']
    n=len(p);results['steps']=n
    if not n or any(len(arr)!=n for arr in (e,a)):
        results['integrity_errors'].append('PRE/EVAL/POST row count mismatch')
        return results
    for i,(u,v,w) in enumerate(zip(p,e,a)):
        if any(x.get('step')!=i for x in (u,v,w)):
            results['integrity_errors'].append(f'Time index mismatch at {i}');break
        if any(x.get('initial_state_id')!=u.get('initial_state_id') for x in (v,w)):
            results['integrity_errors'].append(f'Initial-state mismatch at {i}');break
        if u.get('record_phase')!='pre_action_observation' or w.get('record_phase')!='post_action_state':
            results['integrity_errors'].append(f'PRE/POST phase mismatch at {i}');break
        if i and u.get('previous_action')!=a[i-1].get('executed_action'):
            results['integrity_errors'].append(f'Previous action not executed t-1 at {i}');break
        if any('EVAL_ONLY' in field or field.endswith('_gt') for field in u):
            results['integrity_errors'].append('GT leakage in public row');break
    if any(v.get('observation_rgb_mask_applied_EVAL_ONLY') is not False for v in e):
        results['integrity_errors'].append('Clean policy RGB not verified')
    if any(w.get('action_source')!='frozen_openvla' for w in a):
        results['integrity_errors'].append('Action overridden during clean training collection')
    if not all(v.get('policy_clean_rgb_sha256_EVAL_ONLY') for v in e):
        results['integrity_errors'].append('Missing clean RGB hashes')
    if n and any(u.get('target_pixel_visible') is not None and u.get('target_pixel_missing') for u in p):
        results['integrity_errors'].append('Public target visibility contradiction')
    # The planned missing window does NOT mean all other frames must be visible:
    # natural loss of target is allowed, and should be recorded as perception/coverage.
    planned=entry.get('observation_dropout',[80,87]); start,end=map(int,planned)
    if start<=end:
        bad=[i for i in range(start,min(end+1,n)) if not p[i].get('target_pixel_missing',False)]
        if bad:results['integrity_errors'].append(f'Hidden estimator frame exposed public target at {bad[:8]}')
    known_contact=Counter(str(x.get('gripper_target_contact_EVAL_ONLY')) for x in e)
    known_attached=Counter(str(x.get('attached_label_EVAL_ONLY')) for x in e)
    if known_contact.get('None'):
        results['integrity_errors'].append('Contact geom/readout unavailable in some frames')
    if any(x.get('attached_label_EVAL_ONLY','MISSING') not in (0,1,None) for x in e):
        results['integrity_errors'].append('Invalid attachment label values')
    results['contact_label_counts']=dict(known_contact)
    results['attached_label_counts']=dict(known_attached)
    results['visual_valid_frames']=sum(u.get('target_pixel_visible') is not None for u in p)
    results['visual_missing_frames']=n-results['visual_valid_frames']
    results['initial_state_id']=p[0].get('initial_state_id')
    results['episode_id']=p[0].get('episode_id')
    results['label_coverage_by_initial_state']='single_group'  # combined audit does splits
    if not known_attached.get('1',0): results['coverage_warnings'].append('NO_ATTACHED_POSITIVE')
    if not known_attached.get('0',0): results['coverage_warnings'].append('NO_ATTACHED_NEGATIVE')
    if known_attached.get('None',0)>n*.5: results['coverage_warnings'].append('MORE_THAN_HALF_ATTACHED_UNKNOWN')
    if not known_contact.get('True',0): results['coverage_warnings'].append('NO_CONTACT_POSITIVE')
    if results['visual_valid_frames']<n*.5: results['perception_warnings'].append('RGB_VISIBILITY_BELOW_50_PERCENT')
    if entry.get('tracker','').startswith('human_initial_RGB_bbox'):
        results['perception_warnings'].append('LEGACY_TEMPLATE_TRACKER_NOT_PROVEN_IDENTITY_SAFE')
    for k in ('video_policy_rgb','video_diagnostic_eval_only'):
        if k not in entry or not _resolve(repo_root,entry[k]).is_file():
            results['integrity_errors'].append(f'Missing video evidence: {k}')
    results['integrity_pass']=not results['integrity_errors']
    results['ready_for_attached_training']=False  # only aggregate grouped classes may approve
    return results


def inspect_dataset(data_root: Path, repo_root: Path):
    data_root=Path(data_root)
    rows=[inspect_episode(p,repo_root) for p in sorted((data_root/'raw').glob('task*_ep*')) if p.is_dir()]
    empty_dataset=not rows
    groups=[r.get('initial_state_id') for r in rows if r.get('initial_state_id')]
    duplicated=len(groups)!=len(set(groups))
    if duplicated:
        rows.append({'integrity_errors':['Duplicate initial state across episodes'], 'path':str(data_root)})
    pos=sum(r.get('attached_label_counts',{}).get('1',0) for r in rows)
    neg=sum(r.get('attached_label_counts',{}).get('0',0) for r in rows)
    group_pos=sum(r.get('attached_label_counts',{}).get('1',0)>0 for r in rows)
    group_neg=sum(r.get('attached_label_counts',{}).get('0',0)>0 for r in rows)
    ready=(not empty_dataset and len(set(groups))>=5 and group_pos>=2 and group_neg>=2
           and pos>0 and neg>0 and all(not r.get('integrity_errors') for r in rows))
    result={'episodes':len(groups),'unique_initial_states':len(set(groups)),
            'attached_positive_frames':pos,'attached_negative_frames':neg,
            'positive_initial_states':group_pos,'negative_initial_states':group_neg,
            'all_integrity_pass':bool(rows) and not empty_dataset and all(not r.get('integrity_errors') for r in rows),
            'integrity_errors':['No episode trajectories found under '+str(data_root/'raw')] if empty_dataset else [],
            'train_validation_class_coverage_precheck':ready,
            'ready_for_train_eval':False,
            'train_eval_block_reason':'Need grouped train/val/test split with both classes in train+val; validate after split' if ready else 'Insufficient distinct verified groups/labels or integrity errors',
            'per_episode':rows}
    return result


def paired_binary_summary(path, seed=2026, replicates=20000, comparisons=None):
    """Paired binary outcomes by state; compare explicit (candidate,reference) pairs.

    A missing run is never converted to failure. Every excluded state is listed
    per comparison with missing method(s) and recorded missing-run reason(s).
    Without --compare, return ALL method pairs, never one implicit baseline.
    """
    import itertools
    import numpy as np
    path=Path(path)
    if path.suffix=='.json':
        data=json.loads(path.read_text(encoding='utf-8'))
    else:
        with path.open(newline='',encoding='utf-8') as f:
            data=list(csv.DictReader(f))
    if not isinstance(data,list) or not data:
        raise ValueError('No paired outcomes')
    if not isinstance(replicates,int) or replicates<1:
        raise ValueError('Bootstrap replicates must be >=1')

    def bo(x):
        if x is True or str(x).lower() in ('true','1','success'):return 1
        if x is False or str(x).lower() in ('false','0','failed'):return 0
        raise ValueError(f'Invalid success label: {x}')

    byid={}; absent={}; protocols={}; methods=set()
    provenance_keys=('task_id','policy_checkpoint','mask_plan_hash','max_steps','seed')
    complete_protocol=True
    for row in data:
        if not isinstance(row,dict):raise ValueError('Outcome entries must be objects')
        key=str(row['initial_state_id']).strip();method=str(row['method']).strip()
        if not key or not method:raise ValueError('Empty initial_state_id or method')
        methods.add(method)
        states=byid.setdefault(key,{})
        skipped=absent.setdefault(key,{})
        if method in states or method in skipped:
            raise ValueError(f'Duplicate method/state {key}/{method}')
        value=row.get('success')
        if value is None or str(value).strip()=='':
            skipped[method]=str(row.get('missing_reason') or row.get('run_status') or 'success_not_recorded')
        else:
            states[method]=bo(value)

        has_protocol=all(k in row and row[k] not in (None,'') for k in provenance_keys)
        complete_protocol=complete_protocol and has_protocol
        if has_protocol:
            signature=tuple(str(row[k]) for k in provenance_keys)
            if key in protocols and protocols[key]!=signature:
                raise ValueError('Paired runs have different task/checkpoint/mask/horizon/seed for state '+key)
            protocols[key]=signature

    methods=sorted(methods)
    if len(methods)<2:raise ValueError('Need two methods')
    if comparisons is None:
        # Every pair, with the alphabetical second method as candidate; this
        # preserves the old two-method sign but does not select a global base.
        comparisons=[(candidate,reference) for reference,candidate in itertools.combinations(methods,2)]
    else:
        checked=[]
        for pair in comparisons:
            if isinstance(pair,str):
                parts=pair.split(':')
                if len(parts)!=2:raise ValueError('Use --compare CANDIDATE:REFERENCE')
                candidate,reference=map(str.strip,parts)
            else:
                try:candidate,reference=pair
                except (TypeError,ValueError):raise ValueError('Comparison must contain candidate and reference')
            candidate=str(candidate).strip();reference=str(reference).strip()
            if candidate==reference or candidate not in methods or reference not in methods:
                raise ValueError(f'Invalid comparison {candidate}:{reference}; available methods: {methods}')
            if (candidate,reference) in checked:
                raise ValueError('Duplicate comparison '+candidate+':'+reference)
            checked.append((candidate,reference))
        comparisons=checked
    if not comparisons:raise ValueError('No comparisons requested')

    results=[];rng=np.random.default_rng(seed)
    all_states=sorted(byid)
    for candidate,reference in comparisons:
        shared=[];excluded=[]
        for key in all_states:
            missing=[m for m in (reference,candidate) if m not in byid[key]]
            if missing:
                excluded.append({'initial_state_id':key,'missing_methods':missing,
                    'reasons':{m:absent[key].get(m,'no_run_record') for m in missing}})
            else:shared.append(key)
        if not shared:
            results.append({'comparison':candidate+' - '+reference,'candidate':candidate,
                'reference':reference,'matched_states':0,'excluded_states':excluded,
                'status':'NO_MATCHED_STATES','mean_paired_difference':None,
                'ci95_percentile':None,'rescued':None,'harmed':None,
                'candidate_success_rate':None,'reference_success_rate':None})
            continue
        cand=np.array([byid[k][candidate] for k in shared],dtype=int)
        ref=np.array([byid[k][reference] for k in shared],dtype=int)
        diff=cand-ref
        # Memory bounded by one bootstrap row; no large [replicates x states] matrix.
        boot=np.empty(replicates,dtype=float)
        for j in range(replicates):
            boot[j]=diff[rng.integers(0,len(shared),size=len(shared))].mean()
        results.append({'comparison':candidate+' - '+reference,
            'candidate':candidate,'reference':reference,'matched_states':len(shared),
            'matched_initial_state_ids':shared,'excluded_states':excluded,
            'status':'OK','mean_paired_difference':float(diff.mean()),
            'ci95_percentile':[float(x) for x in np.quantile(boot,[.025,.975])],
            'rescued':int(((ref==0)&(cand==1)).sum()),
            'harmed':int(((ref==1)&(cand==0)).sum()),
            'candidate_success_rate':float(cand.mean()),
            'reference_success_rate':float(ref.mean())})
    if not any(r['status']=='OK' for r in results):
        details={r['comparison']:r['excluded_states'] for r in results}
        raise ValueError('No fully paired states for requested comparison(s): '+json.dumps(details,ensure_ascii=False))
    # Legacy summary fields are based only on states with ALL methods; they are
    # not used for reported pairwise contrasts (which have their own denominators).
    complete=[k for k in all_states if set(byid[k])==set(methods)]
    return {'initial_states_total':len(all_states),'complete_pairs':len(complete),
        'incomplete_states_excluded':len(all_states)-len(complete),
        'protocol_verified':complete_protocol,
        'protocol_warning':None if complete_protocol else
            'PROVISIONAL: missing task_id/policy_checkpoint/mask_plan_hash/max_steps/seed for one or more runs',
        'methods':methods,
        'success_rate_by_method':{m:(float(np.mean([byid[k][m] for k in complete])) if complete else None)
                                  for m in methods},
        'success_rate_scope':'all-method complete states only; use each comparison\'s matched denominator for contrasts',
        'comparisons':results,'ci_type':'paired bootstrap over initial states, descriptive',
        'warning':'Never tune methods, masks, thresholds or horizon on confirmation outcomes'}
