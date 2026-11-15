import argparse,json,math
from pathlib import Path
p=argparse.ArgumentParser();p.add_argument('root',type=Path);a=p.parse_args()
for folder in sorted(a.root.glob('eval_*')):
    entries=[];runs={}
    for f in folder.glob('task0_ep0_*.json'):
        d=json.loads(f.read_text())
        if 'frames' in d:runs[d['method']]=d
        elif 'error' in d:entries.append({'file':f.name,'error':d['error']})
    clean=runs.get('clean',{}).get('frames',[])
    for name,d in runs.items():
        frames=d['frames']; active=[x for x in frames if x.get('active')]
        prefix=[x for x in frames if x['step']<80]
        same=None
        if len(clean)>=80 and len(prefix)==80:
            same=all(x['input_sha256']==y['input_sha256'] and x['action_executed']==y['action_executed'] for x,y in zip(prefix,clean[:80]))
        trace=folder/d.get('eval_only_state_trace_path','missing')
        truths={}
        if trace.is_file(): truths={x['step']:x for x in map(json.loads,trace.read_text().splitlines())}
        errors=[]
        for f in active:
            truth=truths.get(f['step'],{}).get('target_center_px_gt_EVAL_ONLY')
            estimate=f.get('projected_center_px')
            if truth is not None and estimate is not None: errors.append(math.dist(truth,estimate))
        selected={str(i):truths.get(i,{}).get('target_pos_gt_EVAL_ONLY') for i in (79,80,85,87)}
        entries.append({'method':name,'success':d['success'],'termination':d['termination'],'steps':d['steps'],
            'prefix_matches_clean':same,'gate_counts':d['gate_counts'],
            'prediction_error_px_mean':sum(errors)/len(errors) if errors else None,
            'prediction_error_frames':len(errors),'target_positions_eval_only':selected,
            'hidden_policy_gripper':[x.get('action_policy_before_override',x['action_executed'])[-1] for x in active],
            'hidden_executed_gripper':[x['action_executed'][-1] for x in active],
            'overrides':sum(x.get('diagnostic_gripper_override',False) for x in active)})
    report={'rows':entries,'limits':['One development initial state; no statistical success-rate claim.',
        'Pixel errors compare point estimates with projected simulator center, not segmented visible centroid.',
        'Hold arms alter the controller and must be reported separately.',
        'Target/finger pixel coverage remains unverified. Half patches do not isolate semantic objects.',
        'Different closed-loop trajectories make error averages descriptive, not paired state-estimator accuracy.']}
    (folder/'diagnostic_summary.json').write_text(json.dumps(report,indent=2),encoding='utf-8')
    print(folder/'diagnostic_summary.json')
    for e in entries:print(e.get('method',e.get('file')),e.get('success'),e.get('termination'),e.get('prefix_matches_clean'),e.get('hidden_executed_gripper'))
