"""Episode-level, paired analysis. Never treat frames/windows as independent."""
import csv
import math
import json
from collections import defaultdict
from pathlib import Path

import numpy as np


def wilson(successes, total, z=1.96):
    if total == 0:
        return [None, None]
    p = successes/total
    d = 1+z*z/total
    center=(p+z*z/(2*total))/d
    hw=z*np.sqrt(p*(1-p)/total+z*z/(4*total*total))/d
    return [round(float(center-hw), 6),round(float(center+hw), 6)]


def _binomial_cdf(k, n, p):
    return sum(math.comb(n, j)*p**j*(1-p)**(n-j) for j in range(k+1))


def _cp_interval(k, n, alpha=0.025):
    """Exact binomial interval, no SciPy dependency; for small pilot n."""
    def solve(k, target):
        lo, hi = 0., 1.
        for _ in range(65):
            mid = (lo+hi)/2
            if _binomial_cdf(k, n, mid) > target:
                lo = mid
            else:
                hi = mid
        return (lo+hi)/2
    return [0. if k == 0 else solve(k-1, 1-alpha/2),
            1. if k == n else solve(k, alpha/2)]


def paired_bootstrap_diff(x, y, seed=411, repeats=4000):
    """Conservative paired binary CI plus explicitly exploratory bootstrap."""
    x, y = np.asarray(x, float), np.asarray(y, float)
    if x.ndim != 1 or x.shape != y.shape or not np.isin(x, [0,1]).all() or not np.isin(y, [0,1]).all():
        raise ValueError('Expected equally sized binary paired outcomes')
    pairs = x-y
    n = len(pairs)
    out = dict(n=n, difference=float(pairs.mean()) if n else None,
               ci95=[None,None], exploratory_bootstrap_ci95=[None,None])
    wins, losses = int((pairs == 1).sum()), int((pairs == -1).sum())
    out.update(wins=wins, losses=losses, ties=n-wins-losses)
    discordant = wins+losses
    out['exact_mcnemar_p'] = min(1., 2*_binomial_cdf(min(wins,losses), discordant, .5)) if discordant else 1.
    out['ci_method'] = 'Bonferroni simultaneous Clopper-Pearson bounds for win/loss probabilities'
    if n >= 2:
        win_ci, loss_ci = _cp_interval(wins,n), _cp_interval(losses,n)
        out['ci95'] = [win_ci[0]-loss_ci[1], win_ci[1]-loss_ci[0]]
        rng = np.random.default_rng(seed)
        idx = rng.integers(0,n,size=(repeats,n))
        out['exploratory_bootstrap_ci95'] = [float(v) for v in np.quantile(pairs[idx].mean(axis=1), [.025,.975])]
    return out


def analyze(run_dir):
    run_dir=Path(run_dir)
    records=json.loads((run_dir/'episodes.json').read_text(encoding='utf-8'))
    identities=[(r['task_id'],r['episode_id'],r['method']) for r in records]
    if len(identities) != len(set(identities)):
        raise ValueError('Duplicate task/episode/method records')
    manifest_path=run_dir/'manifest.json'
    expected=[]
    if manifest_path.exists():
        settings=json.loads(manifest_path.read_text())['settings']
        expected=[(t,e,m) for t in settings['task_ids'] for e in settings['episode_ids'] for m in settings['methods']]
    missing=[list(k) for k in expected if k not in set(identities)]
    complete=[r for r in records if isinstance(r.get('success'),bool)]
    methods=sorted(set(r['method'] for r in records))
    per_method={}
    for method in methods:
        subset=[r for r in complete if r['method']==method]
        n=len(subset); successes=sum(r['success'] for r in subset)
        mask_total=sum(r.get('mask_coverage_label_available',0) for r in subset)
        mask_covered=sum(r.get('target_center_masked_frames',0) for r in subset)
        per_method[method]=dict(n=n,successes=successes,
                                success_rate=successes/n if n else None,
                                wilson95=wilson(successes,n),
                                target_center_mask_coverage=(mask_covered/mask_total if mask_total else None),
                                mean_peak_reserved_mib=(float(np.mean([r['peak_gpu_reserved_mib'] for r in subset
                                                    if 'peak_gpu_reserved_mib' in r]))
                                                    if any('peak_gpu_reserved_mib' in r for r in subset) else None),
                                excluded_errors=sum(r['method']==method and r.get('success') is None
                                                    for r in records))
    per_task={}
    for task in sorted(set(r['task_id'] for r in complete)):
        per_task[str(task)]={}
        for method in methods:
            subset=[r for r in complete if r['task_id']==task and r['method']==method]
            n=len(subset)
            per_task[str(task)][method]={'n':n, 'success':sum(r['success'] for r in subset),
                                          'success_rate':sum(r['success'] for r in subset)/n if n else None}
    keyed={(r['task_id'],r['episode_id'],r['method']):r for r in complete}
    comparisons={}
    if 'm2' in methods:
        for other in [x for x in ('occluded','m0','m1','cue_control','clean') if x in methods]:
            pairs=[]
            for task_id, ep, method in keyed:
                if method != 'm2':
                    continue
                k=(task_id,ep,other)
                if k in keyed:
                    pairs.append((int(keyed[(task_id,ep,'m2')]['success']),
                                  int(keyed[k]['success'])))
            comparisons[f'm2_minus_{other}']=paired_bootstrap_diff(
                [p[0] for p in pairs],[p[1] for p in pairs])
            comparisons[f'm2_minus_{other}']['unmatched_pair_count']=sum(
                ((t,e,'m2') in keyed) != ((t,e,other) in keyed)
                for t,e in set((r['task_id'],r['episode_id']) for r in records))
            comparisons[f'm2_minus_{other}']['note']='Exploratory pooled across tasks; per-task comparisons below'
            by_task={}
            for task in sorted(set(k[0] for k in keyed)):
                q=[]
                for ep in sorted(set(k[1] for k in keyed if k[0]==task)):
                    a=(task,ep,'m2'); b=(task,ep,other)
                    if a in keyed and b in keyed:
                        q.append((int(keyed[a]['success']),int(keyed[b]['success'])))
                by_task[str(task)]=paired_bootstrap_diff([v[0] for v in q],[v[1] for v in q])
            comparisons[f'm2_minus_{other}']['by_task']=by_task
    errors=[r for r in records if r.get('success') is None]
    summary={'method':per_method,'task':per_task,'paired':comparisons,
             'failed_runs':errors, 'missing_planned_runs':missing,
             'evidence_status':'INCOMPLETE_DO_NOT_CLAIM_GAIN' if errors or missing else 'COMPLETE_RECORDED_RUNS_ONLY',
             'warnings':[
                'Paired unit is TASK/INITIAL_STATE, not frame or overlapping window.',
                'Bootstrap intervals with tiny n are unstable; n=1 has no interval.',
                'No claim of statistical significance from exploration or selected successes.',
                'Check manifest pose-source, camera calibration, mask coverage and contact labels.',
                'Mask coverage metric is target CENTER inside ROI from segregated evaluation-only simulator truth, not object-pixel fraction.',
                'For multiple tasks, inspect per-task paired differences; pooled bootstrap is only exploratory.',
                'A success-rate gain alone does not establish algorithmic novelty.',
             ]}
    (run_dir/'analysis.json').write_text(json.dumps(summary,indent=2),encoding='utf-8')
    with (run_dir/'summary.csv').open('w',newline='',encoding='utf-8') as f:
        writer=csv.writer(f)
        writer.writerow(['method','n','successes','rate','wilson_low','wilson_high','errors'])
        for method, s in per_method.items():
            writer.writerow([method,s['n'],s['successes'],s['success_rate'],
                             *s['wilson95'], s['excluded_errors']])
    return summary


if __name__ == '__main__':
    import argparse
    p=argparse.ArgumentParser();p.add_argument('run_dir'); args=p.parse_args()
    print(json.dumps(analyze(args.run_dir),indent=2))
