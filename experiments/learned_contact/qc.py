"""First-ten-run quality checks; fail closed on unavailable attachment labels."""
import argparse
import json
from pathlib import Path
import numpy as np
from .dataset import load_pairs


def summarize(traces):
    summary={}
    summary['trajectories']=len(traces)
    summary['initial_state_groups']=len(set(t['group'] for t in traces))
    summary['interventions']={name:sum(t['intervention']==name for t in traces)
                              for name in sorted(set(t['intervention'] for t in traces))}
    summary['total_steps']=int(sum(len(t['y']) for t in traces))
    summary['labeled_attached']=int(sum(np.sum(t['y']==1) for t in traces))
    summary['labeled_detached']=int(sum(np.sum(t['y']==0) for t in traces))
    summary['unknown_label_steps']=int(sum(np.sum(t['y']==-1) for t in traces))
    summary['visible_ratio']=(sum(sum(not r['target_pixel_missing'] for r in t['public']) for t in traces)/max(1,summary['total_steps']))
    reasons={}
    for t in traces:
        for reason in t['reason']:
            reasons[reason]=reasons.get(reason,0)+1
    summary['label_reason_counts']=reasons
    issues=[]
    if len(traces)<10:issues.append('Collect >=10 trajectories before making a scale-up decision.')
    if summary['initial_state_groups']<5:issues.append('Insufficient initial-state groups for grouped split.')
    if summary['labeled_attached']<1 or summary['labeled_detached']<1:
        issues.append('No verified attached/detached class balance. Do not train yet.')
    if summary['visible_ratio']<.1:
        issues.append('Too few positive RGB proxy detections; visually audit fixed ROI.')
    if summary['unknown_label_steps']>summary['total_steps']*.7:
        issues.append('Most labels unknown; audit contact-geom tokens, calibration and dynamics.')
    summary['issues']=issues
    summary['ready_for_exploratory_training']=not issues
    return summary


def main():
    ap=argparse.ArgumentParser();ap.add_argument('--data',required=True);ap.add_argument('--output',default=None)
    args=ap.parse_args()
    traces=load_pairs(Path(args.data)/'public',Path(args.data)/'eval_only')
    s=summarize(traces)
    if args.output:Path(args.output).write_text(json.dumps(s,indent=2),encoding='utf-8')
    print(json.dumps(s,indent=2))

if __name__=='__main__':main()
