"""Fail-closed, zero-overwrite evaluation tools for existing LIBERO study.

Never invokes OpenVLA or MuJoCo. Use dedicated sam2_offline CLI for pretrained masks.
"""
import argparse
import hashlib
import json
import sys
from pathlib import Path
from .core import inspect_dataset,paired_binary_summary


def preflight(repo):
    repo=Path(repo)
    expected=('experiments/real_collection/collector.py',
              'experiments/real_collection/run.py',
              'experiments/learned_contact/train_eval.py',
              'experiments/occlusion_eval/run_gpu.py')
    missing=[str(repo/p) for p in expected if not (repo/p).is_file()]
    hashes={p:hashlib.sha256((repo/p).read_bytes()).hexdigest() for p in expected if (repo/p).is_file()}
    result={'project':str(repo.resolve()),'expected_code_present':not missing,
            'missing_files':missing,'source_hashes':hashes,
            'policy_hold_runner_modified_by_this_package':False,
            'pretrained_sam2_present':False,'sam2_real_gpu_validated':False,
            'sampling_performed':False,'tests_are_not_gpu_validation':True}
    try:
        import importlib.metadata as im
        result['installed_numpy']=im.version('numpy')
        result['installed_torch']=im.version('torch')
    except im.PackageNotFoundError:pass
    return result


def main(argv=None):
    p=argparse.ArgumentParser(description=__doc__)
    sp=p.add_subparsers(dest='mode',required=True)
    d=sp.add_parser('preflight');d.add_argument('--repo',default='.')
    d=sp.add_parser('dataset-audit');d.add_argument('--root',required=True);d.add_argument('--repo',default='.')
    d.add_argument('--output',default=None)
    d=sp.add_parser('train-preflight');d.add_argument('--data',required=True);d.add_argument('--seed',type=int,default=2026);d.add_argument('--history',type=int,default=6)
    d.add_argument('--output',default=None)
    d=sp.add_parser('paired-report');d.add_argument('--outcomes',required=True);d.add_argument('--output',default=None)
    d.add_argument('--seed',type=int,default=2026);d.add_argument('--replicates',type=int,default=20000)
    d.add_argument('--compare',action='append',default=None,metavar='CANDIDATE:REFERENCE',
                   help='Repeat to request explicit paired contrasts; without it reports all method pairs')
    a=p.parse_args(argv)
    if a.mode=='preflight':r=preflight(a.repo)
    elif a.mode=='dataset-audit':r=inspect_dataset(a.root,a.repo)
    elif a.mode=='train-preflight':
        from .training_gate import preflight_data
        r=preflight_data(a.data,a.seed,a.history)
    else:r=paired_binary_summary(a.outcomes,seed=a.seed,replicates=a.replicates,comparisons=a.compare)
    result=json.dumps(r,indent=2,ensure_ascii=False,allow_nan=False)
    if getattr(a,'output',None):
        output=Path(a.output)
        if output.exists():p.error('Refusing to overwrite '+str(output))
        output.parent.mkdir(parents=True,exist_ok=True)
        with output.open('x',encoding='utf-8') as f:f.write(result)
    print(result)
    if a.mode=='preflight' and not r['expected_code_present']:return 2
    if a.mode=='dataset-audit' and not r['all_integrity_pass']:return 2
    if a.mode=='train-preflight' and not r.get('ready_for_existing_train_eval', False):return 2
    return 0
if __name__=='__main__':sys.exit(main())
