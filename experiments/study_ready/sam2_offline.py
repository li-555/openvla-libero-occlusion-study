"""Minimal OFFLINE adapter to Meta's official SAM 2.1 video predictor.

Candidate visual features only. No edits to policy code or existing public logs.
Never label masks as verified object identity without independent human review.
Runs only in a separate SAM2 environment with the official pretrained checkpoint.
"""
from __future__ import annotations
import argparse
import hashlib
import json
import tempfile
from pathlib import Path
import numpy as np

OFFICIAL_CONFIG='configs/sam2.1/sam2.1_hiera_t.yaml'
OFFICIAL_WEIGHTS='sam2.1_hiera_tiny.pt'


def validate_box(box, shape):
    if len(box)!=4:raise ValueError('Expected x0,y0,x1,y1')
    x0,y0,x1,y1=map(int,box);h,w=shape[:2]
    if not (0<=x0<x1<=w and 0<=y0<y1<=h and x1-x0>=8 and y1-y0>=8):
        raise ValueError('Invalid manually reviewed initial RGB bbox')
    return (x0,y0,x1,y1)


def frame2record(frame, mask, shape, min_area=12, max_fraction=.22):
    """Mask -> conservative visual candidate. No GT, no calibrated confidence."""
    h,w=shape[:2];m=np.asarray(mask,dtype=bool)
    if m.shape!=(h,w):raise ValueError(f'Mask shape {m.shape} != {h,w}')
    count=int(m.sum())
    record={'step':int(frame),'candidate_pixel_xy':None,'mask_area':count,
            'source':'official_sam2_1_forward_video_prediction','identity_verified':False,
            'confidence_calibrated':False,'reason':None}
    if count<min_area or count>max_fraction*h*w:
        record['reason']='area_rejected_or_absent';return record
    yy,xx=np.nonzero(m)
    record['candidate_pixel_xy']=[float(xx.mean()),float(yy.mean())]
    record['reason']='candidate_REQUIRES_IDENTITY_REVIEW'
    return record


def forward_from_predictor(predictor, video_dir, box, shape, n_frames):
    """Only forward propagation with a SINGLE first-frame RGB box.

    Predictor is official SAM2 API; injection makes API contract testable without
    pretending that pretrained inference ran on this machine.
    """
    state=predictor.init_state(video_path=str(video_dir),offload_video_to_cpu=True,offload_state_to_cpu=True)
    predictor.add_new_points_or_box(inference_state=state,frame_idx=0,obj_id=1,box=np.asarray(box,dtype=np.float32))
    seen=set()
    for idx,obj_ids,logits in predictor.propagate_in_video(inference_state=state,start_frame_idx=0):
        idx=int(idx)
        if idx<0 or idx>=n_frames or idx in seen:raise RuntimeError('SAM2 frame order or index invalid')
        seen.add(idx)
        # Shape convention: [num_objects, 1, H, W], object_id=1.
        objects=[int(i) for i in obj_ids]
        if 1 not in objects:raise RuntimeError('Prompted object id disappeared from official output')
        m=np.asarray(logits[objects.index(1)].detach().cpu().numpy()).squeeze()>0
        yield frame2record(idx,m,shape)
    if seen!=set(range(n_frames)):
        raise RuntimeError(f'SAM2 missing {n_frames-len(seen)} video frames')


def prepare_video_frames(video_path, work_dir):
    """Write numbered frames without altering source or OpenVLA policy RGB."""
    import imageio.v2 as imageio
    folder=Path(work_dir);folder.mkdir(parents=True,exist_ok=False)
    count=0;shape=None
    with imageio.get_reader(str(video_path)) as reader:
        for frame in reader:
            a=np.asarray(frame)
            if a.ndim!=3 or a.shape[-1]!=3:raise ValueError('Expected HWC RGB video')
            if shape is None:shape=a.shape
            if a.shape!=shape:raise ValueError('Video dimensions changed')
            imageio.imwrite(folder/f'{count:05d}.jpg',a,quality=97)
            count+=1
    if not count:raise ValueError('Empty video')
    return shape,count


def sha256_file(path):
    digest=hashlib.sha256()
    with Path(path).open('rb') as f:
        for block in iter(lambda:f.read(1024*1024),b''):
            digest.update(block)
    return digest.hexdigest()


def doctor(checkpoint):
    """Non-destructive checks BEFORE a pretrained GPU run. No pip operations."""
    import importlib.util
    import sys
    p=Path(checkpoint)
    report={'checkpoint_exists':p.is_file(),'checkpoint_bytes':p.stat().st_size if p.is_file() else 0,
            'modules':{name:bool(importlib.util.find_spec(name)) for name in ('torch','sam2','imageio','imageio_ffmpeg')},
            'cuda':False,'torch_version':None,'torchvision_version':None,
            'sam2_api_import_ok':False,'diagnostics':[],
            'sam2_model_inference_tested':False}
    if sys.version_info < (3,10):report['diagnostics'].append('Python >=3.10 required by official SAM2')
    if report['modules']['torch']:
        import torch
        report['cuda']=bool(torch.cuda.is_available())
        report['torch_version']=torch.__version__
        try:
            parts=tuple(int(x) for x in torch.__version__.split('+')[0].split('.')[:2])
            if parts < (2,5):report['diagnostics'].append('Official SAM2 requires torch >=2.5.1; use SEPARATE ENV')
        except ValueError:report['diagnostics'].append('Cannot validate PyTorch version')
    try:
        import torchvision
        report['torchvision_version']=torchvision.__version__
    except Exception as exc:
        report['diagnostics'].append('torchvision import/version mismatch: '+str(exc)[:150])
    if report['modules']['sam2']:
        try:
            from sam2.build_sam import build_sam2_video_predictor
            report['sam2_api_import_ok']=callable(build_sam2_video_predictor)
        except Exception as exc:
            report['diagnostics'].append('Official SAM2 API import failed: '+str(exc)[:180])
    if not report['checkpoint_exists']:report['diagnostics'].append('Official checkpoint not found; NO automatic download')
    report['ready_for_gpu_smoke']=bool(report['checkpoint_exists'] and report['checkpoint_bytes']>1_000_000
        and all(report['modules'].values()) and report['cuda'] and report['sam2_api_import_ok']
        and report['torchvision_version'] and not report['diagnostics'])
    return report


def run(video,output,bbox,checkpoint,model_cfg=OFFICIAL_CONFIG):
    video=Path(video).resolve(); output=Path(output).resolve(); checkpoint=Path(checkpoint).resolve()
    if not video.is_file():raise FileNotFoundError(video)
    if not checkpoint.is_file():raise FileNotFoundError('Official pretrained checkpoint required: '+str(checkpoint))
    readiness=doctor(checkpoint)
    if not readiness['ready_for_gpu_smoke']:raise RuntimeError('SAM2 environment/weights preflight failed: '+json.dumps(readiness))
    if output.exists():raise FileExistsError('Do not overwrite previous candidate run: '+str(output))
    import torch
    if not torch.cuda.is_available():raise RuntimeError('SAM2 requires allocated GPU node; NEVER change libero-eval env to fix this')
    from sam2.build_sam import build_sam2_video_predictor
    with tempfile.TemporaryDirectory(prefix='sam2_causal_forward_') as temp:
        shape,n=prepare_video_frames(video,Path(temp)/'frames')
        box=validate_box(bbox,shape)
        predictor=build_sam2_video_predictor(model_cfg,str(checkpoint),device='cuda')
        with torch.inference_mode(), torch.autocast(device_type='cuda',dtype=torch.bfloat16):
            rows=list(forward_from_predictor(predictor,Path(temp)/'frames',box,shape,n))
    # Write only on complete success; no partial claim or accidental overwrite.
    output.mkdir(parents=True,exist_ok=False)
    with (output/'sam2_candidate_OFFLINE_NOT_TRAIN_APPROVED.jsonl').open('x',encoding='utf-8') as f:
        for row in rows:f.write(json.dumps(row,allow_nan=False)+'\n')
    metadata={'source_video':str(video),'source_sha256':sha256_file(video),
              'model_cfg':model_cfg,'checkpoint':str(checkpoint),
              'checkpoint_sha256':sha256_file(checkpoint),
              'source_bbox':list(box),'bbox_provenance':'human_reviewed_initial_clean_rgb',
              'policy_input_modified':False, 'contains_sim_truth':False,
              'temporal_direction':'forward only',
              'formal_online_causality_verified':False,
              'identity_verified':False,'evaluation_status':'CANDIDATE_REQUIRES_HUMAN_REVIEW',
              'frames':n,'candidates':sum(r['candidate_pixel_xy'] is not None for r in rows),
              'never_merge_directly_to_training':True}
    (output/'manifest.json').write_text(json.dumps(metadata,indent=2),encoding='utf-8')
    return metadata


def main(argv=None):
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--doctor',action='store_true',help='Check isolated SAM2 environment without running inference')
    p.add_argument('--video')
    p.add_argument('--output')
    p.add_argument('--bbox',help='Manually confirmed x0,y0,x1,y1')
    p.add_argument('--checkpoint',required=True,help='Official SAM2.1 tiny weights')
    p.add_argument('--model-config',default=OFFICIAL_CONFIG)
    a=p.parse_args(argv)
    if a.doctor:print(json.dumps(doctor(a.checkpoint),indent=2));return
    if not all((a.video,a.output,a.bbox)):p.error('--video, --output and --bbox required for inference')
    print(json.dumps(run(a.video,a.output,list(map(int,a.bbox.split(','))),a.checkpoint,a.model_config),indent=2))
if __name__=='__main__':main()
