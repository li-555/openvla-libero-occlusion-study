import json
import numpy as np
from pathlib import Path
from PIL import Image
import pytest
from experiments.grasp_v2.libero_replay_probe import (
    FROZEN_EP5_ANCHOR,FROZEN_EP5_INITIAL,FROZEN_EP5_ROI,
    discover_ep5_inputs,_require_real_baseline)


def fixture_files(tmp_path):
    root=tmp_path/'pregrasp_ep5';root.mkdir()
    plan={'schema':'local_occlusion_plan_v1','task_id':0,'episode_id':5,
          'anchor_step':35,'start_step':36,'end_step':51,
          'roi':FROZEN_EP5_ROI,'fill':[128]*3,'image_shape':[224,224,3],
          'anchor_rgb_sha256':FROZEN_EP5_ANCHOR}
    p=root/'plan.json';p.write_text(json.dumps(plan))
    records=[{'step':i,'active':bool(i>=36),'action_executed':[0,0,0,0,0,0,-1]} for i in range(37)]
    record={'task_id':0,'episode_id':5,'method':'occluded',
            'initial_state_sha256':FROZEN_EP5_INITIAL,
            'masked_frames':16,'target_center_masked_frames':16,
            'termination':'horizon','frames':records}
    d=root/'results_occluded'/'eval_real'/'task0_ep5_occluded.json'
    d.parent.mkdir(parents=True);d.write_text(json.dumps(record))
    return root,p,d


def test_auto_discover_only_exact_frozen(tmp_path):
    root,p,d=fixture_files(tmp_path)
    a,b=discover_ep5_inputs(root)
    assert (a,b)==(d,p)


def test_auto_discover_rejects_ambiguous(tmp_path):
    root,p,d=fixture_files(tmp_path)
    q=root/'plan_duplicate.json';q.write_bytes(p.read_bytes())
    with pytest.raises(RuntimeError,match='exactly ONE original frozen'):
        discover_ep5_inputs(root)


def test_real_raw_required(tmp_path):
    root,p,d=fixture_files(tmp_path)
    with pytest.raises(FileNotFoundError,match='ORIGINAL step-36 raw'):
        _require_real_baseline(d,json.loads(p.read_text()))


def test_baseline_prefix_and_real_raw(tmp_path):
    root,p,d=fixture_files(tmp_path)
    rawdir=d.parent/'images'/'task0_ep5_occluded';rawdir.mkdir(parents=True)
    raw=np.zeros((224,224,3),dtype=np.uint8)
    Image.fromarray(raw).save(rawdir/'0036_raw.png')
    base,prefix,sha,path=_require_real_baseline(d,json.loads(p.read_text()))
    assert len(prefix)==36 and len(sha)==64 and path.exists()
    assert all(a.shape==(7,) for a in prefix)


def test_no_mask_in_prefix(tmp_path):
    root,p,d=fixture_files(tmp_path)
    obj=json.loads(d.read_text());obj['frames'][35]['active']=True
    d.write_text(json.dumps(obj))
    with pytest.raises(ValueError,match='Unexpected occlusion'):
        _require_real_baseline(d,json.loads(p.read_text()))
