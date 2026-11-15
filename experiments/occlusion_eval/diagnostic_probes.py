"""Privileged same-state image probes for diagnosis ONLY. Never execute actions."""
import json
import random
import numpy as np
from PIL import Image


def record_probes(raw, roi, fill, cfg, model, description, processor,
                  get_action, normalize, invert, output):
    import torch
    py_state = random.getstate()
    np_state = np.random.get_state()
    cpu_state = torch.get_rng_state()
    gpu_state = torch.cuda.get_rng_state_all()
    previous_path = processor.audit_path
    x0,y0,x1,y1=roi
    middle=(y0+y1)/2
    variants={'clean_privileged':None,'upper_half_patch':(x0,y0,x1,middle),
              'lower_half_patch':(x0,middle,x1,y1)}
    rows={}
    try:
        processor.audit_path=None
        for name, rectangle in variants.items():
            random.setstate(py_state); np.random.set_state(np_state)
            torch.set_rng_state(cpu_state); torch.cuda.set_rng_state_all(gpu_state)
            image=raw.copy()
            if rectangle is not None:
                h,w=image.shape[:2]; xa,ya,xb,yb=rectangle
                image[int(np.floor(ya*h)):int(np.ceil(yb*h)),
                      int(np.floor(xa*w)):int(np.ceil(xb*w))]=fill
            with torch.inference_mode():
                action=np.asarray(get_action(cfg,model,{'full_image':image,'state':np.zeros(8)},
                                             description,processor=processor),dtype=float)
            executed=np.asarray(invert(normalize(action.copy(),binarize=True)))
            rows[name]={'action_raw':action.tolist(),'action_decoded':executed.tolist()}
            Image.fromarray(image).save(output.with_name(name+'.png'))
        output.write_text(json.dumps({'label':'DIAGNOSTIC_ONLY_NOT_EXECUTED',
            'note':'Half patches are geometric probes, not verified target/hand isolation.',
            'variants':rows},indent=2),encoding='utf-8')
    finally:
        random.setstate(py_state);np.random.set_state(np_state)
        torch.set_rng_state(cpu_state);torch.cuda.set_rng_state_all(gpu_state)
        processor.audit_path=previous_path
