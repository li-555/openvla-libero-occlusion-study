"""Explicit, immutable pre-occlusion mask plans. No simulator access."""
import hashlib
import json
from pathlib import Path
import numpy as np
from PIL import Image


def rgb_hash(rgb):
    return hashlib.sha256(np.asarray(rgb, dtype=np.uint8).tobytes()).hexdigest()


def rectangle(shape, roi):
    h, w = shape[:2]
    x0, y0, x1, y1 = map(float, roi)
    if not (0 <= x0 < x1 <= 1 and 0 <= y0 < y1 <= 1):
        raise ValueError('Invalid normalized rectangle')
    mask = np.zeros((h, w), dtype=bool)
    mask[int(np.floor(y0*h)):int(np.ceil(y1*h)),
         int(np.floor(x0*w)):int(np.ceil(x1*w))] = True
    return mask


def read_plan(path):
    p = json.loads(Path(path).read_text(encoding='utf-8'))
    if p.get('schema') != 'local_occlusion_plan_v1':
        raise ValueError('Unknown plan schema')
    if not 0 <= p['anchor_step'] <= p['start_step'] <= p['end_step']:
        raise ValueError('Anchor must precede or equal the first masked observation')
    duration = p['end_step'] - p['start_step'] + 1
    if duration not in (8, 16):
        raise ValueError('Local plans require exactly 8 or 16 policy steps')
    m = rectangle(p['image_shape'], p['roi'])
    if m.mean() > 0.08:
        raise ValueError('Rasterized mask exceeds provisional 8% area limit')
    if len(p['anchor_rgb_sha256']) != 64:
        raise ValueError('Missing anchor hash')
    if len(p['fill']) != 3 or any(not isinstance(v,int) or not 0 <= v <= 255 for v in p['fill']):
        raise ValueError('Invalid fill')
    return p


def apply_plan_args(args):
    """Called AFTER task-spec parsing so task specs cannot silently override plan."""
    path = args.local_mask_plan
    if not path:
        raise ValueError('Supply --local-mask-plan; the old central mask is disabled')
    p = read_plan(path)
    if args.task_ids != [p['task_id']] or args.episode_ids != [p['episode_id']]:
        raise ValueError('One plan is valid only for its declared task and initial-state ID')
    s = args.specs[str(p['task_id'])]
    s.update(mask=tuple(p['roi']), start_step=p['start_step'], end_step=p['end_step'])
    args.mask = tuple(p['roi'])
    args.start_step, args.end_step = p['start_step'], p['end_step']
    args.save_every = 1
    args.local_plan = p
    print('LOCAL MASK ENABLED:', json.dumps({k:p[k] for k in ('roi','fill','anchor_step','start_step','end_step')}))


def validate_anchor(rgb, step, plan):
    if step == plan['anchor_step']:
        if list(rgb.shape) != plan['image_shape'] or rgb_hash(rgb) != plan['anchor_rgb_sha256']:
            raise RuntimeError('Anchor image differs from plan. Stop: do not reuse a mask from another episode or an MP4 extraction. Capture an exact raw PNG from the same clean prefix.')


class PixelAudit:
    """Wrap the existing audit proxy, preserving its PNG/video behavior."""
    def __init__(self, original):
        object.__setattr__(self, '_original', original)
    def __getattr__(self, name):
        return getattr(self._original, name)
    def __setattr__(self, name, value):
        setattr(self._original, name, value)
    def __call__(self, *args, **kwargs):
        path = getattr(self._original, 'audit_path', None)
        result = self._original(*args, **kwargs)
        if path is not None:
            value = result.get('pixel_values')
            if value is None:
                raise RuntimeError('Processor output has no pixel_values')
            arr = value.detach().float().cpu().numpy()
            path = Path(path)
            np.savez_compressed(path.with_suffix('.pixel_values.npz'), pixel_values=arr)
            path.with_suffix('.pixel_values.json').write_text(json.dumps({
                'source': 'actual_processor_return', 'shape': list(value.shape),
                'original_dtype': str(value.dtype), 'saved_dtype': str(arr.dtype),
                'finite': bool(np.isfinite(arr).all()),
                'min': float(arr.min()), 'max': float(arr.max()),
                'float32_sha256': hashlib.sha256(arr.tobytes()).hexdigest()
            }, indent=2), encoding='utf-8')
        return result
