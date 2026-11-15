"""Memory + rigid robot-motion hypotheses with explicit uncertainty handling."""
from dataclasses import dataclass
import numpy as np

@dataclass(frozen=True)
class Mixture:
    probability_attached: float
    memory_pixel: tuple
    rigid_pixel: tuple
    weights: tuple
    decision: str
    cue_pixel: tuple | None
    proxy_geometry: str = 'visible_rgb_centroid_pixel_not_3d_center_projection'


def combine(memory_pixel,rigid_pixel,p,*,low=.25, high=.75, max_age=None,
            age=None):
    if memory_pixel is None or rigid_pixel is None:
        return None
    if not np.isfinite(p) or not 0<=p<=1:
        raise ValueError('P(attached) must lie in [0,1]')
    m=np.asarray(memory_pixel,float);r=np.asarray(rigid_pixel,float)
    if m.shape!=(2,) or r.shape!=(2,) or not np.isfinite(m).all() or not np.isfinite(r).all():
        raise ValueError('invalid component positions')
    uncertain = low < p < high or (max_age is not None and (age is None or age > max_age))
    decision='uncertain' if uncertain else ('rigid' if p>=high else 'memory')
    # No synthesized "best guess" cue in the uncertain region: report both components.
    cue=None if uncertain else tuple((r if decision=='rigid' else m).tolist())
    return Mixture(float(p),tuple(m),tuple(r),(float(1-p),float(p)),decision,cue)
