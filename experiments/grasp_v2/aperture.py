"""One calibrated gripper-aperture contract, shared by probe and online paths.

For the current LIBERO Panda, verified calibration used qpos[0] - qpos[1].
Do not use mean(qpos): the fingers move in opposite directions.
"""
import numpy as np


def aperture_from_qpos(qpos):
    q = np.asarray(qpos, dtype=float)
    if q.shape != (2,) or not np.isfinite(q).all():
        raise ValueError('robot0_gripper_qpos must be two finite joint coordinates')
    return float(q[0] - q[1])
