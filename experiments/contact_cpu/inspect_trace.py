"""Read-only NPZ diagnostics. Never use these privileged summaries as a policy input.

Quaternion norm checks do NOT establish quaternion component order; gripper
min/max from this trace do NOT calibrate its open/closed endpoints.
"""
import argparse
import json
from pathlib import Path
import numpy as np
from evaluate import load


def inspect(d):
    bowl = d['bowl_pos']
    eef = d['eef_pos']
    steps = d['step'].astype(int)
    pick = sorted(set(np.linspace(0, len(steps)-1, min(8, len(steps)), dtype=int)))
    return dict(
        shapes={k:list(v.shape) for k,v in d.items()},
        first_step=int(steps[0]), last_step=int(steps[-1]),
        bowl_total_displacement_coordinate_units=float(np.linalg.norm(bowl[-1]-bowl[0])),
        eef_total_displacement_coordinate_units=float(np.linalg.norm(eef[-1]-eef[0])),
        max_step_gap=int(np.diff(steps).max()),
        gripper_min=d['gripper_qpos'].min(axis=0).tolist(),
        gripper_max=d['gripper_qpos'].max(axis=0).tolist(),
        quaternion_norm_range={k:[float(np.linalg.norm(d[k],axis=1).min()),
                                  float(np.linalg.norm(d[k],axis=1).max())]
                               for k in ('bowl_quat', 'eef_quat')},
        samples=[dict(step=int(steps[i]),
                      gripper_qpos=d['gripper_qpos'][i].tolist(),
                      eef_bowl_center_distance_coordinate_units=float(np.linalg.norm(bowl[i]-eef[i])))
                 for i in pick],
        warnings=[
            'Object pose is privileged offline truth; never pass hidden values to policy or contact gate.',
            'Confirm coordinates, frame, units, both quaternion orders and synchronization in original logger.',
            'Samples and gripper min/max are diagnostics, NOT open/closed calibration or contact labels.',
            'Distance is between pose reference points, not a mesh-to-gripper contact measurement.'
        ])


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('trace', type=Path)
    args = p.parse_args()
    print(json.dumps(inspect(load(args.trace.expanduser())), indent=2))


if __name__ == '__main__':
    main()
