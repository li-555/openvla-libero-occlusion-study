"""Feature contract shared by probe training and on-policy decision time.

All required fields are publicly observable BEFORE taking the candidate action.
GT outcomes are labels and MUST never be added to this feature vector.
"""
from dataclasses import dataclass
import numpy as np

FEATURE_NAMES = (
    'target_minus_eef_u', 'target_minus_eef_v',
    'eef_x_m','eef_y_m','eef_z_m',
    'eef_vel_x','eef_vel_y','eef_vel_z',
    'gripper_aperture_m','last_seen_age_normalized','target_confidence',
    'previous_action_x','previous_action_y','previous_action_z',
    'proposed_action_x','proposed_action_y','proposed_action_z',
    'candidate_action_x','candidate_action_y','candidate_action_z',
    'close_now',
)


@dataclass(frozen=True)
class PreAction:
    """Pre-action public observation; no full LIBERO obs/env or object pose.

    target_px is inferred from pre-occlusion vision and may be stale. Its
    reliability is explicitly encoded, and missing anchors are rejected.
    """
    target_px: tuple
    eef_px: tuple
    eef_xyz_m: tuple
    eef_vel_m: tuple
    gripper_aperture_m: float
    last_seen_age: int
    target_confidence: float
    previous_action_xyz: tuple
    proposed_action_xyz: tuple

    def vector(self, candidate_xyz, close_now):
        target=np.asarray(self.target_px, dtype=float)
        eefpx=np.asarray(self.eef_px, dtype=float)
        xyz=np.asarray(self.eef_xyz_m, dtype=float)
        vel=np.asarray(self.eef_vel_m, dtype=float)
        prior=np.asarray(self.previous_action_xyz, dtype=float)
        prop=np.asarray(self.proposed_action_xyz, dtype=float)
        cand=np.asarray(candidate_xyz, dtype=float)
        if any(a.shape != shape for a,shape in ((target,(2,)),(eefpx,(2,)),
                (xyz,(3,)),(vel,(3,)),(prior,(3,)),(prop,(3,)),(cand,(3,)))):
            raise ValueError('bad pre-action/candidate vector shape')
        if not isinstance(close_now,(bool,np.bool_)):
            raise ValueError('close_now must be bool')
        arr=np.concatenate(((target-eefpx)/224.0, xyz, vel,
            [float(self.gripper_aperture_m), min(int(self.last_seen_age),100)/100.,
             float(self.target_confidence)],prior,prop,cand,[float(close_now)]))
        if arr.shape!=(len(FEATURE_NAMES),) or not np.isfinite(arr).all():
            raise ValueError('invalid or non-finite public input')
        if not 0<=self.target_confidence<=1 or self.last_seen_age<0:
            raise ValueError('invalid target confidence/age')
        return arr


def obs_from_dict(row):
    """Strict whitelist prevents post-action ground-truth leakage."""
    allowed=set(PreAction.__dataclass_fields__)
    required=allowed
    if set(row)!=required:  # including rejecting extras such as object_pos_GT
        raise ValueError(f'pre_action requires EXACT public keys; missing={required-set(row)}, extra={set(row)-allowed}')
    return PreAction(**{k:tuple(v) if isinstance(v,list) else v for k,v in row.items()})
