"""Audit gripper qpos geometry without inferring contact or calibration labels."""
import argparse
import json
from pathlib import Path

import numpy as np


def audit(path):
    with np.load(path, allow_pickle=False) as data:
        if "gripper_qpos" not in data:
            raise ValueError("NPZ must contain gripper_qpos")
        q = np.array(data["gripper_qpos"], dtype=float, copy=True)
        if q.ndim != 2 or q.shape[1] != 2 or not np.isfinite(q).all():
            raise ValueError("gripper_qpos must be finite with shape (n, 2)")
        steps = np.array(data["step"], copy=True) if "step" in data else None
    aperture = q[:, 0] - q[:, 1]
    return {
        "path": str(Path(path).expanduser().resolve()),
        "n_rows": int(len(q)),
        "step_range": [int(steps[0]), int(steps[-1])] if steps is not None else None,
        "joint_min": q.min(axis=0).tolist(),
        "joint_max": q.max(axis=0).tolist(),
        "aperture_definition": "qpos[0] - qpos[1]",
        "aperture_min": float(aperture.min()),
        "aperture_max": float(aperture.max()),
        "joint_sum_max_abs": float(np.max(np.abs(q.sum(axis=1)))),
        "observed_joint_relation": "approximately opposite-sign and equal-magnitude",
        "calibration_status": "not_calibrated",
        "warning": (
            "Observed extrema are not independent fully-open/fully-closed "
            "calibration endpoints and do not establish contact."
        ),
    }


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("trace", type=Path)
    parser.add_argument("--out", type=Path)
    args = parser.parse_args()
    payload = audit(args.trace)
    text = json.dumps(payload, indent=2, ensure_ascii=False) + "\n"
    if args.out:
        args.out.write_text(text, encoding="utf-8")
    else:
        print(text, end="")


if __name__ == "__main__":
    main()
