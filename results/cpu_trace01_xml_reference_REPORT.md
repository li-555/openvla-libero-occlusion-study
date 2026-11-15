# Gripper audit and Physics-branch validation

## What was verified

The installed robosuite Panda gripper XML defines:

- `finger_joint1`: range `0.0..0.04`
- `finger_joint2`: range `-0.04..0.0`
- open/close scalar used by the existing gate: `q0 - q1`
- approximate XML endpoint reference: closed `0.0`, open `0.08`

The recorded trace has `q0-q1` range `0.003573..0.079596` and maximum
`abs(q0+q1)` of `0.000118`, confirming the opposite-sign/equal-magnitude
relationship. This does **not** constitute independent calibration: the trace
contains no separately labeled fully-open and fully-closed sequences without
an object, and contains no contact labels or force/touch evidence.

The new read-only audit is
[`gripper_audit.py`](../../gripper_audit.py), with its tests in
[`test_gripper_audit.py`](../../tests/test_gripper_audit.py).

## Why the original M2 had no Physics selection

With null gripper endpoints, `Config.grip()` returns `unknown`. The Physics
gate requires every historical gripper state to be `closed`, so Physics is
disabled by design. The original trace also contains windows where the object
is static during robot motion; those correctly select Memory instead.

## XML-reference sensitivity run

This independent result uses
[`config.xml_reference.json`](../../v2_gripper_audit_20261106/config.xml_reference.json)
with the XML endpoint values only. It is explicitly a sensitivity run, not a
claim of empirical calibration or contact-classification accuracy.

| Window | M0 position m | M1 position m | M2 conditional m | Memory | Physics | Uncertain |
|---|---:|---:|---:|---:|---:|---:|
| 11:24 | 0.000000 | 0.028171 | N/A | 0 | 0 | 14 |
| 25:39 | 0.000000 | 0.104367 | 0.000000 | 15 | 0 | 0 |
| 40:54 | 0.000000 | 0.091414 | 0.000000 | 14 | 0 | 1 |
| 55:69 | 0.024475 | 0.014391 | 0.024475 | 15 | 0 | 0 |
| 59:73 | 0.053581 | 0.017648 | N/A | 0 | 0 | 15 |
| 70:84 | 0.071716 | 0.000024 | 0.000024 | 0 | 15 | 0 |

Across 89 overlapping diagnostic rows:

- M0: `0.025243 m`
- M1: `0.042832 m`
- M2 conditional: `0.006229 m`
- M2 resolved: `59/89 = 66.29%`
- Selection proportions: Memory `44/89 = 49.44%`, Physics
  `15/89 = 16.85%`, Uncertain `30/89 = 33.71%`
- On the same 59 resolved frames: M0 `0.024455 m`, M1 `0.050736 m`,
  M2 `0.006229 m`

The Physics branch is now exercised in window `70:84`; its M2 result matches
M1 on those selected frames. This is consistent with the rigid co-motion
heuristic, but cannot validate physical contact because the endpoint values
come from XML limits rather than an independent calibration experiment.

Plots, per-frame CSV, predictions, and summary JSON are in this result
directory. Hidden object state remains scoring-only; it is not passed to
`ContactGate.predict()` or used for mode selection.

## Tests and command

**24/24 unit tests passed.**

```bash
cd experiments/contact_cpu
python -m unittest discover -s tests -v
python gripper_audit.py \
  ../../experiments/logs/EVAL-libero_spatial-openvla-2026_11_06-00_52_46--physics_trace_01_ep0_states.npz
python evaluate.py \
  ../../experiments/logs/EVAL-libero_spatial-openvla-2026_11_06-00_52_46--physics_trace_01_ep0_states.npz \
  --windows 11:24,25:39,40:54,55:69,59:73,70:84 \
  --object-quat-order xyzw --eef-quat-order xyzw \
  --config v2_gripper_audit_20261106/config.xml_reference.json \
  --out ../results/cpu_trace01_xml_reference_20261106
```

## Still unresolved

1. Independent fully-open/fully-closed gripper calibration.
2. Contact/force/touch evidence or labeled grasp/release episodes.
3. Independent episodes for generalization; these six windows overlap.
4. Independent provenance confirmation for the bowl quaternion order.
