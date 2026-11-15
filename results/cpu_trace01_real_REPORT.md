# CPU physical prediction evaluation report

## Scope

- One 79-step trace, steps 10-88; offline privileged-visible-prefix diagnostic only.
- M0 Object Memory, M1 rigid SE(3), and M2 Contact-Aware gate.
- Windows: `11:24`, `25:39`, `40:54`, `55:69`, `59:73`, `70:84`. These preserve the predefined window lengths after shifting by +10 because the trace starts at step 10.
- Run assumption: object `xyzw`, eef `xyzw`. The eef order is supported by the logger conversion; object order is not independently proven from the NPZ.

## Mean position error by window (m)

| Window | Frames | M0 | M1 | M2 conditional | M2 resolved |
|---|---:|---:|---:|---:|---:|
| 11:24 | 14 | 0.000000 | 0.028171 | N/A | 0.0% |
| 25:39 | 15 | 0.000000 | 0.104367 | 0.000000 | 100.0% |
| 40:54 | 15 | 0.000000 | 0.091414 | 0.000000 | 100.0% |
| 55:69 | 15 | 0.024475 | 0.014391 | 0.024475 | 100.0% |
| 59:73 | 15 | 0.053581 | 0.017648 | N/A | 0.0% |
| 70:84 | 15 | 0.071716 | 0.000024 | N/A | 0.0% |

Pooled rows are diagnostic only because windows overlap. M0 mean is
`0.025243 m`, M1 mean is `0.042832 m`, and M2 conditional mean is
`0.008158 m` on 50.56% of frames. On those same resolved frames, M0 is
`0.008158 m` and M1 is `0.070057 m`.

## Curves and cases

Per-step error curves are in `plots/`; exact values, modes, reasons, and
hypothesis separation are in `frames.csv`.

- M1 effective cases: windows 55:69, 59:73, and 70:84 (M1 means
  `0.014391`, `0.017648`, and `0.000024 m` versus M0
  `0.024475`, `0.053581`, and `0.071716 m`).
- M1 failure cases: windows 25:39 and 40:54 (M1 means `0.104367` and
  `0.091414 m` while M0 is approximately zero).
- M2 did not select physics. With no gripper calibration, the resolved
  frames selected memory; this is a data limitation, not evidence of M2
  superiority or inferiority.

## Contact state and uncertainty

- 14 frames: `uncertain` from insufficient visible history.
- 45 frames: `memory`.
- 30 frames: `uncertain` from ambiguous evidence.
- Overall M2 resolved fraction: 50.56%; uncertainty fraction: 49.44%.
- Physics selections: 0.

## Data leakage

The current CPU interface passed the causal leakage checks. Hidden object
pose/quaternion replacement leaves predictions and gate metadata invariant;
the corresponding unit test passed. Hidden truth is used only after
prediction for scoring. Full machine-readable results are in
`data_leakage_check.json`.

## Validation and incomplete items

- Unit tests: **22/22 passed**, including leakage invariance and headless
  plotting.
- Command:

```bash
cd experiments/contact_cpu
conda run -n libero-eval python evaluate.py \
  ../../experiments/logs/EVAL-libero_spatial-openvla-2026_11_06-00_52_46--physics_trace_01_ep0_states.npz \
  --windows 11:24,25:39,40:54,55:69,59:73,70:84 \
  --object-quat-order xyzw --eef-quat-order xyzw \
  --config config.example.json \
  --out ../results/cpu_trace01_real_20261106
```

Not completed: independent gripper calibration, independently verified bowl
quaternion provenance, multiple independent episodes, visual-only deployment,
and decision/task-success evaluation. No OpenVLA, MuJoCo, GPU job, training,
source NPZ, or historical result was modified.
