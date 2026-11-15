# Evidence Status

This file prevents development artifacts from being presented as validated
algorithmic results.

## Completed recorded experiments

- `results_15.csv`: recorded Episode 3, 5, and 10 paired runs for clean,
  occluded, hold-fixed, rule, and learned variants.
- `comparison.csv`: post-grasp Episode 5 audit with execution-integrity checks.
- `cpu_trace01_real_REPORT.md`: CPU/offline contact-trace analysis.
- `physics_occlusion_smoke_REPORT.md`: physics-interface smoke-test report.

These are historical records from a small development set, not a statistically
powered benchmark.

## Failure and uncertainty evidence

- `offline_review_REPORT.md` documents non-comparable clean/occluded diagnostics,
  renderer-only checks, and missing online evidence.
- `offline_recovery_README_RESULTS.md` records an offline replay that must not be
  treated as new data or ground truth.
- `media/ep5_learned_contact_failure.mp4` is a selected visual failure case.

## Not validated

Learned Contact and Grasp V2/V2.1 are hypotheses and implementation components.
No claim is made that they improve task success, provide calibrated probabilities,
or outperform the frozen baseline. Real-probe collection, independent training
validation, multi-seed evaluation, and GPU closed-loop validation remain future
work.
