# OpenVLA–LIBERO Occlusion Study

This repository contains a curated, reproducibility-oriented research snapshot of
an OpenVLA–LIBERO occlusion study. It is intended for discussion with robotics and
machine learning researchers, including prospective PhD advisors.

## Scope

The study investigates how visual occlusion and lightweight contact-aware
interventions affect a frozen OpenVLA policy in LIBERO-style manipulation
experiments. The repository currently contains six first-party research modules:

- `occlusion_eval`: occlusion protocols, diagnostics, 2D motion support, and
  paired analysis utilities.
- `learned_contact`: contact-state estimation, data contracts, auditing, and
  CPU-side training/evaluation utilities.
- `grasp_v2`: V2/V2.1 candidate-grasp scoring, action overrides, probe interfaces,
  and validation tests.
- `contact_cpu`: CPU-only contact gates and gripper audits.
- `physics_occlusion`: occlusion compensation interfaces and smoke-testable
  physics adapters.
- `study_ready`: offline readiness checks and training/data-quality gates.

The implementation is research code, not a production robot controller.

## Evidence status

The evidence is deliberately separated into three categories:

1. **Completed recorded experiments**: selected summaries in `results/`, including
   paired real-run records and post-grasp audit metadata.
2. **Failure and uncertainty cases**: selected offline review reports and one
   failed learned-contact demonstration in `media/`.
3. **Unvalidated algorithms**: Learned Contact and Grasp V2/V2.1 remain
   experimental. Their presence does **not** establish improved task success,
   calibrated probabilities, or a deployable policy.

The current snapshot must not be interpreted as a statistically validated
performance comparison. Some results use small development episode sets, and
some diagnostics are offline or CPU-only. The reports preserve these limitations.

## Reproducibility and safety

Commands and configurations are included for CPU/offline inspection and future
reproduction. GPU execution is intentionally not performed as part of this
snapshot. A future GPU study must use an allocated compute node, an independently
verified environment, explicit seeds, and a pre-registered evaluation protocol.

Model weights, checkpoints, complete trajectories, processor tensors, training
datasets, caches, credentials, private server paths, and third-party source code
are intentionally excluded.

## Repository layout

```text
experiments/
  occlusion_eval/
  learned_contact/
  grasp_v2/
  contact_cpu/
  physics_occlusion/
  study_ready/
results/       Curated summaries and reports only
media/         Small, selected evidence images and videos
```

## Interpretation of Learned V2/V2.1

The Learned Contact and Grasp V2/V2.1 modules define hypotheses, data contracts,
and intervention logic. They do not prove that learned intervention improves
success rate. In particular, development scores must not be reported as
closed-loop success rates or calibrated probabilities without additional
multi-seed real-probe data and an independent evaluation split.

## License and attribution

This snapshot contains first-party research additions only. The upstream
OpenVLA, LIBERO, MuJoCo, and other dependencies are not vendored here. Their
licenses and citation requirements must be checked before publishing a full
reproduction package.
