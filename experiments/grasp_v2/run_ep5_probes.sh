#!/usr/bin/env bash
# Run only on a GPU-allocated HKU compute node. Exact original files are discovered
# by their frozen plan hash, window, and initial-state ID; ambiguous -> abort.
set -euo pipefail
cd "$HOME/openvla_eval"
source "$HOME/anaconda3/etc/profile.d/conda.sh"
conda activate libero-eval
export MUJOCO_GL=egl PYOPENGL_PLATFORM=egl TOKENIZERS_PARALLELISM=false
export VLA_OCCLUSION=clean
export PYTHONPATH="$HOME/robot_libero/LIBERO:$PWD:$PWD/experiments/contact_cpu:$PWD/experiments/physics_occlusion:${PYTHONPATH:-}"
OUTPUT="experiments/grasp_v2/data/dev_ep5_step36_eight_probes.jsonl"
BASELINE="experiments/real_collection_runs/dev10/closed_loop_dev15/pregrasp_ep5/results_occluded/eval_20261109T145505287209Z/task0_ep5_occluded.json"
PLAN="experiments/real_collection_runs/dev10/closed_loop_dev15/pregrasp_ep5/plan_36_51_bowl_tight.json"
python -m experiments.grasp_v2.libero_replay_probe \
  --dry-run --output "$OUTPUT" --baseline-json "$BASELINE" --frozen-plan "$PLAN"
python -m experiments.grasp_v2.libero_replay_probe \
  --output "$OUTPUT" --checkpoint <OPENVLA_CHECKPOINT> \
  --split development --xyz-step 0.12 --lift-steps 12 --max-reserved-gib 15 \
  --baseline-json "$BASELINE" --frozen-plan "$PLAN"
