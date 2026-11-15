#!/usr/bin/env bash
set -euo pipefail
cd "$HOME/openvla_eval"
export PYTHONPATH="$HOME/robot_libero/LIBERO:$HOME/openvla_eval${PYTHONPATH:+:$PYTHONPATH}"
python experiments/occlusion_eval/run_gpu.py \
  --pose-source rgb_plane \
  --rgb-roi 0.16517857142857142,0.5267857142857143,0.3169642857142857,0.6607142857142857 \
  --motion-2d-roi 0.16517857142857142,0.5267857142857143,0.3169642857142857,0.6607142857142857 \
  --object-center-height 0.8984 \
  --local-mask-plan experiments/occlusion_eval/postgrasp_dev/postgrasp_plan_80_87_exploratory.json \
  --gate-config experiments/occlusion_eval/postgrasp_dev/m2_gate_dev_from_clean.json \
  --methods occluded,m0,m1_2d,m2_2d \
  --task-ids 0 \
  --episode-ids 0 \
  --max-steps 110 \
  --fail-fast
