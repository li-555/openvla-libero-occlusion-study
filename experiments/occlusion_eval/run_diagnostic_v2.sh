#!/usr/bin/env bash
set -euo pipefail
cd "$(dirname "$0")/../.."
export PYTHONPATH="$HOME/robot_libero/LIBERO:$PWD${PYTHONPATH:+:$PYTHONPATH}"
python -c 'import torch; assert torch.cuda.is_available(), "Activate libero-eval on an allocated GPU node"'
OUT="experiments/occlusion_eval/results/diagnostic_v2_$(date -u +%Y%m%dT%H%M%SZ)_$$"
mkdir -p "$OUT"
set +e
python experiments/occlusion_eval/run_gpu.py \
 --pose-source rgb_plane \
 --motion-2d-roi 0.16517857142857142,0.5267857142857143,0.3169642857142857,0.6607142857142857 \
 --local-mask-plan experiments/occlusion_eval/postgrasp_dev/postgrasp_plan_80_87_exploratory.json \
 --gate-config experiments/occlusion_eval/postgrasp_dev/m2_gate_dev_from_clean.json \
 --methods clean,occluded,m0_2d,m1_2d,m2_2d,occluded_hold,m2_2d_hold \
 --task-ids 0 --episode-ids 0 --seed 7 --max-steps 220 --output-root "$OUT" --fail-fast 2>&1 | tee "$OUT/console.log"
STATUS=${PIPESTATUS[0]}
set -e
python experiments/occlusion_eval/summarize_diagnostic.py "$OUT"
tar -czf "${OUT}.tar.gz" "$OUT"
printf 'Results: %s.tar.gz\n' "$OUT"
exit "$STATUS"
