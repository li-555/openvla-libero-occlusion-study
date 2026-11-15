# Episode 1: offline verification, NOT new data

Original Clean video: `<source-repository>/experiments/real_collection_runs/dev10/raw/task0_ep1/visual_evidence/task0_ep1_seed14_natural_policy_dropout80-87_releasenone/policy_clean_rgb.mp4`

Frames: 220. Contact true: 5. Confirmed attached true: 0.

Original tracker lock frames in lossy-MP4 replay: 124.

Potential shadow candidates following lock: 11 (NOT ID-VERIFIED).

**Do not treat shadow candidates as ground truth or learned features.**

Human must review the original unmarked image, not a GT magenta marker.

Do not revise the stored trajectory or eval labels. No GPU required.

If target becomes clearly independently identifiable, apply a small gated reacquisition fix to FUTURE data only, then offline replay to test false matches.

If indistinguishable, keep unknown and proceed after logging warning.

Inputs: only policy_clean_rgb.mp4, recorded PRE/POST/EVAL, and human-reviewed RGB initialization bbox.
