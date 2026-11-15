"""Offline diagnostic evaluator; truth arrays never enter predict()."""
import argparse
import csv
import hashlib
import json
from pathlib import Path
import numpy as np
from contact_gate import Config, Visible, Robot, ContactGate, pose, errors


FIELDS = dict(step=(), bowl_pos=(3,), bowl_quat=(4,), eef_pos=(3,),
              eef_quat=(4,), gripper_qpos=(2,))


def load(path):
    with np.load(path, allow_pickle=False) as z:
        d = {k: np.array(z[k], copy=True) for k in FIELDS}
    n = len(d["step"])
    if n < 2:
        raise ValueError("Trace requires at least two rows")
    for k, tail in FIELDS.items():
        if d[k].shape != (n,)+tail or not np.isfinite(d[k]).all():
            raise ValueError(f"Invalid {k} shape or nonfinite values")
    s = d["step"]
    if not np.equal(s, s.astype(int)).all() or not (np.diff(s) > 0).all():
        raise ValueError("step must be strictly increasing integers")
    # Validate quaternion integrity, including hidden truth used only for scoring.
    for key in ("bowl_quat", "eef_quat"):
        if (np.linalg.norm(d[key], axis=1) < 1e-12).any():
            raise ValueError(f"Zero quaternion in {key}")
    return d


def predict_window(d, start, end, cfg, obj_order, eef_order):
    """start/end are inclusive STEP LABELS; anchor is exactly start-1.

    This boundary builds a visible prefix before allocating the predictor;
    occluded object rows are never passed to it. Offline visible object poses
    are a privileged measurement proxy, not a deployed visual estimator.
    """
    steps = d["step"].astype(int)
    expected = np.arange(start-1, end+1)
    if start > end or not np.isin(expected, steps).all():
        raise ValueError("Window requires contiguous steps and visible anchor start-1")
    a = int(np.flatnonzero(steps == start-1)[0])
    history = [Visible(int(steps[i]), pose(d["eef_pos"][i], d["eef_quat"][i], eef_order),
                       pose(d["bowl_pos"][i], d["bowl_quat"][i], obj_order),
                       d["gripper_qpos"][i]) for i in range(max(0, a-cfg.history+1), a+1)]
    gate = ContactGate(history, cfg)
    predictions = []
    for i in np.flatnonzero((steps >= start) & (steps <= end)):
        predictions.append(gate.predict(Robot(int(steps[i]),
            pose(d["eef_pos"][i], d["eef_quat"][i], eef_order), d["gripper_qpos"][i])))
    return predictions, gate.metadata()


def score(d, predictions, obj_order, trace, window):
    indices = {int(s): i for i, s in enumerate(d["step"])}
    rows = []
    for p in predictions:
        i = indices[p.step]
        truth = pose(d["bowl_pos"][i], d["bowl_quat"][i], obj_order)
        m, r = errors(p.memory, truth), errors(p.physics, truth)
        selected = errors(p.selected, truth) if p.selected is not None else (None, None)
        rows.append(dict(trace=trace, window=window, step=p.step, mode=p.mode, reason=p.reason,
                         m0_pos_m=m[0], m0_rot_rad=m[1], m1_pos_m=r[0], m1_rot_rad=r[1],
                         m2_selected_pos_m=selected[0], m2_selected_rot_rad=selected[1],
                         hypothesis_best_pos_m_ORACLE_BOUND=min(m[0], r[0]),
                         hypothesis_worst_pos_m=max(m[0], r[0]),
                         hypothesis_separation_m=float(np.linalg.norm(p.memory[:3, 3]-p.physics[:3, 3]))))
    return rows


def summarize(rows):
    selected = [r for r in rows if r["m2_selected_pos_m"] is not None]
    def mean(rs, key):
        return float(np.mean([r[key] for r in rs])) if rs else None
    return dict(n_frames=len(rows), resolved_fraction=len(selected)/len(rows) if rows else None,
                uncertain_frames=len(rows)-len(selected),
                m0_mean_pos_m=mean(rows, "m0_pos_m"), m1_mean_pos_m=mean(rows, "m1_pos_m"),
                m0_mean_rot_rad=mean(rows, "m0_rot_rad"), m1_mean_rot_rad=mean(rows, "m1_rot_rad"),
                m2_CONDITIONAL_mean_pos_m=mean(selected, "m2_selected_pos_m"),
                m2_CONDITIONAL_mean_rot_rad=mean(selected, "m2_selected_rot_rad"),
                m0_SAME_RESOLVED_FRAMES_mean_pos_m=mean(selected, "m0_pos_m"),
                m1_SAME_RESOLVED_FRAMES_mean_pos_m=mean(selected, "m1_pos_m"),
                hypothesis_best_pos_m_ORACLE_BOUND=mean(rows, "hypothesis_best_pos_m_ORACLE_BOUND"),
                hypothesis_worst_mean_pos_m=mean(rows, "hypothesis_worst_pos_m"))


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("traces", nargs="+", type=Path)
    ap.add_argument("--windows", required=True, help="Inclusive hidden step labels, e.g. 15:29,45:59")
    ap.add_argument("--object-quat-order", choices=["xyzw", "wxyz"], required=True)
    ap.add_argument("--eef-quat-order", choices=["xyzw", "wxyz"], required=True)
    ap.add_argument("--config", type=Path)
    ap.add_argument("--out", type=Path, required=True, help="NEW directory; refuses overwriting")
    args = ap.parse_args()
    cfg = Config(**json.loads(args.config.read_text(encoding="utf-8"))) if args.config else Config()
    windows = [tuple(map(int, w.split(":"))) for w in args.windows.split(",")]
    if any(len(w) != 2 for w in windows) or len(set(windows)) != len(windows):
        ap.error("Unique start:end windows required")
    paths = [p.expanduser().resolve() for p in args.traces]
    if len(set(paths)) != len(paths):
        ap.error("Duplicate trace path")
    all_rows, report, inputs, prediction_records = [], [], [], []
    hashes = set()
    for path in paths:
        digest = hashlib.sha256(path.read_bytes()).hexdigest()
        if digest in hashes:
            ap.error("Identical trace contents are not independent episodes")
        hashes.add(digest)
        inputs.append(dict(path=str(path), sha256=digest))
        d = load(path)
        for start, end in windows:
            label = f"{start}:{end}"
            preds, metadata = predict_window(d, start, end, cfg, args.object_quat_order, args.eef_quat_order)
            prediction_records.extend(dict(trace=str(path), window=label, step=p.step,
                mode=p.mode, reason=p.reason, memory=p.memory.tolist(), physics=p.physics.tolist(),
                selected=p.selected.tolist() if p.selected is not None else None) for p in preds)
            rows = score(d, preds, args.object_quat_order, str(path), label)
            all_rows.extend(rows)
            report.append(dict(trace=str(path), window=label, anchor=start-1,
                               metadata=metadata, metrics=summarize(rows)))
    # Each episode is the unit of replication; overlapping windows are NOT IID.
    episode_summary = []
    for p in paths:
        rs = [r for r in all_rows if r["trace"] == str(p)]
        # Use one copy per (episode, window, step); pooled summary is diagnostic only.
        episode_summary.append(dict(trace=str(p), metrics=summarize(rs)))
    payload = dict(protocol="offline privileged-visible-prefix diagnostic; no policy evaluation",
                   inputs=inputs, quaternion_orders=dict(obj=args.object_quat_order, eef=args.eef_quat_order),
                   config_source=str(args.config) if args.config else "unvalidated illustrative defaults",
                   windows=report, episodes=episode_summary,
                   pooled_DIAGNOSTIC_ONLY=summarize(all_rows),
                   warnings=["No calibrated contact probabilities or task success claim.",
                             "Conditional M2 error must be reported with coverage and matched baselines.",
                             "Best-of-two truth-selected error is an ORACLE BOUND, not M2 performance.",
                             "Both hypotheses can be wrong after slip, release, or external object motion.",
                             "Overlapping windows are correlated; bootstrap independent episodes, not rows."])
    args.out.mkdir(parents=True, exist_ok=False)
    with (args.out/"frames.csv").open("w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=list(all_rows[0]))
        w.writeheader()
        w.writerows(all_rows)
    (args.out/"summary.json").write_text(json.dumps(payload, indent=2, allow_nan=False), encoding="utf-8")
    with (args.out/"predictions.jsonl").open("w", encoding="utf-8") as f:
        for r in prediction_records:
            f.write(json.dumps(r, allow_nan=False)+"\n")
    print(f"Wrote {len(all_rows)} frame-window rows to {args.out}")


if __name__ == "__main__":
    main()
