"""Offline audit for pre-occlusion anchored local object masks.

This tool never reads simulator state. It consumes saved RGB/policy images and,
optionally, human-provided masks captured from a legal pre-occlusion frame.
Missing segmentation labels remain UNVERIFIED rather than being inferred.
"""
import argparse
import json
from pathlib import Path

import numpy as np
from PIL import Image, ImageDraw


STAGES = {
    "grasp_pre": range(35, 43),
    "transport_early": range(85, 93),
}
THRESHOLDS = {
    "target": 0.90,
    "gripper": 0.10,
    "finger": 0.15,
    "arm": 0.05,
    "placement": 0.05,
    "area": 0.08,
}


def _load_mask(path, shape):
    if path is None:
        return None
    mask = np.asarray(Image.open(path).convert("L")) > 0
    if mask.shape != shape:
        raise ValueError(f"Mask {path} has shape {mask.shape}, expected {shape}")
    return mask


def _load_annotations(path, shape):
    if path is None:
        return {}
    data = json.loads(Path(path).read_text(encoding="utf-8"))
    result = {}
    for name, mask_path in data.items():
        result[name] = _load_mask(mask_path, shape)
    return result


def _draw_overlay(raw, roi, annotations):
    image = Image.fromarray(raw).convert("RGB")
    draw = ImageDraw.Draw(image)
    h, w = raw.shape[:2]
    x0, y0, x1, y1 = roi
    box = (int(x0 * w), int(y0 * h), int(x1 * w), int(y1 * h))
    draw.rectangle(box, outline=(255, 0, 0), width=2)
    labels = [("target", (255, 255, 0)), ("gripper", (0, 255, 0)),
              ("finger", (0, 200, 255)), ("arm", (255, 128, 0)),
              ("placement", (255, 0, 255))]
    for name, color in labels:
        mask = annotations.get(name)
        if mask is None:
            continue
        ys, xs = np.where(mask)
        if len(xs):
            draw.rectangle((int(xs.min()), int(ys.min()), int(xs.max()), int(ys.max())),
                           outline=color, width=1)
    return np.asarray(image)


def _apply_roi(raw, roi):
    out = raw.copy()
    h, w = out.shape[:2]
    xa, xb = int(roi[0] * w), int(roi[2] * w)
    ya, yb = int(roi[1] * h), int(roi[3] * h)
    out[ya:yb, xa:xb] = 0
    return out


def _mask_panel(raw, roi):
    panel = np.full_like(raw, 255)
    h, w = raw.shape[:2]
    xa, xb = int(roi[0] * w), int(roi[2] * w)
    ya, yb = int(roi[1] * h), int(roi[3] * h)
    panel[ya:yb, xa:xb] = 0
    return panel


def _frame_paths(image_dir, step, raw_dir=None):
    stem = f"{step:04d}"
    extra = raw_dir / f"{stem}_raw.png" if raw_dir else None
    raw_path = extra if extra and extra.exists() else image_dir / f"{stem}_raw.png"
    return (raw_path,
            image_dir / f"{stem}_occluded.png",
            image_dir / f"{stem}_processor.png")


def _display_path(path, run_dir):
    try:
        return str(path.relative_to(run_dir))
    except ValueError:
        return str(path.resolve())


def _video_raw_frame(video_reader, step):
    if video_reader is None:
        return None
    frame = np.asarray(video_reader.get_data(step), dtype=np.uint8)
    return frame[:, :frame.shape[1] // 2]


def audit(run_dir, output_dir, roi, annotation_paths=None, video_path=None, raw_dir=None):
    run_dir, output_dir = Path(run_dir), Path(output_dir)
    image_dir = next((run_dir / "images").glob("task0_ep0_occluded"))
    raw0 = np.asarray(Image.open(image_dir / "0035_raw.png").convert("RGB"))
    annotations = _load_annotations(annotation_paths, raw0.shape[:2])
    video_reader = None
    if video_path is not None:
        import imageio.v2 as imageio
        video_reader = imageio.get_reader(video_path)
    area = (roi[2] - roi[0]) * (roi[3] - roi[1])
    output_dir.mkdir(parents=True, exist_ok=True)
    report = {
        "source_run": str(run_dir),
        "protocol": "Pre-occlusion Anchored Local Object Occlusion",
        "roi_normalized": list(roi),
        "roi_area_fraction": area,
        "thresholds": THRESHOLDS,
        "annotation_status": "VERIFIED" if annotations else "UNVERIFIED",
        "annotation_note": (
            "Masks supplied from legal pre-occlusion observations."
            if annotations else
            "No offline segmentation masks supplied; coverage metrics are UNVERIFIED."
        ),
        "stages": {},
    }
    for stage, steps in STAGES.items():
        frames = []
        reasons = []
        for step in steps:
            raw_path, occluded_path, processor_path = _frame_paths(image_dir, step, raw_dir)
            raw = (np.asarray(Image.open(raw_path).convert("RGB"))
                   if raw_path.exists() else _video_raw_frame(video_reader, step))
            if raw is None:
                reasons.append(f"missing raw frame {step}")
                continue
            occluded = _apply_roi(raw, roi)
            processor = (str(processor_path.relative_to(run_dir))
                         if processor_path.exists() else None)
            metrics = {name: None for name in
                       ("target_coverage", "gripper_coverage", "finger_coverage",
                        "arm_coverage", "placement_coverage")}
            if not annotations:
                reasons.append("segmentation masks unavailable")
            frames.append({
                "step": step,
                "raw": _display_path(raw_path, run_dir) if raw_path.exists() else
                       f"VIDEO_FRAME:{step}",
                "occluded": str(occluded_path.relative_to(run_dir)) if occluded_path.exists() else
                       "GENERATED_OFFLINE_FROM_RAW",
                "processor": processor,
                "processor_status": "AVAILABLE_PNG_ONLY" if processor else "UNVERIFIED",
                "processor_pixel_values": "UNVERIFIED",
                "metrics": metrics,
                "pass": False,
                "status": "UNVERIFIED",
            })
        if frames:
            first = frames[0]
            raw_ref = (Path(first["raw"]) if first["raw"].startswith("/")
                       else run_dir / first["raw"]) if not first["raw"].startswith("VIDEO_FRAME:") else None
            raw = (np.asarray(Image.open(raw_ref).convert("RGB"))
                   if raw_ref is not None else
                   _video_raw_frame(video_reader, first["step"]))
            occluded = _apply_roi(raw, roi)
            processor_path = run_dir / first["processor"] if first["processor"] else None
            processor = (np.asarray(Image.open(processor_path).convert("RGB"))
                         if processor_path and processor_path.exists() else raw)
            panel = np.concatenate((
                _draw_overlay(raw, roi, annotations),
                _mask_panel(raw, roi),
                occluded,
                processor,
            ), axis=1)
            panel_path = output_dir / f"{stage}_comparison.png"
            Image.fromarray(panel).save(panel_path)
        else:
            panel_path = None
        report["stages"][stage] = {
            "steps": frames,
            "pass": False,
            "status": "UNVERIFIED",
            "reasons": reasons or ["segmentation masks unavailable"],
            "comparison": str(panel_path.relative_to(output_dir.parent))
            if panel_path else None,
        }
    (output_dir / "report.json").write_text(
        json.dumps(report, indent=2, allow_nan=False), encoding="utf-8")
    return report


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("run_dir", type=Path)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--roi", default="0.32,0.32,0.70,0.70")
    parser.add_argument("--annotations", type=Path)
    parser.add_argument("--video", type=Path)
    parser.add_argument("--raw-dir", type=Path)
    args = parser.parse_args()
    roi = tuple(float(v) for v in args.roi.split(","))
    if len(roi) != 4 or not (0 <= roi[0] < roi[2] <= 1 and 0 <= roi[1] < roi[3] <= 1):
        parser.error("roi must be x0,y0,x1,y1 in [0,1]")
    report = audit(args.run_dir, args.output_dir, roi, args.annotations, args.video, args.raw_dir)
    print(json.dumps({
        "annotation_status": report["annotation_status"],
        "roi_area_fraction": report["roi_area_fraction"],
        "stages": {k: v["status"] for k, v in report["stages"].items()},
    }, indent=2))


if __name__ == "__main__":
    main()
