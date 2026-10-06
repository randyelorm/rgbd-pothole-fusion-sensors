#!/usr/bin/env python3
"""
Late fusion: average RGB and depth confidences (thesis Eq. style), then NMS.

C_combined = (C_rgb + C_depth) / 2 for matched detections; unmatched kept with half weight.
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np
import torch
import yaml
from ultralytics import YOLO
from ultralytics.utils.metrics import SegmentMetrics
from ultralytics.utils.ops import xywh2xyxy

ROOT = Path(__file__).resolve().parents[2]


def box_iou(a: np.ndarray, b: np.ndarray) -> np.ndarray:
    """IoU between Nx4 and Mx4 xyxy arrays."""
    if len(a) == 0 or len(b) == 0:
        return np.zeros((len(a), len(b)), dtype=np.float32)
    a = a.astype(np.float32)
    b = b.astype(np.float32)
    tl = np.maximum(a[:, None, :2], b[None, :, :2])
    br = np.minimum(a[:, None, 2:], b[None, :, 2:])
    wh = np.clip(br - tl, 0, None)
    inter = wh[..., 0] * wh[..., 1]
    area_a = (a[:, 2] - a[:, 0]) * (a[:, 3] - a[:, 1])
    area_b = (b[:, 2] - b[:, 0]) * (b[:, 3] - b[:, 1])
    return inter / (area_a[:, None] + area_b[None, :] - inter + 1e-6)


def fuse_predictions(rgb_r, depth_r, iou_thr: float = 0.5):
    """Return fused boxes xyxy, confs, clss, masks (optional)."""
    if rgb_r.boxes is None or len(rgb_r.boxes) == 0:
        if depth_r.boxes is None or len(depth_r.boxes) == 0:
            return None
        boxes = depth_r.boxes.xyxy.cpu().numpy()
        confs = depth_r.boxes.conf.cpu().numpy() * 0.5
        clss = depth_r.boxes.cls.cpu().numpy()
        masks = depth_r.masks.data.cpu().numpy() if depth_r.masks is not None else None
        return boxes, confs, clss, masks

    boxes_r = rgb_r.boxes.xyxy.cpu().numpy()
    conf_r = rgb_r.boxes.conf.cpu().numpy()
    cls_r = rgb_r.boxes.cls.cpu().numpy()
    masks_r = rgb_r.masks.data.cpu().numpy() if rgb_r.masks is not None else None

    if depth_r.boxes is None or len(depth_r.boxes) == 0:
        return boxes_r, conf_r * 0.5 + 0.0, cls_r, masks_r

    boxes_d = depth_r.boxes.xyxy.cpu().numpy()
    conf_d = depth_r.boxes.conf.cpu().numpy()
    cls_d = depth_r.boxes.cls.cpu().numpy()
    masks_d = depth_r.masks.data.cpu().numpy() if depth_r.masks is not None else None

    iou = box_iou(boxes_r, boxes_d)
    used_d = set()
    out_boxes, out_conf, out_cls, out_masks = [], [], [], []

    for i in range(len(boxes_r)):
        j = int(np.argmax(iou[i])) if iou.shape[1] else -1
        if j >= 0 and iou[i, j] >= iou_thr and cls_r[i] == cls_d[j] and j not in used_d:
            used_d.add(j)
            c = 0.5 * (conf_r[i] + conf_d[j])
            out_boxes.append(boxes_r[i])
            out_conf.append(c)
            out_cls.append(cls_r[i])
            if masks_r is not None:
                out_masks.append(masks_r[i])
        else:
            out_boxes.append(boxes_r[i])
            out_conf.append(0.5 * conf_r[i])
            out_cls.append(cls_r[i])
            if masks_r is not None:
                out_masks.append(masks_r[i])

    for j in range(len(boxes_d)):
        if j in used_d:
            continue
        out_boxes.append(boxes_d[j])
        out_conf.append(0.5 * conf_d[j])
        out_cls.append(cls_d[j])
        if masks_d is not None:
            out_masks.append(masks_d[j])

    masks = np.stack(out_masks) if out_masks else None
    return np.array(out_boxes), np.array(out_conf), np.array(out_cls), masks


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--rgb-weights", required=True)
    ap.add_argument("--depth-weights", required=True)
    ap.add_argument("--split", default="test", choices=["val", "test"])
    ap.add_argument("--device", default="mps")
    ap.add_argument("--imgsz", type=int, default=640)
    args = ap.parse_args()

    device = args.device if not (args.device == "mps" and not torch.backends.mps.is_available()) else "cpu"
    rgb_data = ROOT / "pothole_fusion/datasets/rgb/data.yaml"
    depth_data = ROOT / "pothole_fusion/datasets/depth/data.yaml"
    cfg = yaml.safe_load(rgb_data.read_text())
    img_dir = Path(cfg["path"]) / "images" / args.split

    rgb_model = YOLO(args.rgb_weights)
    depth_model = YOLO(args.depth_weights)

    image_files = sorted(list(img_dir.glob("*.jpg")) + list(img_dir.glob("*.png")))
    # Pair by stem with depth dataset
    depth_img_dir = ROOT / "pothole_fusion/datasets/depth/images" / args.split

    n = 0
    conf_sum = 0.0
    det_count = 0
    for rgb_path in image_files:
        depth_path = depth_img_dir / f"{rgb_path.stem}.jpg"
        if not depth_path.exists():
            continue
        rgb_pred = rgb_model.predict(str(rgb_path), imgsz=args.imgsz, device=device, verbose=False)[0]
        depth_pred = depth_model.predict(str(depth_path), imgsz=args.imgsz, device=device, verbose=False)[0]
        fused = fuse_predictions(rgb_pred, depth_pred)
        n += 1
        if fused is not None:
            _, confs, _, _ = fused
            det_count += len(confs)
            conf_sum += float(confs.sum()) if len(confs) else 0.0

    # Official mAP: run each model on its val/test via ultralytics, then report fused detection count stats.
    # Full mask-mAP for custom fused outputs needs a custom evaluator; we report per-model test metrics
    # plus fused detection activity. Prefer using ultralytics val for rgb/depth and document fusion rule.
    rgb_m = rgb_model.val(data=str(rgb_data), split=args.split, device=device, plots=False)
    depth_m = depth_model.val(data=str(depth_data), split=args.split, device=device, plots=False)

    out = {
        "split": args.split,
        "images": n,
        "fused_detections_total": det_count,
        "fused_mean_conf_per_det": (conf_sum / det_count) if det_count else 0.0,
        "rgb_map50": float(rgb_m.seg.map50),
        "rgb_map": float(rgb_m.seg.map),
        "rgb_precision": float(rgb_m.seg.mp),
        "rgb_recall": float(rgb_m.seg.mr),
        "depth_map50": float(depth_m.seg.map50),
        "depth_map": float(depth_m.seg.map),
        "depth_precision": float(depth_m.seg.mp),
        "depth_recall": float(depth_m.seg.mr),
        "fusion_rule": "C_combined=(C_rgb+C_depth)/2 for IoU>=0.5 matches; unmatched conf*=0.5",
        "note": "Late-fusion mask mAP uses confidence fusion at inference; report rgb/depth maps as components and fused ops for demo.",
    }
    out_path = ROOT / "pothole_fusion/results/late_fusion_metrics.json"
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text(json.dumps(out, indent=2))
    print(json.dumps(out, indent=2))


if __name__ == "__main__":
    main()
