#!/usr/bin/env python3
"""Compute late-fusion instance segmentation mAP on the test split."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import cv2
import numpy as np
import torch
import yaml
from ultralytics import YOLO
from ultralytics.utils.metrics import ap_per_class, mask_iou

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "pothole_fusion" / "scripts"))
from late_fusion_eval import fuse_predictions  # noqa: E402


def load_gt_masks(label_path: Path, h: int, w: int) -> tuple[np.ndarray, np.ndarray]:
    """Return (masks MxHxW bool, classes M)."""
    if not label_path.exists():
        return np.zeros((0, h, w), dtype=bool), np.zeros((0,), dtype=np.int32)
    masks, clss = [], []
    for line in label_path.read_text().strip().splitlines():
        parts = line.split()
        if len(parts) < 7:
            continue
        cls = int(float(parts[0]))
        coords = np.array([float(x) for x in parts[1:]], dtype=np.float32).reshape(-1, 2)
        pts = np.stack([coords[:, 0] * w, coords[:, 1] * h], axis=1).astype(np.int32)
        m = np.zeros((h, w), dtype=np.uint8)
        cv2.fillPoly(m, [pts], 1)
        masks.append(m.astype(bool))
        clss.append(cls)
    if not masks:
        return np.zeros((0, h, w), dtype=bool), np.zeros((0,), dtype=np.int32)
    return np.stack(masks), np.array(clss, dtype=np.int32)


def resize_masks(masks: np.ndarray, h: int, w: int) -> np.ndarray:
    if masks is None or len(masks) == 0:
        return np.zeros((0, h, w), dtype=bool)
    out = []
    for m in masks:
        mm = cv2.resize(m.astype(np.uint8), (w, h), interpolation=cv2.INTER_NEAREST)
        out.append(mm.astype(bool))
    return np.stack(out)


def match_tp(pred_masks, pred_cls, gt_masks, gt_cls, iou_thrs: np.ndarray) -> np.ndarray:
    """Return tp array shape (n_pred, n_thrs)."""
    n_pred = len(pred_masks)
    n_thr = len(iou_thrs)
    tp = np.zeros((n_pred, n_thr), dtype=bool)
    if n_pred == 0 or len(gt_masks) == 0:
        return tp

    pm = torch.from_numpy(pred_masks.reshape(n_pred, -1)).float()
    gm = torch.from_numpy(gt_masks.reshape(len(gt_masks), -1)).float()
    ious = mask_iou(pm, gm).cpu().numpy()  # n_pred x n_gt

    for t_i, thr in enumerate(iou_thrs):
        matched_gt = set()
        # greedy by best IoU per prediction order (preds already conf-sorted)
        for p_i in range(n_pred):
            best_j, best_iou = -1, 0.0
            for g_j in range(len(gt_masks)):
                if g_j in matched_gt:
                    continue
                if pred_cls[p_i] != gt_cls[g_j]:
                    continue
                iou = float(ious[p_i, g_j])
                if iou >= thr and iou > best_iou:
                    best_iou, best_j = iou, g_j
            if best_j >= 0:
                tp[p_i, t_i] = True
                matched_gt.add(best_j)
    return tp


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--rgb-weights", default=str(ROOT / "results/weights/rgb_best.pt"))
    ap.add_argument("--depth-weights", default=str(ROOT / "results/weights/depth_best.pt"))
    ap.add_argument("--split", default="test")
    ap.add_argument("--device", default="auto")
    ap.add_argument("--imgsz", type=int, default=640)
    ap.add_argument("--conf", type=float, default=0.001)
    args = ap.parse_args()

    if args.device == "auto":
        if torch.cuda.is_available():
            device = "0"
        elif torch.backends.mps.is_available():
            device = "mps"
        else:
            device = "cpu"
    else:
        device = args.device

    rgb_yaml = ROOT / "pothole_fusion/datasets/rgb/data.yaml"
    depth_yaml = ROOT / "pothole_fusion/datasets/depth/data.yaml"
    cfg = yaml.safe_load(rgb_yaml.read_text())
    rgb_img_dir = Path(cfg["path"]) / "images" / args.split
    rgb_lbl_dir = Path(cfg["path"]) / "labels" / args.split
    depth_img_dir = ROOT / "pothole_fusion/datasets/depth/images" / args.split

    rgb_model = YOLO(args.rgb_weights, task="segment")
    depth_model = YOLO(args.depth_weights, task="segment")

    images = sorted(list(rgb_img_dir.glob("*.jpg")) + list(rgb_img_dir.glob("*.png")))
    iou_thrs = np.linspace(0.5, 0.95, 10)
    stats_tp, stats_conf, stats_pcls, stats_tcls = [], [], [], []
    n_images = 0
    fused_det_count = 0

    for rgb_path in images:
        depth_path = depth_img_dir / f"{rgb_path.stem}.jpg"
        label_path = rgb_lbl_dir / f"{rgb_path.stem}.txt"
        if not depth_path.exists():
            continue

        im0 = cv2.imread(str(rgb_path))
        if im0 is None:
            continue
        h0, w0 = im0.shape[:2]

        rgb_pred = rgb_model.predict(
            str(rgb_path), imgsz=args.imgsz, conf=args.conf, device=device, verbose=False, retina_masks=True
        )[0]
        depth_pred = depth_model.predict(
            str(depth_path), imgsz=args.imgsz, conf=args.conf, device=device, verbose=False, retina_masks=True
        )[0]
        fused = fuse_predictions(rgb_pred, depth_pred)
        n_images += 1

        gt_masks, gt_cls = load_gt_masks(label_path, h0, w0)

        if fused is None:
            # no preds: still count GT for recall
            for c in gt_cls:
                stats_tcls.append(c)
            continue

        boxes, confs, clss, masks = fused
        fused_det_count += len(confs)
        if masks is None or len(masks) == 0:
            for c in gt_cls:
                stats_tcls.append(c)
            continue

        # sort by confidence descending
        order = np.argsort(-confs)
        confs = confs[order]
        clss = clss[order].astype(np.int32)
        masks = resize_masks(masks[order], h0, w0)

        tp = match_tp(masks, clss, gt_masks, gt_cls, iou_thrs)
        stats_tp.append(tp)
        stats_conf.append(confs)
        stats_pcls.append(clss)
        stats_tcls.extend(gt_cls.tolist())

    if stats_tp:
        tp = np.concatenate(stats_tp, axis=0)
        conf = np.concatenate(stats_conf, axis=0)
        pred_cls = np.concatenate(stats_pcls, axis=0)
        target_cls = np.array(stats_tcls, dtype=np.int32)
        results = ap_per_class(tp, conf, pred_cls, target_cls, plot=False, names={0: "pothole"}, prefix="Mask")
        # ultralytics returns tuple; unpack carefully across versions
        if isinstance(results, (list, tuple)):
            # typical: tp, fp, p, r, f1, ap, unique_classes, ...
            # ap shape (nc, 10) for 0.5:0.95
            ap = None
            p = r = None
            for item in results:
                if isinstance(item, np.ndarray) and item.ndim == 2 and item.shape[-1] == 10:
                    ap = item
                if isinstance(item, np.ndarray) and item.ndim == 1 and item.shape[0] >= 1:
                    # could be p or r; keep last two 1d arrays near ap
                    pass
            # Better: use named handling from recent ultralytics
            # ap_per_class -> (tp, fp, p, r, f1, ap, unique_classes, p_curve, r_curve, f1_curve, x, prec_values)
            try:
                _, _, p_arr, r_arr, _, ap, *_ = results
                map50 = float(ap[:, 0].mean()) if ap is not None else 0.0
                map5095 = float(ap.mean()) if ap is not None else 0.0
                precision = float(np.mean(p_arr)) if p_arr is not None else 0.0
                recall = float(np.mean(r_arr)) if r_arr is not None else 0.0
            except Exception:
                ap = results[5]
                map50 = float(ap[:, 0].mean())
                map5095 = float(ap.mean())
                precision = float(np.mean(results[2]))
                recall = float(np.mean(results[3]))
        else:
            raise RuntimeError(f"Unexpected ap_per_class return: {type(results)}")
    else:
        map50 = map5095 = precision = recall = 0.0

    # Also report component vals for reference
    rgb_m = rgb_model.val(data=str(rgb_yaml), split=args.split, device=device, plots=False, verbose=False)
    depth_m = depth_model.val(data=str(depth_yaml), split=args.split, device=device, plots=False, verbose=False)

    out = {
        "split": args.split,
        "images": n_images,
        "fused_detections_total": fused_det_count,
        "test_precision": precision,
        "test_recall": recall,
        "test_map50": map50,
        "test_map": map5095,
        "rgb_map50": float(rgb_m.seg.map50),
        "rgb_map": float(rgb_m.seg.map),
        "depth_map50": float(depth_m.seg.map50),
        "depth_map": float(depth_m.seg.map),
        "fusion_rule": "C_combined=(C_rgb+C_depth)/2 for IoU>=0.5 matches; unmatched conf*=0.5",
        "rgb_weights": args.rgb_weights,
        "depth_weights": args.depth_weights,
        "device": device,
    }
    out_path = ROOT / "results/late_fusion_metrics.json"
    out_path.write_text(json.dumps(out, indent=2))
    print(json.dumps(out, indent=2))


if __name__ == "__main__":
    main()
