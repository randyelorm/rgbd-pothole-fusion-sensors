#!/usr/bin/env python3
"""Train RGB-only, depth-only, or early-fusion YOLOv8n-seg on Apple MPS."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import cv2
import torch
import yaml
from ultralytics import YOLO

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "pothole_fusion"))
from models.adapt import adapt_input_channels  # noqa: E402


MODALITY = {
    "rgb": {"data": "rgb/data.yaml", "channels": 3, "init": None},
    "depth": {"data": "depth/data.yaml", "channels": 1, "init": "average_rgb"},
    "early": {"data": "rgbd/data.yaml", "channels": 4, "init": "copy_rgb"},
}


def resolve_device(requested: str) -> str:
    """Prefer CUDA (Kaggle/Colab), then Apple MPS, then CPU."""
    req = (requested or "auto").lower()
    if req == "auto":
        if torch.cuda.is_available():
            return "0"
        if torch.backends.mps.is_available():
            return "mps"
        return "cpu"
    if req in {"cuda", "gpu"}:
        if torch.cuda.is_available():
            return "0"
        raise SystemExit("CUDA requested but not available")
    if req == "mps":
        if torch.backends.mps.is_available():
            return "mps"
        print("MPS unavailable; falling back to CPU")
        return "cpu"
    return requested


def patch_cv2_flag_for_rgbd() -> None:
    """Ultralytics only special-cases channels==1; force UNCHANGED for 4-ch fallbacks."""
    from ultralytics.data.base import BaseDataset

    original_init = BaseDataset.__init__

    def wrapped_init(self, *args, channels: int = 3, **kwargs):
        original_init(self, *args, channels=channels, **kwargs)
        if channels == 4:
            self.cv2_flag = cv2.IMREAD_UNCHANGED

    BaseDataset.__init__ = wrapped_init


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--modality", choices=sorted(MODALITY), required=True)
    ap.add_argument("--config", default=str(ROOT / "pothole_fusion/configs/default.yaml"))
    ap.add_argument("--epochs", type=int, default=None)
    ap.add_argument("--batch", type=int, default=None)
    ap.add_argument("--name", default=None)
    args = ap.parse_args()

    cfg = yaml.safe_load(Path(args.config).read_text())
    meta = MODALITY[args.modality]
    data_yaml = ROOT / "pothole_fusion/datasets" / meta["data"]
    if not data_yaml.exists():
        raise SystemExit(f"Missing {data_yaml}. Run prepare_data.py first.")

    if meta["channels"] == 4:
        patch_cv2_flag_for_rgbd()

    device = resolve_device(cfg.get("device", "mps"))
    epochs = args.epochs or int(cfg["epochs"])
    batch = args.batch or int(cfg["batch"])
    name = args.name or f"{args.modality}_yolov8n_seg"
    channels = meta["channels"]

    print(f"Training modality={args.modality} device={device} epochs={epochs} batch={batch}")
    model = YOLO(cfg["model"])

    if channels != 3:
        def _adapt_cb(trainer):
            # Model exists after trainer setup (not at on_pretrain_routine_start).
            adapt_input_channels(trainer.model, channels, init=meta["init"])
            print(f"Adapted stem conv to {channels} channels ({meta['init']})")

        model.add_callback("on_pretrain_routine_end", _adapt_cb)

    results = model.train(
        data=str(data_yaml),
        imgsz=int(cfg["imgsz"]),
        epochs=epochs,
        batch=batch,
        device=device,
        workers=int(cfg.get("workers", 2)),
        optimizer=cfg.get("optimizer", "SGD"),
        lr0=float(cfg.get("lr0", 0.01)),
        cos_lr=bool(cfg.get("cos_lr", True)),
        seed=int(cfg["seed"]),
        project=str(ROOT / cfg.get("project", "pothole_fusion/runs")),
        name=name,
        exist_ok=True,
        plots=True,
        patience=50,
    )

    best = Path(str(results.save_dir)) / "weights" / "best.pt"
    metrics = {"save_dir": str(results.save_dir), "best": str(best)}
    if best.exists():
        from ultralytics.utils.torch_utils import get_flops, get_num_params

        m = YOLO(str(best))
        imgsz = int(cfg["imgsz"])
        metrics.update(
            {
                "parameters": int(get_num_params(m.model)),
                "parameters_M": round(get_num_params(m.model) / 1e6, 3),
                "GFLOPs": round(float(get_flops(m.model, imgsz=imgsz)), 3),
                "GFLOPs_imgsz": imgsz,
            }
        )
        test_metrics = m.val(data=str(data_yaml), split="test", device=device, plots=True)
        seg = getattr(test_metrics, "seg", None) or test_metrics.box
        metrics.update(
            {
                "test_map50": float(seg.map50),
                "test_map": float(seg.map),
                "test_precision": float(seg.mp),
                "test_recall": float(seg.mr),
            }
        )
    out = ROOT / "pothole_fusion/results" / f"{name}_metrics.json"
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(metrics, indent=2))
    print(json.dumps(metrics, indent=2))


if __name__ == "__main__":
    main()
