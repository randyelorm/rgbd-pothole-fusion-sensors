#!/usr/bin/env python3
"""Pair RGB/depth/labels and build YOLO-seg datasets for RGB, depth, and RGB-D."""

from __future__ import annotations

import argparse
import json
import os
import random
import re
import shutil
from collections import defaultdict
from datetime import datetime
from pathlib import Path

import cv2
import numpy as np
import yaml


TS_DEPTH = re.compile(r"(\d{8}_\d{6})_depth\.npy$")
TS_RGB = re.compile(r"(\d{8}_\d{6})_color_png\.rf\.")


def extract_ts(path: Path) -> str | None:
    name = path.name
    m = TS_DEPTH.search(name)
    if m:
        return m.group(1)
    m = TS_RGB.search(name)
    if m:
        return m.group(1)
    return None


def depth_to_uint8(depth: np.ndarray, clip_mm: float) -> np.ndarray:
    d = depth.astype(np.float32)
    valid = d > 0
    d = np.clip(d, 0, clip_mm)
    if valid.any():
        lo, hi = np.percentile(d[valid], [1, 99])
        if hi <= lo:
            hi = lo + 1.0
        norm = (d - lo) / (hi - lo)
    else:
        norm = np.zeros_like(d)
    norm = np.clip(norm, 0, 1)
    norm[~valid] = 0
    return (norm * 255.0).astype(np.uint8)


def label_ok(label_path: Path) -> bool:
    text = label_path.read_text().strip()
    if not text:
        return False
    for line in text.splitlines():
        parts = line.split()
        if len(parts) < 7:  # class + at least 3 points
            return False
        try:
            float(parts[1])
        except ValueError:
            return False
    return True


def write_yaml(path: Path, names: dict, splits: dict, channels: int) -> None:
    data = {
        "path": str(path.parent.resolve()),
        "train": splits["train"],
        "val": splits["val"],
        "test": splits["test"],
        "names": names,
        "channels": channels,
    }
    path.write_text(yaml.safe_dump(data, sort_keys=False))


def main() -> None:
    root = Path(__file__).resolve().parents[2]
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", default=str(root / "pothole_fusion/configs/default.yaml"))
    args = ap.parse_args()
    cfg = yaml.safe_load(Path(args.config).read_text())

    data_root = Path(cfg["data_root"])
    if not data_root.is_absolute():
        data_root = root / data_root
    out_root = Path(cfg["output_root"])
    if not out_root.is_absolute():
        out_root = root / out_root
    clip = float(cfg.get("depth_clip_mm", 3000))
    seed = int(cfg["seed"])
    fracs = cfg["split"]

    img_dir = data_root / "images"
    depth_dir = data_root / "depths"
    label_dir = data_root / "labels"

    depth_map, image_map, label_map = {}, {}, {}
    for p in depth_dir.glob("*.npy"):
        ts = extract_ts(p)
        if ts:
            depth_map[ts] = p
    for p in list(img_dir.glob("*.jpg")) + list(img_dir.glob("*.png")):
        ts = extract_ts(p)
        if ts:
            image_map[ts] = p
    for p in label_dir.glob("*.txt"):
        ts = extract_ts(p)
        if ts:
            label_map[ts] = p

    common = sorted(set(depth_map) & set(image_map) & set(label_map))
    samples = []
    skipped = defaultdict(int)
    for ts in common:
        if not label_ok(label_map[ts]):
            skipped["bad_label"] += 1
            continue
        depth = np.load(depth_map[ts])
        if depth.ndim != 2 or depth.size == 0:
            skipped["bad_depth"] += 1
            continue
        samples.append(
            {
                "timestamp": ts,
                "image": image_map[ts],
                "depth": depth_map[ts],
                "label": label_map[ts],
                "date": datetime.strptime(ts, "%Y%m%d_%H%M%S").date().isoformat(),
            }
        )

    rng = random.Random(seed)
    rng.shuffle(samples)
    n = len(samples)
    n_train = int(n * fracs["train"])
    n_val = int(n * fracs["val"])
    splits = {
        "train": samples[:n_train],
        "val": samples[n_train : n_train + n_val],
        "test": samples[n_train + n_val :],
    }

    # Fresh output
    if out_root.exists():
        shutil.rmtree(out_root)
    modalities = {
        "rgb": out_root / "rgb",
        "depth": out_root / "depth",
        "rgbd": out_root / "rgbd",
    }

    manifest = {"seed": seed, "clip_mm": clip, "skipped": dict(skipped), "counts": {}, "splits": {}}

    for split_name, items in splits.items():
        manifest["splits"][split_name] = [s["timestamp"] for s in items]
        for mod, base in modalities.items():
            (base / "images" / split_name).mkdir(parents=True, exist_ok=True)
            (base / "labels" / split_name).mkdir(parents=True, exist_ok=True)

        for s in items:
            stem = s["timestamp"]
            rgb = cv2.imread(str(s["image"]), cv2.IMREAD_COLOR)
            if rgb is None:
                skipped["bad_image"] += 1
                continue
            depth = np.load(s["depth"])
            if depth.shape[:2] != rgb.shape[:2]:
                depth = cv2.resize(depth, (rgb.shape[1], rgb.shape[0]), interpolation=cv2.INTER_NEAREST)
            depth_u8 = depth_to_uint8(depth, clip)

            # RGB
            cv2.imwrite(str(modalities["rgb"] / "images" / split_name / f"{stem}.jpg"), rgb)
            shutil.copy2(s["label"], modalities["rgb"] / "labels" / split_name / f"{stem}.txt")

            # Depth as single-channel JPEG (data.yaml channels: 1)
            cv2.imwrite(str(modalities["depth"] / "images" / split_name / f"{stem}.jpg"), depth_u8)
            shutil.copy2(s["label"], modalities["depth"] / "labels" / split_name / f"{stem}.txt")

            # RGB-D: keep a PNG for file discovery + .npy for true 4-channel load (BGR+D)
            rgbd = np.dstack([rgb, depth_u8]).astype(np.uint8)  # HWC, 4
            png_path = modalities["rgbd"] / "images" / split_name / f"{stem}.png"
            npy_path = modalities["rgbd"] / "images" / split_name / f"{stem}.npy"
            cv2.imwrite(str(png_path), rgb)  # placeholder listing file
            np.save(str(npy_path), rgbd)
            shutil.copy2(s["label"], modalities["rgbd"] / "labels" / split_name / f"{stem}.txt")

    names = {0: "pothole"}
    channel_map = {"rgb": 3, "depth": 1, "rgbd": 4}
    for mod, base in modalities.items():
        write_yaml(
            base / "data.yaml",
            names,
            {
                "train": f"images/train",
                "val": f"images/val",
                "test": f"images/test",
            },
            channels=channel_map[mod],
        )
        # ultralytics expects labels parallel to images by replacing /images/ -> /labels/
        manifest["counts"][mod] = {
            k: len(
                [
                    p
                    for p in (base / "images" / k).iterdir()
                    if p.suffix.lower() in {".jpg", ".jpeg", ".png"}
                ]
            )
            for k in ("train", "val", "test")
        }

    manifest["skipped"] = dict(skipped)
    manifest["total_paired"] = len(samples)
    (out_root / "manifest.json").write_text(json.dumps(manifest, indent=2))
    print(json.dumps(manifest["counts"], indent=2))
    print("skipped:", skipped)
    print("wrote", out_root)


if __name__ == "__main__":
    main()
