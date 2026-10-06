"""Adapt YOLOv8 first conv for 1-ch (depth) or 4-ch (RGB-D) inputs."""

from __future__ import annotations

import torch
import torch.nn as nn


def _detection_body(model: nn.Module) -> nn.Sequential:
    """
    Resolve the Sequential backbone/head container.

    Accepts:
      - Ultralytics YOLO wrapper: model.model.model
      - DetectionModel / SegmentationModel: model.model
      - raw Sequential
    """
    if isinstance(model, nn.Sequential):
        return model
    if hasattr(model, "model"):
        inner = model.model
        if isinstance(inner, nn.Sequential):
            return inner
        if hasattr(inner, "model") and isinstance(inner.model, nn.Sequential):
            return inner.model
    raise RuntimeError(f"Could not locate YOLOv8 Sequential body on {type(model)}")


def _first_conv_module(model: nn.Module) -> tuple[nn.Module, str]:
    """Return (parent, attr_name) for the stem Conv2d."""
    body = _detection_body(model)
    stem = body[0]
    if hasattr(stem, "conv") and isinstance(stem.conv, nn.Conv2d):
        return stem, "conv"
    if isinstance(stem, nn.Conv2d):
        return body, "0"
    raise RuntimeError("Could not locate YOLOv8 stem Conv2d")


def adapt_input_channels(model: nn.Module, in_channels: int, init: str = "copy_rgb") -> nn.Module:
    """
    Replace stem Conv2d in_channels while keeping out channels / kernel / stride / padding.

    init:
      - copy_rgb: keep RGB pretrained weights; depth channel copies R (thesis early-fusion)
      - average_rgb: 1-ch weights = mean of RGB pretrained filters (thesis depth-only)
    """
    parent, attr = _first_conv_module(model)
    old: nn.Conv2d = getattr(parent, attr) if attr != "0" else parent[0]
    if attr == "0":
        old = parent[0]

    if old.in_channels == in_channels:
        return model

    new = nn.Conv2d(
        in_channels=in_channels,
        out_channels=old.out_channels,
        kernel_size=old.kernel_size,
        stride=old.stride,
        padding=old.padding,
        dilation=old.dilation,
        groups=old.groups,
        bias=old.bias is not None,
        padding_mode=old.padding_mode,
    )

    with torch.no_grad():
        if old.in_channels == 3 and in_channels == 4 and init == "copy_rgb":
            new.weight[:, :3] = old.weight
            new.weight[:, 3:4] = old.weight[:, 0:1]
        elif old.in_channels == 3 and in_channels == 1 and init == "average_rgb":
            new.weight[:, 0:1] = old.weight.mean(dim=1, keepdim=True)
        else:
            avg = old.weight.mean(dim=1, keepdim=True)
            for c in range(in_channels):
                new.weight[:, c : c + 1] = avg
        if old.bias is not None and new.bias is not None:
            new.bias.copy_(old.bias)

    if attr == "0":
        parent[0] = new
    else:
        setattr(parent, attr, new)

    # Keep yaml channel metadata if present
    for obj in (model, getattr(model, "model", None)):
        if obj is not None and hasattr(obj, "yaml") and isinstance(obj.yaml, dict):
            obj.yaml["ch"] = in_channels
    return model
