"""Box geometry helpers: conversions, IoU, greedy NMS.

Everything here is pure numpy so the service stays light enough for a
free-tier container (no opencv, no torch, no pycocotools).
"""

from __future__ import annotations

from collections.abc import Iterable, Sequence

import numpy as np

__all__ = [
    "xywh_to_xyxy",
    "xyxy_to_xywh",
    "box_area",
    "box_iou_matrix",
    "clip_boxes",
    "greedy_nms",
    "boxes_to_array",
]


def xywh_to_xyxy(box: Sequence[float]) -> list[float]:
    x, y, w, h = (float(v) for v in box[:4])
    return [x, y, x + w, y + h]


def xyxy_to_xywh(box: Sequence[float]) -> list[float]:
    x1, y1, x2, y2 = (float(v) for v in box[:4])
    return [x1, y1, x2 - x1, y2 - y1]


def box_area(boxes: np.ndarray) -> np.ndarray:
    """Area of xyxy boxes. Negative-width boxes clamp to zero."""
    w = np.clip(boxes[:, 2] - boxes[:, 0], 0.0, None)
    h = np.clip(boxes[:, 3] - boxes[:, 1], 0.0, None)
    return w * h


def box_iou_matrix(boxes: np.ndarray) -> np.ndarray:
    """Pairwise IoU for an (N, 4) array of xyxy boxes."""
    n = boxes.shape[0]
    if n == 0:
        return np.zeros((0, 0), dtype=np.float64)

    areas = box_area(boxes)
    x1 = np.maximum(boxes[:, None, 0], boxes[None, :, 0])
    y1 = np.maximum(boxes[:, None, 1], boxes[None, :, 1])
    x2 = np.minimum(boxes[:, None, 2], boxes[None, :, 2])
    y2 = np.minimum(boxes[:, None, 3], boxes[None, :, 3])

    inter = np.clip(x2 - x1, 0.0, None) * np.clip(y2 - y1, 0.0, None)
    union = areas[:, None] + areas[None, :] - inter
    with np.errstate(divide="ignore", invalid="ignore"):
        iou = np.where(union > 0, inter / union, 0.0)
    return iou


def clip_boxes(boxes: np.ndarray, width: float, height: float) -> np.ndarray:
    """Clamp xyxy boxes into the image rectangle."""
    out = boxes.copy()
    out[:, 0] = np.clip(out[:, 0], 0.0, width)
    out[:, 1] = np.clip(out[:, 1], 0.0, height)
    out[:, 2] = np.clip(out[:, 2], 0.0, width)
    out[:, 3] = np.clip(out[:, 3], 0.0, height)
    return out


def greedy_nms(
    scores: Sequence[float],
    iou: np.ndarray,
    iou_threshold: float,
    labels: Sequence[object] | None = None,
    class_agnostic: bool = False,
) -> list[int]:
    """Greedy non-maximum suppression over a precomputed IoU matrix.

    Works for boxes, masks or OKS alike -- the caller decides what the
    similarity matrix means. Returns the kept indices ordered by score
    (highest first).
    """
    order = sorted(range(len(scores)), key=lambda i: scores[i], reverse=True)
    keep: list[int] = []
    suppressed = [False] * len(order)

    for pos, i in enumerate(order):
        if suppressed[pos]:
            continue
        keep.append(i)
        for other_pos in range(pos + 1, len(order)):
            if suppressed[other_pos]:
                continue
            j = order[other_pos]
            if not class_agnostic and labels is not None and labels[i] != labels[j]:
                continue
            if iou[i, j] > iou_threshold:
                suppressed[other_pos] = True

    return keep


def boxes_to_array(boxes: Iterable[Sequence[float] | None], fallback: Sequence[float]) -> np.ndarray:
    """Stack boxes into an (N, 4) array, substituting ``fallback`` for None."""
    rows = [list(b[:4]) if b is not None else list(fallback) for b in boxes]
    if not rows:
        return np.zeros((0, 4), dtype=np.float64)
    return np.asarray(rows, dtype=np.float64)
