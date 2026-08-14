"""Keypoint helpers: layouts, OKS similarity, joint angles, temporal smoothing."""

from __future__ import annotations

import math
from collections.abc import Sequence
from typing import Any

import numpy as np

__all__ = [
    "LAYOUTS",
    "get_layout",
    "parse_keypoints",
    "oks_matrix",
    "bbox_from_keypoints",
    "joint_angles",
    "smooth_keypoints",
]

# --------------------------------------------------------------------------
# layouts
# --------------------------------------------------------------------------

COCO17_NAMES = [
    "nose",
    "left_eye",
    "right_eye",
    "left_ear",
    "right_ear",
    "left_shoulder",
    "right_shoulder",
    "left_elbow",
    "right_elbow",
    "left_wrist",
    "right_wrist",
    "left_hip",
    "right_hip",
    "left_knee",
    "right_knee",
    "left_ankle",
    "right_ankle",
]

# Per-keypoint falloff constants from the COCO evaluation protocol.
COCO17_SIGMAS = [
    0.026, 0.025, 0.025, 0.035, 0.035, 0.079, 0.079, 0.072, 0.072,
    0.062, 0.062, 0.107, 0.107, 0.087, 0.087, 0.089, 0.089,
]

COCO17_SKELETON = [
    (15, 13), (13, 11), (16, 14), (14, 12), (11, 12), (5, 11), (6, 12),
    (5, 6), (5, 7), (6, 8), (7, 9), (8, 10), (1, 2), (0, 1), (0, 2),
    (1, 3), (2, 4), (3, 5), (4, 6),
]

# Joints worth reporting an angle for: (name, [a, vertex, b]) in COCO17 indices.
COCO17_ANGLES = [
    ("left_elbow", (5, 7, 9)),
    ("right_elbow", (6, 8, 10)),
    ("left_shoulder", (7, 5, 11)),
    ("right_shoulder", (8, 6, 12)),
    ("left_hip", (5, 11, 13)),
    ("right_hip", (6, 12, 14)),
    ("left_knee", (11, 13, 15)),
    ("right_knee", (12, 14, 16)),
]


class Layout:
    def __init__(
        self,
        name: str,
        names: list[str],
        sigmas: list[float],
        skeleton: list[tuple[int, int]],
        angles: list[tuple[str, tuple[int, int, int]]] | None = None,
    ) -> None:
        self.name = name
        self.names = names
        self.sigmas = sigmas
        self.skeleton = skeleton
        self.angles = angles or []

    @property
    def num_keypoints(self) -> int:
        return len(self.names)


LAYOUTS: dict[str, Layout] = {
    "coco17": Layout("coco17", COCO17_NAMES, COCO17_SIGMAS, COCO17_SKELETON, COCO17_ANGLES),
}


def get_layout(spec: Any, num_keypoints: int | None = None) -> Layout:
    """Resolve a layout from a name, a custom dict, or the observed keypoint count."""
    if isinstance(spec, dict):
        names = [str(n) for n in spec.get("names", [])]
        count = len(names) or int(spec.get("num_keypoints") or num_keypoints or 0)
        if not names:
            names = [f"kp_{i}" for i in range(count)]
        sigmas = [float(s) for s in spec.get("sigmas", [])] or [0.05] * count
        if len(sigmas) < count:
            sigmas = sigmas + [0.05] * (count - len(sigmas))
        skeleton = [(int(a), int(b)) for a, b in spec.get("skeleton", [])]
        angles = [(str(k), tuple(int(i) for i in v)) for k, v in (spec.get("angles") or {}).items()]
        return Layout(str(spec.get("name", "custom")), names, sigmas[:count], skeleton, angles)  # type: ignore[arg-type]

    if isinstance(spec, str) and spec.lower() in LAYOUTS:
        return LAYOUTS[spec.lower()]

    count = int(num_keypoints or 17)
    if count == 17:
        return LAYOUTS["coco17"]
    return Layout(
        f"generic{count}",
        [f"kp_{i}" for i in range(count)],
        [0.05] * count,
        [],
        [],
    )


# --------------------------------------------------------------------------
# parsing
# --------------------------------------------------------------------------


def parse_keypoints(raw: Any) -> np.ndarray | None:
    """Normalise any keypoint encoding into an (K, 3) float array of x, y, score.

    Accepts flat ``[x, y, v, ...]``, flat ``[x, y, ...]``, nested
    ``[[x, y, v], ...]``, and dict lists ``[{"x": .., "y": .., "score": ..}]``.
    """
    if raw is None:
        return None
    if isinstance(raw, dict):
        for key in ("keypoints", "points", "landmarks", "data"):
            if key in raw:
                return parse_keypoints(raw[key])
        return None
    if not isinstance(raw, (list, tuple)) or not len(raw):
        return None

    first = raw[0]

    if isinstance(first, dict):
        rows = []
        for point in raw:
            x = point.get("x", point.get("X", 0.0))
            y = point.get("y", point.get("Y", 0.0))
            score = point.get("score", point.get("confidence", point.get("v", point.get("visibility", 1.0))))
            rows.append([float(x), float(y), float(score if score is not None else 1.0)])
        return np.asarray(rows, dtype=np.float64)

    if isinstance(first, (list, tuple)):
        rows = []
        for point in raw:
            vals = [float(v) for v in point]
            if len(vals) >= 3:
                rows.append(vals[:3])
            elif len(vals) == 2:
                rows.append([vals[0], vals[1], 1.0])
        return np.asarray(rows, dtype=np.float64) if rows else None

    flat = [float(v) for v in raw]
    if len(flat) % 3 == 0 and len(flat) >= 3:
        arr = np.asarray(flat, dtype=np.float64).reshape(-1, 3)
        # A "third column" that is only ever 0/1/2 is a COCO visibility flag,
        # not a score -- map it onto a confidence so downstream stays uniform.
        third = arr[:, 2]
        if np.all(np.isin(third, (0.0, 1.0, 2.0))) and np.any(third == 2.0):
            arr[:, 2] = np.where(third == 0.0, 0.0, np.where(third == 1.0, 0.5, 1.0))
        return arr
    if len(flat) % 2 == 0 and len(flat) >= 2:
        arr = np.asarray(flat, dtype=np.float64).reshape(-1, 2)
        return np.concatenate([arr, np.ones((arr.shape[0], 1))], axis=1)
    return None


# --------------------------------------------------------------------------
# OKS
# --------------------------------------------------------------------------


def oks_matrix(
    keypoints: Sequence[np.ndarray],
    areas: Sequence[float],
    sigmas: Sequence[float],
    visibility_threshold: float = 0.0,
) -> np.ndarray:
    """Pairwise Object Keypoint Similarity.

    Symmetrised by averaging both directions, because NMS has no notion of
    which instance is ground truth.
    """
    n = len(keypoints)
    if n == 0:
        return np.zeros((0, 0), dtype=np.float64)

    k = max(kp.shape[0] for kp in keypoints)
    sig = np.asarray(list(sigmas)[:k] + [0.05] * max(0, k - len(sigmas)), dtype=np.float64)
    variance = (2.0 * sig) ** 2

    padded = np.zeros((n, k, 3), dtype=np.float64)
    for i, kp in enumerate(keypoints):
        padded[i, : kp.shape[0]] = kp

    out = np.eye(n, dtype=np.float64)
    for i in range(n):
        for j in range(i + 1, n):
            visible = (padded[i, :, 2] > visibility_threshold) & (padded[j, :, 2] > visibility_threshold)
            if not visible.any():
                out[i, j] = out[j, i] = 0.0
                continue
            d2 = np.sum((padded[i, visible, :2] - padded[j, visible, :2]) ** 2, axis=1)
            pair = []
            for area in (areas[i], areas[j]):
                scale = max(float(area), 1.0)
                e = d2 / (variance[visible] * scale * 2.0)
                pair.append(float(np.mean(np.exp(-e))))
            out[i, j] = out[j, i] = sum(pair) / 2.0
    return out


# --------------------------------------------------------------------------
# derived quantities
# --------------------------------------------------------------------------


def bbox_from_keypoints(
    keypoints: np.ndarray,
    threshold: float = 0.0,
    padding: float = 0.0,
) -> list[float] | None:
    """Tight xyxy box around visible keypoints, optionally padded by a ratio."""
    visible = keypoints[keypoints[:, 2] > threshold]
    if visible.shape[0] == 0:
        return None
    x1, y1 = float(visible[:, 0].min()), float(visible[:, 1].min())
    x2, y2 = float(visible[:, 0].max()), float(visible[:, 1].max())
    if padding > 0:
        pad_x = (x2 - x1) * padding
        pad_y = (y2 - y1) * padding
        x1, y1, x2, y2 = x1 - pad_x, y1 - pad_y, x2 + pad_x, y2 + pad_y
    return [x1, y1, x2, y2]


def joint_angles(
    keypoints: np.ndarray,
    layout: Layout,
    threshold: float = 0.0,
) -> dict[str, float]:
    """Interior angle in degrees at each configured joint, skipping occluded ones."""
    result: dict[str, float] = {}
    for name, (a_idx, v_idx, b_idx) in layout.angles:
        if max(a_idx, v_idx, b_idx) >= keypoints.shape[0]:
            continue
        a, v, b = keypoints[a_idx], keypoints[v_idx], keypoints[b_idx]
        if min(a[2], v[2], b[2]) <= threshold:
            continue
        v1 = a[:2] - v[:2]
        v2 = b[:2] - v[:2]
        n1, n2 = float(np.linalg.norm(v1)), float(np.linalg.norm(v2))
        if n1 == 0 or n2 == 0:
            continue
        cos = float(np.dot(v1, v2)) / (n1 * n2)
        result[name] = round(math.degrees(math.acos(max(-1.0, min(1.0, cos)))), 2)
    return result


def smooth_keypoints(
    current: np.ndarray,
    previous: np.ndarray,
    alpha: float,
    max_jump: float | None = None,
) -> np.ndarray:
    """Exponential moving average against the previous frame's keypoints.

    ``alpha`` is the weight of the *current* observation, so 1.0 disables
    smoothing. Keypoints that move further than ``max_jump`` pixels are treated
    as a genuine fast motion (or a re-detection) and left unsmoothed.
    """
    if previous is None or previous.shape != current.shape or alpha >= 1.0:
        return current

    out = current.copy()
    for i in range(current.shape[0]):
        if current[i, 2] <= 0 or previous[i, 2] <= 0:
            continue
        if max_jump is not None:
            dist = float(np.linalg.norm(current[i, :2] - previous[i, :2]))
            if dist > max_jump:
                continue
        out[i, :2] = alpha * current[i, :2] + (1.0 - alpha) * previous[i, :2]
    return out
