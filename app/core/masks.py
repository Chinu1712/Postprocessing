"""Mask codecs and raster ops for segmentation postprocessing.

Supported input encodings
-------------------------
* COCO polygons              ``[[x1, y1, x2, y2, ...], ...]``
* Point-pair polygons        ``[[[x, y], [x, y], ...], ...]``
* Uncompressed COCO RLE      ``{"size": [h, w], "counts": [int, ...]}``
* Compressed COCO RLE        ``{"size": [h, w], "counts": "..."}``  (LEB128 string)
* Dense bitmaps              nested lists / flat list + ``size``
* Probability maps           float arrays, binarised at a threshold

Output encodings are polygons and/or uncompressed COCO RLE. Uncompressed RLE
is emitted deliberately: it round-trips through plain JSON without needing
pycocotools on the consumer side.

All raster work happens on a bounded working canvas (see ``fit_canvas``) so a
4K frame with thirty instances cannot blow the container's memory budget.
"""

from __future__ import annotations

import base64
from collections.abc import Sequence
from typing import Any

import numpy as np

__all__ = [
    "fit_canvas",
    "decode_mask",
    "rle_decode",
    "rle_encode",
    "polygons_to_mask",
    "mask_to_polygons",
    "mask_iou_matrix",
    "bbox_from_mask",
    "fill_holes",
    "remove_small_components",
    "resize_mask",
    "simplify_polygon",
]

# --------------------------------------------------------------------------
# canvas sizing
# --------------------------------------------------------------------------


def fit_canvas(width: int, height: int, max_side: int) -> tuple[int, int, float]:
    """Return ``(work_w, work_h, scale)`` for a bounded raster canvas.

    ``scale`` maps original pixel coordinates into working coordinates, so
    ``orig_x * scale == work_x``. Never upscales.
    """
    width = max(int(width), 1)
    height = max(int(height), 1)
    longest = max(width, height)
    if max_side <= 0 or longest <= max_side:
        return width, height, 1.0
    scale = max_side / float(longest)
    return max(int(round(width * scale)), 1), max(int(round(height * scale)), 1), scale


# --------------------------------------------------------------------------
# RLE
# --------------------------------------------------------------------------


def _decode_leb128_counts(payload: str | bytes) -> list[int]:
    """Decode COCO's compressed-RLE counts string.

    Port of ``rleFrString`` from the COCO API: base-32 digits offset by 48,
    continuation bit 0x20, sign bit 0x10, and every count after the second is
    delta-coded against the one two positions back.
    """
    data = payload.encode("ascii") if isinstance(payload, str) else payload
    counts: list[int] = []
    p = 0
    while p < len(data):
        value = 0
        shift = 0
        more = True
        char = 0
        while more:
            char = data[p] - 48
            value |= (char & 0x1F) << (5 * shift)
            more = bool(char & 0x20)
            p += 1
            shift += 1
            if not more and (char & 0x10):
                value |= -1 << (5 * shift)
        if len(counts) > 2:
            value += counts[-2]
        counts.append(value)
    return counts


def rle_decode(rle: dict[str, Any], height: int | None = None, width: int | None = None) -> np.ndarray:
    """Decode a COCO RLE dict into a bool mask of shape (H, W)."""
    size = rle.get("size") or rle.get("shape")
    if size and len(size) >= 2:
        h, w = int(size[0]), int(size[1])
    elif height and width:
        h, w = int(height), int(width)
    else:
        raise ValueError("RLE mask needs a 'size' field or an image size")

    counts = rle.get("counts")
    if counts is None:
        raise ValueError("RLE mask has no 'counts'")
    if isinstance(counts, (str, bytes)):
        counts = _decode_leb128_counts(counts)
    counts = [int(c) for c in counts]

    flat = np.zeros(h * w, dtype=bool)
    pos = 0
    value = False
    for run in counts:
        end = min(pos + run, flat.size)
        if value and end > pos:
            flat[pos:end] = True
        pos = end
        value = not value
        if pos >= flat.size:
            break
    # COCO RLE is column-major.
    return flat.reshape((w, h)).T if rle.get("order", "F").upper() == "F" else flat.reshape((h, w))


def rle_encode(mask: np.ndarray) -> dict[str, Any]:
    """Encode a bool mask as uncompressed (JSON-safe) COCO RLE."""
    h, w = mask.shape
    flat = np.asarray(mask, dtype=bool).T.reshape(-1)  # column-major
    if flat.size == 0:
        return {"size": [h, w], "counts": [], "order": "F"}

    changes = np.flatnonzero(np.diff(flat)) + 1
    boundaries = np.concatenate(([0], changes, [flat.size]))
    runs = np.diff(boundaries).tolist()
    if flat[0]:  # counts always start with a background run
        runs = [0] + runs
    return {"size": [h, w], "counts": runs, "order": "F"}


# --------------------------------------------------------------------------
# polygons
# --------------------------------------------------------------------------


def _normalize_polygon(poly: Sequence[Any]) -> list[tuple[float, float]]:
    """Accept ``[x, y, x, y, ...]`` or ``[[x, y], ...]`` and return point pairs."""
    if not len(poly):
        return []
    first = poly[0]
    if isinstance(first, (list, tuple)) and len(first) >= 2:
        return [(float(p[0]), float(p[1])) for p in poly]
    if isinstance(first, dict):
        return [(float(p.get("x", 0.0)), float(p.get("y", 0.0))) for p in poly]
    flat = [float(v) for v in poly]
    return [(flat[i], flat[i + 1]) for i in range(0, len(flat) - 1, 2)]


def polygons_to_mask(polygons: Sequence[Sequence[Any]], height: int, width: int) -> np.ndarray:
    """Scanline-fill polygons into a bool mask (even-odd rule, so holes work)."""
    mask = np.zeros((height, width), dtype=bool)
    rings = [_normalize_polygon(p) for p in polygons]
    rings = [r for r in rings if len(r) >= 3]
    if not rings:
        return mask

    all_y = [pt[1] for ring in rings for pt in ring]
    y_start = max(int(np.floor(min(all_y))), 0)
    y_end = min(int(np.ceil(max(all_y))), height - 1)

    for y in range(y_start, y_end + 1):
        sample = y + 0.5
        crossings: list[float] = []
        for ring in rings:
            n = len(ring)
            for i in range(n):
                x1, y1 = ring[i]
                x2, y2 = ring[(i + 1) % n]
                if y1 == y2:
                    continue
                if (y1 <= sample < y2) or (y2 <= sample < y1):
                    t = (sample - y1) / (y2 - y1)
                    crossings.append(x1 + t * (x2 - x1))
        if not crossings:
            continue
        crossings.sort()
        for i in range(0, len(crossings) - 1, 2):
            xa = int(np.ceil(crossings[i] - 0.5))
            xb = int(np.floor(crossings[i + 1] - 0.5))
            xa = max(xa, 0)
            xb = min(xb, width - 1)
            if xb >= xa:
                mask[y, xa : xb + 1] = True
    return mask


_MOORE = [(-1, 0), (-1, 1), (0, 1), (1, 1), (1, 0), (1, -1), (0, -1), (-1, -1)]


def _trace_boundary(mask: np.ndarray, start: tuple[int, int]) -> list[tuple[int, int]]:
    """Moore-neighbour boundary trace with Jacob's stopping criterion."""
    h, w = mask.shape

    def solid(y: int, x: int) -> bool:
        return 0 <= y < h and 0 <= x < w and bool(mask[y, x])

    contour = [start]
    current = start
    backtrack = (start[0], start[1] - 1)  # row scan guarantees left is background
    first_step: tuple[tuple[int, int], tuple[int, int]] | None = None
    limit = 4 * (h + w) * 4 + 64

    for _ in range(limit):
        offset = (backtrack[0] - current[0], backtrack[1] - current[1])
        idx = _MOORE.index(offset) if offset in _MOORE else 6
        moved = False
        for k in range(1, 9):
            dy, dx = _MOORE[(idx + k) % 8]
            ny, nx = current[0] + dy, current[1] + dx
            if solid(ny, nx):
                pdy, pdx = _MOORE[(idx + k - 1) % 8]
                backtrack = (current[0] + pdy, current[1] + pdx)
                current = (ny, nx)
                moved = True
                break
        if not moved:
            break  # isolated pixel
        if first_step is None:
            first_step = (current, backtrack)
        elif (current, backtrack) == first_step:
            break
        contour.append(current)
    return contour


def mask_to_polygons(
    mask: np.ndarray,
    min_area: float = 1.0,
    tolerance: float = 1.0,
) -> list[list[float]]:
    """Trace outer contours of every connected component into flat polygons.

    Returns ``[[x1, y1, x2, y2, ...], ...]`` in mask pixel coordinates.
    """
    labels, count = connected_components(mask)
    polygons: list[list[float]] = []
    for comp in range(1, count + 1):
        component = labels == comp
        if component.sum() < min_area:
            continue
        ys, xs = np.nonzero(component)
        start = (int(ys[0]), int(xs[np.flatnonzero(ys == ys[0])[0]]))
        contour = _trace_boundary(component, start)
        if len(contour) < 3:
            continue
        points = [(float(x) + 0.5, float(y) + 0.5) for y, x in contour]
        if tolerance > 0:
            points = simplify_polygon(points, tolerance)
        if len(points) < 3:
            continue
        polygons.append([coord for point in points for coord in point])
    return polygons


def simplify_polygon(points: list[tuple[float, float]], tolerance: float) -> list[tuple[float, float]]:
    """Ramer-Douglas-Peucker simplification of a closed ring."""
    if len(points) < 3 or tolerance <= 0:
        return points

    def rdp(pts: list[tuple[float, float]]) -> list[tuple[float, float]]:
        if len(pts) < 3:
            return pts
        (x1, y1), (x2, y2) = pts[0], pts[-1]
        dx, dy = x2 - x1, y2 - y1
        norm = (dx * dx + dy * dy) ** 0.5
        best_idx, best_dist = 0, -1.0
        for i in range(1, len(pts) - 1):
            px, py = pts[i]
            if norm == 0:
                dist = ((px - x1) ** 2 + (py - y1) ** 2) ** 0.5
            else:
                dist = abs(dy * px - dx * py + x2 * y1 - y2 * x1) / norm
            if dist > best_dist:
                best_idx, best_dist = i, dist
        if best_dist <= tolerance:
            return [pts[0], pts[-1]]
        left = rdp(pts[: best_idx + 1])
        right = rdp(pts[best_idx:])
        return left[:-1] + right

    # Split the ring at its extreme point so simplification is orientation-stable.
    closed = points + [points[0]]
    simplified = rdp(closed)
    if len(simplified) > 1 and simplified[0] == simplified[-1]:
        simplified = simplified[:-1]
    return simplified


# --------------------------------------------------------------------------
# raster ops
# --------------------------------------------------------------------------


def connected_components(mask: np.ndarray) -> tuple[np.ndarray, int]:
    """Label 8-connected components using row runs + union-find.

    Run-based labelling keeps this fast in pure Python: the inner loop walks
    runs per row, not pixels.
    """
    h, w = mask.shape
    parent: list[int] = [0]

    def find(a: int) -> int:
        while parent[a] != a:
            parent[a] = parent[parent[a]]
            a = parent[a]
        return a

    def union(a: int, b: int) -> None:
        ra, rb = find(a), find(b)
        if ra != rb:
            parent[max(ra, rb)] = min(ra, rb)

    runs_by_row: list[list[tuple[int, int, int]]] = []  # (start, end_exclusive, label)
    for y in range(h):
        row = mask[y]
        if not row.any():
            runs_by_row.append([])
            continue
        padded = np.concatenate(([False], row, [False]))
        edges = np.flatnonzero(padded[1:] != padded[:-1])
        starts, ends = edges[0::2], edges[1::2]
        runs: list[tuple[int, int, int]] = []
        for s, e in zip(starts.tolist(), ends.tolist(), strict=True):
            parent.append(len(parent))
            runs.append((s, e, len(parent) - 1))
        # 8-connectivity: runs touch if they overlap when widened by one column.
        if y > 0:
            for s, e, label in runs:
                for ps, pe, plabel in runs_by_row[y - 1]:
                    if ps < e + 1 and s < pe + 1:
                        union(label, plabel)
        runs_by_row.append(runs)

    remap: dict[int, int] = {}
    labels = np.zeros((h, w), dtype=np.int32)
    for y, runs in enumerate(runs_by_row):
        for s, e, label in runs:
            root = find(label)
            if root not in remap:
                remap[root] = len(remap) + 1
            labels[y, s:e] = remap[root]
    return labels, len(remap)


def remove_small_components(mask: np.ndarray, min_area: float) -> np.ndarray:
    """Drop connected components smaller than ``min_area`` pixels."""
    if min_area <= 0:
        return mask
    labels, count = connected_components(mask)
    if count == 0:
        return mask
    sizes = np.bincount(labels.reshape(-1), minlength=count + 1)
    keep = np.flatnonzero(sizes >= min_area)
    keep = keep[keep > 0]
    return np.isin(labels, keep)


def fill_holes(mask: np.ndarray) -> np.ndarray:
    """Fill background regions that do not touch the image border."""
    background = ~mask
    labels, count = connected_components(background)
    if count == 0:
        return mask
    border = np.concatenate([labels[0], labels[-1], labels[:, 0], labels[:, -1]])
    outside = set(int(v) for v in np.unique(border) if v > 0)
    holes = (labels > 0) & ~np.isin(labels, list(outside) or [0])
    return mask | holes


def resize_mask(mask: np.ndarray, height: int, width: int) -> np.ndarray:
    """Nearest-neighbour resize (adequate for binary masks, and allocation-free)."""
    h, w = mask.shape
    if (h, w) == (height, width):
        return mask
    ys = np.clip((np.arange(height) + 0.5) * h / height, 0, h - 1).astype(np.int64)
    xs = np.clip((np.arange(width) + 0.5) * w / width, 0, w - 1).astype(np.int64)
    return mask[ys[:, None], xs[None, :]]


def mask_iou_matrix(masks: Sequence[np.ndarray]) -> np.ndarray:
    """Pairwise IoU over bool masks, computed on packed bits."""
    n = len(masks)
    if n == 0:
        return np.zeros((0, 0), dtype=np.float64)

    flat = np.stack([np.asarray(m, dtype=bool).reshape(-1) for m in masks]).astype(np.float32)
    inter = flat @ flat.T
    areas = flat.sum(axis=1)
    union = areas[:, None] + areas[None, :] - inter
    with np.errstate(divide="ignore", invalid="ignore"):
        iou = np.where(union > 0, inter / union, 0.0)
    return iou.astype(np.float64)


def bbox_from_mask(mask: np.ndarray) -> list[float] | None:
    """Tight xyxy box around the set pixels, or None for an empty mask."""
    if not mask.any():
        return None
    ys, xs = np.nonzero(mask)
    return [float(xs.min()), float(ys.min()), float(xs.max() + 1), float(ys.max() + 1)]


# --------------------------------------------------------------------------
# tolerant decoder
# --------------------------------------------------------------------------


def _as_array(value: Any) -> np.ndarray | None:
    try:
        arr = np.asarray(value, dtype=np.float32)
    except (ValueError, TypeError):
        return None
    return arr if arr.ndim >= 2 else None


def decode_mask(
    raw: Any,
    height: int,
    width: int,
    binarize_threshold: float = 0.5,
    image_height: int | None = None,
    image_width: int | None = None,
) -> np.ndarray | None:
    """Decode any supported mask encoding to a bool mask of shape (height, width).

    ``height``/``width`` are the *working* canvas size; ``image_*`` describe the
    coordinate space the raw mask lives in (defaults to the working size).
    Returns None when ``raw`` holds nothing mask-shaped.
    """
    if raw is None:
        return None

    src_h = int(image_height or height)
    src_w = int(image_width or width)

    # {"counts": ..., "size": ...} -- COCO RLE, compressed or not
    if isinstance(raw, dict):
        if "counts" in raw:
            mask = rle_decode(raw, src_h, src_w)
            return resize_mask(mask, height, width)
        for key in ("mask", "segmentation", "polygons", "polygon", "contours", "data"):
            if key in raw:
                return decode_mask(raw[key], height, width, binarize_threshold, src_h, src_w)
        if "base64" in raw or "png" in raw:
            return _decode_base64_mask(raw, height, width, binarize_threshold)
        return None

    if isinstance(raw, str):
        return _decode_base64_mask({"base64": raw, "size": [src_h, src_w]}, height, width, binarize_threshold)

    if not isinstance(raw, (list, tuple)):
        return None
    if not len(raw):
        return None

    first = raw[0]

    # [[x, y, ...], ...] or [[[x, y], ...], ...] -- polygon rings
    if isinstance(first, (list, tuple)) and len(first) and not isinstance(first[0], (list, tuple, dict)):
        looks_like_bitmap = len(raw) == src_h and len(first) == src_w
        if not looks_like_bitmap:
            scale_x = width / float(src_w)
            scale_y = height / float(src_h)
            rings = [_normalize_polygon(r) for r in raw]
            scaled = [[(x * scale_x, y * scale_y) for x, y in ring] for ring in rings]
            return polygons_to_mask(scaled, height, width)
    if isinstance(first, (list, tuple)) and len(first) and isinstance(first[0], (list, tuple, dict)):
        scale_x = width / float(src_w)
        scale_y = height / float(src_h)
        rings = [_normalize_polygon(r) for r in raw]
        scaled = [[(x * scale_x, y * scale_y) for x, y in ring] for ring in rings]
        return polygons_to_mask(scaled, height, width)

    # a single flat polygon: [x, y, x, y, ...]
    if isinstance(first, (int, float)) and len(raw) >= 6 and len(raw) % 2 == 0 and len(raw) != src_h * src_w:
        scale_x = width / float(src_w)
        scale_y = height / float(src_h)
        ring = [(x * scale_x, y * scale_y) for x, y in _normalize_polygon(raw)]
        return polygons_to_mask([ring], height, width)

    # dense bitmap / probability map
    arr = _as_array(raw)
    if arr is None:
        if len(raw) == src_h * src_w:
            arr = np.asarray(raw, dtype=np.float32).reshape((src_h, src_w))
        else:
            return None
    if arr.ndim > 2:
        arr = arr.reshape(arr.shape[-2], arr.shape[-1]) if arr.shape[0] == 1 else arr[0]
    mask = arr > binarize_threshold if arr.max() <= 1.0 + 1e-6 else arr > (binarize_threshold * 255.0)
    return resize_mask(mask, height, width)


def _decode_base64_mask(raw: dict[str, Any], height: int, width: int, threshold: float) -> np.ndarray | None:
    """Decode a base64 payload: raw bytes when a size is given, else a PNG via Pillow."""
    payload = raw.get("base64") or raw.get("png") or raw.get("data")
    if not isinstance(payload, str):
        return None
    if "," in payload[:64] and payload.lstrip().startswith("data:"):
        payload = payload.split(",", 1)[1]
    try:
        blob = base64.b64decode(payload, validate=False)
    except Exception:
        return None

    size = raw.get("size") or raw.get("shape")
    if size and len(size) >= 2:
        h, w = int(size[0]), int(size[1])
        if len(blob) == h * w:
            arr = np.frombuffer(blob, dtype=np.uint8).reshape((h, w))
            return resize_mask(arr > (threshold * 255.0), height, width)

    try:  # PNG/JPEG payload -- optional dependency, absent on the slim image
        import io

        from PIL import Image  # type: ignore

        with Image.open(io.BytesIO(blob)) as img:
            arr = np.asarray(img.convert("L"), dtype=np.uint8)
        return resize_mask(arr > (threshold * 255.0), height, width)
    except Exception:
        return None
