"""Tolerant normalisation of whatever the upstream model server POSTs us.

The pipeline's postprocessing node forwards *raw* model results, and every
serving stack spells them differently. Rather than demand one schema, we sniff
the common ones -- ultralytics, detectron2, torchserve, HuggingFace pipelines,
Triton -- and reduce them to a single ``Instance`` shape.

If this module cannot find anything detection-shaped it says so loudly: the
caller turns that into a 422 so the pipeline falls back to its built-in filter
instead of silently receiving zero instances.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any

import numpy as np

from .core.geometry import xywh_to_xyxy
from .core.keypoints import parse_keypoints

# Keys that plausibly hold the list of predictions, most specific first.
_LIST_KEYS = (
    "instances",
    "predictions",
    "detections",
    "segments",
    "poses",
    "keypoints_results",
    "results",
    "objects",
    "outputs",
    "output",
    "items",
    "data",
)

_SCORE_KEYS = ("score", "confidence", "conf", "probability", "prob", "det_score")
_LABEL_KEYS = ("label", "class_name", "category_name", "category", "name", "class", "cls_name")
_LABEL_ID_KEYS = ("label_id", "class_id", "category_id", "cls", "class_idx", "index")
_BOX_KEYS = ("bbox", "box", "bounding_box", "bbox_xyxy", "xyxy", "rect", "boundingBox")
_MASK_KEYS = ("mask", "segmentation", "seg", "mask_rle", "polygons", "polygon", "contours", "mask_polygon")
_KEYPOINT_KEYS = ("keypoints", "kpts", "pose", "landmarks", "joints", "points")
_TRACK_KEYS = ("track_id", "id", "instance_id", "object_id", "tracker_id")

_RESERVED = set(
    _SCORE_KEYS + _LABEL_KEYS + _LABEL_ID_KEYS + _BOX_KEYS + _MASK_KEYS + _KEYPOINT_KEYS + _TRACK_KEYS
) | {"bbox_format", "area", "segmentation_format"}


class NormalizationError(ValueError):
    """Raised when the payload holds nothing that looks like model output."""


@dataclass
class Instance:
    """One candidate detection, in original image pixel coordinates."""

    source_index: int
    score: float = 1.0
    label: str | None = None
    label_id: int | None = None
    bbox: list[float] | None = None  # xyxy
    raw_mask: Any = None
    keypoints: np.ndarray | None = None  # (K, 3) -> x, y, score
    track_id: Any = None
    extra: dict[str, Any] = field(default_factory=dict)

    # filled in by the pipelines
    mask: np.ndarray | None = None
    area: float | None = None


@dataclass
class NormalizedInput:
    instances: list[Instance]
    image_width: int
    image_height: int
    task: str | None = None
    model_id: str | None = None
    frame_id: Any = None
    image_size_known: bool = True
    container: str | None = None  # which key the predictions were found under
    warnings: list[str] = field(default_factory=list)
    passthrough: dict[str, Any] = field(default_factory=dict)


# --------------------------------------------------------------------------
# small helpers
# --------------------------------------------------------------------------


def _first(source: dict[str, Any], keys: tuple[str, ...]) -> Any:
    for key in keys:
        if key in source and source[key] is not None:
            return source[key]
    lowered = {k.lower(): v for k, v in source.items()}
    for key in keys:
        if key in lowered and lowered[key] is not None:
            return lowered[key]
    return None


def _as_float(value: Any, default: float | None = None) -> float | None:
    try:
        return float(value)
    except (TypeError, ValueError):
        return default


def _find_prediction_list(payload: Any, depth: int = 0) -> tuple[list[Any], str | None]:
    """Locate the list of per-instance dicts inside an arbitrary payload."""
    if isinstance(payload, list):
        return payload, None
    if not isinstance(payload, dict) or depth > 4:
        return [], None

    for key in _LIST_KEYS:
        value = payload.get(key)
        if isinstance(value, list):
            if not value or isinstance(value[0], dict):
                return value, key
        if isinstance(value, dict):
            nested, nested_key = _find_prediction_list(value, depth + 1)
            if nested:
                return nested, f"{key}.{nested_key}" if nested_key else key

    # Last resort: any list of dicts that carries a score or a box.
    for key, value in payload.items():
        if isinstance(value, list) and value and isinstance(value[0], dict):
            if _first(value[0], _SCORE_KEYS) is not None or _first(value[0], _BOX_KEYS) is not None:
                return value, key
    return [], None


def _extract_image_size(payload: Any) -> tuple[int | None, int | None]:
    if not isinstance(payload, dict):
        return None, None

    for key in ("image", "image_size", "frame", "meta", "metadata", "info", "original_size", "orig_shape"):
        node = payload.get(key)
        if isinstance(node, dict):
            w = _as_float(_first(node, ("width", "w", "image_width")))
            h = _as_float(_first(node, ("height", "h", "image_height")))
            if w and h:
                return int(w), int(h)
        elif isinstance(node, (list, tuple)) and len(node) >= 2:
            a, b = _as_float(node[0]), _as_float(node[1])
            if a and b:  # (h, w) is the near-universal convention for shapes
                return int(b), int(a)

    w = _as_float(_first(payload, ("width", "image_width", "img_width", "frame_width")))
    h = _as_float(_first(payload, ("height", "image_height", "img_height", "frame_height")))
    if w and h:
        return int(w), int(h)

    for key in ("shape", "size", "image_shape"):
        node = payload.get(key)
        if isinstance(node, (list, tuple)) and len(node) >= 2:
            a, b = _as_float(node[0]), _as_float(node[1])
            if a and b:
                return int(b), int(a)
    return None, None


def _coerce_bbox(raw: Any, bbox_format: str) -> tuple[list[float] | None, str]:
    """Return (xyxy, detected_format)."""
    if raw is None:
        return None, bbox_format
    if isinstance(raw, dict):
        if all(k in raw for k in ("x1", "y1", "x2", "y2")):
            return [float(raw["x1"]), float(raw["y1"]), float(raw["x2"]), float(raw["y2"])], "xyxy"
        if all(k in raw for k in ("left", "top", "right", "bottom")):
            return [float(raw["left"]), float(raw["top"]), float(raw["right"]), float(raw["bottom"])], "xyxy"
        if all(k in raw for k in ("x", "y", "width", "height")):
            return xywh_to_xyxy([raw["x"], raw["y"], raw["width"], raw["height"]]), "xywh"
        if all(k in raw for k in ("x", "y", "w", "h")):
            return xywh_to_xyxy([raw["x"], raw["y"], raw["w"], raw["h"]]), "xywh"
        return None, bbox_format

    if not isinstance(raw, (list, tuple)) or len(raw) < 4:
        return None, bbox_format
    vals = [float(v) for v in raw[:4]]

    if bbox_format == "xywh":
        return xywh_to_xyxy(vals), "xywh"
    if bbox_format == "cxcywh":
        cx, cy, w, h = vals
        return [cx - w / 2, cy - h / 2, cx + w / 2, cy + h / 2], "cxcywh"
    if bbox_format == "xyxy":
        return vals, "xyxy"

    # auto: xywh is the only reading that stays consistent when x2 < x1.
    if vals[2] <= vals[0] or vals[3] <= vals[1]:
        return xywh_to_xyxy(vals), "xywh"
    return vals, "xyxy"


def _looks_normalized(values: list[float]) -> bool:
    finite = [v for v in values if v == v]
    if not finite:
        return False
    return max(abs(v) for v in finite) <= 1.5


# --------------------------------------------------------------------------
# entry point
# --------------------------------------------------------------------------


def normalize_payload(
    payload: Any,
    bbox_format: str = "auto",
    coordinate_space: str = "auto",
    default_width: int = 1920,
    default_height: int = 1080,
) -> NormalizedInput:
    """Reduce an arbitrary model-server payload to a ``NormalizedInput``."""
    warnings: list[str] = []
    raw_items, container = _find_prediction_list(payload)

    if not isinstance(raw_items, list):
        raise NormalizationError("could not locate a list of predictions in the payload")
    if raw_items and not isinstance(raw_items[0], dict):
        columnar = _from_columnar(payload)
        if columnar is None:
            raise NormalizationError(
                "predictions must be a list of objects, or a columnar dict of "
                "parallel 'boxes'/'scores'/'masks'/'keypoints' arrays"
            )
        raw_items = columnar
        container = "columnar"
    elif not raw_items and isinstance(payload, dict):
        columnar = _from_columnar(payload)
        if columnar is not None:
            raw_items = columnar
            container = "columnar"
        elif not _has_any_known_container(payload):
            raise NormalizationError(
                "no predictions found -- expected one of "
                + ", ".join(_LIST_KEYS[:6])
                + " or a columnar boxes/scores payload"
            )

    width, height = _extract_image_size(payload)
    size_known = bool(width and height)
    if not size_known:
        width, height = default_width, default_height

    instances: list[Instance] = []
    seen_formats: set[str] = set()

    for idx, raw in enumerate(raw_items):
        if not isinstance(raw, dict):
            continue
        instance, detected = _parse_instance(idx, raw, bbox_format)
        if detected:
            seen_formats.add(detected)
        instances.append(instance)

    if bbox_format == "auto" and len(seen_formats) > 1:
        warnings.append(
            "mixed bbox formats inferred across instances; set params.bbox_format "
            "explicitly ('xyxy', 'xywh' or 'cxcywh') if boxes look wrong"
        )

    _apply_coordinate_space(instances, coordinate_space, width, height, size_known, warnings)

    if not size_known and instances:
        inferred_w, inferred_h = _infer_size_from_instances(instances)
        if inferred_w and inferred_h:
            width, height = inferred_w, inferred_h
            warnings.append(
                f"image size not supplied; inferred {width}x{height} from prediction extents. "
                "Send image.width / image.height for exact mask rasterisation."
            )
        else:
            warnings.append(f"image size not supplied; assuming {width}x{height}")

    passthrough: dict[str, Any] = {}
    if isinstance(payload, dict):
        for key in ("frame_id", "frame_index", "timestamp", "source", "stream_id", "camera_id"):
            if key in payload:
                passthrough[key] = payload[key]

    return NormalizedInput(
        instances=instances,
        image_width=int(width),
        image_height=int(height),
        task=_first(payload, ("task", "task_type", "type")) if isinstance(payload, dict) else None,
        model_id=_first(payload, ("model_id", "model", "model_name")) if isinstance(payload, dict) else None,
        frame_id=passthrough.get("frame_id", passthrough.get("frame_index")),
        image_size_known=size_known,
        container=container,
        warnings=warnings,
        passthrough=passthrough,
    )


def _has_any_known_container(payload: dict[str, Any]) -> bool:
    return any(key in payload for key in _LIST_KEYS)


def _parse_instance(idx: int, raw: dict[str, Any], bbox_format: str) -> tuple[Instance, str | None]:
    score = _as_float(_first(raw, _SCORE_KEYS), 1.0)

    label_raw = _first(raw, _LABEL_KEYS)
    label = str(label_raw) if label_raw is not None else None

    label_id_raw = _first(raw, _LABEL_ID_KEYS)
    label_id = None
    if label_id_raw is not None:
        try:
            label_id = int(label_id_raw)
        except (TypeError, ValueError):
            if label is None:
                label = str(label_id_raw)

    per_item_format = raw.get("bbox_format") or bbox_format
    bbox, detected = _coerce_bbox(_first(raw, _BOX_KEYS), str(per_item_format))

    mask = _first(raw, _MASK_KEYS)
    keypoints = parse_keypoints(_first(raw, _KEYPOINT_KEYS))

    track_raw = _first(raw, _TRACK_KEYS)

    extra = {k: v for k, v in raw.items() if k not in _RESERVED and k.lower() not in _RESERVED}

    return (
        Instance(
            source_index=idx,
            score=score if score is not None else 1.0,
            label=label,
            label_id=label_id,
            bbox=bbox,
            raw_mask=mask,
            keypoints=keypoints,
            track_id=track_raw,
            extra=extra,
        ),
        detected if bbox is not None else None,
    )


def _from_columnar(payload: Any) -> list[dict[str, Any]] | None:
    """Rebuild per-instance dicts from parallel arrays (ultralytics/Triton style)."""
    if not isinstance(payload, dict):
        return None

    def column(keys: tuple[str, ...]) -> list[Any] | None:
        value = _first(payload, keys)
        return value if isinstance(value, list) else None

    boxes = column(("boxes", "bboxes", "xyxy", "rects"))
    scores = column(("scores", "confidences", "confs", "probabilities"))
    labels = column(("labels", "classes", "class_ids", "class_names", "categories"))
    masks = column(("masks", "segmentations", "segments", "polygons"))
    keypoint_col = column(("keypoints", "kpts", "poses", "landmarks"))

    columns = [c for c in (boxes, scores, labels, masks, keypoint_col) if c is not None]
    if not columns:
        return None
    count = max(len(c) for c in columns)
    if count == 0:
        return []

    items: list[dict[str, Any]] = []
    for i in range(count):
        item: dict[str, Any] = {}
        if boxes is not None and i < len(boxes):
            item["bbox"] = boxes[i]
        if scores is not None and i < len(scores):
            item["score"] = scores[i]
        if labels is not None and i < len(labels):
            value = labels[i]
            item["label" if isinstance(value, str) else "class_id"] = value
        if masks is not None and i < len(masks):
            item["mask"] = masks[i]
        if keypoint_col is not None and i < len(keypoint_col):
            item["keypoints"] = keypoint_col[i]
        items.append(item)
    return items


def _apply_coordinate_space(
    instances: list[Instance],
    coordinate_space: str,
    width: int,
    height: int,
    size_known: bool,
    warnings: list[str],
) -> None:
    """Scale 0..1 coordinates up to pixels when the model emits normalised output."""
    if coordinate_space == "pixel":
        return

    samples: list[float] = []
    for inst in instances:
        if inst.bbox:
            samples.extend(inst.bbox)
        if inst.keypoints is not None and inst.keypoints.size:
            samples.extend(inst.keypoints[:, :2].reshape(-1).tolist())

    if coordinate_space == "auto":
        if not samples or not _looks_normalized(samples):
            return
        if not size_known:
            warnings.append(
                "coordinates look normalised (0..1) but no image size was supplied; "
                "leaving them as-is. Send image.width / image.height to rescale."
            )
            return
        warnings.append("coordinates detected as normalised (0..1) and scaled to pixels")

    for inst in instances:
        if inst.bbox:
            inst.bbox = [
                inst.bbox[0] * width,
                inst.bbox[1] * height,
                inst.bbox[2] * width,
                inst.bbox[3] * height,
            ]
        if inst.keypoints is not None and inst.keypoints.size:
            inst.keypoints[:, 0] *= width
            inst.keypoints[:, 1] *= height


def _infer_size_from_instances(instances: list[Instance]) -> tuple[int | None, int | None]:
    max_x = max_y = 0.0
    for inst in instances:
        if inst.bbox:
            max_x = max(max_x, inst.bbox[2])
            max_y = max(max_y, inst.bbox[3])
        if inst.keypoints is not None and inst.keypoints.size:
            max_x = max(max_x, float(inst.keypoints[:, 0].max()))
            max_y = max(max_y, float(inst.keypoints[:, 1].max()))
    if max_x <= 0 or max_y <= 0:
        return None, None
    return int(np.ceil(max_x)) + 1, int(np.ceil(max_y)) + 1
