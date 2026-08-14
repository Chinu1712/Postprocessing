"""Segmentation postprocessing.

Pipeline order (each step feeds the next):

1. normalise the payload into instances
2. confidence threshold
3. label allow/deny filter
4. decode + clean masks (speckle removal, hole filling)
5. area filters
6. mask-IoU (or box-IoU) deduplication of same-label overlaps
7. optional same-label merge
8. sort, truncate, encode

Masks are rasterised on a bounded canvas so a 4K frame with many instances
stays within a small container's memory.
"""

from __future__ import annotations

import time
from typing import Any

import numpy as np

from ..core import masks as M
from ..core.geometry import box_iou_matrix, boxes_to_array, clip_boxes, greedy_nms, xyxy_to_xywh
from ..normalize import Instance, NormalizedInput
from ..schemas import SegmentationParams


def run(data: NormalizedInput, params: SegmentationParams) -> dict[str, Any]:
    started = time.perf_counter()
    stats: dict[str, Any] = {
        "input_count": len(data.instances),
        "source_container": data.container,
        "dropped_low_confidence": 0,
        "dropped_by_label_filter": 0,
        "dropped_no_mask": 0,
        "dropped_by_area": 0,
        "suppressed_by_nms": 0,
        "merged": 0,
    }
    warnings = list(data.warnings)

    work_w, work_h, scale = M.fit_canvas(data.image_width, data.image_height, params.mask_max_side)
    if scale < 1.0:
        warnings.append(
            f"masks rasterised at {work_w}x{work_h} (scale {scale:.3f}) to stay within mask_max_side="
            f"{params.mask_max_side}; polygon output is scaled back to full resolution"
        )

    kept = _filter_by_score_and_label(data.instances, params, stats)
    kept = _decode_and_clean(kept, params, work_h, work_w, scale, data, stats, warnings)
    kept = _filter_by_area(kept, params, work_h, work_w, scale, stats)
    kept = _deduplicate(kept, params, stats)

    if params.merge_same_label:
        kept = _merge_same_label(kept, work_h, work_w, stats)

    kept = _sort_and_truncate(kept, params)

    instances = [
        _serialize(inst, i, params, scale, data, work_h, work_w) for i, inst in enumerate(kept)
    ]
    stats["kept"] = len(instances)
    stats["duration_ms"] = round((time.perf_counter() - started) * 1000, 3)

    return {
        "task": "segmentation",
        "model_id": data.model_id,
        "frame_id": data.frame_id,
        "image": {"width": data.image_width, "height": data.image_height},
        "mask_canvas": {"width": work_w, "height": work_h, "scale": scale},
        "count": len(instances),
        "instances": instances,
        "detections": instances,  # alias for consumers expecting the detection key
        "stats": stats,
        "params": params.model_dump(exclude_none=True),
        "warnings": warnings,
        **data.passthrough,
    }


# --------------------------------------------------------------------------
# steps
# --------------------------------------------------------------------------


def _filter_by_score_and_label(
    instances: list[Instance], params: SegmentationParams, stats: dict[str, Any]
) -> list[Instance]:
    allowed = {label.lower() for label in params.allowed_labels} if params.allowed_labels else None
    denied = {label.lower() for label in params.denied_labels} if params.denied_labels else set()

    out: list[Instance] = []
    for inst in instances:
        if inst.score < params.confidence_threshold:
            stats["dropped_low_confidence"] += 1
            continue
        key = (inst.label or "").lower()
        if allowed is not None and key not in allowed:
            stats["dropped_by_label_filter"] += 1
            continue
        if key and key in denied:
            stats["dropped_by_label_filter"] += 1
            continue
        out.append(inst)
    return out


def _decode_and_clean(
    instances: list[Instance],
    params: SegmentationParams,
    work_h: int,
    work_w: int,
    scale: float,
    data: NormalizedInput,
    stats: dict[str, Any],
    warnings: list[str],
) -> list[Instance]:
    out: list[Instance] = []
    undecodable = 0

    for inst in instances:
        mask = None
        if inst.raw_mask is not None:
            try:
                mask = M.decode_mask(
                    inst.raw_mask,
                    work_h,
                    work_w,
                    binarize_threshold=params.mask_binarize_threshold,
                    image_height=data.image_height,
                    image_width=data.image_width,
                )
            except Exception as exc:  # a single bad mask must not sink the frame
                warnings.append(f"instance {inst.source_index}: mask decode failed ({exc})")
                mask = None
            if mask is None:
                undecodable += 1

        if mask is not None:
            if params.min_component_area > 0:
                mask = M.remove_small_components(mask, params.min_component_area * (scale * scale))
            if params.fill_holes:
                mask = M.fill_holes(mask)
            if not mask.any():
                mask = None

        if mask is None and params.require_mask:
            stats["dropped_no_mask"] += 1
            continue

        inst.mask = mask
        if mask is not None:
            inst.area = float(mask.sum())
        elif inst.bbox:
            inst.area = max(inst.bbox[2] - inst.bbox[0], 0.0) * max(inst.bbox[3] - inst.bbox[1], 0.0)
        out.append(inst)

    if undecodable:
        warnings.append(
            f"{undecodable} instance(s) had a mask field that could not be decoded; "
            "they were kept as box-only detections"
        )
    return out


def _filter_by_area(
    instances: list[Instance],
    params: SegmentationParams,
    work_h: int,
    work_w: int,
    scale: float,
    stats: dict[str, Any],
) -> list[Instance]:
    """Apply area limits. ``min_area`` is quoted in original image pixels, so it
    is converted into working-canvas pixels; the ratio limits are scale-free."""
    frame_area = float(work_h * work_w)
    min_absolute = params.min_area * (scale * scale)
    min_from_ratio = params.min_area_ratio * frame_area
    max_from_ratio = params.max_area_ratio * frame_area

    out: list[Instance] = []
    for inst in instances:
        if inst.mask is None:
            out.append(inst)
            continue
        area = float(inst.mask.sum())
        if area < min_absolute or area < min_from_ratio or area > max_from_ratio:
            stats["dropped_by_area"] += 1
            continue
        out.append(inst)
    return out


def _deduplicate(
    instances: list[Instance], params: SegmentationParams, stats: dict[str, Any]
) -> list[Instance]:
    # With mixed mask/box results, box IoU is the only similarity defined for all.
    use_mask = params.nms == "mask" and all(inst.mask is not None for inst in instances)
    stats["nms_similarity"] = "none" if params.nms == "none" else ("mask_iou" if use_mask else "box_iou")

    if params.nms == "none" or len(instances) < 2:
        return instances

    scores = [inst.score for inst in instances]
    labels = [inst.label if inst.label is not None else inst.label_id for inst in instances]

    if use_mask:
        similarity = M.mask_iou_matrix([inst.mask for inst in instances])  # type: ignore[misc]
    else:
        boxes = boxes_to_array([_effective_box(inst) for inst in instances], [0.0, 0.0, 0.0, 0.0])
        similarity = box_iou_matrix(boxes)

    keep = greedy_nms(
        scores,
        similarity,
        params.iou_threshold,
        labels=labels,
        class_agnostic=params.class_agnostic_nms,
    )
    stats["suppressed_by_nms"] = len(instances) - len(keep)
    return [instances[i] for i in sorted(keep)]


def _effective_box(inst: Instance) -> list[float] | None:
    if inst.bbox:
        return inst.bbox
    if inst.mask is not None:
        return M.bbox_from_mask(inst.mask)
    return None


def _merge_same_label(
    instances: list[Instance], work_h: int, work_w: int, stats: dict[str, Any]
) -> list[Instance]:
    groups: dict[str, list[Instance]] = {}
    for inst in instances:
        groups.setdefault(str(inst.label if inst.label is not None else inst.label_id), []).append(inst)

    merged: list[Instance] = []
    for members in groups.values():
        if len(members) == 1:
            merged.append(members[0])
            continue
        best = max(members, key=lambda i: i.score)
        union = np.zeros((work_h, work_w), dtype=bool)
        has_mask = False
        for member in members:
            if member.mask is not None:
                union |= member.mask
                has_mask = True
        if has_mask:
            best.mask = union
            best.area = float(union.sum())
            best.bbox = None  # recomputed from the merged mask downstream
        best.extra = dict(best.extra)
        best.extra["merged_from"] = len(members)
        stats["merged"] += len(members) - 1
        merged.append(best)
    return merged


def _sort_and_truncate(instances: list[Instance], params: SegmentationParams) -> list[Instance]:
    if params.sort_by == "score":
        instances = sorted(instances, key=lambda i: i.score, reverse=True)
    elif params.sort_by == "area":
        instances = sorted(instances, key=lambda i: i.area or 0.0, reverse=True)
    return instances[: params.max_detections]


# --------------------------------------------------------------------------
# serialisation
# --------------------------------------------------------------------------


def _serialize(
    inst: Instance,
    index: int,
    params: SegmentationParams,
    scale: float,
    data: NormalizedInput,
    work_h: int,
    work_w: int,
) -> dict[str, Any]:
    inv = 1.0 / scale if scale else 1.0

    bbox = inst.bbox
    if inst.mask is not None and (params.bbox_from_mask or bbox is None):
        work_box = M.bbox_from_mask(inst.mask)
        bbox = [v * inv for v in work_box] if work_box else bbox

    if bbox and params.clip_to_image:
        bbox = clip_boxes(
            np.asarray([bbox], dtype=np.float64), data.image_width, data.image_height
        )[0].tolist()

    area = None
    if inst.mask is not None:
        area = float(inst.mask.sum()) * inv * inv
    elif bbox:
        area = max(bbox[2] - bbox[0], 0.0) * max(bbox[3] - bbox[1], 0.0)

    out: dict[str, Any] = {
        "id": index,
        "source_index": inst.source_index,
        "label": inst.label,
        "label_id": inst.label_id,
        "score": round(float(inst.score), 6),
        "bbox": [round(v, 2) for v in bbox] if bbox else None,
        "bbox_format": "xyxy",
        "bbox_xywh": [round(v, 2) for v in xyxy_to_xywh(bbox)] if bbox else None,
        "area": round(area, 2) if area is not None else None,
        "has_mask": inst.mask is not None,
    }

    if inst.track_id is not None:
        out["track_id"] = inst.track_id

    if inst.mask is not None and params.output_mask != "none":
        out["mask"] = _encode_mask(inst.mask, params, inv, work_h, work_w, data)
    elif params.output_mask != "none":
        out["mask"] = None

    if params.keep_extra_fields and inst.extra:
        out["extra"] = inst.extra
    return out


def _encode_mask(
    mask: np.ndarray,
    params: SegmentationParams,
    inv: float,
    work_h: int,
    work_w: int,
    data: NormalizedInput,
) -> dict[str, Any]:
    encoded: dict[str, Any] = {"size": [work_h, work_w], "scale": round(1.0 / inv, 6)}

    if params.output_mask in ("polygon", "both"):
        polygons = M.mask_to_polygons(mask, min_area=1.0, tolerance=params.simplify_tolerance)
        encoded["format"] = "polygon"
        encoded["polygons"] = [
            [round(coord * inv, 2) for coord in polygon] for polygon in polygons
        ]
        encoded["coordinate_space"] = "image_pixels"

    if params.output_mask in ("rle", "both"):
        encoded["rle"] = M.rle_encode(mask)
        encoded["rle_note"] = (
            "uncompressed COCO RLE, column-major, at 'size' resolution "
            f"(image is {data.image_width}x{data.image_height})"
        )
        if params.output_mask == "rle":
            encoded["format"] = "rle"

    if params.output_mask == "both":
        encoded["format"] = "polygon+rle"
    return encoded
