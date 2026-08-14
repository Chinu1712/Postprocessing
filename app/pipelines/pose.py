"""Pose postprocessing.

Pipeline order:

1. normalise the payload into instances
2. confidence threshold + label allow/deny filter
3. per-keypoint visibility threshold
4. drop instances with too few visible keypoints
5. optional temporal smoothing against the previous frame
6. OKS (or box-IoU) deduplication of overlapping people
7. derive boxes, joint angles, skeleton; sort, truncate

OKS is used rather than box IoU because two detections of the same person
frequently have near-identical boxes but different skeletons -- and two people
standing close together have overlapping boxes but distinct skeletons. Box NMS
gets both cases wrong.
"""

from __future__ import annotations

import time
from typing import Any

import numpy as np

from ..core.geometry import box_iou_matrix, boxes_to_array, clip_boxes, greedy_nms, xyxy_to_xywh
from ..core.keypoints import (
    Layout,
    bbox_from_keypoints,
    get_layout,
    joint_angles,
    oks_matrix,
    parse_keypoints,
    smooth_keypoints,
)
from ..normalize import Instance, NormalizedInput
from ..schemas import PoseParams


def run(data: NormalizedInput, params: PoseParams) -> dict[str, Any]:
    started = time.perf_counter()
    stats: dict[str, Any] = {
        "input_count": len(data.instances),
        "source_container": data.container,
        "dropped_low_confidence": 0,
        "dropped_by_label_filter": 0,
        "dropped_no_keypoints": 0,
        "dropped_few_visible_keypoints": 0,
        "suppressed_by_nms": 0,
        "smoothed": 0,
    }
    warnings = list(data.warnings)

    observed = max(
        (inst.keypoints.shape[0] for inst in data.instances if inst.keypoints is not None),
        default=0,
    )
    layout = get_layout(params.layout, observed or None)
    if observed and observed != layout.num_keypoints:
        warnings.append(
            f"model emitted {observed} keypoints but layout '{layout.name}' defines "
            f"{layout.num_keypoints}; names and sigmas are matched positionally"
        )

    kept = _filter_by_score_and_label(data.instances, params, stats)
    kept = _apply_visibility(kept, params, stats)

    if params.smoothing.enabled:
        kept = _smooth(kept, params, layout, stats, warnings)

    kept = _deduplicate(kept, params, layout, stats)
    kept = _sort_and_truncate(kept, params)

    instances = [_serialize(inst, i, params, layout, data) for i, inst in enumerate(kept)]
    stats["kept"] = len(instances)
    stats["duration_ms"] = round((time.perf_counter() - started) * 1000, 3)

    response: dict[str, Any] = {
        "task": "pose",
        "model_id": data.model_id,
        "frame_id": data.frame_id,
        "image": {"width": data.image_width, "height": data.image_height},
        "count": len(instances),
        "keypoint_layout": {
            "name": layout.name,
            "names": layout.names,
            "num_keypoints": layout.num_keypoints,
        },
        "instances": instances,
        "detections": instances,  # alias for consumers expecting the detection key
        "stats": stats,
        "params": params.model_dump(exclude_none=True, exclude={"previous"}),
        "warnings": warnings,
        **data.passthrough,
    }
    if params.include_skeleton:
        response["keypoint_layout"]["skeleton"] = [list(pair) for pair in layout.skeleton]
    return response


# --------------------------------------------------------------------------
# steps
# --------------------------------------------------------------------------


def _filter_by_score_and_label(
    instances: list[Instance], params: PoseParams, stats: dict[str, Any]
) -> list[Instance]:
    allowed = {label.lower() for label in params.allowed_labels} if params.allowed_labels else None
    denied = {label.lower() for label in params.denied_labels} if params.denied_labels else set()

    out: list[Instance] = []
    for inst in instances:
        if inst.keypoints is None or inst.keypoints.size == 0:
            stats["dropped_no_keypoints"] += 1
            continue
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


def _apply_visibility(
    instances: list[Instance], params: PoseParams, stats: dict[str, Any]
) -> list[Instance]:
    """Zero out low-confidence keypoints, then drop skeletons that lost too many."""
    out: list[Instance] = []
    for inst in instances:
        keypoints = inst.keypoints
        assert keypoints is not None
        keypoints = keypoints.copy()
        keypoints[keypoints[:, 2] < params.keypoint_threshold, 2] = 0.0
        visible = int((keypoints[:, 2] > 0).sum())
        if visible < params.min_visible_keypoints:
            stats["dropped_few_visible_keypoints"] += 1
            continue
        inst.keypoints = keypoints
        inst.area = _instance_area(inst)
        out.append(inst)
    return out


def _instance_area(inst: Instance) -> float:
    """Object scale for OKS: the model's box if it has one, else the keypoint hull."""
    if inst.bbox:
        area = max(inst.bbox[2] - inst.bbox[0], 0.0) * max(inst.bbox[3] - inst.bbox[1], 0.0)
        if area > 0:
            return area
    if inst.keypoints is not None:
        box = bbox_from_keypoints(inst.keypoints, threshold=0.0)
        if box:
            return max((box[2] - box[0]) * (box[3] - box[1]), 1.0)
    return 1.0


def _smooth(
    instances: list[Instance],
    params: PoseParams,
    layout: Layout,
    stats: dict[str, Any],
    warnings: list[str],
) -> list[Instance]:
    previous = _parse_previous(params.previous)
    if not previous:
        warnings.append("smoothing is enabled but no usable 'previous' frame was supplied; skipped")
        return instances

    by_track = {str(track): kp for track, kp in previous if track is not None}
    untracked = [kp for track, kp in previous if track is None]

    for inst in instances:
        match: np.ndarray | None = None

        if inst.track_id is not None and str(inst.track_id) in by_track:
            match = by_track[str(inst.track_id)]
        elif untracked or by_track:
            candidates = list(by_track.values()) + untracked
            match = _best_oks_match(inst, candidates, layout, params.smoothing.match_threshold)

        if match is not None and inst.keypoints is not None:
            inst.keypoints = smooth_keypoints(
                inst.keypoints, match, params.smoothing.alpha, params.smoothing.max_jump
            )
            stats["smoothed"] += 1
    return instances


def _parse_previous(previous: Any) -> list[tuple[Any, np.ndarray]]:
    """Accept a full previous response, its ``instances`` list, or bare keypoints."""
    if previous is None:
        return []
    items: Any = previous
    if isinstance(previous, dict):
        items = previous.get("instances") or previous.get("detections") or previous.get("predictions") or []
    if not isinstance(items, list):
        return []

    out: list[tuple[Any, np.ndarray]] = []
    for item in items:
        if not isinstance(item, dict):
            keypoints = parse_keypoints(item)
            if keypoints is not None:
                out.append((None, keypoints))
            continue
        keypoints = parse_keypoints(item.get("keypoints") or item.get("keypoints_flat"))
        if keypoints is not None:
            out.append((item.get("track_id"), keypoints))
    return out


def _best_oks_match(
    inst: Instance, candidates: list[np.ndarray], layout: Layout, threshold: float
) -> np.ndarray | None:
    if inst.keypoints is None or not candidates:
        return None
    area = inst.area or _instance_area(inst)
    best_score, best = 0.0, None
    for candidate in candidates:
        if candidate.shape != inst.keypoints.shape:
            continue
        score = float(oks_matrix([inst.keypoints, candidate], [area, area], layout.sigmas)[0, 1])
        if score > best_score:
            best_score, best = score, candidate
    return best if best_score >= threshold else None


def _deduplicate(
    instances: list[Instance], params: PoseParams, layout: Layout, stats: dict[str, Any]
) -> list[Instance]:
    if params.nms == "none" or len(instances) < 2:
        stats["nms_similarity"] = "none" if params.nms == "none" else params.nms
        return instances

    scores = [inst.score for inst in instances]
    labels = [inst.label if inst.label is not None else inst.label_id for inst in instances]

    if params.nms == "oks":
        similarity = oks_matrix(
            [inst.keypoints for inst in instances],  # type: ignore[misc]
            [inst.area or _instance_area(inst) for inst in instances],
            layout.sigmas,
        )
        threshold = params.oks_threshold
        stats["nms_similarity"] = "oks"
    else:
        boxes = boxes_to_array(
            [inst.bbox or bbox_from_keypoints(inst.keypoints) for inst in instances],  # type: ignore[arg-type]
            [0.0, 0.0, 0.0, 0.0],
        )
        similarity = box_iou_matrix(boxes)
        threshold = params.iou_threshold
        stats["nms_similarity"] = "box_iou"

    # Poses are near-always a single class, so suppression defaults to agnostic.
    keep = greedy_nms(
        scores,
        similarity,
        threshold,
        labels=labels,
        class_agnostic=params.class_agnostic_nms or all(label is None for label in labels),
    )
    stats["suppressed_by_nms"] = len(instances) - len(keep)
    return [instances[i] for i in sorted(keep)]


def _sort_and_truncate(instances: list[Instance], params: PoseParams) -> list[Instance]:
    if params.sort_by == "score":
        instances = sorted(instances, key=lambda i: i.score, reverse=True)
    elif params.sort_by == "area":
        instances = sorted(instances, key=lambda i: i.area or 0.0, reverse=True)
    return instances[: params.max_detections]


# --------------------------------------------------------------------------
# serialisation
# --------------------------------------------------------------------------


def _serialize(
    inst: Instance, index: int, params: PoseParams, layout: Layout, data: NormalizedInput
) -> dict[str, Any]:
    keypoints = inst.keypoints
    assert keypoints is not None

    if params.clip_to_image:
        keypoints = keypoints.copy()
        keypoints[:, 0] = np.clip(keypoints[:, 0], 0.0, data.image_width)
        keypoints[:, 1] = np.clip(keypoints[:, 1], 0.0, data.image_height)

    bbox = _resolve_bbox(inst, keypoints, params)
    if bbox and params.clip_to_image:
        bbox = clip_boxes(
            np.asarray([bbox], dtype=np.float64), data.image_width, data.image_height
        )[0].tolist()

    points: list[dict[str, Any]] = []
    for i in range(keypoints.shape[0]):
        x, y, score = keypoints[i]
        visible = bool(score > 0)
        if params.drop_invisible_keypoints and not visible:
            continue
        points.append(
            {
                "index": i,
                "name": layout.names[i] if i < len(layout.names) else f"kp_{i}",
                "x": round(float(x), 2),
                "y": round(float(y), 2),
                "score": round(float(score), 6),
                "visible": visible,
            }
        )

    visible_count = int((keypoints[:, 2] > 0).sum())
    out: dict[str, Any] = {
        "id": index,
        "source_index": inst.source_index,
        "label": inst.label or "person",
        "label_id": inst.label_id,
        "score": round(float(inst.score), 6),
        "bbox": [round(v, 2) for v in bbox] if bbox else None,
        "bbox_format": "xyxy",
        "bbox_xywh": [round(v, 2) for v in xyxy_to_xywh(bbox)] if bbox else None,
        "num_keypoints": keypoints.shape[0],
        "num_visible_keypoints": visible_count,
        "mean_keypoint_score": round(float(keypoints[keypoints[:, 2] > 0, 2].mean()), 6)
        if visible_count
        else 0.0,
        "keypoints": points,
        # Flat COCO-style triplets, for renderers that want them without a reshape.
        "keypoints_flat": [round(float(v), 2) for v in keypoints.reshape(-1)],
    }

    if inst.track_id is not None:
        out["track_id"] = inst.track_id
    if params.compute_angles:
        out["angles"] = joint_angles(keypoints, layout, threshold=0.0)
    if params.keep_extra_fields and inst.extra:
        out["extra"] = inst.extra
    return out


def _resolve_bbox(inst: Instance, keypoints: np.ndarray, params: PoseParams) -> list[float] | None:
    if params.bbox_mode == "none":
        return None
    if params.bbox_mode == "given":
        return inst.bbox
    if params.bbox_mode == "keypoints":
        return bbox_from_keypoints(keypoints, threshold=0.0, padding=params.bbox_padding)
    return inst.bbox or bbox_from_keypoints(keypoints, threshold=0.0, padding=params.bbox_padding)
