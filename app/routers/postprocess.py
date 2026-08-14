"""Postprocessing endpoints.

The pipeline's ``External REST Connector`` POSTs raw model output as the request
body. Everything is tuned through ``params`` inside that body or through the
query string, so the connector needs no schema of its own.
"""

from __future__ import annotations

import json
import logging
from typing import Any

from fastapi import APIRouter, Depends, HTTPException, Request, status
from pydantic import ValidationError

from ..config import get_settings
from ..models_registry import KNOWN_MODELS, infer_task
from ..normalize import NormalizationError, normalize_payload
from ..openapi_bodies import (
    AUTO_EXAMPLE,
    BATCH_EXAMPLE,
    POSE_EXAMPLE,
    SEGMENTATION_EXAMPLE,
    request_body,
)
from ..pipelines import pose as pose_pipeline
from ..pipelines import segmentation as segmentation_pipeline
from ..schemas import (
    CommonParams,
    PoseParams,
    SegmentationParams,
    describe_validation_error,
    resolve_params,
)
from ..security import require_api_key

log = logging.getLogger("postprocess")

router = APIRouter(prefix="/postprocess", tags=["postprocess"], dependencies=[Depends(require_api_key)])


# --------------------------------------------------------------------------
# helpers
# --------------------------------------------------------------------------


async def _read_body(request: Request) -> Any:
    raw = await request.body()
    if not raw:
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail="empty request body -- POST the raw model output as JSON",
        )
    try:
        return json.loads(raw)
    except json.JSONDecodeError as exc:
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail=f"request body is not valid JSON: {exc}",
        ) from exc


def _normalize(payload: Any, params: CommonParams) -> Any:
    settings = get_settings()
    try:
        data = normalize_payload(
            payload,
            bbox_format=params.bbox_format,
            coordinate_space=params.coordinate_space,
            default_width=settings.default_image_width,
            default_height=settings.default_image_height,
        )
    except NormalizationError as exc:
        # A 4xx here is deliberate: the pipeline node falls back to its built-in
        # filter on a failed connector call, which beats returning zero instances.
        raise HTTPException(
            status_code=422,
            detail=f"could not interpret the payload: {exc}",
        ) from exc

    if len(data.instances) > settings.max_instances_in:
        raise HTTPException(
            status_code=413,
            detail=(
                f"payload holds {len(data.instances)} instances, over the "
                f"MAX_INSTANCES_IN limit of {settings.max_instances_in}"
            ),
        )
    return data


# --------------------------------------------------------------------------
# endpoints
# --------------------------------------------------------------------------


@router.post(
    "/segmentation",
    summary="Postprocess instance-segmentation output",
    description=(
        "Confidence threshold, label filtering, mask decoding and cleanup, "
        "mask-IoU deduplication, and polygon/RLE encoding. Accepts masks as COCO "
        "polygons, compressed or uncompressed RLE, dense bitmaps, or probability maps."
        "\n\nThe example body below is runnable as-is: of its four predictions the "
        "0.19 pallet is dropped on confidence and the 0.58 person is removed as a "
        "duplicate of the 0.94 one, leaving two instances."
    ),
    openapi_extra=request_body(SEGMENTATION_EXAMPLE, "Raw segmentation model output"),
)
async def postprocess_segmentation(request: Request) -> dict[str, Any]:
    body = await _read_body(request)
    params, payload = resolve_params(body, dict(request.query_params), SegmentationParams)
    data = _normalize(payload, params)
    return segmentation_pipeline.run(data, params)  # type: ignore[arg-type]


@router.post(
    "/pose",
    summary="Postprocess pose / keypoint output",
    description=(
        "Confidence threshold, per-keypoint visibility threshold, minimum-visible-"
        "keypoint filtering, OKS deduplication, derived boxes, optional joint angles "
        "and optional temporal smoothing against the previous frame."
        "\n\nThe example body below is runnable as-is: the 0.44 person is the same "
        "skeleton as the 0.93 one and is removed by OKS deduplication, leaving one "
        "instance with joint angles attached."
    ),
    openapi_extra=request_body(POSE_EXAMPLE, "Raw pose / keypoint model output"),
)
async def postprocess_pose(request: Request) -> dict[str, Any]:
    body = await _read_body(request)
    params, payload = resolve_params(body, dict(request.query_params), PoseParams)
    data = _normalize(payload, params)
    return pose_pipeline.run(data, params)  # type: ignore[arg-type]


@router.post(
    "",
    summary="Postprocess, routing on task or model id",
    description=(
        "Dispatches to the segmentation or pose pipeline based on `task` in the body, "
        "the `?task=` query parameter, or the model id. Falls back to sniffing the "
        "payload shape: `keypoints` on the first prediction means pose, `mask` or "
        "`segmentation` means segmentation. Returns 400 when the task cannot be "
        "determined."
    ),
    openapi_extra=request_body(AUTO_EXAMPLE, "Raw model output, with a task or model id to route on"),
)
async def postprocess_auto(request: Request) -> dict[str, Any]:
    body = await _read_body(request)
    query = dict(request.query_params)
    task = _resolve_task(body, query)

    if task == "pose":
        params, payload = resolve_params(body, query, PoseParams)
        return pose_pipeline.run(_normalize(payload, params), params)  # type: ignore[arg-type]
    if task == "segmentation":
        params, payload = resolve_params(body, query, SegmentationParams)
        return segmentation_pipeline.run(_normalize(payload, params), params)  # type: ignore[arg-type]

    raise HTTPException(
        status_code=status.HTTP_400_BAD_REQUEST,
        detail=(
            "could not determine the task. Send task='pose' or task='segmentation' "
            "in the body or as a query parameter, a known model_id, or POST directly "
            "to /postprocess/pose or /postprocess/segmentation."
        ),
    )


def _resolve_task(body: Any, query: dict[str, Any]) -> str | None:
    declared = query.get("task")
    model_id = query.get("model_id")
    hint = None

    if isinstance(body, dict):
        declared = declared or body.get("task") or body.get("task_type")
        model_id = model_id or body.get("model_id") or body.get("model") or body.get("model_name")
        hint = _sniff_task(body)

    task = infer_task(model_id, declared, hint)
    if task in {"detection", "ocr"}:
        # Detection output has neither masks nor keypoints; the segmentation
        # pipeline degrades to box NMS, which is the right behaviour for it.
        return "segmentation" if task == "detection" else None
    return task


def _sniff_task(body: dict[str, Any]) -> str | None:
    """Peek at the first prediction: keypoints mean pose, masks mean segmentation."""
    for key in ("instances", "predictions", "detections", "results", "outputs", "poses", "segments"):
        items = body.get(key)
        if isinstance(items, list) and items and isinstance(items[0], dict):
            first = items[0]
            if any(k in first for k in ("keypoints", "kpts", "landmarks", "joints")):
                return "pose"
            if any(k in first for k in ("mask", "segmentation", "polygons", "contours")):
                return "segmentation"
            return None
    if isinstance(body.get("keypoints"), list):
        return "pose"
    if isinstance(body.get("masks"), list):
        return "segmentation"
    return None


@router.post(
    "/batch",
    summary="Postprocess several frames in one call",
    description=(
        "Body: `{\"items\": [<payload>, ...], \"params\": {...}, \"task\": \"pose\"}`. "
        "Each item is processed independently; a failure on one item is reported in "
        "that item's slot rather than failing the batch."
    ),
    openapi_extra=request_body(BATCH_EXAMPLE, "Several frames, with optional shared params"),
)
async def postprocess_batch(request: Request) -> dict[str, Any]:
    body = await _read_body(request)
    settings = get_settings()

    if isinstance(body, list):
        items, envelope = body, {}
    elif isinstance(body, dict):
        envelope = body
        raw_items = body.get("items") or body.get("frames") or body.get("batch")
        if not isinstance(raw_items, list):
            raise HTTPException(
                status_code=status.HTTP_400_BAD_REQUEST,
                detail="batch body must contain an 'items' array of payloads",
            )
        items = raw_items
    else:
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail="batch body must be an object or array",
        )

    if len(items) > settings.max_batch_items:
        raise HTTPException(
            status_code=413,
            detail=f"batch of {len(items)} exceeds MAX_BATCH_ITEMS={settings.max_batch_items}",
        )

    query = dict(request.query_params)
    shared_params = envelope.get("params") if isinstance(envelope.get("params"), dict) else {}

    results: list[dict[str, Any]] = []
    failures = 0
    for index, item in enumerate(items):
        merged: Any = item
        if isinstance(item, dict) and shared_params:
            merged = {**item, "params": {**shared_params, **(item.get("params") or {})}}
        elif shared_params:
            merged = {"predictions": item, "params": shared_params}

        task = _resolve_task(merged if isinstance(merged, dict) else {}, {**query, **_task_from(envelope)})
        try:
            if task == "pose":
                params, payload = resolve_params(merged, query, PoseParams)
                results.append(pose_pipeline.run(_normalize(payload, params), params))  # type: ignore[arg-type]
            elif task == "segmentation":
                params, payload = resolve_params(merged, query, SegmentationParams)
                results.append(segmentation_pipeline.run(_normalize(payload, params), params))  # type: ignore[arg-type]
            else:
                raise HTTPException(status_code=400, detail="could not determine the task for this item")
        except HTTPException as exc:
            failures += 1
            results.append({"index": index, "error": exc.detail, "status_code": exc.status_code})
        except ValidationError as exc:
            failures += 1
            results.append({"index": index, "error": describe_validation_error(exc), "status_code": 422})
        except Exception as exc:  # noqa: BLE001 - one bad frame must not fail the batch
            failures += 1
            log.exception("batch item %s failed", index)
            results.append({"index": index, "error": str(exc), "status_code": 500})

    return {"count": len(results), "failed": failures, "results": results}


def _task_from(envelope: dict[str, Any]) -> dict[str, Any]:
    task = envelope.get("task") or envelope.get("task_type")
    model_id = envelope.get("model_id") or envelope.get("model")
    out: dict[str, Any] = {}
    if task:
        out["task"] = task
    if model_id:
        out["model_id"] = model_id
    return out


@router.get("/params", summary="Effective default parameters for both pipelines")
async def describe_params() -> dict[str, Any]:
    return {
        "segmentation": _describe(SegmentationParams),
        "pose": _describe(PoseParams),
        "note": (
            "Send any of these under 'params' in the request body, or as query "
            "parameters. Query parameters win over body params."
        ),
    }


def _describe(model_cls: type[CommonParams]) -> dict[str, Any]:
    defaults = model_cls()
    described: dict[str, Any] = {}
    for name, field in model_cls.model_fields.items():
        if name == "previous":
            continue
        described[name] = {
            "default": getattr(defaults, name).model_dump()
            if hasattr(getattr(defaults, name), "model_dump")
            else getattr(defaults, name),
            "description": field.description,
        }
    return described


@router.get("/models", summary="Known model ids and the task each routes to")
async def list_models() -> dict[str, Any]:
    return {
        "models": [{"model_id": key, "task": value} for key, value in sorted(KNOWN_MODELS.items())],
        "supported_tasks": ["segmentation", "pose"],
        "note": (
            "Unlisted ids are matched on keywords: '*seg*', '*mask*' route to "
            "segmentation and '*pose*', '*keypoint*' route to pose."
        ),
    }
