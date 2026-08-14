"""Request/response models.

The postprocessing node POSTs raw model output as the request body, so the
endpoints accept a free-form object. Tuning knobs are read from ``params``
inside that body, or from the query string -- see :func:`resolve_params`.
"""

from __future__ import annotations

from typing import Any, Literal

from pydantic import BaseModel, ConfigDict, Field, ValidationError

from .config import get_settings

BBoxFormat = Literal["auto", "xyxy", "xywh", "cxcywh"]
CoordinateSpace = Literal["auto", "pixel", "normalized"]
MaskOutput = Literal["polygon", "rle", "both", "none"]


class SmoothingParams(BaseModel):
    """Stateless temporal smoothing: the caller supplies the previous frame."""

    model_config = ConfigDict(extra="forbid")

    enabled: bool = False
    alpha: float = Field(0.6, ge=0.0, le=1.0, description="Weight of the current frame; 1.0 = no smoothing")
    max_jump: float | None = Field(
        None, gt=0, description="Skip smoothing for keypoints that moved further than this many pixels"
    )
    match_threshold: float = Field(
        0.5, ge=0.0, le=1.0, description="Minimum OKS to treat two instances as the same"
    )


class CommonParams(BaseModel):
    model_config = ConfigDict(extra="ignore")

    confidence_threshold: float = Field(
        default_factory=lambda: get_settings().confidence_threshold,
        ge=0.0,
        le=1.0,
        description="Instances scoring below this are dropped",
    )
    max_detections: int = Field(
        default_factory=lambda: get_settings().max_detections, ge=1, le=10_000
    )
    allowed_labels: list[str] | None = Field(None, description="Keep only these labels (case-insensitive)")
    denied_labels: list[str] | None = Field(
        None, description="Drop these labels (applied after allowed_labels)"
    )
    class_agnostic_nms: bool = Field(False, description="Suppress across labels rather than within a label")
    clip_to_image: bool = True
    bbox_format: BBoxFormat = "auto"
    coordinate_space: CoordinateSpace = "auto"
    keep_extra_fields: bool = Field(True, description="Echo unrecognised per-instance fields back in 'extra'")
    sort_by: Literal["score", "area", "none"] = "score"


class SegmentationParams(CommonParams):
    """Tuning for ``/postprocess/segmentation``."""

    iou_threshold: float = Field(
        default_factory=lambda: get_settings().iou_threshold,
        ge=0.0,
        le=1.0,
        description="Overlapping same-label instances above this IoU are deduplicated",
    )
    nms: Literal["mask", "box", "none"] = Field("mask", description="Similarity used for deduplication")
    mask_binarize_threshold: float = Field(0.5, ge=0.0, le=1.0, description="Cut-off for probability masks")
    mask_max_side: int = Field(
        default_factory=lambda: get_settings().mask_max_side,
        ge=64,
        le=4096,
        description="Longest side of the internal raster canvas; caps memory on 4K frames",
    )
    min_area: float = Field(
        0.0, ge=0.0, description="Drop instances whose mask is smaller than this many image pixels"
    )
    min_area_ratio: float = Field(
        0.0, ge=0.0, le=1.0, description="Same, expressed as a fraction of the frame area"
    )
    max_area_ratio: float = Field(
        1.0, ge=0.0, le=1.0, description="Drop instances covering more than this fraction of the frame"
    )
    min_component_area: float = Field(
        0.0, ge=0.0, description="Remove speckle: connected components smaller than this many pixels"
    )
    fill_holes: bool = Field(False, description="Fill interior holes in each mask")
    merge_same_label: bool = Field(
        False, description="Union all masks sharing a label into one instance (semantic-style output)"
    )
    simplify_tolerance: float = Field(
        1.0, ge=0.0, description="Douglas-Peucker tolerance in pixels for polygon output; 0 disables"
    )
    output_mask: MaskOutput = Field("polygon", description="Mask encoding in the response")
    bbox_from_mask: bool = Field(True, description="Recompute each bbox from its mask")
    require_mask: bool = Field(False, description="Drop instances that carry no decodable mask")


class PoseParams(CommonParams):
    """Tuning for ``/postprocess/pose``."""

    keypoint_threshold: float = Field(
        default_factory=lambda: get_settings().keypoint_threshold,
        ge=0.0,
        le=1.0,
        description="Keypoints below this score are marked not-visible",
    )
    oks_threshold: float = Field(
        default_factory=lambda: get_settings().oks_threshold,
        ge=0.0,
        le=1.0,
        description="Instances above this OKS are deduplicated",
    )
    iou_threshold: float = Field(
        default_factory=lambda: get_settings().iou_threshold,
        ge=0.0,
        le=1.0,
        description="Box IoU threshold, used when nms='box'",
    )
    nms: Literal["oks", "box", "none"] = "oks"
    layout: str | dict[str, Any] = Field(
        "coco17",
        description="Keypoint layout name, or a custom {names, sigmas, skeleton, angles} object",
    )
    min_visible_keypoints: int = Field(
        3, ge=0, description="Drop instances with fewer visible keypoints than this"
    )
    drop_invisible_keypoints: bool = Field(
        False, description="Omit not-visible keypoints entirely instead of returning them with visible=false"
    )
    bbox_mode: Literal["auto", "keypoints", "given", "none"] = Field(
        "auto", description="'auto' uses the model's box when present and derives one otherwise"
    )
    bbox_padding: float = Field(
        0.0, ge=0.0, le=1.0, description="Pad derived boxes by this fraction of their size"
    )
    include_skeleton: bool = Field(True, description="Include the layout's limb connectivity in the response")
    compute_angles: bool = Field(False, description="Report interior joint angles in degrees")
    smoothing: SmoothingParams = Field(default_factory=SmoothingParams)
    previous: dict[str, Any] | list[Any] | None = Field(
        None, description="Previous frame's response (or its instances), used when smoothing is enabled"
    )


def resolve_params(
    body: Any,
    query: dict[str, Any],
    model_cls: type[CommonParams],
) -> tuple[CommonParams, Any]:
    """Merge params from the body and the query string, and return the raw payload.

    Precedence: query string > body ``params`` > environment defaults.
    """
    payload = body
    raw_params: dict[str, Any] = {}

    param_keys = ("params", "postprocess_params", "options", "config")

    if isinstance(body, dict):
        for key in param_keys:
            candidate = body.get(key)
            if isinstance(candidate, dict):
                raw_params.update(candidate)
        # An explicit {"payload": ..., "params": ...} envelope, rather than raw
        # model output with params mixed in alongside it.
        inner = body.get("payload")
        if inner is not None and any(key in body for key in param_keys):
            payload = inner
        # Pose smoothing needs the previous frame, which rides alongside the payload.
        if "previous" in body and "previous" not in raw_params:
            raw_params["previous"] = body["previous"]

    cleaned = {k: v for k, v in query.items() if v is not None and k not in {"api_key"}}
    raw_params.update(_coerce_query(cleaned, model_cls))

    return model_cls.model_validate(raw_params), payload


def describe_validation_error(exc: ValidationError) -> list[dict[str, Any]]:
    """Render a pydantic error into plain JSON-safe dicts."""
    described: list[dict[str, Any]] = []
    for error in exc.errors(include_url=False):
        described.append(
            {
                "param": ".".join(str(part) for part in error.get("loc", ())),
                "message": error.get("msg"),
                "received": repr(error.get("input")),
            }
        )
    return described


_LIST_FIELDS = {"allowed_labels", "denied_labels"}


def _coerce_query(query: dict[str, Any], model_cls: type[CommonParams]) -> dict[str, Any]:
    """Query strings arrive as text; convert them to the field's declared type."""
    out: dict[str, Any] = {}
    fields = model_cls.model_fields
    for key, value in query.items():
        if key not in fields or not isinstance(value, str):
            if key in fields:
                out[key] = value
            continue
        if key in _LIST_FIELDS:
            out[key] = [part.strip() for part in value.split(",") if part.strip()]
            continue
        lowered = value.strip().lower()
        if lowered in {"true", "false"}:
            out[key] = lowered == "true"
            continue
        try:
            out[key] = int(value) if value.strip().lstrip("-").isdigit() else float(value)
        except ValueError:
            out[key] = value
    return out
