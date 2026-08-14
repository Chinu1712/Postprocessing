"""Model id -> task routing.

Used by ``/postprocess`` to pick a pipeline when the caller does not name a
task explicitly. Unknown ids fall back to keyword matching on the id itself,
so a newly registered ``*-seg`` or ``*-pose`` model routes correctly without a
code change here.
"""

from __future__ import annotations

# Known serving endpoints and the task each one performs.
KNOWN_MODELS: dict[str, str] = {
    "discovered-yolo11-object-detection": "detection",
    "discovered-base-detr-resnet-serve": "detection",
    "discovered-base-detectron2-maskrcnn-serve": "segmentation",
    "discovered-vc-yolov8-seg": "segmentation",
    "discovered-base-yolov8n-pose-serve": "pose",
    "discovered-base-yolov8n-pose-v2-serve": "pose",
    "discovered-base-trocr-ocr-serve": "ocr",
}

_POSE_HINTS = ("pose", "keypoint", "kpt", "hrnet", "openpose", "movenet", "blazepose", "rtmpose")
_SEGMENTATION_HINTS = ("seg", "mask", "maskrcnn", "sam", "deeplab", "unet", "panoptic", "instance-seg")


def task_for_model(model_id: str | None) -> str | None:
    """Best-effort task for a model id, or None when it cannot be decided."""
    if not model_id:
        return None

    key = str(model_id).strip().lower()
    if key in KNOWN_MODELS:
        return KNOWN_MODELS[key]

    # Registry ids get prefixed and suffixed in transit; match on the stem too.
    for known, task in KNOWN_MODELS.items():
        if known in key or key in known:
            return task

    if any(hint in key for hint in _POSE_HINTS):
        return "pose"
    if any(hint in key for hint in _SEGMENTATION_HINTS):
        return "segmentation"
    return None


def infer_task(
    model_id: str | None, declared_task: str | None, payload_hint: str | None = None
) -> str | None:
    """Resolve a task from, in order: an explicit task, the model id, a payload hint."""
    for candidate in (declared_task, payload_hint):
        if not candidate:
            continue
        key = str(candidate).strip().lower()
        if key in {"pose", "keypoint", "keypoints", "pose_estimation", "keypoint_detection"}:
            return "pose"
        if key in {"segmentation", "segment", "instance_segmentation", "semantic_segmentation", "mask"}:
            return "segmentation"
        if key in {"detection", "object_detection", "detect"}:
            return "detection"

    return task_for_model(model_id)
