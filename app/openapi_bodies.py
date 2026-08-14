"""Request-body declarations for the OpenAPI schema.

The postprocessing endpoints read the raw request body themselves so they can
accept any model-server payload shape. That flexibility costs us the automatic
schema FastAPI would otherwise derive from a Pydantic parameter -- and without a
declared ``requestBody`` the Swagger UI at ``/docs`` shows no editor and sends an
empty body, which is useless for hand-testing.

So we declare the body here explicitly, with a runnable example for each
endpoint. The schema stays permissive (a free-form object) because the whole
point is that we accept more than one shape; the examples are what make ``/docs``
a working test console.
"""

from __future__ import annotations

from typing import Any

# A rough standing figure in COCO-17 order, as flat x, y, score triplets.
_POSE_KEYPOINTS = [
    385, 120, 0.97, 375, 110, 0.94, 395, 110, 0.95, 362, 114, 0.81, 408, 114, 0.83,
    340, 190, 0.92, 430, 190, 0.91, 315, 300, 0.88, 455, 300, 0.87,
    300, 400, 0.84, 470, 400, 0.82, 352, 390, 0.90, 418, 390, 0.90,
    348, 520, 0.86, 422, 520, 0.85, 344, 650, 0.79, 426, 650, 0.78,
]

# The same person again, jittered and low-confidence: a duplicate detection that
# OKS deduplication should remove.
_POSE_KEYPOINTS_DUPLICATE = [
    387, 124, 0.55, 377, 114, 0.52, 397, 114, 0.53, 364, 118, 0.31, 410, 118, 0.33,
    342, 194, 0.51, 432, 194, 0.50, 317, 304, 0.47, 457, 304, 0.46,
    302, 404, 0.43, 472, 404, 0.41, 354, 394, 0.49, 420, 394, 0.49,
    350, 524, 0.45, 424, 524, 0.44, 346, 654, 0.38, 428, 654, 0.37,
]

SEGMENTATION_EXAMPLE: dict[str, Any] = {
    "model_id": "discovered-vc-yolov8-seg",
    "image": {"width": 1280, "height": 720},
    "frame_id": 1042,
    "predictions": [
        {
            "label": "person",
            "class_id": 0,
            "score": 0.94,
            "bbox": [320, 140, 520, 640],
            "segmentation": [[320, 140, 520, 140, 520, 640, 320, 640]],
        },
        {
            "label": "person",
            "class_id": 0,
            "score": 0.58,
            "bbox": [324, 146, 516, 634],
            "segmentation": [[324, 146, 516, 146, 516, 634, 324, 634]],
        },
        {
            "label": "forklift",
            "class_id": 7,
            "score": 0.82,
            "bbox": [800, 300, 1100, 600],
            "segmentation": [[800, 300, 1100, 300, 1100, 600, 800, 600]],
        },
        {
            "label": "pallet",
            "class_id": 9,
            "score": 0.19,
            "bbox": [40, 600, 120, 680],
            "segmentation": [[40, 600, 120, 600, 120, 680, 40, 680]],
        },
    ],
    "params": {"confidence_threshold": 0.3, "iou_threshold": 0.7, "output_mask": "polygon"},
}

POSE_EXAMPLE: dict[str, Any] = {
    "model_id": "discovered-base-yolov8n-pose-serve",
    "image": {"width": 1280, "height": 720},
    "frame_id": 1042,
    "predictions": [
        {"label": "person", "score": 0.93, "bbox": [300, 90, 470, 660], "keypoints": _POSE_KEYPOINTS},
        {
            "label": "person",
            "score": 0.44,
            "bbox": [304, 96, 466, 654],
            "keypoints": _POSE_KEYPOINTS_DUPLICATE,
        },
    ],
    "params": {
        "confidence_threshold": 0.3,
        "keypoint_threshold": 0.3,
        "oks_threshold": 0.7,
        "compute_angles": True,
    },
}

AUTO_EXAMPLE: dict[str, Any] = {"task": "pose", **POSE_EXAMPLE}

BATCH_EXAMPLE: dict[str, Any] = {
    "task": "pose",
    "params": {"confidence_threshold": 0.3, "compute_angles": True},
    "items": [
        {"image": {"width": 1280, "height": 720}, "frame_id": 1, "predictions": POSE_EXAMPLE["predictions"]},
        {"image": {"width": 1280, "height": 720}, "frame_id": 2, "predictions": POSE_EXAMPLE["predictions"]},
    ],
}


def request_body(example: dict[str, Any], description: str) -> dict[str, Any]:
    """Build the ``openapi_extra`` fragment that gives ``/docs`` a working editor."""
    return {
        "requestBody": {
            "required": True,
            "description": description,
            "content": {
                "application/json": {
                    "schema": {
                        "type": "object",
                        "description": (
                            "Raw model output. The shape is intentionally not constrained -- see the "
                            "endpoint description for the payload forms accepted."
                        ),
                    },
                    "example": example,
                }
            },
        }
    }
