"""End-to-end tests against the HTTP surface, using realistic payload shapes."""

from __future__ import annotations

import numpy as np
import pytest
from fastapi.testclient import TestClient

from app.config import get_settings
from app.core import masks as M
from app.main import app

client = TestClient(app)


# --------------------------------------------------------------------------
# fixtures / payload builders
# --------------------------------------------------------------------------


def square_polygon(x0, y0, x1, y1):
    return [[float(x0), float(y0), float(x1), float(y0), float(x1), float(y1), float(x0), float(y1)]]


def seg_payload(**overrides):
    """detectron2 / maskrcnn shaped output."""
    payload = {
        "model_id": "discovered-base-detectron2-maskrcnn-serve",
        "task": "segmentation",
        "image": {"width": 640, "height": 480},
        "predictions": [
            {"label": "person", "class_id": 0, "score": 0.94, "bbox": [100, 100, 200, 300],
             "segmentation": square_polygon(100, 100, 200, 300)},
            {"label": "person", "class_id": 0, "score": 0.55, "bbox": [102, 104, 198, 298],
             "segmentation": square_polygon(102, 104, 198, 298)},  # duplicate of the first
            {"label": "car", "class_id": 2, "score": 0.81, "bbox": [400, 200, 520, 280],
             "segmentation": square_polygon(400, 200, 520, 280)},
            {"label": "dog", "class_id": 16, "score": 0.11, "bbox": [10, 10, 30, 30],
             "segmentation": square_polygon(10, 10, 30, 30)},  # below threshold
        ],
    }
    payload.update(overrides)
    return payload


def coco_person(offset=0.0, score=0.9, keypoint_score=0.9):
    """17 COCO keypoints laid out as a rough standing figure."""
    base = [
        (100, 50), (95, 45), (105, 45), (90, 47), (110, 47),
        (80, 80), (120, 80), (70, 120), (130, 120), (65, 160), (135, 160),
        (85, 160), (115, 160), (85, 220), (115, 220), (85, 280), (115, 280),
    ]
    flat = []
    for x, y in base:
        flat.extend([float(x) + offset, float(y), keypoint_score])
    return {"score": score, "keypoints": flat}


def pose_payload(**overrides):
    payload = {
        "model_id": "discovered-base-yolov8n-pose-serve",
        "task": "pose",
        "image": {"width": 640, "height": 480},
        "predictions": [coco_person(offset=0.0, score=0.92)],
    }
    payload.update(overrides)
    return payload


# --------------------------------------------------------------------------
# meta
# --------------------------------------------------------------------------


def test_health():
    response = client.get("/health")
    assert response.status_code == 200
    assert response.json()["status"] == "ok"


def test_root_lists_endpoints():
    body = client.get("/").json()
    assert "POST /postprocess/segmentation" in body["endpoints"]
    assert "POST /postprocess/pose" in body["endpoints"]


def test_params_endpoint_documents_defaults():
    body = client.get("/postprocess/params").json()
    assert body["segmentation"]["confidence_threshold"]["default"] == 0.30
    assert body["pose"]["oks_threshold"]["default"] == 0.70
    assert body["pose"]["keypoint_threshold"]["description"]


def test_models_endpoint_routes_known_ids():
    models = {m["model_id"]: m["task"] for m in client.get("/postprocess/models").json()["models"]}
    assert models["discovered-vc-yolov8-seg"] == "segmentation"
    assert models["discovered-base-yolov8n-pose-v2-serve"] == "pose"


def test_openapi_is_generated():
    assert client.get("/openapi.json").status_code == 200


@pytest.mark.parametrize(
    "path",
    ["/postprocess/segmentation", "/postprocess/pose", "/postprocess", "/postprocess/batch"],
)
def test_openapi_declares_a_request_body_with_an_example(path):
    """Without this, Swagger UI at /docs renders no editor and posts an empty body."""
    schema = client.get("/openapi.json").json()
    content = schema["paths"][path]["post"]["requestBody"]["content"]["application/json"]
    assert content["example"], f"{path} has no example body for /docs to prefill"


@pytest.mark.parametrize(
    "path",
    ["/postprocess/segmentation", "/postprocess/pose", "/postprocess", "/postprocess/batch"],
)
def test_documented_example_bodies_actually_work(path):
    """The example shown in /docs must succeed when the user presses Execute."""
    schema = client.get("/openapi.json").json()
    example = schema["paths"][path]["post"]["requestBody"]["content"]["application/json"]["example"]

    response = client.post(path, json=example)
    assert response.status_code == 200, response.text

    body = response.json()
    if path == "/postprocess/batch":
        assert body["failed"] == 0
        assert all(result["count"] >= 1 for result in body["results"])
    else:
        assert body["count"] >= 1


# --------------------------------------------------------------------------
# segmentation
# --------------------------------------------------------------------------


def test_segmentation_thresholds_and_dedupes():
    body = client.post("/postprocess/segmentation", json=seg_payload()).json()

    assert body["task"] == "segmentation"
    assert body["stats"]["input_count"] == 4
    assert body["stats"]["dropped_low_confidence"] == 1  # the 0.11 dog
    assert body["stats"]["suppressed_by_nms"] == 1  # the duplicate person
    assert body["count"] == 2

    labels = sorted(inst["label"] for inst in body["instances"])
    assert labels == ["car", "person"]
    assert body["instances"][0]["score"] == 0.94  # sorted by score


def test_segmentation_returns_polygons_in_image_coordinates():
    body = client.post("/postprocess/segmentation", json=seg_payload()).json()
    person = next(i for i in body["instances"] if i["label"] == "person")

    assert person["mask"]["format"] == "polygon"
    polygon = person["mask"]["polygons"][0]
    xs, ys = polygon[0::2], polygon[1::2]
    assert min(xs) == pytest.approx(100, abs=2)
    assert max(xs) == pytest.approx(200, abs=2)
    assert min(ys) == pytest.approx(100, abs=2)
    assert max(ys) == pytest.approx(300, abs=2)
    assert person["area"] == pytest.approx(100 * 200, rel=0.05)


def test_segmentation_bbox_is_recomputed_from_the_mask():
    payload = seg_payload(predictions=[
        {"label": "person", "score": 0.9, "bbox": [0, 0, 640, 480],  # deliberately wrong
         "segmentation": square_polygon(100, 100, 200, 300)},
    ])
    body = client.post("/postprocess/segmentation", json=payload).json()
    bbox = body["instances"][0]["bbox"]
    assert bbox[0] == pytest.approx(100, abs=2)
    assert bbox[2] == pytest.approx(200, abs=2)


def test_segmentation_accepts_rle_masks():
    mask = np.zeros((480, 640), dtype=bool)
    mask[100:200, 50:150] = True
    payload = {
        "image": {"width": 640, "height": 480},
        "predictions": [{"label": "road", "score": 0.9, "segmentation": M.rle_encode(mask)}],
    }
    body = client.post("/postprocess/segmentation?output_mask=both", json=payload).json()

    assert body["count"] == 1
    instance = body["instances"][0]
    assert instance["has_mask"] is True
    assert instance["area"] == pytest.approx(100 * 100, rel=0.05)
    assert "rle" in instance["mask"]
    assert instance["mask"]["rle"]["counts"]


def test_segmentation_accepts_probability_maps():
    probs = np.zeros((48, 64), dtype=float)
    probs[10:30, 10:30] = 0.95
    payload = {
        "image": {"width": 64, "height": 48},
        "predictions": [{"label": "sky", "score": 0.9, "mask": probs.tolist()}],
    }
    body = client.post("/postprocess/segmentation", json=payload).json()
    assert body["count"] == 1
    assert body["instances"][0]["area"] == pytest.approx(400, rel=0.1)


def test_segmentation_keeps_box_only_detections():
    payload = {
        "image": {"width": 640, "height": 480},
        "predictions": [{"label": "person", "score": 0.9, "bbox": [10, 10, 50, 50]}],
    }
    body = client.post("/postprocess/segmentation", json=payload).json()
    assert body["count"] == 1
    assert body["instances"][0]["has_mask"] is False
    assert body["stats"]["nms_similarity"] == "box_iou"


def test_segmentation_require_mask_drops_box_only_detections():
    payload = {
        "image": {"width": 640, "height": 480},
        "predictions": [{"label": "person", "score": 0.9, "bbox": [10, 10, 50, 50]}],
        "params": {"require_mask": True},
    }
    body = client.post("/postprocess/segmentation", json=payload).json()
    assert body["count"] == 0
    assert body["stats"]["dropped_no_mask"] == 1


def test_segmentation_label_allowlist():
    payload = seg_payload()
    payload["params"] = {"allowed_labels": ["car"]}
    body = client.post("/postprocess/segmentation", json=payload).json()
    assert body["count"] == 1
    assert body["instances"][0]["label"] == "car"
    # The 0.11 dog is already gone on confidence, so the label filter sees 3 and drops 2.
    assert body["stats"]["dropped_low_confidence"] == 1
    assert body["stats"]["dropped_by_label_filter"] == 2


def test_segmentation_label_denylist_via_query_string():
    body = client.post("/postprocess/segmentation?denied_labels=car", json=seg_payload()).json()
    assert {i["label"] for i in body["instances"]} == {"person"}


def test_segmentation_query_params_override_body_params():
    payload = seg_payload()
    payload["params"] = {"confidence_threshold": 0.01}
    body = client.post("/postprocess/segmentation?confidence_threshold=0.9", json=payload).json()
    assert body["params"]["confidence_threshold"] == 0.9
    assert body["count"] == 1  # only the 0.94 person survives


def test_segmentation_min_area_filter():
    payload = seg_payload()
    payload["params"] = {"min_area": 15000, "confidence_threshold": 0.5}
    body = client.post("/postprocess/segmentation", json=payload).json()
    # person = 20000 px stays, car = 9600 px goes
    assert {i["label"] for i in body["instances"]} == {"person"}
    assert body["stats"]["dropped_by_area"] >= 1


def test_segmentation_fill_holes_and_speckle_removal():
    mask = np.zeros((100, 100), dtype=bool)
    mask[20:60, 20:60] = True
    mask[35:45, 35:45] = False  # hole
    mask[90, 90] = True  # speckle
    payload = {
        "image": {"width": 100, "height": 100},
        "predictions": [{"label": "blob", "score": 0.9, "mask": M.rle_encode(mask)}],
        "params": {"fill_holes": True, "min_component_area": 5, "output_mask": "rle"},
    }
    body = client.post("/postprocess/segmentation", json=payload).json()
    decoded = M.rle_decode(body["instances"][0]["mask"]["rle"])
    assert decoded[40, 40]  # hole filled
    assert not decoded[90, 90]  # speckle gone


def test_segmentation_merge_same_label():
    payload = {
        "image": {"width": 200, "height": 200},
        "predictions": [
            {"label": "road", "score": 0.9, "segmentation": square_polygon(0, 0, 50, 50)},
            {"label": "road", "score": 0.8, "segmentation": square_polygon(120, 120, 180, 180)},
        ],
        "params": {"merge_same_label": True, "nms": "none"},
    }
    body = client.post("/postprocess/segmentation", json=payload).json()
    assert body["count"] == 1
    assert body["instances"][0]["extra"]["merged_from"] == 2


def test_segmentation_class_agnostic_nms_merges_across_labels():
    payload = {
        "image": {"width": 640, "height": 480},
        "predictions": [
            {"label": "person", "score": 0.9, "segmentation": square_polygon(10, 10, 110, 110)},
            {"label": "pedestrian", "score": 0.7, "segmentation": square_polygon(11, 11, 109, 109)},
        ],
    }
    default = client.post("/postprocess/segmentation", json=payload).json()
    agnostic = client.post("/postprocess/segmentation?class_agnostic_nms=true", json=payload).json()
    assert default["count"] == 2
    assert agnostic["count"] == 1


def test_segmentation_large_frame_uses_a_bounded_canvas():
    payload = {
        "image": {"width": 3840, "height": 2160},
        "predictions": [{"label": "person", "score": 0.9, "segmentation": square_polygon(0, 0, 1920, 1080)}],
        "params": {"mask_max_side": 512},
    }
    body = client.post("/postprocess/segmentation", json=payload).json()
    assert body["mask_canvas"]["width"] == 512
    assert body["mask_canvas"]["scale"] == pytest.approx(512 / 3840)
    # Polygons still come back in full-resolution image coordinates.
    xs = body["instances"][0]["mask"]["polygons"][0][0::2]
    assert max(xs) == pytest.approx(1920, rel=0.05)
    assert any("mask_max_side" in w for w in body["warnings"])


def test_segmentation_max_detections():
    payload = {
        "image": {"width": 1000, "height": 1000},
        "predictions": [
            {"label": f"obj{i}", "score": 0.9 - i * 0.01, "bbox": [i * 10, 0, i * 10 + 5, 5]}
            for i in range(20)
        ],
        "params": {"max_detections": 5},
    }
    body = client.post("/postprocess/segmentation", json=payload).json()
    assert body["count"] == 5


# --------------------------------------------------------------------------
# pose
# --------------------------------------------------------------------------


def test_pose_basic():
    body = client.post("/postprocess/pose", json=pose_payload()).json()

    assert body["task"] == "pose"
    assert body["count"] == 1
    assert body["keypoint_layout"]["name"] == "coco17"
    assert len(body["keypoint_layout"]["skeleton"]) == 19

    instance = body["instances"][0]
    assert instance["num_keypoints"] == 17
    assert instance["num_visible_keypoints"] == 17
    assert instance["keypoints"][0]["name"] == "nose"
    assert instance["bbox"] is not None
    assert len(instance["keypoints_flat"]) == 51


def test_pose_keypoint_threshold_marks_low_confidence_points_invisible():
    person = coco_person(score=0.9, keypoint_score=0.9)
    person["keypoints"][2] = 0.1  # nose confidence
    person["keypoints"][5] = 0.1  # left eye confidence
    payload = pose_payload(predictions=[person], params={"keypoint_threshold": 0.5})

    body = client.post("/postprocess/pose", json=payload).json()
    instance = body["instances"][0]
    assert instance["num_visible_keypoints"] == 15
    assert instance["keypoints"][0]["visible"] is False


def test_pose_drop_invisible_keypoints():
    person = coco_person()
    person["keypoints"][2] = 0.0
    payload = pose_payload(predictions=[person], params={"drop_invisible_keypoints": True})
    body = client.post("/postprocess/pose", json=payload).json()
    assert len(body["instances"][0]["keypoints"]) == 16


def test_pose_min_visible_keypoints_drops_sparse_skeletons():
    person = coco_person(keypoint_score=0.05)  # everything below threshold
    payload = pose_payload(predictions=[person])
    body = client.post("/postprocess/pose", json=payload).json()
    assert body["count"] == 0
    assert body["stats"]["dropped_few_visible_keypoints"] == 1


def test_pose_oks_nms_removes_a_duplicate_person():
    payload = pose_payload(predictions=[
        coco_person(offset=0.0, score=0.92),
        coco_person(offset=2.0, score=0.61),  # same person, jittered
    ])
    body = client.post("/postprocess/pose", json=payload).json()
    assert body["stats"]["nms_similarity"] == "oks"
    assert body["stats"]["suppressed_by_nms"] == 1
    assert body["count"] == 1
    assert body["instances"][0]["score"] == 0.92


def test_pose_oks_nms_keeps_two_distinct_people():
    payload = pose_payload(predictions=[
        coco_person(offset=0.0, score=0.92),
        coco_person(offset=200.0, score=0.88),
    ])
    body = client.post("/postprocess/pose", json=payload).json()
    assert body["count"] == 2


def test_box_nms_merges_two_people_that_oks_keeps_apart():
    """Side-by-side people overlap heavily in box space but not in skeleton space.

    At a 20px separation box IoU is 0.56 (suppressed at 0.5) while OKS is 0.45
    (kept at 0.7) -- which is exactly why pose defaults to OKS.
    """
    payload = pose_payload(predictions=[
        coco_person(offset=0.0, score=0.92),
        coco_person(offset=20.0, score=0.88),
    ])
    oks = client.post("/postprocess/pose", json=payload).json()
    box = client.post("/postprocess/pose?nms=box&iou_threshold=0.5", json=payload).json()
    assert oks["count"] == 2
    assert box["count"] == 1


def test_pose_confidence_threshold():
    payload = pose_payload(predictions=[coco_person(score=0.1)])
    body = client.post("/postprocess/pose", json=payload).json()
    assert body["count"] == 0
    assert body["stats"]["dropped_low_confidence"] == 1


def test_pose_joint_angles():
    person = coco_person()
    # Bend the left arm into a right angle: shoulder(5), elbow(7), wrist(9).
    person["keypoints"][15:18] = [0.0, 0.0, 0.9]
    person["keypoints"][21:24] = [0.0, 10.0, 0.9]
    person["keypoints"][27:30] = [10.0, 10.0, 0.9]
    payload = pose_payload(predictions=[person], params={"compute_angles": True})

    body = client.post("/postprocess/pose", json=payload).json()
    assert body["instances"][0]["angles"]["left_elbow"] == pytest.approx(90.0, abs=0.5)


def test_pose_bbox_derived_from_keypoints_when_model_gives_none():
    body = client.post("/postprocess/pose", json=pose_payload()).json()
    bbox = body["instances"][0]["bbox"]
    assert bbox[0] == pytest.approx(65, abs=1)
    assert bbox[3] == pytest.approx(280, abs=1)


def test_pose_bbox_mode_given_prefers_the_model_box():
    person = coco_person()
    person["bbox"] = [0, 0, 640, 480]
    payload = pose_payload(predictions=[person], params={"bbox_mode": "given"})
    body = client.post("/postprocess/pose", json=payload).json()
    assert body["instances"][0]["bbox"] == [0.0, 0.0, 640.0, 480.0]


def test_pose_temporal_smoothing():
    previous = client.post("/postprocess/pose", json=pose_payload()).json()
    moved = pose_payload(predictions=[coco_person(offset=10.0, score=0.92)])
    moved["params"] = {"smoothing": {"enabled": True, "alpha": 0.5}}
    moved["previous"] = previous

    body = client.post("/postprocess/pose", json=moved).json()
    assert body["stats"]["smoothed"] == 1
    # Nose was at x=100, moved to x=110; alpha 0.5 lands it halfway.
    assert body["instances"][0]["keypoints"][0]["x"] == pytest.approx(105.0, abs=0.1)


def test_pose_smoothing_without_previous_frame_warns_and_continues():
    payload = pose_payload()
    payload["params"] = {"smoothing": {"enabled": True, "alpha": 0.5}}
    body = client.post("/postprocess/pose", json=payload).json()
    assert body["count"] == 1
    assert any("smoothing" in w for w in body["warnings"])


def test_pose_accepts_nested_keypoint_triplets():
    payload = {
        "image": {"width": 640, "height": 480},
        "predictions": [{"score": 0.9, "keypoints": [[10, 10, 0.9], [20, 20, 0.9], [30, 30, 0.9]]}],
        "params": {"min_visible_keypoints": 3},
    }
    body = client.post("/postprocess/pose", json=payload).json()
    assert body["count"] == 1
    assert body["instances"][0]["num_keypoints"] == 3
    assert any("17" in w for w in body["warnings"])


def test_pose_custom_layout():
    payload = {
        "image": {"width": 640, "height": 480},
        "predictions": [{"score": 0.9, "keypoints": [[10, 10, 0.9], [20, 20, 0.9]]}],
        "params": {
            "layout": {"name": "hand", "names": ["wrist", "tip"], "sigmas": [0.05, 0.05],
                       "skeleton": [[0, 1]]},
            "min_visible_keypoints": 2,
        },
    }
    body = client.post("/postprocess/pose", json=payload).json()
    assert body["keypoint_layout"]["name"] == "hand"
    assert body["instances"][0]["keypoints"][0]["name"] == "wrist"


# --------------------------------------------------------------------------
# input-shape tolerance
# --------------------------------------------------------------------------


def test_accepts_a_bare_list_of_predictions():
    payload = [{"label": "person", "score": 0.9, "bbox": [10, 10, 50, 50]}]
    body = client.post("/postprocess/segmentation", json=payload).json()
    assert body["count"] == 1


def test_accepts_columnar_output():
    payload = {
        "image": {"width": 640, "height": 480},
        "boxes": [[10, 10, 50, 50], [100, 100, 150, 150]],
        "scores": [0.9, 0.8],
        "labels": ["person", "car"],
    }
    body = client.post("/postprocess/segmentation", json=payload).json()
    assert body["count"] == 2
    assert {i["label"] for i in body["instances"]} == {"person", "car"}


def test_accepts_nested_results_container():
    payload = {
        "image": {"width": 640, "height": 480},
        "results": {"predictions": [{"label": "person", "score": 0.9, "bbox": [10, 10, 50, 50]}]},
    }
    assert client.post("/postprocess/segmentation", json=payload).json()["count"] == 1


def test_xywh_boxes_are_detected():
    payload = {
        "image": {"width": 640, "height": 480},
        "predictions": [{"label": "person", "score": 0.9, "bbox": [10, 10, 40, 40], "bbox_format": "xywh"}],
    }
    body = client.post("/postprocess/segmentation", json=payload).json()
    assert body["instances"][0]["bbox"] == [10.0, 10.0, 50.0, 50.0]


def test_normalized_coordinates_are_scaled_to_pixels():
    payload = {
        "image": {"width": 640, "height": 480},
        "predictions": [{"label": "person", "score": 0.9, "bbox": [0.1, 0.2, 0.5, 0.8]}],
    }
    body = client.post("/postprocess/segmentation", json=payload).json()
    assert body["instances"][0]["bbox"] == [64.0, 96.0, 320.0, 384.0]
    assert any("normalised" in w for w in body["warnings"])


def test_unknown_fields_are_echoed_back():
    payload = {
        "image": {"width": 640, "height": 480},
        "predictions": [{"label": "person", "score": 0.9, "bbox": [10, 10, 50, 50], "zone": "entrance"}],
    }
    body = client.post("/postprocess/segmentation", json=payload).json()
    assert body["instances"][0]["extra"]["zone"] == "entrance"


def test_track_ids_survive():
    payload = {
        "image": {"width": 640, "height": 480},
        "predictions": [{"label": "person", "score": 0.9, "bbox": [10, 10, 50, 50], "track_id": 7}],
    }
    assert client.post("/postprocess/segmentation", json=payload).json()["instances"][0]["track_id"] == 7


def test_missing_image_size_is_inferred_and_flagged():
    payload = {"predictions": [{"label": "person", "score": 0.9, "bbox": [10, 10, 50, 50]}]}
    body = client.post("/postprocess/segmentation", json=payload).json()
    assert body["count"] == 1
    assert any("image size" in w for w in body["warnings"])


def test_frame_metadata_is_passed_through():
    payload = seg_payload(frame_id=42, timestamp="2026-08-13T10:00:00Z")
    body = client.post("/postprocess/segmentation", json=payload).json()
    assert body["frame_id"] == 42
    assert body["timestamp"] == "2026-08-13T10:00:00Z"


# --------------------------------------------------------------------------
# routing, batching, errors
# --------------------------------------------------------------------------


def test_auto_routes_on_model_id():
    payload = pose_payload()
    payload.pop("task")
    assert client.post("/postprocess", json=payload).json()["task"] == "pose"


def test_auto_routes_on_payload_shape():
    payload = pose_payload()
    payload.pop("task")
    payload.pop("model_id")
    assert client.post("/postprocess", json=payload).json()["task"] == "pose"


def test_auto_routes_segmentation_on_query_param():
    payload = seg_payload()
    payload.pop("task")
    payload.pop("model_id")
    assert client.post("/postprocess?task=segmentation", json=payload).json()["task"] == "segmentation"


def test_auto_returns_400_when_task_is_undecidable():
    payload = {"image": {"width": 10, "height": 10}, "predictions": [{"score": 0.9, "bbox": [1, 1, 2, 2]}]}
    response = client.post("/postprocess", json=payload)
    assert response.status_code == 400
    assert "task" in response.json()["detail"]


def test_batch_processes_each_frame():
    body = client.post(
        "/postprocess/batch",
        json={"task": "pose", "items": [pose_payload(), pose_payload()], "params": {"compute_angles": True}},
    ).json()
    assert body["count"] == 2
    assert body["failed"] == 0
    assert all(result["count"] == 1 for result in body["results"])
    assert "angles" in body["results"][0]["instances"][0]


def test_batch_isolates_a_failing_frame():
    body = client.post(
        "/postprocess/batch",
        json={"task": "segmentation", "items": [seg_payload(), {"nothing": "useful"}]},
    ).json()
    assert body["failed"] == 1
    assert body["results"][0]["count"] == 2
    assert "error" in body["results"][1]


def test_unparseable_payload_returns_422_so_the_caller_falls_back():
    response = client.post("/postprocess/segmentation", json={"totally": "unrelated"})
    assert response.status_code == 422
    assert "could not interpret" in response.json()["detail"]


def test_empty_body_returns_400():
    response = client.post("/postprocess/pose", content=b"")
    assert response.status_code == 400


def test_invalid_json_returns_400():
    response = client.post(
        "/postprocess/pose", content=b"{not json", headers={"Content-Type": "application/json"}
    )
    assert response.status_code == 400


def test_invalid_param_value_returns_422():
    payload = seg_payload()
    payload["params"] = {"confidence_threshold": 5.0}
    assert client.post("/postprocess/segmentation", json=payload).status_code == 422


def test_empty_prediction_list_is_a_valid_empty_result():
    body = client.post(
        "/postprocess/pose", json={"image": {"width": 640, "height": 480}, "predictions": []}
    ).json()
    assert body["count"] == 0
    assert body["instances"] == []


def test_request_id_header_is_returned():
    response = client.get("/health", headers={"X-Request-ID": "abc123"})
    assert response.headers["X-Request-ID"] == "abc123"
    assert "X-Process-Time-Ms" in response.headers


# --------------------------------------------------------------------------
# auth
# --------------------------------------------------------------------------


@pytest.fixture
def secured(monkeypatch):
    monkeypatch.setenv("API_KEYS", "secret-one,secret-two")
    get_settings.cache_clear()
    yield
    monkeypatch.delenv("API_KEYS", raising=False)
    get_settings.cache_clear()


def test_api_key_required_when_configured(secured):
    assert client.post("/postprocess/pose", json=pose_payload()).status_code == 401
    assert client.get("/health").status_code == 200  # probes stay open


def test_api_key_accepted_in_header_and_as_bearer(secured):
    assert client.post(
        "/postprocess/pose", json=pose_payload(), headers={"X-API-Key": "secret-one"}
    ).status_code == 200
    assert client.post(
        "/postprocess/pose", json=pose_payload(), headers={"Authorization": "Bearer secret-two"}
    ).status_code == 200


def test_wrong_api_key_is_rejected(secured):
    assert client.post(
        "/postprocess/pose", json=pose_payload(), headers={"X-API-Key": "nope"}
    ).status_code == 401
