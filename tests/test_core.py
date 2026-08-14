"""Unit tests for the geometry, mask and keypoint primitives."""

from __future__ import annotations

import numpy as np
import pytest

from app.core import masks as M
from app.core.geometry import box_iou_matrix, greedy_nms, xywh_to_xyxy, xyxy_to_xywh
from app.core.keypoints import (
    bbox_from_keypoints,
    get_layout,
    joint_angles,
    oks_matrix,
    parse_keypoints,
    smooth_keypoints,
)

# --------------------------------------------------------------------------
# geometry
# --------------------------------------------------------------------------


def test_box_conversions_round_trip():
    assert xywh_to_xyxy([10, 20, 30, 40]) == [10, 20, 40, 60]
    assert xyxy_to_xywh([10, 20, 40, 60]) == [10, 20, 30, 40]


def test_box_iou_matrix():
    boxes = np.array([[0, 0, 10, 10], [0, 0, 10, 10], [20, 20, 30, 30]], dtype=float)
    iou = box_iou_matrix(boxes)
    assert iou[0, 1] == pytest.approx(1.0)
    assert iou[0, 2] == pytest.approx(0.0)


def test_box_iou_half_overlap():
    boxes = np.array([[0, 0, 10, 10], [5, 0, 15, 10]], dtype=float)
    # intersection 50, union 150
    assert box_iou_matrix(boxes)[0, 1] == pytest.approx(1 / 3)


def test_greedy_nms_respects_labels():
    iou = np.array([[1.0, 0.9], [0.9, 1.0]])
    scores = [0.9, 0.8]

    assert greedy_nms(scores, iou, 0.7, labels=["car", "car"]) == [0]
    assert sorted(greedy_nms(scores, iou, 0.7, labels=["car", "person"])) == [0, 1]
    assert greedy_nms(scores, iou, 0.7, labels=["car", "person"], class_agnostic=True) == [0]


def test_greedy_nms_keeps_highest_score():
    iou = np.array([[1.0, 0.95], [0.95, 1.0]])
    assert greedy_nms([0.3, 0.95], iou, 0.7, labels=["a", "a"]) == [1]


# --------------------------------------------------------------------------
# masks
# --------------------------------------------------------------------------


def _square_mask(size=20, x0=4, y0=4, x1=14, y1=14):
    mask = np.zeros((size, size), dtype=bool)
    mask[y0:y1, x0:x1] = True
    return mask


def test_rle_round_trip():
    mask = _square_mask()
    decoded = M.rle_decode(M.rle_encode(mask))
    assert np.array_equal(decoded, mask)


def test_rle_round_trip_full_and_empty():
    for mask in (np.ones((8, 6), dtype=bool), np.zeros((8, 6), dtype=bool)):
        assert np.array_equal(M.rle_decode(M.rle_encode(mask)), mask)


def test_compressed_rle_decodes():
    # Reference vector from the COCO API for a 4x4 mask with a 2x2 block set.
    mask = np.zeros((4, 4), dtype=bool)
    mask[1:3, 1:3] = True
    uncompressed = M.rle_encode(mask)
    compressed = _compress(uncompressed["counts"])
    decoded = M.rle_decode({"size": [4, 4], "counts": compressed})
    assert np.array_equal(decoded, mask)


def _compress(counts: list[int]) -> str:
    """Mirror of COCO's rleToString, used only to exercise the decoder."""
    out = []
    for i, value in enumerate(counts):
        x = int(value)
        if i > 2:
            x -= int(counts[i - 2])
        more = True
        while more:
            c = x & 0x1F
            x >>= 5
            more = (x != -1) if (c & 0x10) else (x != 0)
            if more:
                c |= 0x20
            out.append(chr(c + 48))
    return "".join(out)


def test_polygon_to_mask_and_back():
    polygon = [[4.0, 4.0, 14.0, 4.0, 14.0, 14.0, 4.0, 14.0]]
    mask = M.polygons_to_mask(polygon, 20, 20)
    assert mask.sum() == 100
    assert M.bbox_from_mask(mask) == [4.0, 4.0, 14.0, 14.0]


def test_mask_to_polygons_traces_a_square():
    mask = _square_mask()
    polygons = M.mask_to_polygons(mask, tolerance=0.5)
    assert len(polygons) == 1
    xs = polygons[0][0::2]
    ys = polygons[0][1::2]
    assert min(xs) == pytest.approx(4.5, abs=0.6)
    assert max(xs) == pytest.approx(13.5, abs=0.6)
    assert min(ys) == pytest.approx(4.5, abs=0.6)
    assert max(ys) == pytest.approx(13.5, abs=0.6)


def test_mask_to_polygons_finds_each_component():
    mask = np.zeros((30, 30), dtype=bool)
    mask[2:8, 2:8] = True
    mask[20:26, 20:26] = True
    assert len(M.mask_to_polygons(mask)) == 2


def test_mask_iou_matrix():
    a = _square_mask()
    b = _square_mask(x0=9, x1=19)  # half overlap horizontally
    iou = M.mask_iou_matrix([a, b])
    assert iou[0, 0] == pytest.approx(1.0)
    assert iou[0, 1] == pytest.approx(50 / 150, abs=1e-6)


def test_fill_holes():
    mask = _square_mask()
    mask[8:10, 8:10] = False
    assert not mask[8, 8]
    assert M.fill_holes(mask)[8, 8]


def test_remove_small_components():
    mask = np.zeros((30, 30), dtype=bool)
    mask[2:12, 2:12] = True  # 100 px
    mask[25, 25] = True  # speckle
    cleaned = M.remove_small_components(mask, min_area=10)
    assert cleaned.sum() == 100
    assert not cleaned[25, 25]


def test_connected_components_uses_8_connectivity():
    mask = np.zeros((6, 6), dtype=bool)
    mask[1, 1] = True
    mask[2, 2] = True  # diagonal neighbour
    _, count = M.connected_components(mask)
    assert count == 1


def test_fit_canvas_downscales_only_when_needed():
    assert M.fit_canvas(800, 600, 1024) == (800, 600, 1.0)
    w, h, scale = M.fit_canvas(4000, 2000, 1000)
    assert (w, h) == (1000, 500)
    assert scale == pytest.approx(0.25)


def test_decode_mask_accepts_probability_map():
    probs = np.zeros((10, 10), dtype=float)
    probs[2:6, 2:6] = 0.9
    mask = M.decode_mask(probs.tolist(), 10, 10, binarize_threshold=0.5)
    assert mask is not None and mask.sum() == 16


def test_decode_mask_rescales_polygons_to_working_canvas():
    polygon = [[0.0, 0.0, 100.0, 0.0, 100.0, 100.0, 0.0, 100.0]]
    mask = M.decode_mask(polygon, 50, 50, image_height=100, image_width=100)
    assert mask is not None
    assert mask.sum() == pytest.approx(2500, rel=0.05)


def test_decode_mask_returns_none_for_junk():
    assert M.decode_mask({"unrelated": 1}, 10, 10) is None
    assert M.decode_mask([], 10, 10) is None


def test_simplify_polygon_reduces_collinear_points():
    ring = [(float(x), 0.0) for x in range(10)] + [(9.0, 5.0), (0.0, 5.0)]
    assert len(M.simplify_polygon(ring, tolerance=0.5)) < len(ring)


# --------------------------------------------------------------------------
# keypoints
# --------------------------------------------------------------------------


def test_parse_keypoints_flat_triplets():
    kp = parse_keypoints([1.0, 2.0, 0.9, 3.0, 4.0, 0.8])
    assert kp is not None and kp.shape == (2, 3)
    assert kp[1].tolist() == [3.0, 4.0, 0.8]


def test_parse_keypoints_maps_coco_visibility_flags():
    kp = parse_keypoints([1.0, 2.0, 2.0, 3.0, 4.0, 0.0, 5.0, 6.0, 1.0])
    assert kp is not None
    assert kp[0, 2] == 1.0  # v=2 -> visible
    assert kp[1, 2] == 0.0  # v=0 -> absent
    assert kp[2, 2] == 0.5  # v=1 -> labelled but occluded


def test_parse_keypoints_nested_and_dicts():
    assert parse_keypoints([[1, 2, 0.5], [3, 4, 0.6]]).shape == (2, 3)
    parsed = parse_keypoints([{"x": 1, "y": 2, "score": 0.7}])
    assert parsed is not None and parsed[0].tolist() == [1.0, 2.0, 0.7]


def test_parse_keypoints_xy_pairs_default_to_full_confidence():
    kp = parse_keypoints([1.0, 2.0, 3.0, 4.0])
    assert kp is not None and kp.shape == (2, 3)
    assert kp[:, 2].tolist() == [1.0, 1.0]


def test_oks_identical_poses_is_one():
    kp = np.array([[10.0, 10.0, 1.0], [20.0, 20.0, 1.0], [30.0, 10.0, 1.0]])
    layout = get_layout("coco17")
    oks = oks_matrix([kp, kp.copy()], [10_000.0, 10_000.0], layout.sigmas)
    assert oks[0, 1] == pytest.approx(1.0)


def test_oks_falls_off_with_distance():
    a = np.array([[10.0, 10.0, 1.0], [20.0, 20.0, 1.0]])
    b = a.copy()
    b[:, :2] += 60.0
    layout = get_layout("coco17")
    assert oks_matrix([a, b], [1000.0, 1000.0], layout.sigmas)[0, 1] < 0.2


def test_oks_ignores_invisible_keypoints():
    a = np.array([[10.0, 10.0, 1.0], [900.0, 900.0, 0.0]])
    b = np.array([[10.0, 10.0, 1.0], [5.0, 5.0, 0.0]])
    layout = get_layout("coco17")
    assert oks_matrix([a, b], [1000.0, 1000.0], layout.sigmas)[0, 1] == pytest.approx(1.0)


def test_bbox_from_keypoints_skips_invisible():
    kp = np.array([[10.0, 10.0, 0.9], [20.0, 30.0, 0.9], [999.0, 999.0, 0.0]])
    assert bbox_from_keypoints(kp, threshold=0.0) == [10.0, 10.0, 20.0, 30.0]


def test_bbox_from_keypoints_padding():
    kp = np.array([[0.0, 0.0, 1.0], [10.0, 10.0, 1.0]])
    assert bbox_from_keypoints(kp, padding=0.1) == [-1.0, -1.0, 11.0, 11.0]


def test_joint_angles_right_angle():
    layout = get_layout("coco17")
    kp = np.zeros((17, 3))
    kp[:, 2] = 1.0
    kp[5] = [0.0, 0.0, 1.0]  # left shoulder
    kp[7] = [0.0, 10.0, 1.0]  # left elbow
    kp[9] = [10.0, 10.0, 1.0]  # left wrist
    assert joint_angles(kp, layout)["left_elbow"] == pytest.approx(90.0, abs=0.01)


def test_smooth_keypoints_blends_towards_previous():
    current = np.array([[10.0, 10.0, 1.0]])
    previous = np.array([[0.0, 0.0, 1.0]])
    assert smooth_keypoints(current, previous, alpha=0.5)[0, :2].tolist() == [5.0, 5.0]


def test_smooth_keypoints_skips_large_jumps():
    current = np.array([[500.0, 500.0, 1.0]])
    previous = np.array([[0.0, 0.0, 1.0]])
    smoothed = smooth_keypoints(current, previous, alpha=0.5, max_jump=100.0)
    assert smoothed[0, :2].tolist() == [500.0, 500.0]


def test_get_layout_custom_spec():
    layout = get_layout({"name": "hand", "names": ["wrist", "thumb"], "sigmas": [0.1, 0.2]})
    assert layout.num_keypoints == 2
    assert layout.names == ["wrist", "thumb"]


def test_get_layout_falls_back_to_generic_for_unusual_counts():
    layout = get_layout(None, num_keypoints=21)
    assert layout.name == "generic21"
    assert layout.num_keypoints == 21


def test_get_layout_defaults_to_coco17_for_17_keypoints():
    assert get_layout(None, num_keypoints=17).name == "coco17"
