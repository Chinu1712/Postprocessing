# Postprocess API

Stateless REST postprocessing for **segmentation** and **pose** model output.

A pipeline's postprocessing node POSTs raw model results here; the service applies
confidence thresholding, label filtering, mask/keypoint cleanup and overlap
deduplication, and returns a normalised result. It replaces the node's built-in
filter (confidence threshold + same-label IoU dedupe) with logic that actually
understands masks and skeletons.

Nothing is persisted and nothing is shared between requests, so it scales
horizontally and deploys to Render as a plain web service.

---

## Why this exists

The built-in postprocessing does two things: drop detections below a confidence
threshold, and deduplicate same-label boxes above an IoU threshold. That is the
right behaviour for **detection** models. For the other two task types it is
wrong in specific ways:

| | Built-in behaviour | What this service does |
|---|---|---|
| **Segmentation** | Dedupes on *box* IoU. Two instances with near-identical boxes but disjoint masks (a person behind a railing, overlapping produce on a belt) get merged. | Dedupes on *mask* IoU. Also decodes, cleans (speckle removal, hole filling) and re-encodes masks, and recomputes each box from its mask. |
| **Pose** | Dedupes on box IoU. Two people standing side by side have ~0.6 box IoU and get merged; a duplicate detection of one person and a genuine second person are indistinguishable. | Dedupes on **OKS** (Object Keypoint Similarity), the COCO metric that compares skeletons rather than boxes. Also thresholds per-keypoint confidence, drops skeletons with too few visible joints, derives boxes from keypoints, and optionally smooths across frames. |

Concretely, for two people 20 px apart in a 1280×720 frame: box IoU is 0.56
(suppressed at the default 0.5) while OKS is 0.45 (kept at the default 0.7).
There is a test pinning exactly this — `test_box_nms_merges_two_people_that_oks_keeps_apart`.

---

## Endpoints

| Method | Path | Purpose |
|---|---|---|
| `POST` | `/postprocess/segmentation` | Postprocess instance-segmentation output |
| `POST` | `/postprocess/pose` | Postprocess pose / keypoint output |
| `POST` | `/postprocess` | Route to one of the above using `task` or `model_id` |
| `POST` | `/postprocess/batch` | Several frames in one call |
| `GET` | `/postprocess/params` | Every tuning parameter, its default and its description |
| `GET` | `/postprocess/models` | Known model ids and the task each routes to |
| `GET` | `/health` | Liveness probe (never requires auth) |
| `GET` | `/docs` | Interactive OpenAPI docs |

---

## Request contract

POST the raw model output as the JSON body. Tuning parameters go under
`params` in that body, or in the query string (query string wins).

```jsonc
{
  "model_id": "discovered-vc-yolov8-seg",     // optional, used for auto-routing
  "task": "segmentation",                      // optional, used for auto-routing
  "image": { "width": 1280, "height": 720 },   // strongly recommended
  "frame_id": 1042,                            // echoed back untouched
  "predictions": [ /* per-instance objects */ ],
  "params": { "confidence_threshold": 0.3 }    // optional
}
```

`image.width` / `image.height` matter: without them masks cannot be rasterised
at the right scale and normalised (0–1) coordinates cannot be converted to
pixels. The service infers a size from prediction extents when they are missing
and says so in `warnings`.

### Input shapes it accepts

The payload does not have to match one schema. The normaliser sniffs the common
serving formats:

- **Prediction list under any of** `instances`, `predictions`, `detections`,
  `results`, `outputs`, `objects`, `segments`, `poses`, `items`, `data` —
  including nested (`results.predictions`).
- **A bare JSON array** of prediction objects.
- **Columnar / parallel arrays**: `{"boxes": [...], "scores": [...], "labels": [...], "masks": [...], "keypoints": [...]}` (ultralytics, Triton).
- **Per-field aliases**: score / confidence / conf / probability; label /
  class_name / category / name; bbox / box / bounding_box / xyxy / rect.
- **Box formats**: `xyxy`, `xywh`, `cxcywh`, `{x1,y1,x2,y2}`, `{x,y,w,h}`,
  `{left,top,right,bottom}`. Auto-detected, or pinned with `params.bbox_format`.
- **Normalised coordinates**: 0–1 boxes and keypoints are detected and scaled to
  pixels (`params.coordinate_space` to force it either way).
- **Masks**: COCO polygons (flat or point-pair), compressed COCO RLE (the LEB128
  string form), uncompressed RLE, dense bitmaps, float probability maps, and
  base64 raw/PNG bitmaps.
- **Keypoints**: flat `[x,y,v,...]` triplets, flat `[x,y,...]` pairs, nested
  `[[x,y,v],...]`, and dict lists `[{"x":..,"y":..,"score":..}]`. COCO
  visibility flags (`0`/`1`/`2`) are mapped onto confidences.

Unrecognised per-instance fields are echoed back under `extra`, so `track_id`,
zone tags and similar metadata survive the round trip.

If the payload contains nothing detection-shaped, the service returns **422**
rather than an empty result — a failed connector call makes the pipeline fall
back to its built-in filter, which is far better than silently dropping every
detection in the frame.

---

## Response

```jsonc
{
  "task": "segmentation",
  "model_id": "discovered-vc-yolov8-seg",
  "frame_id": 1042,
  "image": { "width": 1280, "height": 720 },
  "mask_canvas": { "width": 1024, "height": 576, "scale": 0.8 },
  "count": 2,
  "instances": [ /* see below */ ],
  "detections": [ /* the same array, aliased for consumers expecting this key */ ],
  "stats": { "input_count": 4, "source_container": "predictions",
             "dropped_low_confidence": 1, "suppressed_by_nms": 1,
             "nms_similarity": "mask_iou", "kept": 2, "duration_ms": 42.6 },
  "params": { /* the parameters actually applied */ },
  "warnings": []
}
```

`stats` is designed to be readable in a pipeline log: every filter reports how
many instances it removed, so a run that returns nothing tells you *which* stage
emptied it. `source_container` names the key the predictions were found under,
which answers the other common question when a tolerant parser sees an
unfamiliar payload.

### Segmentation instance

```jsonc
{
  "id": 0,
  "source_index": 0,
  "label": "person",
  "label_id": 0,
  "score": 0.94,
  "bbox": [320.0, 140.0, 520.0, 640.0],
  "bbox_format": "xyxy",
  "bbox_xywh": [320.0, 140.0, 200.0, 500.0],
  "area": 100000.0,
  "has_mask": true,
  "mask": {
    "format": "polygon",
    "polygons": [[320.62, 140.62, 519.38, 140.62, 519.38, 639.38, 320.62, 639.38]],
    "coordinate_space": "image_pixels",
    "size": [576, 1024],
    "scale": 0.8
  }
}
```

Polygons are always returned in **full-resolution image pixels**, whatever the
internal canvas size. `size` / `scale` describe the canvas and only matter if you
ask for RLE (`params.output_mask` = `rle` or `both`), which is emitted at canvas
resolution as uncompressed, column-major COCO RLE — plain JSON integers, so the
consumer needs no pycocotools.

### Pose instance

```jsonc
{
  "id": 0,
  "label": "person",
  "score": 0.93,
  "bbox": [300.0, 90.0, 470.0, 660.0],
  "num_keypoints": 17,
  "num_visible_keypoints": 17,
  "mean_keypoint_score": 0.871765,
  "keypoints": [
    { "index": 0, "name": "nose", "x": 385.0, "y": 120.0, "score": 0.97, "visible": true }
  ],
  "keypoints_flat": [385.0, 120.0, 0.97, "..."],
  "angles": { "left_elbow": 175.73, "left_knee": 180.0 }
}
```

The response also carries `keypoint_layout` with the joint `names` and the
`skeleton` limb pairs, so a renderer can draw the figure without hardcoding COCO.

Full worked request/response pairs live in [`examples/`](examples/).

---

## Parameters

Send under `params` in the body or as query parameters. `GET /postprocess/params`
returns this table live from the running service.

### Shared

| Parameter | Default | Meaning |
|---|---|---|
| `confidence_threshold` | `0.30` | Instances scoring below this are dropped |
| `max_detections` | `300` | Cap after all filtering |
| `allowed_labels` | `null` | Keep only these labels (case-insensitive) |
| `denied_labels` | `null` | Drop these labels |
| `class_agnostic_nms` | `false` | Suppress across labels rather than within a label |
| `clip_to_image` | `true` | Clamp boxes and keypoints to the frame |
| `bbox_format` | `auto` | `auto` / `xyxy` / `xywh` / `cxcywh` |
| `coordinate_space` | `auto` | `auto` / `pixel` / `normalized` |
| `keep_extra_fields` | `true` | Echo unrecognised per-instance fields back |
| `sort_by` | `score` | `score` / `area` / `none` |

### Segmentation

| Parameter | Default | Meaning |
|---|---|---|
| `iou_threshold` | `0.70` | Overlapping same-label instances above this are deduplicated |
| `nms` | `mask` | `mask` (mask IoU) / `box` (box IoU) / `none` |
| `mask_binarize_threshold` | `0.5` | Cut-off for probability masks |
| `mask_max_side` | `1024` | Longest side of the internal raster canvas |
| `min_area` | `0` | Drop masks smaller than this many image pixels |
| `min_area_ratio` | `0` | Same, as a fraction of frame area |
| `max_area_ratio` | `1.0` | Drop masks covering more than this fraction |
| `min_component_area` | `0` | Speckle removal: drop connected components below this size |
| `fill_holes` | `false` | Fill interior holes in each mask |
| `merge_same_label` | `false` | Union all masks sharing a label (semantic-style output) |
| `simplify_tolerance` | `1.0` | Douglas-Peucker tolerance in pixels; `0` disables |
| `output_mask` | `polygon` | `polygon` / `rle` / `both` / `none` |
| `bbox_from_mask` | `true` | Recompute each box from its mask |
| `require_mask` | `false` | Drop instances with no decodable mask |

### Pose

| Parameter | Default | Meaning |
|---|---|---|
| `keypoint_threshold` | `0.30` | Keypoints below this are marked not-visible |
| `oks_threshold` | `0.70` | Instances above this OKS are deduplicated |
| `iou_threshold` | `0.70` | Box IoU threshold, used when `nms="box"` |
| `nms` | `oks` | `oks` / `box` / `none` |
| `layout` | `coco17` | Layout name, or a custom `{names, sigmas, skeleton, angles}` object |
| `min_visible_keypoints` | `3` | Drop skeletons with fewer visible joints |
| `drop_invisible_keypoints` | `false` | Omit invisible keypoints instead of flagging them |
| `bbox_mode` | `auto` | `auto` / `keypoints` / `given` / `none` |
| `bbox_padding` | `0.0` | Pad derived boxes by this fraction |
| `include_skeleton` | `true` | Include limb connectivity in the response |
| `compute_angles` | `false` | Report interior joint angles in degrees |
| `smoothing` | disabled | `{enabled, alpha, max_jump, match_threshold}` — see below |

#### Temporal smoothing

The service is stateless, so smoothing works by the caller handing back the
previous frame's response:

```jsonc
{
  "image": { "width": 1280, "height": 720 },
  "predictions": [ /* this frame */ ],
  "previous": { /* the previous frame's response, or just its instances array */ },
  "params": { "smoothing": { "enabled": true, "alpha": 0.6, "max_jump": 80 } }
}
```

Instances are matched by `track_id` when present, otherwise by best OKS above
`match_threshold`. `alpha` is the weight of the current observation, so `1.0`
disables smoothing and `0.3` is heavy damping. Keypoints that move further than
`max_jump` pixels are left unsmoothed — that is a real fast motion or a
re-detection, not jitter.

---

## Wiring it into the pipeline

In the **Postprocessing** node, set **External REST Connector** to the deployed
URL:

```
https://<your-service>.onrender.com/postprocess/segmentation
```

or, for pose models:

```
https://<your-service>.onrender.com/postprocess/pose
```

If the connector cannot send a per-call body of parameters, put them in the
query string instead — the node only needs one URL:

```
https://<your-service>.onrender.com/postprocess/pose?confidence_threshold=0.4&keypoint_threshold=0.35&min_visible_keypoints=5
```

A single URL can serve every model if you use the routing endpoint, which
decides from `model_id` or the payload shape:

```
https://<your-service>.onrender.com/postprocess
```

Known ids route automatically:

| Model id | Task |
|---|---|
| `discovered-base-detectron2-maskrcnn-serve` | segmentation |
| `discovered-vc-yolov8-seg` | segmentation |
| `discovered-base-yolov8n-pose-serve` | pose |
| `discovered-base-yolov8n-pose-v2-serve` | pose |
| `discovered-yolo11-object-detection` | detection → box NMS path |
| `discovered-base-detr-resnet-serve` | detection → box NMS path |

Unlisted ids fall back to keyword matching (`*seg*`/`*mask*` → segmentation,
`*pose*`/`*keypoint*` → pose).

**On failure the node falls back to its built-in filter**, so this service is
deliberately loud: it returns 4xx on payloads it cannot interpret rather than
returning an empty frame that looks like a valid "nothing detected" result.

---

## Local development

```bash
python -m venv .venv
.venv/Scripts/activate           # Windows;  source .venv/bin/activate on Unix
pip install -r requirements-dev.txt

uvicorn app.main:app --reload    # http://127.0.0.1:8000/docs
```

Try it:

```bash
curl -X POST http://127.0.0.1:8000/postprocess/segmentation \
  -H "Content-Type: application/json" \
  -d @examples/segmentation_request.json

curl -X POST http://127.0.0.1:8000/postprocess/pose \
  -H "Content-Type: application/json" \
  -d @examples/pose_request.json
```

Run the tests (95 of them, no network or model server needed):

```bash
pytest
```

Smoke-test a running instance, local or deployed:

```bash
python scripts/smoke_test.py                        # local
python scripts/smoke_test.py https://xxx.onrender.com --api-key SECRET
```

---

## Deploying to Render

1. Push this directory to a Git repository.
2. In Render: **New → Blueprint**, point it at the repo. [`render.yaml`](render.yaml)
   defines the service — Python runtime, health check on `/health`, and the
   default thresholds as environment variables.
   *(Or **New → Web Service** manually: build `pip install -r requirements.txt`,
   start `uvicorn app.main:app --host 0.0.0.0 --port $PORT`.)*
3. Optionally set `API_KEYS` in the dashboard to require authentication.
4. Verify: `curl https://<your-service>.onrender.com/health`
5. Paste the endpoint URL into the pipeline node's External REST Connector field.

A [`Dockerfile`](Dockerfile) is included if you prefer a Docker service.

### Notes on the free plan

Free instances sleep after 15 minutes idle and take ~30 s to wake, which shows
up as one slow frame after a quiet period. For a live video pipeline use a paid
instance; the workload is CPU-light but latency-sensitive.

### Configuration

All optional — see [`.env.example`](.env.example).

| Variable | Default | Purpose |
|---|---|---|
| `API_KEYS` | *(empty)* | Comma-separated keys. Empty = open access |
| `API_KEY_HEADER` | `X-API-Key` | Header to read the key from (Bearer also accepted) |
| `CORS_ORIGINS` | `*` | Comma-separated allowed origins |
| `LOG_LEVEL` | `INFO` | Standard Python levels |
| `DEFAULT_CONFIDENCE_THRESHOLD` | `0.30` | Starting default; still overridable per request |
| `DEFAULT_IOU_THRESHOLD` | `0.70` | " |
| `DEFAULT_KEYPOINT_THRESHOLD` | `0.30` | " |
| `DEFAULT_OKS_THRESHOLD` | `0.70` | " |
| `MASK_MAX_SIDE` | `1024` | Raster canvas cap |
| `MAX_INSTANCES_IN` | `2000` | Reject payloads larger than this (413) |
| `MAX_BATCH_ITEMS` | `64` | Batch size cap (413) |
| `DEFAULT_IMAGE_WIDTH` / `_HEIGHT` | `1920` / `1080` | Used only when a payload omits the image size |

---

## Layout

```
app/
  main.py              FastAPI app, request-id middleware, error handling
  config.py            Environment-sourced settings
  schemas.py           Parameter models, query/body merging
  security.py          Optional API-key auth
  normalize.py         Tolerant payload -> Instance normalisation
  models_registry.py   model_id -> task routing
  core/
    geometry.py        Box conversions, IoU, greedy NMS
    masks.py           RLE codecs, polygon fill, contour tracing, raster ops
    keypoints.py       Layouts, OKS, joint angles, smoothing
  pipelines/
    segmentation.py    The segmentation pipeline
    pose.py            The pose pipeline
  routers/
    health.py          Probes and service description
    postprocess.py     The postprocessing endpoints
tests/                 95 tests: 35 unit, 60 end-to-end
examples/              Worked request/response pairs
scripts/smoke_test.py  Stdlib-only checker for a running instance
```

## Design notes

**Pure numpy, no OpenCV / torch / pycocotools.** Everything raster — RLE codecs,
scanline polygon fill, Moore-neighbour contour tracing, run-based connected
components, Douglas-Peucker simplification — is implemented here in ~450 lines
of [`app/core/masks.py`](app/core/masks.py).
The image stays under 100 MB and cold starts in about a second, which matters on
a small instance. Pillow is used only if present, and only for base64 PNG masks.

**Bounded raster canvas.** Masks are rasterised with the longest side capped at
`mask_max_side` (default 1024). Thirty full-resolution 4K bool masks would be
~250 MB; capped, they are ~15 MB. Polygon output is scaled back to
full-resolution image coordinates, so the cap is invisible to the consumer and
costs only sub-pixel boundary precision.

**Failures are loud.** An uninterpretable payload is a 422 and an oversized one
is a 413, because the calling node treats a failed connector call as "use the
built-in filter" — which is a much better outcome than a 200 carrying an empty
instance list.

**One bad instance does not sink a frame.** A mask that fails to decode
downgrades that instance to box-only and adds a warning; a failing item in a
batch is reported in its own slot while the rest of the batch completes.
