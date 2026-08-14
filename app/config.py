"""Runtime configuration, sourced from environment variables.

Every default here can still be overridden per-request; these only set the
starting point so an operator can retune a deployment without touching the
pipeline's node config.
"""

from __future__ import annotations

import os
from functools import lru_cache


def _env_float(name: str, default: float) -> float:
    try:
        return float(os.environ[name])
    except (KeyError, ValueError):
        return default


def _env_int(name: str, default: int) -> int:
    try:
        return int(os.environ[name])
    except (KeyError, ValueError):
        return default


def _env_bool(name: str, default: bool) -> bool:
    raw = os.environ.get(name)
    if raw is None:
        return default
    return raw.strip().lower() in {"1", "true", "yes", "on"}


def _env_list(name: str) -> list[str]:
    raw = os.environ.get(name, "")
    return [item.strip() for item in raw.split(",") if item.strip()]


class Settings:
    """Process-wide settings. Instantiated once via :func:`get_settings`."""

    def __init__(self) -> None:
        self.app_name = os.environ.get("APP_NAME", "postprocess-api")
        self.version = os.environ.get("APP_VERSION", "1.0.0")
        self.log_level = os.environ.get("LOG_LEVEL", "INFO").upper()

        # Auth: when unset the service is open (fine behind a private network).
        self.api_keys = set(_env_list("API_KEYS"))
        self.api_key_header = os.environ.get("API_KEY_HEADER", "X-API-Key")

        self.cors_origins = _env_list("CORS_ORIGINS") or ["*"]

        # Defaults mirroring the pipeline node's built-in filter.
        self.confidence_threshold = _env_float("DEFAULT_CONFIDENCE_THRESHOLD", 0.30)
        self.iou_threshold = _env_float("DEFAULT_IOU_THRESHOLD", 0.70)
        self.keypoint_threshold = _env_float("DEFAULT_KEYPOINT_THRESHOLD", 0.30)
        self.oks_threshold = _env_float("DEFAULT_OKS_THRESHOLD", 0.70)

        self.max_detections = _env_int("DEFAULT_MAX_DETECTIONS", 300)
        self.mask_max_side = _env_int("MASK_MAX_SIDE", 1024)
        self.max_instances_in = _env_int("MAX_INSTANCES_IN", 2000)
        self.max_batch_items = _env_int("MAX_BATCH_ITEMS", 64)

        self.default_image_width = _env_int("DEFAULT_IMAGE_WIDTH", 1920)
        self.default_image_height = _env_int("DEFAULT_IMAGE_HEIGHT", 1080)

        self.echo_request_on_error = _env_bool("ECHO_REQUEST_ON_ERROR", False)
        self.request_log_bodies = _env_bool("REQUEST_LOG_BODIES", False)


@lru_cache(maxsize=1)
def get_settings() -> Settings:
    return Settings()
