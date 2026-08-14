"""Application entry point.

Run locally:      uvicorn app.main:app --reload
Run in Render:    uvicorn app.main:app --host 0.0.0.0 --port $PORT
"""

from __future__ import annotations

import logging
import time
import uuid

from fastapi import FastAPI, Request
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse
from pydantic import ValidationError

from .config import get_settings
from .routers import health, postprocess
from .schemas import describe_validation_error

settings = get_settings()

logging.basicConfig(
    level=getattr(logging, settings.log_level, logging.INFO),
    format="%(asctime)s %(levelname)s %(name)s %(message)s",
)
log = logging.getLogger("postprocess")

app = FastAPI(
    title="Vision Postprocessing API",
    version=settings.version,
    description=(
        "Stateless REST postprocessing for segmentation and pose model output.\n\n"
        "Point a pipeline's external postprocessing connector at "
        "`/postprocess/segmentation` or `/postprocess/pose` and POST the raw model "
        "results. Tune behaviour with `params` in the body or query parameters; see "
        "`GET /postprocess/params`."
    ),
    docs_url="/docs",
    redoc_url="/redoc",
    openapi_url="/openapi.json",
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=settings.cors_origins,
    allow_credentials=False,
    allow_methods=["*"],
    allow_headers=["*"],
)


@app.middleware("http")
async def request_context(request: Request, call_next):
    """Attach a request id, time the call, and never let an unhandled error 500 silently."""
    request_id = request.headers.get("X-Request-ID") or uuid.uuid4().hex[:12]
    started = time.perf_counter()

    try:
        response = await call_next(request)
    except Exception:
        elapsed = (time.perf_counter() - started) * 1000
        log.exception("request %s %s failed after %.1fms", request.method, request.url.path, elapsed)
        return JSONResponse(
            status_code=500,
            content={
                "error": "internal_error",
                "detail": "postprocessing failed; the caller should fall back to its built-in filter",
                "request_id": request_id,
            },
            headers={"X-Request-ID": request_id},
        )

    elapsed = (time.perf_counter() - started) * 1000
    response.headers["X-Request-ID"] = request_id
    response.headers["X-Process-Time-Ms"] = f"{elapsed:.2f}"
    if request.url.path.startswith("/postprocess"):
        log.info(
            "%s %s -> %s in %.1fms [%s]",
            request.method,
            request.url.path,
            response.status_code,
            elapsed,
            request_id,
        )
    return response


@app.exception_handler(ValidationError)
async def invalid_params(request: Request, exc: ValidationError) -> JSONResponse:
    """Bad tuning parameters are the caller's mistake, not a server fault."""
    return JSONResponse(
        status_code=422,
        content={"error": "invalid_params", "detail": describe_validation_error(exc)},
    )


app.include_router(health.router)
app.include_router(postprocess.router)
