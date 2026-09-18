"""HTTP layer for the Decision-Maker engine (Phase 4).

Load-once pattern: the engine is constructed ONCE and injected when the app is
created; the /health endpoint reports readiness (model loaded). Endpoints are
stateless. Error mapping: 400 malformed JSON, 422 contract validation, 503 not
ready, 500 engine malfunction (typed, never a bare 500).
"""
from __future__ import annotations

import json
import logging
from typing import Any

from fastapi import FastAPI, Request
from fastapi.responses import JSONResponse

from .engine import EngineError

logger = logging.getLogger("decisionmaker")


def _error(status: int, etype: str, message: str) -> JSONResponse:
    return JSONResponse({"error": {"type": etype, "message": message}}, status_code=status)


def create_app(engine: Any) -> FastAPI:
    """engine: any object exposing `.predict(payload)->dict`, `.config.model`,
    and a truthy `.backbone` once loaded (DecisionEngine satisfies this)."""
    app = FastAPI(title="Decision-Maker Service", version="0.1.0")

    @app.get("/health")
    async def health() -> JSONResponse:
        ready = bool(engine.backbone is not None)
        if not ready:
            return _error(503, "not_ready", "model not loaded; retry after re-checking /health")
        return JSONResponse({
            "status": "ready",
            "model": engine.config.model,
            "uptime_seconds": 0.0,  # podman healthcheck reports uptime from container metadata
        })

    @app.post("/v1/decisionmaker")
    async def evaluate(request: Request) -> JSONResponse:
        raw = await request.body()
        try:
            payload: Any = json.loads(raw)
        except ValueError:
            return _error(400, "validation", "malformed JSON body")
        if not isinstance(payload, dict):
            return _error(422, "validation", "request body must be a JSON object")

        try:
            result = engine.predict(payload)
        except EngineError as exc:
            status = {"validation": 422, "not_ready": 503, "rate_limit": 429}.get(exc.code, 500)
            body = {"error": exc.asdict()}
            return JSONResponse(body, status_code=status)
        except Exception:
            # Keep typed errors distinct from genuine faults, and ALWAYS surface
            # the traceback (logs feed systemd) so a 500 is debuggable.
            logger.exception("engine failed on request")
            return _error(500, "internal", "unexpected engine failure")
        return JSONResponse(result)

    return app