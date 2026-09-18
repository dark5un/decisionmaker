"""HTTP-layer unit tests (Phase 4) — run against a fake engine, no model/GPU.

These verify the service contract mapping (health readiness, typed errors) with
a stub, so the HTTP logic is testable before the model/container exists.
"""
import json
from types import SimpleNamespace

from fastapi.testclient import TestClient

from server.src.engine import EngineError
from server.src.app import create_app


class ReadyEngine:
    config = SimpleNamespace(model="Qwen/Qwen3-0.6B-Instruct")
    backbone = "loaded"  # truthy -> /health ready

    def predict(self, payload):
        return {
            "model": self.config.model,
            "answers": {qid: {"type": q["type"], "boolean": 0.9} if q["type"] == "boolean"
                        else {"type": q["type"], "choice": "x", "probabilities": {"x": 1.0},
                              "confidence": 1.0}
                        for qid, q in payload["questions"].items()},
            "usage": {"input_tokens": 10, "output_tokens": 0},
        }


class NotLoadedEngine:
    config = SimpleNamespace(model="x")
    backbone = None  # falsy -> /health not ready

    def predict(self, payload):
        return ReadyEngine().predict(payload)


class RaiseValidationEngine:
    config = SimpleNamespace(model="x")
    backbone = "loaded"

    def predict(self, payload):
        raise EngineError("validation", "contract violation", {"field": "questions"})


class RaiseInternalEngine:
    config = SimpleNamespace(model="x")
    backbone = "loaded"

    def predict(self, payload):
        raise EngineError("internal", "non-finite logits")


def client(engine):
    return TestClient(create_app(engine))


def test_health_ready():
    with client(ReadyEngine()) as c:
        r = c.get("/health")
        assert r.status_code == 200
        body = r.json()
        assert body["status"] == "ready"
        assert body["model"] == "Qwen/Qwen3-0.6B-Instruct"


def test_health_not_loaded():
    with client(NotLoadedEngine()) as c:
        r = c.get("/health")
        assert r.status_code == 503
        assert r.json()["error"]["type"] == "not_ready"


def test_evaluate_valid_request():
    with client(ReadyEngine()) as c:
        r = c.post("/v1/decisionmaker", content=json.dumps({
            "state": "Help!", "questions": {"u": {"type": "boolean", "instructions": "urgent?"}}
        }))
        assert r.status_code == 200
        body = r.json()
        assert "answers" in body and "usage" in body
        assert body["answers"]["u"]["type"] == "boolean"


def test_evaluate_malformed_json_400():
    with client(ReadyEngine()) as c:
        r = c.post("/v1/decisionmaker", content=b"not-json{")
        assert r.status_code == 400
        assert r.json()["error"]["type"] == "validation"


def test_evaluate_rejects_non_object_422():
    with client(ReadyEngine()) as c:
        r = c.post("/v1/decisionmaker", content=json.dumps([1, 2, 3]))
        assert r.status_code == 422
        assert r.json()["error"]["type"] == "validation"


def test_evaluate_validation_error_422_details():
    with client(RaiseValidationEngine()) as c:
        r = c.post("/v1/decisionmaker", content=json.dumps({"state": "x", "questions": {}}))
        assert r.status_code == 422
        err = r.json()["error"]
        assert err["type"] == "validation"
        assert err["details"] == {"field": "questions"}


def test_evaluate_internal_error_500_typed():
    with client(RaiseInternalEngine()) as c:
        r = c.post("/v1/decisionmaker", content=json.dumps({"state": "x", "questions": {"q": {"type": "boolean"}}}))
        assert r.status_code == 500
        assert r.json()["error"]["type"] == "internal"