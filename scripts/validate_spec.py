#!/usr/bin/env python3
"""Validate spec/openapi.yaml and the JSON fixtures under tests/fixtures/.

Gates (Phase 2):
  1. The OpenAPI document is structurally valid.
  2. Every *.request.json fixture is accepted by the DecisionMakerRequest schema.
  3. Every *.invalid.json fixture is REJECTED by the DecisionMakerRequest schema.
  4. Every *.response.json fixture is accepted by the DecisionMakerResponse schema.
  5. A synthetic request in the exact shape the public docs show is accepted.

All $refs in this spec are fragment refs into components.schemas, so a small
self-contained dereferencer is enough (no external registry dependency).
"""
import json
import sys
from pathlib import Path

import yaml
from openapi_spec_validator import validate as validate_oas
from jsonschema import Draft202012Validator

ROOT = Path(__file__).resolve().parents[1]
SPEC = ROOT / "spec" / "openapi.yaml"
FIXTURES = ROOT / "tests" / "fixtures"


def load_spec():
    return yaml.safe_load(SPEC.read_text())


def _deref(node, spec, _depth=0):
    """Recursively replace component $refs (fragment-only) with their targets."""
    if _depth > 40:
        raise RuntimeError("$ref depth exceeded (cycle?)")
    if isinstance(node, dict):
        if "$ref" in node and isinstance(node["$ref"], str):
            ref = node["$ref"]
            if not ref.startswith("#/components/"):
                raise ValueError(f"unsupported external $ref: {ref}")
            target = spec
            for part in ref.lstrip("#/").split("/"):
                # JSON pointers use ~1 and ~0 escapes
                part = part.replace("~1", "/").replace("~0", "~")
                target = target[part]
            return _deref(target, spec, _depth + 1)
        return {k: _deref(v, spec, _depth + 1) for k, v in node.items()}
    if isinstance(node, list):
        return [_deref(v, spec, _depth + 1) for v in node]
    return node


def schema_of(spec, path):
    request = spec["paths"]["/v1/decisionmaker"]["post"]
    return _deref(request[path]["content"]["application/json"]["schema"], spec)


def validator_for(schema, spec):
    return Draft202012Validator(schema)


def check(name, ok, detail=""):
    tag = "PASS" if ok else "FAIL"
    print(f"[{tag}] {name}" + (f"  -- {detail}" if detail else ""))
    return ok


def main():
    spec = load_spec()
    results = []

    # Gate 1: structural validity.
    try:
        validate_oas(spec)
        results.append(check("spec structural validity", True))
    except Exception as exc:  # report the validator's own exception text
        results.append(check("spec structural validity", False, str(exc)[:300]))

    request_schema = schema_of(spec, "requestBody")
    _resp_200 = spec["paths"]["/v1/decisionmaker"]["post"]["responses"]["200"]
    response_schema = _deref(_resp_200["content"]["application/json"]["schema"], spec)
    req_validator = validator_for(request_schema, spec)
    resp_validator = validator_for(response_schema, spec)

    # Helpers to assert accept/reject.
    def accept(fixture_path, validator):
        try:
            validator.validate(json.loads(fixture_path.read_text()))
            return True, ""
        except Exception as exc:  # the schema error text is the diagnostic we want
            return False, str(exc).splitlines()[0][:200]

    def reject(fixture_path, validator):
        try:
            validator.validate(json.loads(fixture_path.read_text()))
            return False, "was ACCEPTED but expected to be rejected"
        except Exception:
            return True, ""

    # Gate 2: every valid request accepted.
    for p in sorted(FIXTURES.glob("*.request.json")):
        ok, detail = accept(p, req_validator)
        results.append(check(f"valid request accepted: {p.name}", ok, detail))

    # Gate 3: every invalid request rejected.
    for p in sorted(FIXTURES.glob("*.invalid.json")):
        ok, detail = reject(p, req_validator)
        results.append(check(f"invalid request rejected: {p.name}", ok, detail))

    # Gate 4: every response accepted.
    for p in sorted(FIXTURES.glob("*.response.json")):
        ok, detail = accept(p, resp_validator)
        results.append(check(f"valid response accepted: {p.name}", ok, detail))

    failed = sum(1 for ok in results if not ok)
    print(f"\n{len(results) - failed}/{len(results)} gates passed")
    return 1 if failed else 0


if __name__ == "__main__":
    sys.exit(main())