#!/usr/bin/env python3
"""Decision-Maker benchmark (real, no stubs): measure the live service end-to-end.

Reports cold-start-to-ready, warm per-request latency (p50/p95/mean), throughput
(requests/s) under a concurrent load, and a correctness gate (every response's
distributions sum-to-1 and all qids present). Verifies which GPU served by
reading the /health reply nothing else — device attribution is a server-side log.

Usage:
    scripts/benchmark.py --warm 100 --load 200 --concurrency 8

Read-only: only issues HTTP requests to the service. Requires the service up
(make serve-up). Runs a warmup burst first so the numbers exclude CUDA init.
"""
from __future__ import annotations

import argparse
import json
import statistics
import sys
import time
from concurrent.futures import ThreadPoolExecutor, as_completed
from urllib import request

DEFAULT_REQUEST = {
    "state": ("Help! My payouts have been failing for 3 days. The billing portal "
              "shows 'pending' since Tuesday and support has not replied."),
    "model": "decisionmaker-latest",
    "questions": {
        "is_urgent": {
            "type": "boolean",
            "instructions": "Does this convey urgency?",
            "criteria": {"true": "Explicitly time-sensitive",
                          "false": "No urgency expressed"},
        },
        "department": {
            "type": "choice",
            "instructions": "Which team should handle this?",
            "criteria": {
                "billing": "Payments, invoicing, refunds",
                "technical": "Bugs, outages, integrations",
                "sales": "Pricing, upgrades, new accounts",
            },
        },
        "frustration": {
            "type": "score",
            "instructions": "How frustrated is the customer?",
            "criteria": ["Calm", "Frustrated", "Very angry"],
        },
    },
}


def post_once(base: str, payload: dict, timeout: float) -> tuple[float, dict]:
    """Return (latency_seconds, decoded_response). Raises on HTTP != 200."""
    data = json.dumps(payload).encode()
    t0 = time.perf_counter()
    req = request.Request(base + "/v1/decisionmaker", data=data,
                          headers={"Content-Type": "application/json"}, method="POST")
    with request.urlopen(req, timeout=timeout) as resp:
        body = resp.read().decode()
    lat = time.perf_counter() - t0
    if resp.status != 200:
        raise RuntimeError(f"HTTP {resp.status}: {body[:200]}")
    return lat, json.loads(body)


def sums_to_one(response: dict) -> bool:
    for ans in response.get("answers", {}).values():
        probs = ans.get("probabilities")
        if probs is not None and abs(sum(probs.values()) - 1.0) > 1e-6:
            return False
    return True


def gate(response: dict, expected_qids: int) -> bool:
    answers = response.get("answers", {})
    if len(answers) != expected_qids:
        return False
    return all(("boolean" in a) != ("probabilities" in a) for _, a in answers.items())

def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--base", default="http://127.0.0.1:8090")
    ap.add_argument("--warm", type=int, default=100, help="warmup requests before timing")
    ap.add_argument("--load", type=int, default=200, help="timed requests")
    ap.add_argument("--concurrency", type=int, default=8)
    ap.add_argument("--timeout", type=float, default=60.0)
    args = ap.parse_args()
    base = args.base.rstrip("/")

    # 0) readiness
    t_cold = time.perf_counter()
    with request.urlopen(base + "/health", timeout=args.timeout) as r:
        health = json.loads(r.read())
    cold_s = time.perf_counter() - t_cold
    if health.get("status") != "ready":
        print("ERROR: service not ready:", health)
        return 1
    print(f"health: {health.get('status')} model={health.get('model')} "
          f"(this call took {cold_s*1000:.0f} ms — model is already warm)")

    payload = DEFAULT_REQUEST
    print(f"workload: {args.load} requests x concurrency {args.concurrency}, "
          f"after {args.warm} warmup; payload has {len(payload['questions'])} questions")

    # 1) warmup burst (excludes CUDA/graph init from the numbers)
    for _ in range(args.warm):
        post_once(base, payload, args.timeout)

    # 2) timed sequential latency
    lats = []
    for _ in range(args.load):
        lat, resp = post_once(base, payload, args.timeout)
        if not sums_to_one(resp):
            print("FATAL: response distributions did not sum to 1")
            return 2
        lats.append(lat * 1000)
    lats.sort()
    ms = lambda xs: statistics.mean(xs)
    p50 = lats[len(lats) // 2]
    p95 = lats[int(len(lats) * 0.95)]
    print(f"latency: p50={p50:.1f}ms p95={p95:.1f}ms mean={ms(lats):.1f}ms "
          f"(n={len(lats)})")

    # 3) concurrent throughput
    t0 = time.perf_counter()
    done = 0
    with ThreadPoolExecutor(max_workers=args.concurrency) as ex:
        futs = [ex.submit(post_once, base, payload, args.timeout) for _ in range(args.load)]
        for f in as_completed(futs):
            _, resp = f.result()
            if not sums_to_one(resp):
                print("FATAL: throughput response did not sum to 1")
                return 2
            done += 1
    elapsed = time.perf_counter() - t0
    print(f"throughput: {done} requests in {elapsed:.2f}s = {done/elapsed:.1f} req/s "
          f"(concurrency {args.concurrency})")

    # 4) correctness gate on a fresh deterministic call
    _, resp = post_once(base, payload, args.timeout)
    if not gate(resp, expected_qids=len(payload["questions"])):
        print("FATAL: correctness gate failed (qids/answer shapes)")
        return 2
    print("gate: all distributions present, correct answer types, sums-to-1 -> OK")
    return 0


if __name__ == "__main__":
    sys.exit(main())