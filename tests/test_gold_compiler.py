"""Gold Compiler gates (plan 05) — CI-checkable, deterministic.

Covers the acceptance gates from research/plans/BUILD_BRIEF.md phases 0-4 and
the plan-06 gold-loader contract:
  - every corpus table is schema-valid (decision.schema.json)
  - all 18 corpus tables reach `verified` (verify-corpus)
  - injected-bad tables are caught (bad field / bad outcome / bad behavior)
  - synthesize double-run is byte-identical
  - synthesized train.jsonl loads in the unchanged harness and gold sums to 1
  - probe referee is read-only and reports measured vs predicted
  - history_to_gold: reproducible, harness-loadable, time-hold-out, refusal
"""
from __future__ import annotations

import hashlib
import json
import subprocess
import sys
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from research.compiler import corpus as _corpus  # noqa: E402
from research.compiler import core  # noqa: E402
from research.compiler import anchors as _anchors  # noqa: E402

SCHEMA = core.load_schema()


def test_all_corpus_tables_schema_valid():
    for t in _corpus.ALL:
        errs = core.validate_table(t, SCHEMA)
        assert not errs, f"{t['skill']} {t['decision_id']}: {errs}"


def test_corpus_sizes():
    assert len(_corpus.SHORTLIST) == 10
    assert len(_corpus.ALL) >= 16  # 10 shortlist + held-out


# ---------------------------------------------------------------- phase 2 ---

def test_verify_corpus_all_verified():
    for t in _corpus.ALL:
        a = _anchors.anchors_for(t["decision_id"])
        rep = core.verify(t, anchors=a)
        assert rep["status"] == "verified", (
            f"{t['decision_id']} not verified: {rep['per_rule']}")


def test_inject_bad_field_caught():
    import copy
    t = copy.deepcopy(next(x for x in _corpus.ALL
                           if x["decision_id"] == "is-service-working"))
    t["rules"][0]["condition"] = {"all": [
        {"field": "front_port_up", "operator": "eq", "value": True},
        {"field": "bogus_field_xyz", "operator": "eq", "value": 1}]}
    rep = core.verify(t, anchors=_anchors.anchors_for("is-service-working"))
    assert rep["status"] != "verified"


def test_inject_bad_outcome_caught():
    import copy
    t = copy.deepcopy(next(x for x in _corpus.ALL
                           if x["decision_id"] == "is-service-working"))
    t["rules"][1]["outcome"] = "not_a_criterion"
    rep = core.verify(t, anchors=_anchors.anchors_for("is-service-working"))
    assert rep["status"] != "verified"


def test_inject_bad_behavior_caught():
    import copy
    t = copy.deepcopy(next(x for x in _corpus.ALL
                           if x["decision_id"] == "root-cause-before-fix"))
    t["rules"][0]["outcome"] = "false"  # r1 should fire 'true' when known
    rep = core.verify(t, anchors=_anchors.anchors_for("root-cause-before-fix"))
    assert rep["status"] != "verified"
    assert any(r["verdict"] == "contradicted" for r in rep["per_rule"].values())


# ---------------------------------------------------------------- phase 3 ---

def test_synthesize_double_run_byte_identical(tmp_path):
    d1, d2 = tmp_path / "r1", tmp_path / "r2"
    subprocess.run([sys.executable, "scripts/gold_compiler.py", "pipeline", "corpus",
                    "--out", str(d1)], cwd=ROOT, check=True, capture_output=True)
    subprocess.run([sys.executable, "scripts/gold_compiler.py", "pipeline", "corpus",
                    "--out", str(d2)], cwd=ROOT, check=True, capture_output=True)
    assert (d1 / "train.jsonl").read_bytes() == (d2 / "train.jsonl").read_bytes()


def test_synthesized_rows_load_in_harness(tmp_path):
    from train.train import load_rows, group_splits, gold_vector
    d = tmp_path / "r"
    subprocess.run([sys.executable, "scripts/gold_compiler.py", "pipeline", "corpus",
                    "--out", str(d)], cwd=ROOT, check=True, capture_output=True)
    rows = load_rows(d / "train.jsonl")
    splits = group_splits(rows)
    assert all(k in splits for k in ("train", "dev", "calibration", "test", "ood"))
    for r in rows:
        for qid, q in r["questions"].items():
            gv = gold_vector(q["type"], q.get("criteria"), r["gold_probs"][qid])
            assert abs(sum(gv) - 1.0) < 1e-6


def test_no_family_leak(tmp_path):
    from collections import defaultdict
    d = tmp_path / "r"
    subprocess.run([sys.executable, "scripts/gold_compiler.py", "pipeline", "corpus",
                    "--out", str(d)], cwd=ROOT, check=True, capture_output=True)
    rows = [json.loads(l) for l in (d / "train.jsonl").read_text().splitlines()]
    fam = defaultdict(set)
    for r in rows:
        fam[r["metadata"]["source_group_id"]].add(r["split"])
    assert all(len(s) == 1 for s in fam.values())


# ---------------------------------------------------------------- phase 4 ---

def test_probe_referee_report(tmp_path):
    d = tmp_path / "probe"
    subprocess.run([sys.executable, "scripts/gold_compiler.py", "probe", "corpus",
                    "--out", str(d)], cwd=ROOT, check=True, capture_output=True)
    reports = [json.loads(l) for l in (d / "probe_report.jsonl").read_text().splitlines()]
    assert reports
    # all reports either measure cleanly or declare non-observable
    for r in reports:
        assert r["discrepancy"] in (True, False, None)


# ---------------------------------------------------------------- plan 06 ---

def test_history_to_gold_reproducible_and_loads(tmp_path):
    csv_path = ROOT / "research" / "data" / "business_history.csv"
    if not csv_path.exists():
        pytest.skip("business_history.csv fixture not generated")
    o1, o2 = tmp_path / "b1.jsonl", tmp_path / "b2.jsonl"
    base = ["scripts/history_to_gold.py", "--input", str(csv_path),
            "--state", "segment,campaign,dealsize", "--decision-col", "decision",
            "--decision-id", "renew", "--candidate-col", "candidate",
            "--outcome-col", "outcome", "--time", "quarter"]
    subprocess.run(base + ["--out", str(o1)], cwd=ROOT, check=True, capture_output=True)
    subprocess.run(base + ["--out", str(o2)], cwd=ROOT, check=True, capture_output=True)
    assert o1.read_bytes() == o2.read_bytes()
    from train.train import load_rows, group_splits, gold_vector
    rows = load_rows(o1)
    assert len(rows) > 0 and "test" in group_splits(rows)
    for r in rows:
        for qid, q in r["questions"].items():
            assert abs(sum(gold_vector(q["type"], q.get("criteria"),
                                       r["gold_probs"][qid])) - 1.0) < 1e-6


def test_history_to_gold_refusal(tmp_path):
    csv_path = ROOT / "research" / "data" / "business_history.csv"
    if not csv_path.exists():
        pytest.skip("fixture missing")
    o = tmp_path / "r.jsonl"
    subprocess.run([sys.executable, "scripts/history_to_gold.py",
                    "--input", str(csv_path), "--state", "segment,campaign,dealsize",
                    "--decision-col", "decision", "--decision-id", "renew",
                    "--candidate-col", "candidate", "--outcome-col", "outcome",
                    "--min-count", "100000", "--out", str(o)],
                   cwd=ROOT, check=True, capture_output=True)
    assert (o.read_text(encoding="utf-8").strip() == "")  # sub-min-count refused