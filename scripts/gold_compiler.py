#!/usr/bin/env python3
"""Gold Compiler — raw prose / rulebook -> verified -> exact-gold train.jsonl.

The single buildable tool of plan 05 (research/plans/05-gold-compiler-plan.md).
Turns a decision-dense document (a SKILL.md or any rule-governed prose) into
calibrated Decision-Maker training data WITHOUT a teacher LLM as the source of
truth: extract -> verify (constructive: static binding + behavioral oracle over
hand-marked anchors) -> synthesize (compute gold by running the interpreter).

Exit taxonomy (the frozen contract):
  0 = emitted (tables / train.jsonl)
  2 = refused (no_gold — judgment-prose, or no extractor available)
  3 = emitted with human-review flags (some rule incomplete / contradicted,
      or a probe discrepancy) — the output is valid but needs human eyes.

Subcommands (identical to phases):
  extract    <input> [--out dir]      derive candidate rule tables
  verify     <table> [--anchors]      constructive evidence+oracle check
  synthesize <verified-table> [--out] compute exact-gold train.jsonl
  probe      <table>                  Plan-C referee (measured vs predicted)
  --input SKILL.md --out DIR          full pipeline in one invocation

Outputs land in `--out` (default ./compiler-out):
  tables.jsonl      candidate/verified decision tables
  train.jsonl       exact-gold rows (seed.jsonl schema, data_sha256 stamped)
  verify_manifest.jsonl  per-rule verdicts + review flags + regen log
  probe_report.jsonl     probe discrepancies (read-only)

The corpus (research/compiler/corpus.py) holds the 10 shortlist + held-out
tables; anchors (research/compiler/anchors.py) are the only human-in-loop input.
"""
from __future__ import annotations

import argparse
import json
import sys
import hashlib
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from research.compiler import core, corpus  # noqa: E402
from research.compiler import anchors as _anchors  # noqa: E402


def _write(out: Path, name: str, objs) -> Path:
    out.mkdir(parents=True, exist_ok=True)
    p = out / name
    blob = "".join(json.dumps(o, ensure_ascii=False, sort_keys=True) + "\n" for o in objs)
    p.write_text(blob, encoding="utf-8")
    return p


def _read_objs(path: Path):
    return [json.loads(line) for line in path.read_text().splitlines()]


def _load_or_extract(args):
    """Resolve `--input` to a list of candidate tables.

    Accepts (in order): the literal 'corpus' (all bundled tables),
    a `.jsonl` rulebook of tables, or prose text derived from the
    deterministic rulebook extractor.
    """
    inp = args.input
    if inp in ("corpus", "all", "shortlist"):
        skill = args.skill
        return [dict(t) for t in corpus.ALL
                if skill is None or t["skill"] == skill], None
    p = Path(inp)
    if p.is_file() and p.suffix in (".jsonl", ".json"):
        return _read_objs(p), None
    return core.extract_text(inp, skill=args.skill, use_llm=args.use_llm)


# ------------------------------------------------ subcommands --------------

def cmd_extract(args) -> int:
    inp = args.input
    is_prose = not (inp in ("corpus", "all", "shortlist")
                    or Path(inp).is_file() and Path(inp).suffix in (".jsonl", ".json"))
    text = ""
    if is_prose:
        text = Path(inp).read_text() if Path(inp).exists() else inp
        if args.refuse_if_judgment:
            refuse = core.refuse_reason(text)
            if refuse:
                ev = {"reason": refuse, "evidence": [m for m in core._JUDGMENT_MARKERS if m in text.lower()][:8]}
                print(json.dumps(ev, indent=2))
                print(f"REFUSED exit=2 ({refuse}): no pinnable answer -> no_gold")
                return 2
        if args.use_llm:
            tables, why = core.extract_text(text, skill=args.skill, use_llm=True)
            if why and not tables:
                print(json.dumps({"reason": why, "evidence": ["no extractor registered"]}))
                return 2
            src_text = text
        else:
            tables, _ = core.extract_text(text, skill=args.skill)
            src_text = text
    else:
        tables, _ = _load_or_extract(args)
        src_text = args.source_text

    out = Path(args.out)
    _write(out, "tables.jsonl", tables)
    for t in tables:
        print(json.dumps({"skill": t["skill"], "decision_id": t["decision_id"],
                          "verification": t["verification"],
                          "hash": t["canonical_hash"]}))
    print(f"extracted {len(tables)} candidate table(s) -> {out}/tables.jsonl")
    return 0


def cmd_verify(args) -> int:
    tables = _read_objs(Path(args.input))
    out = Path(args.out)
    manifests, verified, flagged = [], [], []
    srctext = Path(args.source_text).read_text() if args.source_text else None
    for t in tables:
        a = _anchors.anchors_for(t["decision_id"])
        rep = core.verify(t, anchors=a, source_text=srctext, max_regens=args.max_regens)
        t["verification"] = rep["status"]
        manifests.append({"decision": t["decision_id"], **rep})
        (verified if rep["status"] == "verified" else flagged).append(t)
        for r in rep["per_rule"].values():
            print(f"  {t['decision_id']} {r['verdict']:>12} anchors={r['anchors_covered']}")
    _write(out, "tables.verified.jsonl", verified + flagged)
    _write(out, "verify_manifest.jsonl", manifests)
    print(f"verified={len(verified)} flagged={len(flagged)} total={len(tables)}")
    for m in manifests:
        for r in m["per_rule"].values():
            if r["verdict"] != "verified":
                print(f"  FLAG {m['decision_id']}: {r['verdict']}")
    return 0 if not flagged else 3


def cmd_synthesize(args) -> int:
    tables = _read_objs(Path(args.input))
    out = Path(args.out)
    rows = []
    for t in tables:
        fid = f"{t['skill']}-{t['decision_id']}"
        split = _synthesize_splitplan().get(t["decision_id"], "train")
        rows.extend(core.make_rows(t, split, fid, args.seed, args.k_phrasings))
    ordered = sorted(rows, key=lambda r: r["id"])
    p = _write(out, "train.jsonl", ordered)
    sha = hashlib.sha256(p.read_bytes()).hexdigest()
    _write(out, "train.meta.json", [{"rows": len(ordered), "data_sha256": f"sha256:{sha}"}])
    from collections import Counter
    print(f"wrote {len(ordered)} rows -> {p}")
    print("splits:", dict(Counter(r["split"] for r in ordered)))
    print(f"data_sha256=sha256:{sha}")
    return 0


def _synthesize_splitplan():
    """Deterministic split assignment (group-separated; booleans to test/ood
    so the ECE gate has boolean coverage). Overridable per decision."""
    d = {
        # shortlist — reuse planA's proven assignment (booleans to test/ood)
        "root-cause-before-fix": "test", "tight-loop-before-theory": "ood",
        "dpms-wake-keys": "ood", "idle-pipeline-running": "test",
        "gpu-uuid-pinning": "dev", "temp-sharpening-catch": "dev",
        "gpu-cdi-presence": "calibration", "rule-of-three": "calibration",
        "is-service-working": "train", "rootless-podman-broken": "train",
        # held-out
        "pairing-source-of-truth": "test", "disconnect-connect-drop": "ood",
        "box-home-permissions": "test", "box-vs-host-path": "ood",
        "commit-fix-vs-oneoff": "dev", "newuidmap-caps-failure": "test",
        "mdns-self-resolution": "ood", "listening-port-vs-backend": "dev",
    }
    return d


def cmd_probe(args) -> int:
    tables, _ = _load_or_extract(args)
    out = Path(args.out)
    reports = []
    for t in tables:
        r = core.probe_decision(t)
        reports.append(r)
        if r["discrepancy"]:
            print(f"  DISCREPANCY {r['decision_id']}: predicted={r['rule_predicted']} measured={r['measured']}")
        else:
            print(f"  {r['decision_id']}: measured==predicted ({r.get('rule_predicted')})")
    _write(out, "probe_report.jsonl", reports)
    n_disc = sum(1 for r in reports if r.get("discrepancy"))
    print(f"probed={len(reports)} discrepancies={n_disc}")
    return 3 if n_disc else 0


def cmd_pipeline(args) -> int:
    """extract -> verify -> synthesize -> probe in one invocation."""
    out = Path(args.out); out.mkdir(parents=True, exist_ok=True)
    tables, why = _load_or_extract(args)
    if why and not tables:
        print(json.dumps({"reason": why})); return 2
    # extract exit-2 refusal for prose judgment input
    if args.input and not Path(args.input).suffix in (".jsonl", ".json") \
       and args.input not in ("corpus", "all", "shortlist"):
        text = Path(args.input).read_text() if Path(args.input).exists() else args.input
        refuse = core.refuse_reason(text)
        if refuse and args.refuse_if_judgment:
            print(json.dumps({"reason": refuse, "evidence": ["judgment markers matched"]}))
            return 2
    # verify -> verified + manifest
    manifests, flagged = [], []
    for t in tables:
        a = _anchors.anchors_for(t["decision_id"])
        rep = core.verify(t, anchors=a)
        t["verification"] = rep["status"]
        manifests.append(rep)
        if rep["status"] != "verified":
            flagged.append(t["decision_id"])
    _write(out, "tables.verified.jsonl", tables)
    _write(out, "verify_manifest.jsonl", manifests)
    # synthesize -> train.jsonl
    rows = []
    for t in tables:
        fid = f"{t['skill']}-{t['decision_id']}"
        split = _synthesize_splitplan().get(t["decision_id"], "train")
        rows.extend(core.make_rows(t, split, fid, args.seed, args.k_phrasings))
    ordered = sorted(rows, key=lambda r: r["id"])
    p = _write(out, "train.jsonl", ordered)
    sha = hashlib.sha256(p.read_bytes()).hexdigest()
    _write(out, "train.meta.json", [{"rows": len(ordered), "data_sha256": f"sha256:{sha}"}])
    from collections import Counter
    print(f"rows={len(ordered)} splits={dict(Counter(r['split'] for r in ordered))}")
    print(f"data_sha256=sha256:{sha}")
    print(f"verified={len(tables)-len(flagged)}/{len(tables)}; flags={flagged}")
    return 0 if not flagged else 3


def build_parser():
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--input", "--in", dest="input", help="input file (or inline text)")
    ap.add_argument("--out", default="compiler-out", help="output directory")
    ap.add_argument("--skill", default=None, help="filter to one skill")
    ap.add_argument("--refuse-if-judgment", action="store_true",
                    help="exit 2 on judgment-prose instead of trying")
    ap.add_argument("--use-llm", action="store_true",
                    help="use the registered LLM extractor (refuses if none)")
    ap.add_argument("--max-regens", type=int, default=3)
    ap.add_argument("--seed", type=int, default=17)
    ap.add_argument("--k-phrasings", type=int, default=None)
    ap.add_argument("--source-text", default=None,
                    help="full source SKILL.md to check verbatim spans against")
    sub = ap.add_subparsers(dest="command")
    for name in ("extract", "verify", "synthesize", "probe", "pipeline"):
        sp = sub.add_parser(name)
        sp.add_argument("input", nargs="?", help="input file or inline text")
        sp.add_argument("--out", default="compiler-out")
        sp.add_argument("--skill", default=None)
        sp.add_argument("--refuse-if-judgment", action="store_true")
        sp.add_argument("--use-llm", action="store_true")
        sp.add_argument("--max-regens", type=int, default=3)
        sp.add_argument("--seed", type=int, default=17)
        sp.add_argument("--k-phrasings", type=int, default=None)
        sp.add_argument("--source-text", default=None)
    return ap


def main(argv=None) -> int:
    args = build_parser().parse_args(argv)
    cmd = args.command
    if cmd == "extract":
        return cmd_extract(args)
    if cmd == "verify":
        return cmd_verify(args)
    if cmd == "synthesize":
        return cmd_synthesize(args)
    if cmd == "probe":
        return cmd_probe(args)
    if cmd == "pipeline" or args.input:
        return cmd_pipeline(args)
    print("missing --input or subcommand"); build_parser().print_help()
    return 2


if __name__ == "__main__":
    sys.exit(main())