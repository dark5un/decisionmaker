#!/usr/bin/env python3
"""Gold Compiler CI (plan 05) — the three non-pytest gate targets.

Run via:  make gold-compiler-lint
          make gold-compiler-verify-corpus
          make gold-compiler-determinism

Each is deterministic and CI-checkable (used by pre-commit too). Exits non-zero
on gate failure.
"""
from __future__ import annotations

import subprocess
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

PY = sys.executable


def sh(*args, **kw):
    return subprocess.run([str(a) if not isinstance(a, Path) else str(a) for a in args],
                          cwd=ROOT, capture_output=True, text=True, **kw)


def lint():
    import py_compile
    from pathlib import Path as _P
    ok = True
    for f in ["scripts/gold_compiler.py", "scripts/history_to_gold.py",
              "research/compiler/corpus.py", "research/compiler/core.py",
              "research/compiler/anchors.py", "research/compiler/renderers.py"]:
        try:
            py_compile.compile(str(ROOT / f), doraise=True)
            print(f"OK  compile {f}")
        except py_compile.PyCompileError as exc:
            ok = False
            print(f"FAIL compile {f}: {exc}")
    # CLI --help must not crash (import + argparse contract lock)
    for f in ["scripts/gold_compiler.py", "scripts/history_to_gold.py"]:
        r = sh(PY, f, "--help")
        if r.returncode != 0:
            ok = False
            print(f"FAIL --help {f}:\n{r.stderr}")
        else:
            print(f"OK  --help {f}")
    return 0 if ok else 1


def verify_corpus():
    from research.compiler import corpus, core
    from research.compiler import anchors
    total = 0
    bad = []
    for t in corpus.ALL:
        total += 1
        rep = core.verify(t, anchors=anchors.anchors_for(t["decision_id"]))
        status = rep["status"]
        if status != "verified":
            bad.append((t["decision_id"], status))
        print(f"  {t['decision_id']:28} {status}")
    if bad:
        print(f"verify-corpus FAIL: {bad}")
        return 1
    print(f"verify-corpus: all {total} corpus tables verified")
    return 0


def determinism(tmp=None):
    import tempfile
    tmp = tmp or Path(tempfile.mkdtemp())
    d1, d2 = tmp / "r1", tmp / "r2"
    meta1, meta2 = [], []
    for d in (d1, d2):
        r = sh(PY, "scripts/gold_compiler.py", "pipeline", "corpus", "--out", str(d))
        if r.returncode != 0:
            print("determinism pipeline failed:\n", r.stdout, r.stderr)
            return 1
    b1, b2 = (d1 / "train.jsonl").read_bytes(), (d2 / "train.jsonl").read_bytes()
    same = b1 == b2
    import hashlib
    print(f"run1 sha256={hashlib.sha256(b1).hexdigest()}")
    print(f"run2 sha256={hashlib.sha256(b2).hexdigest()}")
    print("determinism:", "OK byte-identical" if same else "FAIL differ")
    return 0 if same else 1


if __name__ == "__main__":
    which = sys.argv[1] if len(sys.argv) > 1 else "all"
    fns = {"lint": lint, "verify-corpus": verify_corpus, "determinism": determinism}
    if which == "all":
        rc = 0
        for name in ("lint", "verify-corpus", "determinism"):
            rc |= fns[name]()
        sys.exit(rc)
    sys.exit(fns[which]())