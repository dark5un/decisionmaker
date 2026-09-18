#!/usr/bin/env python3
"""Cross-language golden parity: for every valid request fixture under
tests/fixtures/, the service's OWN curl answer must equal the Go client's and
the Rust client's answer (identical per-qid distributions + rejections).

Canonicalization invariant: type-safe SDKs re-encode structured state/questions
through their native map encoders, which sort JSON object keys alphabetically
(Go `encoding/json` of a map; Rust `serde_json` BTreeMap-backed Value*). A raw
curl sends the fixture body byte-for-byte, so if a fixture's object keys aren't
already sorted, curl vs SDK would send different leaf text to the engine and get
different (if tiny) distributions. To make the comparison exact, the harness
canonicalizes each fixture (recursively sorted keys, arrays order-preserved)
into a temp file and feeds the SAME canonical bytes to curl, Go, and Rust. This
is THE documented wire invariant: structured object keys travel in SDK-canonical
(sorted) order. A raw fixture with unsorted keys is still round-tripped, but key
order is normalized by the client encodersbefore transmission.
"""
from __future__ import annotations

import argparse, json, os, subprocess, sys, tempfile
from pathlib import Path

REPO = Path(__file__).resolve().parents[1]
FIXDIR = REPO / "tests" / "fixtures"


def run(cmd: list[str], **kw) -> subprocess.CompletedProcess:
    return subprocess.run(cmd, capture_output=True, text=True, check=True, **kw)


def build_clis(tmp: Path) -> tuple[str, str]:
    """Build the Go + Rust example CLIs once; return (go_bin, rust_bin)."""
    go_bin = tmp / "decisionmaker_go"
    go_env = dict(os.environ)
    go_env["GOCACHE"] = str(tmp / "go-cache")
    go_env["CGO_ENABLED"] = "0"
    run(
        ["go", "build", "-o", str(go_bin), "./cmd/cli"],
        cwd=REPO / "sdk" / "go", env=go_env,
    )
    rust_env = dict(os.environ)
    rust_env["CARGO_TARGET_DIR"] = str(tmp / "rust-target")
    run(
        ["cargo", "build", "--example", "cli"],
        cwd=REPO / "sdk" / "rust", env=rust_env,
    )
    rust_bin = tmp / "rust-target" / "debug" / "examples" / "cli"
    return str(go_bin), str(rust_bin)


def normalize(resp: dict) -> dict:
    """Keep only the distribution contract per qid (drops usage; model is a
    constant for this server). Rounds to 1e-9 so cross-language float
    formatting never flakes."""
    answers = resp.get("answers") or {}
    out = {}
    for qid, a in sorted(answers.items()):
        keep = {"type": a.get("type")}
        for k in ("boolean", "choice", "score"):
            if k in a:
                keep[k] = a[k]
        if "probabilities" in a:
            keep["probabilities"] = {
                k: round(float(v), 9)
                for k, v in sorted(a["probabilities"].items())
            }
        if "confidence" in a:
            keep["confidence"] = round(float(a["confidence"]), 9)
        out[qid] = keep
    return {"answers": out}


def main() -> int:
    import argparse
    ap = argparse.ArgumentParser()
    ap.add_argument("--base", default="http://127.0.0.1:8090")
    args = ap.parse_args()
    base = args.base.rstrip("/")

    health = subprocess.run(
        ["curl", "-s", "-m", "5", "-o", "/dev/null", "-w", "%{http_code}",
         base + "/health"],
        capture_output=True, text=True).stdout
    if health != "200":
        print("ERROR: service not ready at", base, "(run make serve-up)")
        return 1

    fixtures = sorted(FIXDIR.glob("*.request.json"))
    if not fixtures:
        print("ERROR: no request fixtures under", FIXDIR)
        return 1

    with tempfile.TemporaryDirectory(prefix="decisionmaker-parity-") as tmpd:
        tmp = Path(tmpd)
        go_bin, rust_bin = build_clis(tmp)
        failed = 0
        checked = 0
        canon_dir = tmp / "canonical"
        canon_dir.mkdir(exist_ok=True)
        for fix in fixtures:
            print(f"  [{fix.name}]:", end=" ", flush=True)
            # Canonicalize: recursively sorted keys -> the SDK-canonical wire
            # form both Go and Rust emit. curl must send the SAME bytes.
            canon = canon_dir / fix.name
            canon.write_text(json.dumps(json.loads(fix.read_text()), sort_keys=True))

            curl = run(["curl", "-s", "-X", "POST", base + "/v1/decisionmaker",
                        "-H", "Content-Type: application/json",
                        "-d", "@" + str(canon)]).stdout
            go_out = run([go_bin, base, str(canon)]).stdout
            rust_out = run([rust_bin, base, str(canon)]).stdout
            try:
                a = normalize(json.loads(curl))
                b = normalize(json.loads(go_out))
                c = normalize(json.loads(rust_out))
            except json.JSONDecodeError as e:
                print("FAIL (bad JSON: %s)" % e)
                failed += 1
                continue
            checked += 1
            if a != b or a != c or b != c:
                print("MISMATCH")
                for name, x in (("curl", a), ("go", b), ("rust", c)):
                    print("    ", name, json.dumps(x, sort_keys=True))
                failed += 1
                continue
            # sanity: sums-to-1 on every returned distribution
            sums_ok = True
            for q, an in a["answers"].items():
                if "probabilities" in an:
                    if abs(sum(an["probabilities"].values()) - 1) > 1e-6:
                        sums_ok = False
                        print("  NOT SUMMING for", q, an["probabilities"])
            print("OK" if sums_ok else "SUM-FAIL")
            if not sums_ok:
                failed += 1

    if failed:
        print(f"\nPARITY: {failed} fixture(s) failed")
        return 1
    print(f"\nPARITY: {checked}/{len(fixtures)} fixtures identical across curl/Go/Rust, all sums-to-1")
    return 0


if __name__ == "__main__":
    sys.exit(main())
