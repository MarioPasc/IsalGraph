"""F-1 re-verification: does `isalgraph_pruned` survive as the reference arm?

F-1 was frozen on a probe that encoded ONE GRAPH PER SUBPROCESS, so every
measurement was a cold first call and paid engine warm-up -- inflating one
COIL-DEL graph from a true 4.95 ms to a measured 578.95 ms (x117). The probe's
kill counts (canonical 20/45 on the three largest datasets) are what F-1 rests
on, so they have to be re-taken with warm-up amortised.

A first attempt via the competitors registry failed for an unrelated reason:
it refuses `isalgraph_canonical` above n = 12 with `SuiteScopeError` before
attempting an encode, so it measured a scope guard. This calls
`isalgraph.core.backends` directly.

Parent enforces a per-graph wall-clock deadline by killing a streaming worker
when a BEGIN is not followed by its DONE, then restarting past the culprit.
Sequential and single-process by design: the budget is a wall-clock kill and
concurrency would inflate the very quantity under test.
"""

from __future__ import annotations

import json
import subprocess
import sys
import time
from pathlib import Path

import numpy as np

PY = "/home/mpascual/.conda/envs/isalgraph-cpp/bin/python"
WORKER = str(Path(__file__).with_name("f1_worker.py"))
COHORT = Path(
    "/media/mpascual/Sandisk2TB/research/ISAL/completed/isalgraph"
    "/data/source/APPROX_GED/exported_suite2"
)
BUDGET_S = 60.0
LIMIT = 200
SEED = 42


def run_band(npz: Path, encoder: str, idx: list[int]) -> dict[int, float | None]:
    """Encode the given indices, returning seconds per index or None if killed."""
    out: dict[int, float | None] = {}
    pos = 0
    while pos < len(idx):
        lo, hi = idx[pos], idx[-1] + 1
        proc = subprocess.Popen(
            [PY, WORKER, str(npz), encoder, str(lo), str(hi)],
            stdout=subprocess.PIPE,
            text=True,
            bufsize=1,
        )
        current: int | None = None
        deadline = time.monotonic() + BUDGET_S
        killed_at: int | None = None
        assert proc.stdout is not None
        while True:
            # Poll the pipe with a deadline; a BEGIN with no DONE means a kill.
            if time.monotonic() > deadline and current is not None:
                proc.kill()
                killed_at = current
                break
            line = proc.stdout.readline()
            if not line:
                break
            parts = line.split()
            if parts[0] == "BEGIN":
                current = int(parts[1])
                deadline = time.monotonic() + BUDGET_S
            elif parts[0] == "DONE":
                k = int(parts[1])
                out[k] = float(parts[2])
                current = None
                deadline = time.monotonic() + BUDGET_S
        proc.wait()
        if killed_at is None:
            break
        out[killed_at] = None  # censored
        nxt = [i for i in idx if i > killed_at]
        if not nxt:
            break
        pos = idx.index(nxt[0])
    return out


def main() -> None:
    rng = np.random.default_rng(SEED)
    report: dict = {"budget_s": BUDGET_S, "limit": LIMIT, "seed": SEED, "datasets": {}}
    print(
        f"{'dataset':14s} {'encoder':10s} {'ok':>4s} {'killed':>7s} {'rate':>7s} "
        f"{'med ms':>8s} {'max s':>7s}"
    )
    for key in ["protein", "coil_del", "mutagenicity"]:
        npz = COHORT / f"{key}.npz"
        with np.load(npz, allow_pickle=False) as z:
            n_graphs = len(z["n_nodes"])
        idx = sorted(rng.choice(n_graphs, size=min(LIMIT, n_graphs), replace=False).tolist())
        report["datasets"][key] = {}
        for enc in ("canonical", "pruned"):
            res = run_band(npz, enc, idx)
            done = [v for v in res.values() if v is not None]
            killed = sum(1 for v in res.values() if v is None)
            report["datasets"][key][enc] = {
                "attempted": len(idx),
                "ok": len(done),
                "killed": killed,
                "median_s": float(np.median(done)) if done else None,
                "max_s": float(np.max(done)) if done else None,
            }
            print(
                f"{key:14s} {enc:10s} {len(done):4d} {killed:7d} "
                f"{100 * killed / len(idx):6.1f}% "
                f"{1000 * np.median(done) if done else float('nan'):8.2f} "
                f"{np.max(done) if done else float('nan'):7.2f}",
                flush=True,
            )
    dest = sys.argv[1] if len(sys.argv) > 1 else "f1_verify.json"
    Path(dest).write_text(json.dumps(report, indent=2))
    print(f"\n[done] wrote {dest}")
    print("F1_VERIFY_DONE")


if __name__ == "__main__":
    main()
