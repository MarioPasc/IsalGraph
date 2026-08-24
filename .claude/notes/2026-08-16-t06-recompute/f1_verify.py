"""F-1 re-verification: is `canonical_string` computable on Suite-2 graphs?

F-1 was frozen on a probe that encoded ONE GRAPH PER SUBPROCESS, so every
measurement was a cold first call and paid engine warm-up -- inflating one
COIL-DEL graph from a true 4.95 ms to a measured 578.95 ms (x117). The probe's
kill counts (canonical 20/45 on the three largest datasets) are what F-1 rests
on, so they have to be re-taken with warm-up amortised.

A first attempt via the competitors registry failed for an unrelated reason: it
refuses `isalgraph_canonical` above n = 12 with `SuiteScopeError` before
attempting an encode, so it measured a scope guard (design note S11.1). This
calls `isalgraph.core.backends` directly.

Parent enforces a per-graph wall-clock deadline by killing a streaming worker
when a BEGIN is not followed by its DONE, then restarting past the culprit.
Sequential and single-process by design: the budget is a wall-clock kill and
concurrency would inflate the very quantity under test.

SCOPE OF THE ANSWER. This measures fact 1 of design note S11.1 only -- whether
the encoder is fast enough. Fact 2, whether the packaged backend would emit
Suite-2 columns, is settled separately: `SUITE1_ONLY` is a frozen T-04 policy,
not a performance outcome. A fast result here therefore does NOT reopen F-1; it
goes to the PI.

BUDGET CAVEAT. The kill budget here is 60 s, while D14's frozen encoding budget
is 300 s. A graph killed at 60 s might survive at 300 s, so a kill count here is
an UPPER bound on the true 300 s kill count. `max_ok_s` is reported precisely so
that gap can be judged: if no survivor comes close to 60 s the cut is clean and
raising the budget 5x would change little.

TWO DEFECTS CORRECTED 2026-08-23 by [T06-subagent]; the first launch produced a
zero-byte log and no result because of them.

1. RANGE. The parent passed the worker `idx[pos] .. idx[-1] + 1`, so the worker
   encoded every graph in the CONTIGUOUS SPAN of the sample, not the sample:
   2.8x the intended work on protein, 19.3x on coil_del, 19.7x on mutagenicity.
   The worker now takes an explicit index file.
2. DEADLINE NEVER FIRED. `proc.stdout.readline()` blocks, so the loop only
   re-checked `time.monotonic() > deadline` AFTER a line arrived. Once a graph
   hung past the budget no line ever arrived, the check was never reached and
   the parent blocked indefinitely instead of killing. Reading now happens on a
   separate thread feeding a queue the parent polls with a timeout.
"""

from __future__ import annotations

import json
import queue
import subprocess
import sys
import threading
import time
from pathlib import Path
from typing import IO

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
DATASETS = ["protein", "coil_del", "mutagenicity"]


def _pump(stream: IO[str], sink: queue.Queue[str | None]) -> None:
    """Feed every line of ``stream`` into ``sink``, then a ``None`` sentinel at EOF."""
    for line in stream:
        sink.put(line)
    sink.put(None)


def run_band(npz: Path, encoder: str, idx: list[int], idx_file: Path) -> dict[int, float | None]:
    """Encode the sampled indices, returning seconds per index or None if killed.

    Parameters
    ----------
    npz
        Cohort CSR archive for one dataset.
    encoder
        ``"canonical"`` or ``"pruned"``.
    idx
        The sampled graph indices, ascending.
    idx_file
        File holding ``idx``, one per line, that the worker reads.

    Returns
    -------
    dict
        Maps each sampled index to its encode time in seconds, or ``None`` when
        the per-graph wall-clock budget killed it.
    """
    out: dict[int, float | None] = {}
    pos = 0
    while pos < len(idx):
        proc = subprocess.Popen(
            [PY, WORKER, str(npz), encoder, str(idx_file), str(pos)],
            stdout=subprocess.PIPE,
            text=True,
            bufsize=1,
        )
        assert proc.stdout is not None
        sink: queue.Queue[str | None] = queue.Queue()
        reader = threading.Thread(target=_pump, args=(proc.stdout, sink), daemon=True)
        reader.start()

        current: int | None = None
        killed_at: int | None = None
        deadline = time.monotonic() + BUDGET_S
        while True:
            remaining = deadline - time.monotonic()
            if remaining <= 0:
                if current is not None:
                    proc.kill()
                    killed_at = current
                    break
                deadline = time.monotonic() + BUDGET_S
                continue
            try:
                line = sink.get(timeout=min(remaining, 1.0))
            except queue.Empty:
                continue
            if line is None:  # worker reached EOF: band finished cleanly
                break
            parts = line.split()
            if not parts:
                continue
            if parts[0] == "BEGIN":
                current = int(parts[1])
                deadline = time.monotonic() + BUDGET_S
            elif parts[0] == "DONE":
                out[int(parts[1])] = float(parts[2])
                current = None
                deadline = time.monotonic() + BUDGET_S
        proc.kill()
        proc.wait()
        if killed_at is None:
            break
        out[killed_at] = None  # censored
        pos = idx.index(killed_at) + 1
    return out


def provenance() -> dict[str, str]:
    """Record the engine build and the checkout whose ``src/`` was actually imported."""
    import isalgraph

    src = subprocess.run(
        ["git", "-C", "/home/mpascual/research/code/IsalGraph", "rev-parse", "HEAD"],
        capture_output=True,
        text=True,
        check=False,
    ).stdout.strip()
    return {
        "engine": isalgraph.engine(),
        "isalgraph_build_hash": str(isalgraph.build_info()["build_hash"]),
        "src_commit": src,
        "isalgraph_file": isalgraph.__file__,
    }


def main() -> None:
    dest = Path(sys.argv[1] if len(sys.argv) > 1 else "f1_verify.json")
    prov = provenance()
    if prov["engine"] != "cpp":
        raise SystemExit(f"refusing to measure on engine={prov['engine']!r}; need 'cpp'")

    rng = np.random.default_rng(SEED)
    report: dict = {
        "budget_s": BUDGET_S,
        "limit": LIMIT,
        "seed": SEED,
        "provenance": prov,
        "note": "60 s kill budget; D14's frozen encoding budget is 300 s, so a kill "
        "count here is an upper bound on the 300 s kill count",
        "datasets": {},
    }
    print(json.dumps(prov, indent=2), flush=True)
    print(
        f"\n{'dataset':14s} {'encoder':10s} {'att':>4s} {'ok':>4s} {'kill':>5s} "
        f"{'rate':>7s} {'med ms':>9s} {'max ok s':>9s} {'n ok max':>9s} {'n kill min':>11s}",
        flush=True,
    )
    for key in DATASETS:
        npz = COHORT / f"{key}.npz"
        with np.load(npz, allow_pickle=False) as z:
            n_nodes = z["n_nodes"]
        n_graphs = len(n_nodes)
        idx = sorted(rng.choice(n_graphs, size=min(LIMIT, n_graphs), replace=False).tolist())
        idx_file = dest.with_name(f"f1_idx_{key}.txt")
        idx_file.write_text("\n".join(str(i) for i in idx))
        report["datasets"][key] = {"n_graphs": int(n_graphs), "sampled": len(idx)}

        for enc in ("canonical", "pruned"):
            t_cell = time.monotonic()
            res = run_band(npz, enc, idx, idx_file)
            # The sample is the unit of account: a worker that emitted lines for
            # anything outside `idx` would be the range defect returning.
            stray = sorted(set(res) - set(idx))
            done = {k: v for k, v in res.items() if v is not None}
            killed = sorted(k for k, v in res.items() if v is None)
            missing = sorted(set(idx) - set(res))
            n_ok = [int(n_nodes[k]) for k in done]
            n_kill = [int(n_nodes[k]) for k in killed]
            cell = {
                "attempted": len(idx),
                "ok": len(done),
                "killed": len(killed),
                "missing": len(missing),
                "stray": len(stray),
                "kill_rate": len(killed) / len(idx),
                "median_s": float(np.median(list(done.values()))) if done else None,
                "max_ok_s": float(np.max(list(done.values()))) if done else None,
                "n_ok_max": max(n_ok) if n_ok else None,
                "n_kill_min": min(n_kill) if n_kill else None,
                "killed_indices": killed,
                "cell_wall_s": round(time.monotonic() - t_cell, 1),
            }
            report["datasets"][key][enc] = cell
            print(
                f"{key:14s} {enc:10s} {len(idx):4d} {len(done):4d} {len(killed):5d} "
                f"{100 * cell['kill_rate']:6.1f}% "
                f"{1000 * cell['median_s'] if done else float('nan'):9.2f} "
                f"{cell['max_ok_s'] if done else float('nan'):9.3f} "
                f"{str(cell['n_ok_max']):>9s} {str(cell['n_kill_min']):>11s}",
                flush=True,
            )
            # Write after every cell so a kill mid-run still leaves a usable record.
            dest.write_text(json.dumps(report, indent=2))

    dest.write_text(json.dumps(report, indent=2))
    print(f"\n[done] wrote {dest}", flush=True)
    print("F1_VERIFY_DONE", flush=True)


if __name__ == "__main__":
    main()
