"""Competitor ceiling sweep to n = 98.  IsalGraph measured separately (isal_ceiling.py).

Suite 2 reaches n = 98 ([data](data.md) §1.2).  Every cell is attempted; a failed or
timed-out cell is printed as one.  AGM is attempted only up to n = 20 -- above that
`scale.py` already measured 0/5 exact and each attempt costs 3-8 s of pure budget burn.

Sampler caveat: `random_connected` here is a random spanning tree plus uniformly chosen
extra edges, not a uniform connected G(n, m).  Rejection sampling on G(n, m) does not
terminate at m ~ n above n ~ 30.  The cohort table in `scale.py` used the uniform
rejection sampler; this ceiling table does not.  The two are not interchangeable.
"""

from __future__ import annotations

import json
import random
import signal
import statistics
import time

import agm_cam
import backends as B
from sweep import random_connected

PROFILES = [(n, int(r * n)) for n in (20, 30, 50, 70, 98) for r in (1.05, 2.0)]
REPS = 3
AGM_NODES = 300_000
AGM_MAX_N = 20
MDFSC_TIMEOUT_S = 60


class _Timeout(Exception):
    pass


def _alarm(_sig, _frm):
    raise _Timeout


signal.signal(signal.SIGALRM, _alarm)


def timed(fn, G, limit=0):
    t0 = time.perf_counter()
    if limit:
        signal.alarm(limit)
    try:
        out = fn(G)
        return out, time.perf_counter() - t0, "ok"
    except _Timeout:
        return None, time.perf_counter() - t0, "TIMEOUT"
    finally:
        signal.alarm(0)


def main() -> None:
    rng = random.Random(42)
    rows = []
    print(
        f"{'n':>4}{'m':>5}"
        f"{'g6 bits':>9}{'s6 bits':>9}{'adj bits':>10}{'mDFSC bits':>12}"
        f"{'g6 ms':>8}{'s6 ms':>8}{'nauty ms':>10}{'mDFSC ms':>12}"
        f"{'AGM':>18}",
        flush=True,
    )
    for n, m in PROFILES:
        Gs = [random_connected(n, m, rng) for _ in range(REPS)]
        t = {k: [] for k in ("g6", "s6", "nauty", "mdfsc")}
        bits = {k: [] for k in ("g6", "s6", "adj", "mdfsc")}
        md_status = "ok"
        agm_note = "not attempted"
        for G in Gs:
            s, dt, _ = timed(B.graph6, G)
            t["g6"].append(dt)
            bits["g6"].append(len(s) * 6)
            s, dt, _ = timed(B.sparse6, G)
            t["s6"].append(dt)
            bits["s6"].append(len(s) * 6)
            _, dt, _ = timed(B.nauty_canon_graph6, G)
            t["nauty"].append(dt)
            bits["adj"].append(n * (n - 1) // 2)
            _, dt, st = timed(B.mdfsc, G, MDFSC_TIMEOUT_S)
            t["mdfsc"].append(dt)
            if st == "TIMEOUT":
                md_status = f"TIMEOUT >{MDFSC_TIMEOUT_S}s"
            else:
                bits["mdfsc"].append(m * 2 * max(n - 1, 1).bit_length())
            if n <= AGM_MAX_N:
                try:
                    B.agm_code(G, budget=AGM_NODES)
                    agm_note = "exact"
                except agm_cam.AGMBudgetExceeded:
                    agm_note = "budget exceeded"
        med = {k: (statistics.median(v) if v else float("nan")) for k, v in t.items()}
        mb = {k: (statistics.median(v) if v else None) for k, v in bits.items()}
        rows.append(
            {
                "n": n,
                "m": m,
                "bits": mb,
                "ms": {k: v * 1e3 for k, v in med.items()},
                "mdfsc_status": md_status,
                "agm": agm_note,
            }
        )
        md_bits = f"{mb['mdfsc']:.0f}" if mb["mdfsc"] is not None else "FAIL"
        md_ms = f"{med['mdfsc'] * 1e3:.1f}" if md_status == "ok" else md_status
        print(
            f"{n:>4}{m:>5}{mb['g6']:>9.0f}{mb['s6']:>9.0f}{mb['adj']:>10.0f}"
            f"{md_bits:>12}"
            f"{med['g6'] * 1e3:>8.3f}{med['s6'] * 1e3:>8.3f}{med['nauty'] * 1e3:>10.3f}"
            f"{md_ms:>12}{agm_note:>18}",
            flush=True,
        )
    with open("ceiling.json", "w") as fh:
        json.dump(rows, fh, indent=2)
    print("\nwrote ceiling.json")


if __name__ == "__main__":
    main()
