"""Probe: what does the realised-bytes convention decide, and is it even-handed?

Recomputes the per-stratum intersection--union test of ``claim_a_strata.json``
from the raw encoding ``.npz`` cells under four realised-bit conventions. The
question is not "which number is largest" -- it is which rule charges every
representation the same way.

``frozen``
    ``realised_bits`` exactly as ``competitors/bits.py`` emits it. **Not
    even-handed, and that is the finding**: the adjacency triangle is charged
    its payload packed into whole bytes (8 of 8 payload bits per stored byte),
    graph6 and sparse6 are charged their own published six-bit-per-byte wire
    (6 of 8), and the instruction string is charged eight bits for a symbol
    drawn from a nine-letter alphabet (3.17 of 8). gSpan min-DFS suffers the
    same rendering artefact and ``bits.py`` already flags it ``inflated``;
    IsalGraph does not carry that flag.

``ours_packed``
    The instruction string gets a **specified wire format** -- two symbols to a
    byte, ``8*ceil(L/2)``, since ``|Sigma| = 9 <= 16``. Every competitor keeps
    the wire its own format defines. This does not change the rule
    ``competitors.md`` 5 locked (*"the actual serialized length as the format
    defines it"*); it supplies the missing definition for the one format that
    never had one. A ``.g6`` file really is eight bits per character and is
    charged as such; a bit vector really is packed and is charged as such.

``all_packed``
    Every representation is charged its own information content rounded up to
    whole bytes, ``8*ceil(entropy_bits/8)``. Even-handed by construction, and
    idealised: it charges graph6 a file size no implementation of graph6
    produces. Reported so the comparison between it and ``ours_packed`` is
    visible rather than assumed.

``entropy_only``
    Not a realised convention at all -- the marginal Wilcoxon on the entropy
    bound alone. ``competitors.md`` 5 always required both marginals to be
    reported; this is one of them, and it needs no new definition.

The ``frozen`` arm is the reproduction check: it must land on the published
364 / 630 / 584 tally before any other arm means anything.

Usage::

    python .claude/notes/review/tasks/t06_bit_convention.py
"""

from __future__ import annotations

import collections
import glob
import json
import math
import os

import numpy as np
from scipy import stats

BASE = "/media/mpascual/Sandisk2TB/research/ISAL/completed/isalgraph/data/source/T06/encodings"
OUT = os.environ.get("T06_PROBE_OUT", os.path.dirname(os.path.abspath(__file__)))
REF = "isalgraph_pruned"
COMPS = ("graph6", "sparse6", "nauty_graph6", "adjacency", "agm_cam", "min_dfs", "sparse6_nauty")
MIN_GRAPHS = 8
ALPHA = 0.05

#: Conventions this probe evaluates, in reporting order.
ARMS = ("frozen", "ours_packed", "all_packed", "entropy_only")

#: The four metric-admissible comparators. ``agm_cam`` is excluded from the
#: n > 20 predicate because its scope guard stops at n = 12, so it contributes
#: no stratum there and requiring it would empty the predicate.
ADMISSIBLE_ABOVE_20 = frozenset({"nauty_graph6", "sparse6_nauty", "min_dfs"})


def load(suite: str) -> dict[tuple[str, str], dict[str, np.ndarray]]:
    """Load every encoding cell of one suite.

    Args:
        suite: ``suite1`` or ``suite2``.

    Returns:
        ``{(dataset, representation): arrays}``.
    """
    out = {}
    for path in glob.glob(f"{BASE}/{suite}/*.npz"):
        dataset, representation = os.path.basename(path)[:-4].split("__")
        z = np.load(path, allow_pickle=True)
        out[(dataset, representation)] = {
            "ids": z["graph_ids"],
            "n": z["node_counts"],
            "e": z["entropy_bits"],
            "r": z["realised_bits"],
            "st": z["status"],
            "L": z["length"],
        }
    return out


def realised(cell: dict[str, np.ndarray], representation: str, arm: str) -> np.ndarray:
    """Return the realised bit count for *cell* under *arm*.

    Args:
        cell: One loaded encoding cell.
        representation: Backend name.
        arm: One of :data:`ARMS`.

    Returns:
        Realised bits per graph.

    Raises:
        ValueError: If *arm* is not a realised convention.
    """
    if arm == "frozen":
        return np.asarray(cell["r"], dtype=float)
    if arm == "ours_packed":
        if representation.startswith("isalgraph_"):
            return 8.0 * np.ceil(np.asarray(cell["L"], dtype=float) / 2.0)
        return np.asarray(cell["r"], dtype=float)
    if arm == "all_packed":
        return 8.0 * np.ceil(np.asarray(cell["e"], dtype=float) / 8.0)
    raise ValueError(f"{arm!r} is not a realised convention")


def _wilcoxon(gap: np.ndarray) -> float:
    """Two-sided Wilcoxon signed-rank p-value, 1.0 on an all-zero gap."""
    if np.all(gap == 0):
        return 1.0
    return float(stats.wilcoxon(gap, alternative="two-sided", zero_method="wilcox").pvalue)


def strata(arm: str) -> list[dict[str, object]]:
    """Return one verdict row per node-count stratum under *arm*.

    Args:
        arm: One of :data:`ARMS`.

    Returns:
        Rows carrying the verdict, both median gaps and the combined p-value.
    """
    rows: list[dict[str, object]] = []
    for suite in ("suite1", "suite2"):
        cells = load(suite)
        for dataset in sorted({d for d, _ in cells}):
            if (dataset, REF) not in cells:
                continue
            ref = cells[(dataset, REF)]
            position = {gid: i for i, gid in enumerate(ref["ids"])}
            for competitor in COMPS:
                if (dataset, competitor) not in cells:
                    continue
                cmp_ = cells[(dataset, competitor)]
                pairs = [(position[g], j) for j, g in enumerate(cmp_["ids"]) if g in position]
                if not pairs:
                    continue
                ri = np.array([a for a, _ in pairs])
                ci = np.array([b for _, b in pairs])
                keep = (
                    (ref["st"][ri] != "error")
                    & (cmp_["st"][ci] != "error")
                    & np.isfinite(ref["e"][ri])
                    & np.isfinite(cmp_["e"][ci])
                )
                ri, ci = ri[keep], ci[keep]
                ns = ref["n"][ri]
                gap_e = cmp_["e"][ci] - ref["e"][ri]
                if arm == "entropy_only":
                    gap_r = gap_e
                else:
                    gap_r = realised(cmp_, competitor, arm)[ci] - realised(ref, REF, arm)[ri]
                for n in np.unique(ns):
                    m = ns == n
                    if int(m.sum()) < MIN_GRAPHS:
                        continue
                    p = (
                        _wilcoxon(gap_e[m])
                        if arm == "entropy_only"
                        else max(_wilcoxon(gap_e[m]), _wilcoxon(gap_r[m]))
                    )
                    me, mr = float(np.median(gap_e[m])), float(np.median(gap_r[m]))
                    if p < ALPHA and me > 0 and mr > 0:
                        v = "isalgraph_shorter"
                    elif p < ALPHA and me < 0 and mr < 0:
                        v = "competitor_shorter"
                    else:
                        v = "tie"
                    rows.append(
                        {
                            "suite": suite,
                            "dataset": dataset,
                            "comparator": competitor,
                            "n": int(n),
                            "n_graphs": int(m.sum()),
                            "verdict": v,
                            "median_gap_entropy": me,
                            "median_gap_realised": mr,
                            "p": p,
                        }
                    )
    return rows


def predicate_above_20(rows: list[dict[str, object]]) -> tuple[int, int, int]:
    """Return the three counts of the "most compact admissible" predicate.

    Args:
        rows: Verdict rows from :func:`strata`.

    Returns:
        ``(strata, strictly shortest, never beaten)`` over strata above
        ``n = 20`` where every admissible comparator is present.
    """
    by: dict[tuple[str, str, int], dict[str, str]] = collections.defaultdict(dict)
    for row in rows:
        if int(row["n"]) > 20 and row["comparator"] in ADMISSIBLE_ABOVE_20:
            by[(str(row["suite"]), str(row["dataset"]), int(row["n"]))][str(row["comparator"])] = (
                str(row["verdict"])
            )
    full = [k for k, v in by.items() if len(v) == len(ADMISSIBLE_ABOVE_20)]
    strict = sum(1 for k in full if all(v == "isalgraph_shorter" for v in by[k].values()))
    never = sum(1 for k in full if all(v != "competitor_shorter" for v in by[k].values()))
    return len(full), strict, never


def report(arm: str, rows: list[dict[str, object]]) -> None:
    """Print the per-band and per-comparator summary for one arm.

    Args:
        arm: The convention.
        rows: Its verdict rows.
    """
    tally = collections.Counter(str(r["verdict"]) for r in rows)
    print(f"\n=== {arm} === {len(rows)} strata  {dict(tally)}")
    bands = ((1, 5, "1-5"), (6, 10, "6-10"), (11, 20, "11-20"), (21, 40, "21-40"), (41, 999, "41+"))
    print("  band     strata   win   win%")
    for lo, hi, label in bands:
        band = [r for r in rows if lo <= int(r["n"]) <= hi]
        wins = sum(1 for r in band if r["verdict"] == "isalgraph_shorter")
        share = 100 * wins / len(band) if band else 0.0
        print(f"  {label:8s} {len(band):6d} {wins:5d}  {share:5.1f}%")
    print("  per comparator (n > 20):")
    for competitor in COMPS:
        band = [r for r in rows if r["comparator"] == competitor and int(r["n"]) > 20]
        if not band:
            continue
        wins = sum(1 for r in band if r["verdict"] == "isalgraph_shorter")
        losses = sum(1 for r in band if r["verdict"] == "competitor_shorter")
        print(
            f"    {competitor:16s} strata={len(band):4d} win={wins:4d} "
            f"loss={losses:4d} win%={100 * wins / len(band):5.1f}"
        )
    total, strict, never = predicate_above_20(rows)
    print(
        f"  admissible predicate over {total} strata: "
        f"strictly shortest={strict}, never beaten={never}"
    )


def overhead_table() -> None:
    """Print, per representation, how many payload bits each stored byte carries.

    This is the whole argument in one table: a convention that charges one
    format 8 payload bits per byte and another 3.17 is not measuring the
    encodings, it is measuring how wasteful each format's rendering happens to
    be.
    """
    print("\n=== payload bits per stored byte, frozen convention ===")
    ratios: dict[str, list[float]] = collections.defaultdict(list)
    for suite in ("suite1", "suite2"):
        for (_, representation), cell in load(suite).items():
            keep = (cell["st"] != "error") & np.isfinite(cell["e"]) & (cell["r"] > 0)
            if not keep.any():
                continue
            ratios[representation] += list(
                8.0 * np.asarray(cell["e"])[keep] / np.asarray(cell["r"])[keep]
            )
    for representation in sorted(ratios):
        values = np.asarray(ratios[representation])
        print(
            f"  {representation:22s} median {np.median(values):4.2f} of 8   "
            f"overhead {8.0 / np.median(values):4.2f}x"
        )
    print(f"  (IsalGraph's alphabet is 9 symbols, so log2 9 = {math.log2(9):.4f} bits per symbol)")


def main() -> int:
    """Entry point.

    Returns:
        ``0`` when the frozen arm reproduces the published tally, ``1``
        otherwise -- a mismatch there invalidates every other arm.
    """
    overhead_table()
    results = {}
    for arm in ARMS:
        rows = strata(arm)
        report(arm, rows)
        results[arm] = rows
        with open(f"{OUT}/claimA_{arm}.json", "w") as handle:
            json.dump(rows, handle)
    published = {"isalgraph_shorter": 364, "tie": 630, "competitor_shorter": 584}
    observed = dict(collections.Counter(str(r["verdict"]) for r in results["frozen"]))
    ok = observed == published
    print(f"\nfrozen arm reproduces the published tally: {ok}  ({observed})")
    print(f"wrote claimA_<arm>.json for {', '.join(ARMS)} under {OUT}")
    return 0 if ok else 1


if __name__ == "__main__":
    raise SystemExit(main())
