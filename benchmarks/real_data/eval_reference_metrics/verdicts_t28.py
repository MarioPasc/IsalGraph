"""T-28 verdicts from the PAIRED graph-level bootstrap, not from point estimates.

Reads either the merged ``rho_table.json`` or a directory of ``f2_partials/``
and reports, per (reference, competitor), how many cells the IsalGraph arm wins,
ties and loses.

**A difference whose 95 % interval covers zero is a TIE**, never a loss. Every
verdict here comes from ``difference_vs_reference_arm`` --- the paired difference
computed on identical pairs under one graph-level resample --- and never from two
overlapping marginal intervals, which is a different and weaker test.

Under the ``wl`` reference the ``wl_subtree`` arm is degenerate: its distance IS
the reference, so its rho is exactly 1.0 and the paired difference is exactly the
arm's deficit against a perfect score. It is printed, marked, and excluded from
the tallies.
"""

from __future__ import annotations

import argparse
import json
from collections import defaultdict
from pathlib import Path
from typing import Any

ARM = "isalgraph_pruned"
DEGENERATE = {("wl", "wl_subtree")}
REF_ORDER = ("exact", "lb", "ub", "wl", "spectral", "spectral_comb", "spectral_adj", "spectral_esd")
REF_LABEL = {
    "exact": "GED exact",
    "lb": "GED lower bd",
    "ub": "GED upper bd",
    "wl": "WL kernel",
    "spectral": "spectral norm-L",
    "spectral_comb": "spectral comb-L",
    "spectral_adj": "spectral adj",
    "spectral_esd": "spectral ESD",
}


def load(path: Path) -> list[dict[str, Any]]:
    """Load rho rows from a merged table or a directory of shard partials.

    Args:
        path: ``rho_table.json``, a ``f2_partials`` directory, or a families
            directory containing one.

    Returns:
        Every rho row found.

    Raises:
        FileNotFoundError: If *path* holds nothing readable.
    """
    if path.is_file():
        payload = json.loads(path.read_text())
        return list(payload.get("rows") or payload.get("rho_rows") or [])
    directory = path if path.name == "f2_partials" else path / "f2_partials"
    rows: list[dict[str, Any]] = []
    for shard in sorted(directory.glob("*.json")):
        rows.extend(json.loads(shard.read_text()).get("rho_rows", []))
    if not rows:
        raise FileNotFoundError(f"no rho rows under {path}")
    return rows


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--results", type=Path, required=True)
    ap.add_argument("--view", default="all_pairs")
    ap.add_argument("--detail", action="store_true")
    args = ap.parse_args()

    rows = [r for r in load(args.results) if r.get("view") == args.view]

    cells = sorted({(r["suite"], r["dataset"]) for r in rows})
    # (reference, competitor) -> [win, tie, loss]
    tally: dict[tuple[str, str], list[int]] = defaultdict(lambda: [0, 0, 0])
    detail: dict[str, list[str]] = defaultdict(list)
    nulls: dict[str, list[int]] = defaultdict(lambda: [0, 0])  # clears, total

    for r in rows:
        ref, rep = r["reference"], r["representation"]
        if rep == ARM:
            excess = r.get("excess_over_size_null")
            if isinstance(excess, dict):
                nulls[ref][1] += 1
                if excess["ci_low"] > 0.0:
                    nulls[ref][0] += 1
            continue
        if rep.startswith("isalgraph"):
            continue
        diff = r.get("difference_vs_reference_arm")
        if not isinstance(diff, dict):
            continue
        lo, hi, pt = diff["ci_low"], diff["ci_high"], diff["point"]
        degenerate = (ref, rep) in DEGENERATE
        if lo <= 0.0 <= hi:
            verdict, slot = "TIE ", 1
        elif pt > 0:
            verdict, slot = "WIN ", 0
        else:
            verdict, slot = "LOSS", 2
        if not degenerate:
            tally[(ref, rep)][slot] += 1
        detail[ref].append(
            f"  {'DEGEN' if degenerate else verdict}  {r['suite'][-1]}/{r['dataset']:<16} "
            f"vs {rep:<15} {pt:+.4f} [{lo:+.4f},{hi:+.4f}]"
        )

    competitors = sorted({c for _, c in tally})
    print(f"\n=== T-28 paired-bootstrap verdicts | view={args.view} | arm={ARM} ===")
    print(f"{len(cells)} cells: {', '.join(f'{s[-1]}/{d}' for s, d in cells)}")
    print("\nwin / tie / loss, from the paired difference; a CI covering 0 is a TIE\n")
    header = f"{'reference':<18}" + "".join(f"{c[:14]:>18}" for c in competitors) + "   clears null"
    print(header)
    print("-" * len(header))
    for ref in REF_ORDER:
        if not any((ref, c) in tally for c in competitors):
            continue
        line = f"{REF_LABEL.get(ref, ref):<18}"
        for c in competitors:
            if (ref, c) not in tally:
                line += f"{'-':>18}"
                continue
            w, t, ls = tally[(ref, c)]
            line += f"{w}W {t}T {ls}L".rjust(18)
        if ref in nulls:
            line += f"   {nulls[ref][0]}/{nulls[ref][1]}"
        print(line)

    if args.detail:
        for ref in REF_ORDER:
            if ref in detail:
                print(f"\n--- {REF_LABEL.get(ref, ref)} ---")
                print("\n".join(sorted(detail[ref])))
    print("\nDegenerate (excluded from tallies):", ", ".join(f"{r}/{c}" for r, c in DEGENERATE))
    print("'clears null' counts cells whose excess-over-size-null CI is strictly above 0.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
