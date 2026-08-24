"""Claim A stratified by graph size, with the intersection-union verdict.

``claim_a.json`` reports bits per ``(dataset, representation)``. That is one
number per cell over a cohort spanning ``n = 2`` to ``n = 98``, and it cannot
answer the question a methodology decision actually turns on: **does the bit
advantage hold at every size, or only where the graphs are small?**

The question is live because the *other* claim's advantage does not survive
size. Within-``n`` Spearman against GED falls from 1.0000 at ``n = 3`` to 0.2608
at ``n = 12`` and to noise above ``n = 40``. If the bit advantage decayed the
same way the two claims would share a cause; if it does not, they are
independent and Claim A is the more robust of the two. Either finding changes
what the paper leads with, and neither is currently measured.

**The test is the same one A1 uses, applied per stratum.** Wilcoxon signed-rank
on paired bit counts, run under both conventions, combined by the
intersection-union rule ``p = max(p_entropy, p_realised)`` --- so Claim A is
read as the conjunction it is, *fewer bits under both conventions*, and no
primary convention has to be named after the data exist (design note 18.8).

**Strata are node counts, not bands.** A band would smear the boundary the
question is about. Strata below :data:`MIN_GRAPHS` are skipped rather than
reported at a width that means nothing, and the count of skipped graphs is
carried so the coverage is auditable.
"""

from __future__ import annotations

import argparse
import json
import logging
from collections.abc import Sequence
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Final

import numpy as np

from benchmarks.real_data.eval_stats.multiplicity import _wilcoxon as wilcoxon_pair
from benchmarks.real_data.eval_stats.t06_f2_inputs import (
    BIT_CONVENTIONS,
    REFERENCE_ARM,
    ArmEncodings,
    load_encodings,
)

LOGGER: Final = logging.getLogger(__name__)

#: The six Claim-A serialisations (``preregistration`` 4.1).
CLAIM_A_COMPARATORS: Final[tuple[str, ...]] = (
    "graph6",
    "sparse6",
    "nauty_graph6",
    "adjacency",
    "agm_cam",
    "min_dfs",
)

#: Minimum graphs in a stratum before a signed-rank test is worth reporting.
#: Below this the Wilcoxon null distribution is too coarse for the p-value to
#: mean anything, so the stratum is skipped and counted rather than printed.
MIN_GRAPHS: Final[int] = 8


class ClaimAStrataError(Exception):
    """Raised when a stratified Claim-A cell cannot be assembled."""


@dataclass(frozen=True)
class StratumVerdict:
    """One ``(dataset, representation, n)`` cell.

    Attributes:
        suite: Suite key.
        dataset: Dataset key.
        representation: The comparator.
        n: Node count defining the stratum.
        n_graphs: Graphs both arms encoded at this size.
        iut_p: ``max`` of the two marginal p-values --- the reported value.
        p_entropy: Wilcoxon p under the entropy bound.
        p_realised: Wilcoxon p under realised bytes.
        median_gap_entropy: Median ``competitor - IsalGraph`` bits; positive
            means IsalGraph is shorter.
        median_gap_realised: The same under realised bytes.
        fraction_shorter_entropy: Graphs where IsalGraph is strictly shorter.
        fraction_shorter_realised: The same under realised bytes.
        verdict: ``isalgraph_shorter``, ``competitor_shorter`` or ``tie``.
        discordant: Whether the conventions disagree in direction.
    """

    suite: str
    dataset: str
    representation: str
    n: int
    n_graphs: int
    iut_p: float
    p_entropy: float
    p_realised: float
    median_gap_entropy: float
    median_gap_realised: float
    fraction_shorter_entropy: float
    fraction_shorter_realised: float
    verdict: str
    discordant: bool


def aligned_with_size(
    reference: ArmEncodings, competitor: ArmEncodings, arm: str = "primary"
) -> tuple[dict[str, np.ndarray], dict[str, np.ndarray], np.ndarray]:
    """Join two arms on ``graph_ids`` and carry the node counts through.

    Args:
        reference: The IsalGraph arm.
        competitor: The comparator.
        arm: ``primary`` or ``complete_case``.

    Returns:
        ``(reference_bits, competitor_bits, node_counts)`` where the bit dicts
        are keyed by convention and every array is the same length and refers to
        the same graphs in the same order.
    """
    ref_pos = {gid: i for i, gid in enumerate(reference.graph_ids)}
    com_pos = {gid: i for i, gid in enumerate(competitor.graph_ids)}
    ref_ok, com_ok = reference.usable(arm), competitor.usable(arm)
    keep = [
        (ref_pos[gid], com_pos[gid])
        for gid in np.intersect1d(reference.graph_ids, competitor.graph_ids)
        if ref_ok[ref_pos[gid]] and com_ok[com_pos[gid]]
    ]
    if not keep:
        empty = {c: np.empty(0) for c in BIT_CONVENTIONS}
        return empty, empty, np.empty(0, dtype=np.int64)

    ref_idx = np.array([i for i, _ in keep])
    com_idx = np.array([j for _, j in keep])
    finite = np.ones(ref_idx.size, dtype=bool)
    for convention in BIT_CONVENTIONS:
        finite &= np.isfinite(reference.bits[convention][ref_idx])
        finite &= np.isfinite(competitor.bits[convention][com_idx])
    ref_bits = {c: reference.bits[c][ref_idx][finite] for c in BIT_CONVENTIONS}
    com_bits = {c: competitor.bits[c][com_idx][finite] for c in BIT_CONVENTIONS}
    return ref_bits, com_bits, reference.node_counts[ref_idx][finite]


def _verdict(gap_entropy: float, gap_realised: float, iut_p: float, alpha: float) -> str:
    """Classify a stratum, treating a non-significant result as a tie."""
    if iut_p > alpha:
        return "tie"
    if gap_entropy > 0 and gap_realised > 0:
        return "isalgraph_shorter"
    if gap_entropy < 0 and gap_realised < 0:
        return "competitor_shorter"
    return "tie"


def stratify(
    reference: ArmEncodings,
    competitor: ArmEncodings,
    suite: str,
    dataset: str,
    *,
    alpha: float = 0.05,
    arm: str = "primary",
) -> tuple[list[StratumVerdict], int]:
    """Run the IUT per node-count stratum for one cell.

    Args:
        reference: The IsalGraph arm.
        competitor: The comparator.
        suite: Suite key.
        dataset: Dataset key.
        alpha: Level for the per-stratum verdict. This is a **descriptive**
            stratification, not a BH family member, so the level is applied
            locally and labelled as such.
        arm: ``primary`` or ``complete_case``.

    Returns:
        ``(verdicts, n_graphs_skipped)``.
    """
    ref_bits, com_bits, sizes = aligned_with_size(reference, competitor, arm)
    if sizes.size == 0:
        return [], 0

    verdicts: list[StratumVerdict] = []
    skipped = 0
    for n in sorted({int(v) for v in sizes}):
        selection = sizes == n
        count = int(selection.sum())
        if count < MIN_GRAPHS:
            skipped += count
            continue
        marginal: dict[str, float] = {}
        gaps: dict[str, float] = {}
        shorter: dict[str, float] = {}
        for convention in BIT_CONVENTIONS:
            ref = ref_bits[convention][selection]
            com = com_bits[convention][selection]
            marginal[convention] = wilcoxon_pair(com, ref)[1]
            gaps[convention] = float(np.median(com - ref))
            shorter[convention] = float(np.mean(ref < com))
        iut_p = max(marginal.values())
        verdicts.append(
            StratumVerdict(
                suite=suite,
                dataset=dataset,
                representation=competitor.representation,
                n=n,
                n_graphs=count,
                iut_p=iut_p,
                p_entropy=marginal["entropy_bits"],
                p_realised=marginal["realised_bits"],
                median_gap_entropy=gaps["entropy_bits"],
                median_gap_realised=gaps["realised_bits"],
                fraction_shorter_entropy=shorter["entropy_bits"],
                fraction_shorter_realised=shorter["realised_bits"],
                verdict=_verdict(gaps["entropy_bits"], gaps["realised_bits"], iut_p, alpha),
                discordant=(gaps["entropy_bits"] > 0) != (gaps["realised_bits"] > 0),
            )
        )
    return verdicts, skipped


def build_parser() -> argparse.ArgumentParser:
    """Return the CLI parser."""
    ap = argparse.ArgumentParser(description="Claim A stratified by graph size.")
    ap.add_argument("--encodings", type=Path, required=True, help="the encodings/ tree")
    ap.add_argument("--out", type=Path, required=True, help="claim_a_strata.json")
    ap.add_argument("--alpha", type=float, default=0.05)
    ap.add_argument("--arm", choices=("primary", "complete_case"), default="primary")
    return ap


def main(argv: Sequence[str] | None = None) -> int:
    """Entry point.

    Args:
        argv: Argument vector, or ``None`` for ``sys.argv``.

    Returns:
        0 on success, 1 when nothing was measured.
    """
    args = build_parser().parse_args(argv)
    logging.basicConfig(level=logging.INFO, format="%(levelname)s %(message)s")

    rows: list[StratumVerdict] = []
    skipped_total = 0
    for suite in ("suite1", "suite2"):
        directory = args.encodings / suite
        if not directory.is_dir():
            continue
        datasets = sorted({p.stem.split("__", 1)[0] for p in directory.glob("*.npz")})
        for dataset in datasets:
            reference = load_encodings(args.encodings, suite, dataset, REFERENCE_ARM)
            if reference is None:
                continue
            for name in CLAIM_A_COMPARATORS:
                competitor = load_encodings(args.encodings, suite, dataset, name)
                if competitor is None:
                    continue
                verdicts, skipped = stratify(
                    reference, competitor, suite, dataset, alpha=args.alpha, arm=args.arm
                )
                rows.extend(verdicts)
                skipped_total += skipped
            LOGGER.info(
                "%s/%-16s %3d strata across %d comparators",
                suite,
                dataset,
                len([r for r in rows if r.dataset == dataset and r.suite == suite]),
                len(CLAIM_A_COMPARATORS),
            )

    if not rows:
        LOGGER.error("no strata measured under %s", args.encodings)
        return 1

    tally: dict[str, int] = {}
    for row in rows:
        tally[row.verdict] = tally.get(row.verdict, 0) + 1

    payload = {
        "schema_version": "t06.claim_a_strata.1",
        "ticket": "T-06",
        "descriptive": True,
        "note": (
            "Claim A per node-count stratum. DESCRIPTIVE: the confirmatory A1 cells are "
            "per (dataset, representation) and live in family_F2.json under BH over "
            "N_actual = 79. The level here is applied locally and per stratum."
        ),
        "test": (
            "Wilcoxon signed-rank on paired bit counts under BOTH conventions, combined by "
            "the intersection-union rule p = max(p_entropy, p_realised). Claim A is a "
            "conjunction -- fewer bits under both -- so the IUT is the matching procedure and "
            "no primary convention has to be named (design note 18.8)."
        ),
        "arm": args.arm,
        "alpha": args.alpha,
        "min_graphs_per_stratum": MIN_GRAPHS,
        "n_graphs_skipped_thin_strata": skipped_total,
        "verdict_tally": tally,
        "n_rows": len(rows),
        "rows": [asdict(r) for r in rows],
    }
    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(json.dumps(payload, indent=2, default=str))
    print(f"wrote {args.out} ({len(rows)} strata, tally {tally})")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
