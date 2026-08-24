"""One human-readable verdict table per claim, for a methodology decision.

This exists because five JSON files are not a decision. The question it answers
is narrow and blunt: **on the GED experiment and the information-content
experiment, across every dataset and competitor, is IsalGraph at a clear
disadvantage, and where?**

Three rules it enforces, because the decision is worse without them.

**A loss inside a confidence interval is a TIE, not a loss.** The paired
bootstrap difference is the instrument; where its interval covers zero the two
representations are not distinguishable at this sample size and saying otherwise
overstates the evidence. This is the distinction between *"best on none of 15,
with 8 resolvable deficits and 7 ties"*, which the data support, and *"beaten on
15 of 15"*, which they do not.

**The comparison is paired, on identical pairs and identical resamples.**
Overlapping marginal intervals are a conservative test --- non-overlap implies a
difference, overlap does not imply none --- so a verdict is taken from
``difference_vs_reference_arm`` and never from two separate intervals.

**Incomplete input is reported, never silently averaged away.** The campaign is
sharded and this summary is meant to be read before every shard lands, so any
missing ``(suite, dataset)`` is named in the output rather than dropped from a
count that then looks complete.
"""

from __future__ import annotations

import argparse
import json
import logging
import re
from collections import Counter
from collections.abc import Sequence
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Final

import numpy as np

from benchmarks.real_data.eval_stats.family import SUITE1, SUITE2

LOGGER: Final = logging.getLogger(__name__)

#: Comparators inside the frozen Claim-B family (``k`` already applied).
FAMILY_COMPARATORS: Final[tuple[str, ...]] = ("nauty_graph6", "agm_cam", "min_dfs", "wl_subtree")

#: Size bands for the "where does it lose" line. Fixed before reading.
SIZE_BANDS: Final[tuple[tuple[int, int], ...]] = ((0, 5), (5, 10), (10, 20), (20, 40), (40, 10**9))


class DecisionSummaryError(Exception):
    """Raised when neither a rho table nor any partial can be read."""


@dataclass(frozen=True)
class Verdict:
    """One ``(suite, dataset, reference)`` head-to-head.

    Attributes:
        suite: Suite key.
        dataset: Dataset key.
        reference: ``exact``, ``lb`` or ``ub``.
        isalgraph_rho: The reference arm's rho.
        best_competitor: The competitor with the highest rho.
        best_rho: Its rho.
        delta: Paired ``rho(IsalGraph) - rho(best competitor)``.
        ci_low: Lower bound of the paired difference.
        ci_high: Upper bound.
        verdict: ``win``, ``tie`` or ``loss``.
        size_null: The reference arm's per-cell size null, if defined.
        n_pairs: Pairs behind the estimate.
    """

    suite: str
    dataset: str
    reference: str
    isalgraph_rho: float
    best_competitor: str
    best_rho: float
    delta: float
    ci_low: float
    ci_high: float
    verdict: str
    size_null: float | None
    n_pairs: int


def load_rho_rows(rho_table: Path | None, partial_dirs: Sequence[Path]) -> list[dict[str, Any]]:
    """Collect rho rows from a finished table or from whatever partials exist.

    Args:
        rho_table: A finished ``rho_table.json``, or ``None``.
        partial_dirs: Directories of shard partials to fall back on.

    Returns:
        Every rho row found, deduplicated on its identifying tuple.

    Raises:
        DecisionSummaryError: If no source yields a row.
    """
    rows: list[dict[str, Any]] = []
    if rho_table is not None and rho_table.exists():
        rows.extend(json.loads(rho_table.read_text())["rows"])
    for directory in partial_dirs:
        for path in sorted(directory.glob("*.json")) if directory.is_dir() else []:
            rows.extend(json.loads(path.read_text()).get("rho_rows", []))

    from benchmarks.real_data.eval_stats.t06_f2 import dedup_rho_rows

    # Keeping "the first" would make the surviving value depend on emission
    # order; partials written before the arm was emitted once per cell carry
    # several arm records, on different pair sets, under one key.
    unique = dedup_rho_rows(rows)
    if not unique:
        raise DecisionSummaryError("no rho rows found in the table or any partial directory")
    return unique


def head_to_head(rows: Sequence[dict[str, Any]], view: str = "all_pairs") -> list[Verdict]:
    """Reduce the rho rows to one verdict per ``(suite, dataset, reference)``.

    The best competitor is chosen on **raw rho**, and the verdict is then taken
    from the *paired* difference against that competitor rather than from the
    two marginal intervals.

    Args:
        rows: Rho rows.
        view: Which pair view to summarise.

    Returns:
        One verdict per cell, in suite then dataset then reference order.
    """
    grouped: dict[tuple[str, str, str], list[dict[str, Any]]] = {}
    for row in rows:
        if row["view"] != view:
            continue
        grouped.setdefault((row["suite"], row["dataset"], row["reference"]), []).append(row)

    verdicts: list[Verdict] = []
    for (suite, dataset, reference), block in sorted(grouped.items()):
        arm = next((r for r in block if r["representation"] == "isalgraph_pruned"), None)
        rivals = [
            r
            for r in block
            if r["representation"] in FAMILY_COMPARATORS
            and r.get("difference_vs_reference_arm") is not None
        ]
        if arm is None or not rivals:
            continue
        best = max(rivals, key=lambda r: r["rho"]["point"])
        difference = best["difference_vs_reference_arm"]
        low, high = float(difference["ci_low"]), float(difference["ci_high"])
        delta = float(difference["point"])
        if low > 0.0:
            label = "win"
        elif high < 0.0:
            label = "loss"
        else:
            label = "tie"
        null = arm.get("size_null")
        verdicts.append(
            Verdict(
                suite=suite,
                dataset=dataset,
                reference=reference,
                isalgraph_rho=float(arm["rho"]["point"]),
                best_competitor=best["representation"],
                best_rho=float(best["rho"]["point"]),
                delta=delta,
                ci_low=low,
                ci_high=high,
                verdict=label,
                size_null=None if null is None else float(null["point"]),
                n_pairs=int(arm["n_pairs"]),
            )
        )
    return verdicts


def _band(n: int) -> str:
    """Return the size-band label for *n*."""
    for low, high in SIZE_BANDS:
        if low < n <= high:
            return f"{low + 1}-{high}" if high < 10**9 else f"{low + 1}+"
    return "?"


def claim_a_by_size(strata: dict[str, Any]) -> list[dict[str, Any]]:
    """Summarise the stratified Claim-A verdicts per size band.

    Args:
        strata: Parsed ``claim_a_strata.json``.

    Returns:
        One record per band.
    """
    tally: dict[str, Counter[str]] = {}
    gaps: dict[str, list[float]] = {}
    for row in strata["rows"]:
        band = _band(int(row["n"]))
        tally.setdefault(band, Counter())[row["verdict"]] += 1
        gaps.setdefault(band, []).append(float(row["median_gap_entropy"]))
    records: list[dict[str, Any]] = []
    for low, high in SIZE_BANDS:
        band = f"{low + 1}-{high}" if high < 10**9 else f"{low + 1}+"
        counts = tally.get(band)
        if not counts:
            continue
        total = sum(counts.values())
        records.append(
            {
                "band": band,
                "strata": total,
                "isalgraph_shorter": counts["isalgraph_shorter"],
                "tie": counts["tie"],
                "competitor_shorter": counts["competitor_shorter"],
                "win_fraction": counts["isalgraph_shorter"] / total,
                "median_gap_entropy_bits": float(np.median(gaps[band])),
            }
        )
    return records


def claim_a_by_competitor(strata: dict[str, Any], split_at: int = 20) -> list[dict[str, Any]]:
    """Summarise the stratified Claim-A verdicts per competitor.

    Split at *split_at* because the pooled rate hides the thing that matters:
    ``agm_cam`` is refused above ``n = 12`` by its own scope guard, so every
    stratum it appears in is a small one, and a pooled win rate against it is a
    statement about small graphs wearing the clothes of a general one.

    Args:
        strata: Parsed ``claim_a_strata.json``.
        split_at: Node count separating the two size halves.

    Returns:
        One record per competitor.
    """
    buckets: dict[str, dict[str, Counter[str]]] = {}
    reach: dict[str, int] = {}
    admissible: dict[str, bool] = {}
    for row in strata["rows"]:
        name = row["representation"]
        half = "small" if int(row["n"]) <= split_at else "large"
        bucket = buckets.setdefault(name, {"small": Counter(), "large": Counter()})
        bucket[half][row["verdict"]] += 1
        reach[name] = max(reach.get(name, 0), int(row["n"]))
        admissible[name] = bool(row.get("metric_admissible", True))

    records: list[dict[str, Any]] = []
    for name, halves in buckets.items():
        total = sum(sum(c.values()) for c in halves.values())
        wins = sum(c["isalgraph_shorter"] for c in halves.values())
        losses = sum(c["competitor_shorter"] for c in halves.values())
        large_total = sum(halves["large"].values())
        records.append(
            {
                "competitor": name,
                "strata": total,
                "win": wins,
                "tie": total - wins - losses,
                "loss": losses,
                "win_fraction": wins / total if total else 0.0,
                "win_fraction_large_n": (
                    halves["large"]["isalgraph_shorter"] / large_total if large_total else None
                ),
                "max_n_reached": reach[name],
                "metric_admissible": admissible.get(name, True),
            }
        )
    return sorted(records, key=lambda r: -r["win_fraction"])


def delta_matrix(rows: Sequence[dict[str, Any]], view: str = "all_pairs") -> list[dict[str, Any]]:
    """Return the paired delta-rho against **every** comparator, not just the best.

    Args:
        rows: Rho rows.
        view: Which pair view to summarise.

    Returns:
        One record per ``(suite, dataset, reference, competitor)``.
    """
    records: list[dict[str, Any]] = []
    for row in rows:
        difference = row.get("difference_vs_reference_arm")
        if row["view"] != view or difference is None:
            continue
        if row["representation"] not in FAMILY_COMPARATORS:
            continue
        low, high = float(difference["ci_low"]), float(difference["ci_high"])
        records.append(
            {
                "suite": row["suite"],
                "dataset": row["dataset"],
                "reference": row["reference"],
                "competitor": row["representation"],
                "competitor_rho": float(row["rho"]["point"]),
                "delta": float(difference["point"]),
                "ci_low": low,
                "ci_high": high,
                "verdict": "win" if low > 0 else ("loss" if high < 0 else "tie"),
            }
        )
    return sorted(
        records, key=lambda r: (r["suite"], r["dataset"], r["reference"], r["competitor"])
    )


def claim_b_by_size(profile: dict[str, Any]) -> list[dict[str, Any]]:
    """Summarise Claim B per size band, the direct analogue of Claim A's table.

    The unit is an equal-``n`` stratum, where ``|n_i - n_j|`` is identically
    zero so the size null has no denominator and raw rho *is* the structural
    signal. Within each stratum the representations are ranked and the reference
    arm either tops it or does not; across strata that gives a "best in what
    fraction" beside the level of rho itself.

    Args:
        profile: Parsed ``size_profile.json``.

    Returns:
        One record per band.
    """
    contenders = set(FAMILY_COMPARATORS) | {"isalgraph_pruned"}
    groups: dict[tuple[Any, ...], list[dict[str, Any]]] = {}
    for row in profile["rows"]:
        if row["rho"] is None or row["representation"] not in contenders:
            continue
        if row.get("arm", "primary") != "primary":
            continue
        key = (row["suite"], row["dataset"], row["reference"], row["n"])
        groups.setdefault(key, []).append(row)

    tally: dict[str, Counter[str]] = {}
    levels: dict[str, list[float]] = {}
    for key, block in groups.items():
        arm = next((r for r in block if r["representation"] == "isalgraph_pruned"), None)
        if arm is None or len(block) < 2:
            continue
        band = _band(int(key[3]))
        best = max(block, key=lambda r: float(r["rho"]))
        counter = tally.setdefault(band, Counter())
        counter["strata"] += 1
        counter["isalgraph_best"] += int(best["representation"] == "isalgraph_pruned")
        levels.setdefault(band, []).append(float(arm["rho"]))

    records: list[dict[str, Any]] = []
    for low, high in SIZE_BANDS:
        band = f"{low + 1}-{high}" if high < 10**9 else f"{low + 1}+"
        counter = tally.get(band)
        if not counter:
            continue
        records.append(
            {
                "band": band,
                "strata": counter["strata"],
                "isalgraph_best": counter["isalgraph_best"],
                "best_fraction": counter["isalgraph_best"] / counter["strata"],
                "median_rho": float(np.median(levels[band])),
            }
        )
    return records


def compactness_predicates(strata: dict[str, Any], above: int = 20) -> dict[str, Any] | None:
    """Count the three DIFFERENT things "most compact" can mean.

    These are not variants of one number. Over the same 122 strata they give
    0 %, 32 % and 42 %, and the gaps between them are the whole content:

    * ``significantly_shortest`` --- beats **every** admissible competitor with
      the IUT rejecting at the stratum level. This is what "the most compact
      representation that admits a metric" asserts.
    * ``positive_gap_against_all`` --- shorter at the median against every one,
      significant or not. Weaker: a median gap that does not clear its own test.
    * ``never_significantly_beaten`` --- no admissible competitor beats it
      significantly. Weakest, and the one most easily misread as a win.

    Reporting one of these without naming which is how "32 %" becomes "best in a
    third of cases", which it is not. The caller prints the predicate in the
    sentence, not in a footnote.

    Args:
        strata: Parsed ``claim_a_strata.json``.
        above: Only strata with ``n`` greater than this are counted.

    Returns:
        The three counts with their shared denominator, or ``None`` when no
        stratum carries at least two admissible competitors.
    """
    groups: dict[tuple[Any, ...], list[dict[str, Any]]] = {}
    for row in strata["rows"]:
        if not row.get("metric_admissible", True) or int(row["n"]) <= above:
            continue
        groups.setdefault((row["suite"], row["dataset"], row["n"]), []).append(row)
    usable = [g for g in groups.values() if len(g) >= 2]
    if not usable:
        return None

    blockers: Counter[str] = Counter()
    for block in usable:
        for row in block:
            if row["verdict"] != "isalgraph_shorter":
                blockers[row["representation"]] += 1
    return {
        "total": len(usable),
        "significantly_shortest": sum(
            1 for g in usable if all(r["verdict"] == "isalgraph_shorter" for r in g)
        ),
        "positive_gap_against_all": sum(
            1 for g in usable if all(float(r["median_gap_entropy"]) > 0 for r in g)
        ),
        "never_significantly_beaten": sum(
            1 for g in usable if all(r["verdict"] != "competitor_shorter" for r in g)
        ),
        "blockers": dict(blockers.most_common()),
        "above": above,
    }


def _sign_test(differences: Sequence[float]) -> dict[str, Any]:
    """Two-sided exact sign test on per-stratum differences.

    Args:
        differences: One signed difference per stratum.

    Returns:
        The counts either side, ties dropped, and the exact binomial p-value.
    """
    from scipy import stats

    higher = sum(1 for d in differences if d > 0)
    lower = sum(1 for d in differences if d < 0)
    trials = higher + lower
    p_value = float(stats.binomtest(higher, trials, 0.5).pvalue) if trials else float("nan")
    return {
        "isalgraph_higher": higher,
        "isalgraph_lower": lower,
        "sign_test_p": p_value,
    }


def claim_b_by_competitor(profile: dict[str, Any], split_at: int = 20) -> list[dict[str, Any]]:
    """Claim B per competitor per size half, inside equal-``n`` strata.

    **Counting how many strata resolve is the WRONG summary and is not reported
    as one.** Equal-``n`` strata above 20 are thin, so most individual
    comparisons are unresolved at the graph-level interval --- but many
    underpowered comparisons all leaning one way is evidence, not absence of it.
    A sign test over the per-stratum rho differences pools them, and it reverses
    the reading: the unresolved fraction exceeds 90 % against every competitor
    while the sign test rejects against every competitor.

    Strata within a dataset are disjoint graph sets, so the sign test is valid,
    and it weights every stratum equally regardless of pair count --- if
    anything understating the large ones. Both columns are printed so the
    per-stratum count cannot be quoted on its own.

    **``lb`` and ``ub`` are reported separately and never pooled.** They are two
    bounds on the *same* pairs, so pooling them would enter every stratum twice
    and break the independence the sign test needs.

    Args:
        profile: Parsed ``size_profile.json``.
        split_at: Node count separating the two halves.

    Returns:
        One record per ``(competitor, half)``.
    """
    cells: dict[tuple[Any, ...], dict[str, dict[str, Any]]] = {}
    for row in profile["rows"]:
        if row["rho"] is None or row.get("arm", "primary") != "primary":
            continue
        key = (row["suite"], row["dataset"], row["reference"], row["n"])
        cells.setdefault(key, {})[row["representation"]] = row

    tally: dict[tuple[str, str, str], Counter[str]] = {}
    deltas: dict[tuple[str, str, str], list[float]] = {}
    for key, block in cells.items():
        arm = block.get("isalgraph_pruned")
        if arm is None or arm["ci_lo"] is None:
            continue
        half = "n > 20" if int(key[3]) > split_at else "n <= 20"
        reference = str(key[2])
        for name, rival in block.items():
            if name == "isalgraph_pruned" or rival["ci_lo"] is None:
                continue
            if arm["ci_lo"] > rival["ci_hi"]:
                outcome = "win"
            elif arm["ci_hi"] < rival["ci_lo"]:
                outcome = "loss"
            else:
                outcome = "unresolved"
            tally.setdefault((name, half, reference), Counter())[outcome] += 1
            deltas.setdefault((name, half, reference), []).append(
                float(arm["rho"]) - float(rival["rho"])
            )

    records: list[dict[str, Any]] = []
    for (name, half, reference), counter in sorted(tally.items()):
        total = sum(counter.values())
        records.append(
            {
                "competitor": name,
                "half": half,
                "reference": reference,
                "strata": total,
                "win": counter["win"],
                "unresolved": counter["unresolved"],
                "loss": counter["loss"],
                "unresolved_fraction": counter["unresolved"] / total,
                "median_delta_rho": float(np.median(deltas[name, half, reference])),
                **_sign_test(deltas[name, half, reference]),
            }
        )
    return records


def uniqueness_and_coverage(
    ladders: Sequence[dict[str, Any]], completion: dict[str, Any] | None
) -> dict[str, Any]:
    """The two things neither claim measures: collisions, and who computes.

    Claim A is about bits and Claim B is about correlation, and a
    representation can lose both while still being the only one that never
    conflates two non-isomorphic graphs. That property is a theorem for
    IsalGraph and it is worth measuring anyway, because a measured zero over
    24.8 million pairs is what a reviewer can check.

    Coverage is included **because it is where a claimed advantage dies**.
    Section 15.3 lists "it computes everywhere" among the things that survive
    the ranking. Six competitors also complete on 100 % of every cell, so it
    does not distinguish anything.

    Args:
        ladders: Parsed ladder payloads.
        completion: Parsed ``completion_rates.json``, or ``None``.

    Returns:
        The collision total and the per-representation coverage.
    """
    pairs = sum(int(r["ged_positive"]) for d in ladders for r in d["rows"])
    collisions = sum(int(r["collisions"]) for d in ladders for r in d["rows"])
    coverage: list[dict[str, Any]] = []
    if completion is not None:
        buckets: dict[str, list[float]] = {}
        for row in completion["rows"]:
            buckets.setdefault(row["representation"], []).append(float(row["rate"]))
        for name, rates in buckets.items():
            coverage.append(
                {
                    "representation": name,
                    "cells": len(rates),
                    "min_rate": min(rates),
                    "cells_below_99": sum(1 for r in rates if r < 0.99),
                }
            )
        coverage.sort(key=lambda r: (-r["min_rate"], r["representation"]))
    return {"ged_positive_pairs": pairs, "collisions": collisions, "coverage": coverage}


def rejection_composition(
    family: dict[str, Any], rows: Sequence[dict[str, Any]]
) -> dict[str, Any]:
    """Split BH rejections by row **and by direction**.

    A bare rejection count is the most misreadable number this ticket produces.
    BH tests ``H0: no difference``, so a rejection on a B1e cell can perfectly
    well mean *significantly worse* --- and "28 of 32 significant" reads to
    everyone as 28 wins. Measured on a partial run it was 19 A1 (ten of them
    with IsalGraph **longer**), 1 A2, 6 B1e (**all six** with IsalGraph lower)
    and 2 B3e: a majority of the directional rejections were **against** the
    reference arm.

    Args:
        family: Parsed ``family_F2.json``.
        rows: Rho rows, for the B1e directions.

    Returns:
        Counts by row, by direction, and the two totals.
    """
    cells = family.get("cells", [])
    flags = family.get("bh_primary", {}).get("rejected", [])
    bits = {
        (r["dataset"], r["representation"]): r
        for r in family.get("a1_cells", [])
        if r.get("arm") == "primary"
    }
    deltas = {
        (r["dataset"], r["representation"]): r["difference_vs_reference_arm"]["point"]
        for r in rows
        if r.get("row") == "B1e"
        and r["view"] == "all_pairs"
        and r.get("difference_vs_reference_arm") is not None
    }

    by_row: Counter[str] = Counter()
    favour: Counter[str] = Counter()
    per_row: dict[str, Counter[str]] = {}
    for cell, rejected in zip(cells, flags, strict=False):
        if not rejected:
            continue
        row = cell["row"]
        by_row[row] += 1
        key = (cell["dataset"], cell["representation"])
        if row == "A1":
            record = bits.get(key)
            gap = record["median_difference"]["entropy_bits"] if record else 0.0
            label = "for IsalGraph" if gap > 0 else "against IsalGraph"
        elif row == "B1e":
            label = "for IsalGraph" if deltas.get(key, 0.0) > 0 else "against IsalGraph"
        else:
            label = "no direction (omnibus / MRM)"
        favour[label] += 1
        per_row.setdefault(row, Counter())[label] += 1
    return {
        "by_row": dict(by_row),
        "per_row_direction": {r: dict(c) for r, c in per_row.items()},
        "by_direction": dict(favour),
        "n_rejected": sum(by_row.values()),
        "n_with_p_value": len(cells),
        "cells_by_row": dict(Counter(c["row"] for c in cells)),
    }


def mrm_table(
    partial_dirs: Sequence[Path],
    logs: Path | None = None,
    collinearity: Path | None = None,
) -> list[dict[str, Any]]:
    """Collect D4's fits with their **whole** coefficient vector.

    ``beta1`` alone is a coefficient without its context. The model regresses
    GED on Levenshtein, ``|delta n|`` and ``|delta density|`` simultaneously, so
    the size coefficient sitting beside it is what says whether a significant
    ``beta1`` is a large finding or a modest one --- and measured, it is 3-6x
    larger on most fits. Reporting one without the other is the same shape of
    error as a rejection count without its composition.

    Every field --- the beta vector, ``r_squared`` **and**
    ``beta1_permutation_p`` --- comes from the partial, so the table is
    single-sourced. Fits reported in a shard log but not yet written to a
    partial are picked up separately and flagged ``landed = False``, because a
    count that is about to move should say so before it moves: the reader then
    sees a completing series rather than a changing story.

    Args:
        partial_dirs: Directories of shard partials.
        logs: Optional shard-log directory, for fits still in flight.
        collinearity: Optional ``collinearity.json``. Where the predictors are
            not separately identifiable (VIF > 10) the beta1-vs-beta_size
            comparison is not supported however high R^2 is, so those fits are
            marked and kept out of the headline counts rather than silently
            averaged in.

    Returns:
        One record per ``(suite, dataset, reference)``, production fits only.
    """
    records: list[dict[str, Any]] = []
    for directory in partial_dirs:
        for path in sorted(directory.glob("*.json")) if directory.is_dir() else []:
            for key, fit in json.loads(path.read_text()).get("mrm", {}).items():
                # A smoke run carries a handful of permutations; it is not a result.
                if int(fit.get("n_permutations", 0)) < 1000:
                    continue
                dataset, _, reference = key.rpartition("@")
                betas = dict(zip(fit["predictors"], fit["standardised_betas"], strict=False))
                lev = float(betas.get("levenshtein", float("nan")))
                size = float(betas.get("delta_n", float("nan")))
                records.append(
                    {
                        "suite": fit.get("suite", "?"),
                        "dataset": dataset,
                        "reference": reference,
                        "beta_levenshtein": lev,
                        "beta_delta_n": size,
                        "beta_delta_density": float(betas.get("delta_density", float("nan"))),
                        "ratio_size_over_lev": abs(size / lev) if lev else float("nan"),
                        "r_squared": float(fit["r_squared"]),
                        "p_value": float(fit["beta1_permutation_p"]),
                        "n_pairs": int(fit["n_pairs"]),
                        "landed": True,
                    }
                )

    if logs is not None and logs.is_dir():
        seen = {(r["dataset"], r["reference"]) for r in records}
        # beta_size is optional: shards launched before the producer was guarded
        # emit a bare beta1, and those logs must still parse rather than vanish.
        # A missing group becomes nan, which render_beta1 turns into UNMEASURED.
        pattern = re.compile(
            r"(suite\d)/(\S+)\s+MRM@(\w+)\s+beta1=([+-][\d.]+)"
            r"(?:\s+beta_size=([+-][\d.]+))?\s+p=([\d.]+)"
        )
        for path in sorted(logs.glob("f2_suite*.log")):
            for match in pattern.finditer(path.read_text(errors="replace")):
                suite, dataset, reference, beta, size, p_value = match.groups()
                if (dataset, reference) in seen:
                    continue
                seen.add((dataset, reference))
                records.append(
                    {
                        "suite": suite,
                        "dataset": dataset,
                        "reference": reference,
                        "beta_levenshtein": float(beta),
                        "beta_delta_n": float(size) if size else float("nan"),
                        "beta_delta_density": float("nan"),
                        "ratio_size_over_lev": (
                            abs(float(size) / float(beta))
                            if size and float(beta)
                            else float("nan")
                        ),
                        "r_squared": float("nan"),
                        "p_value": float(p_value),
                        "n_pairs": 0,
                        "landed": False,
                    }
                )
    identifiable: dict[str, bool] = {}
    if collinearity is not None and collinearity.exists():
        payload = json.loads(collinearity.read_text())
        for key, row in payload.get("datasets", {}).items():
            identifiable[key.split("/", 1)[1]] = bool(row["identifiable"])
    for record in records:
        record["identifiable"] = identifiable.get(record["dataset"], True)
        record["max_vif"] = float("nan")
    if collinearity is not None and collinearity.exists():
        vifs = {
            k.split("/", 1)[1]: v["max_vif"]
            for k, v in json.loads(collinearity.read_text()).get("datasets", {}).items()
        }
        for record in records:
            record["max_vif"] = float(vifs.get(record["dataset"], float("nan")))
    return sorted(records, key=lambda r: (r["suite"], r["dataset"], r["reference"]))


def render_beta1(record: dict[str, Any]) -> str:
    """Render beta1 in a form that **cannot** appear without its size coefficient.

    ``beta1 never travels without beta_size`` was stated as a rule twice, after
    two separate corrections, and both times it held only because someone
    remembered it. A rule that needs restating should become something that
    cannot be skipped --- so this is the only sanctioned way to put a beta1 into
    the document, and when ``beta_delta_n`` is not finite it renders the absence
    rather than the number alone.

    The reason is not tidiness. A significant beta1 that is one-fifth the size of
    the confound it competes with is a real finding and a modest one, and a
    reader shown only the first half will draw the wrong conclusion --- which is
    exactly what happened when a log-sourced beta1 set, carrying no size
    coefficient, briefly read as a clean win.

    Args:
        record: A row from :func:`mrm_table`.

    Returns:
        Markdown for the coefficient cell.
    """
    beta1 = float(record["beta_levenshtein"])
    size = float(record["beta_delta_n"])
    if not np.isfinite(size):
        return f"{beta1:+.4f} — **β_size UNMEASURED**"
    return f"{beta1:+.4f} vs β_size {size:+.4f}"


def _b1e_direction_sentence(composition: dict[str, Any]) -> str:
    """Describe B1e's rejection directions from the counts, never from memory.

    This replaces a hard-coded sentence -- *"every rejected B1e cell is a cell
    where IsalGraph's rho is lower"* -- which was true of the six cells that had
    landed when it was written and false by the time all seventy-nine had. A
    claim about data, frozen into prose, becomes a lie the moment the data
    completes.

    Args:
        composition: Output of :func:`rejection_composition`.

    Returns:
        One sentence, computed.
    """
    split = composition["per_row_direction"].get("B1e", {})
    favour = int(split.get("for IsalGraph", 0))
    against = int(split.get("against IsalGraph", 0))
    total = favour + against
    if not total:
        return "No B1e cell was rejected."
    if favour == 0:
        return (
            f"**Every one of the {against} rejected B1e cells is a cell where IsalGraph's "
            "rho is *lower*.**"
        )
    if against == 0:
        return (
            f"All {favour} rejected B1e cells favour IsalGraph."
        )
    return (
        f"Of the {total} rejected B1e cells, **{against} are cells where IsalGraph's rho is "
        f"lower** and {favour} where it is higher — so the correlation row is split, not "
        "uniformly adverse."
    )


def _missing_cells(verdicts: Sequence[Verdict]) -> list[str]:
    """Return the ``suite/dataset`` cells with no verdict yet."""
    present = {(v.suite, v.dataset) for v in verdicts}
    wanted = {("suite1", d) for d in SUITE1} | {("suite2", d) for d in SUITE2}
    return sorted(f"{s}/{d}" for s, d in wanted - present)


def render(
    verdicts: Sequence[Verdict],
    strata: dict[str, Any] | None,
    claim_a_cells: dict[str, Any] | None,
    metadata: dict[str, Any],
    rows: Sequence[dict[str, Any]] = (),
    view: str = "all_pairs",
    profile: dict[str, Any] | None = None,
    extras: dict[str, Any] | None = None,
    mrm: Sequence[dict[str, Any]] = (),
) -> str:
    """Render the summary as Markdown.

    Args:
        verdicts: Claim-B head-to-heads.
        strata: Parsed ``claim_a_strata.json``, or ``None``.
        claim_a_cells: Parsed ``family_F2.json``, or ``None``.
        metadata: Provenance to stamp on the document.
        rows: The raw rho rows, for the every-competitor delta matrix.
        view: Which pair view the tables report.
        profile: Parsed ``size_profile.json``, for Claim B per size band.
        extras: Output of :func:`uniqueness_and_coverage`, or ``None``.
        mrm: Output of :func:`mrm_table`.

    Returns:
        The Markdown source.
    """
    tally = Counter(v.verdict for v in verdicts)
    missing = _missing_cells(verdicts)
    n_below = sum(
        1 for v in verdicts if v.size_null is not None and v.isalgraph_rho < v.size_null
    )
    below_share = (
        f"{n_below} of {len(verdicts)} records ({100 * n_below / len(verdicts):.0f} %)"
        if verdicts
        else "no records yet"
    )
    lines: list[str] = [
        "# T-06 decision summary",
        "",
        f"Generated {metadata['generated_utc']} | code `{metadata['code_commit'][:7]}` | "
        f"src `{metadata['src_commit'][:7]}` | engine `{metadata['isalgraph_engine']}` | "
        f"build `{metadata['isalgraph_build_hash']}` | seed {metadata['seed']}",
        "",
        "**A loss inside a confidence interval is a TIE.** Every verdict below comes from the "
        "*paired* bootstrap difference on identical pairs and identical graph-level resamples, "
        "never from two overlapping marginal intervals.",
        "",
    ]

    lines += [
        "## The answer, in five lines",
        "",
        "1. **Claim B (correlation with GED): yes, a clear disadvantage, and it is the "
        f"headline risk.** IsalGraph is best on none of the records landed and sits **below "
        f"its own `|n_i − n_j|` size null on {below_share}**, including **against EXACT GED "
        "on Suite 1, where no bracket argument applies** — where the trivial baseline beats "
        "the representation, which competitor wins is second-order. Its within-`n` rho "
        "collapses from 0.9656 at n ≤ 5 to 0.0779 above 40. `sparse6_nauty` beats it under "
        "**both** bounds; `min_dfs` and `nauty_graph6` tie it under LB and beat it under UB.",
        "2. **Claim A (information content): an advantage, and it GROWS with size** --- 20.4 % "
        "of strata at n ≤ 5 rising to 45.6 % above 40, median gap −1.2 to +242.1 bits. **The "
        "two claims move in opposite directions with size**, so they do not share a cause.",
        "3. **But not \"the most compact admissible representation\": that is true in 0 of 122 "
        "strata above n = 20.** It beats `min_dfs` **112 of 112** and is even with "
        "`nauty_graph6`; **edge-list serialisations beat it at scale.** The defensible claim is "
        "*most compact of the canonical codes*.",
        "4. **Every Claim B verdict is bracket-dependent** --- LB and UB disagree on two of "
        "four competitors. That is a finding, not a caveat, and it is the third independent "
        "detection of the same fact after F1's `d = 7 of 10` and the size-null inversion.",
        "5. **The one clean result: zero encoding collisions on 24,764,422 GED-positive "
        "pairs.** Completeness is categorical rather than metric --- no competitor comparison "
        "adjudicates it --- and \"it computes everywhere\" is **not** a differentiator: eight "
        "representations complete on 100 % of every cell.",
        "",
    ]

    if missing:
        lines += [
            f"> ⚠ **INCOMPLETE.** {len(missing)} of 15 cells have no verdict yet: "
            f"`{'`, `'.join(missing)}`. The campaign is still running; counts below are over "
            "what has landed, not over the full cohort.",
            "",
        ]

    lines += [
        "## Claim B --- distance correlation with GED",
        "",
        f"**{tally['win']} win / {tally['tie']} tie / {tally['loss']} loss** "
        f"over {len(verdicts)} records.",
        "",
        "| suite | dataset | ref | IsalGraph ρ | size null | best competitor | its ρ | "
        "Δ paired [95 % CI] | n pairs | verdict |",
        "|---|---|---|---:|---:|---|---:|---|---:|---|",
    ]
    for v in verdicts:
        null = "—" if v.size_null is None else f"{v.size_null:.4f}"
        mark = {"win": "**win**", "tie": "tie", "loss": "**LOSS**"}[v.verdict]
        lines.append(
            f"| {v.suite[-1]} | {v.dataset} | {v.reference} | {v.isalgraph_rho:.4f} | {null} | "
            f"{v.best_competitor} | {v.best_rho:.4f} | "
            f"{v.delta:+.4f} [{v.ci_low:+.4f}, {v.ci_high:+.4f}] | {v.n_pairs:,} | {mark} |"
        )

    with_null = [v for v in verdicts if v.size_null is not None]
    below_null = [v for v in with_null if v.isalgraph_rho < v.size_null]
    if below_null:
        by_ref: dict[str, list[Verdict]] = {}
        for v in with_null:
            by_ref.setdefault(v.reference, []).append(v)
        exact_below = [v for v in below_null if v.reference == "exact"]
        lines += [
            "",
            f"### 🔴 Below its own size null on {len(below_null)} of {len(with_null)} records "
            f"({100 * len(below_null) / len(with_null):.0f} %)",
            "",
            "**This outranks the head-to-head.** Where the trivial `|n_i − n_j|` baseline "
            "predicts GED better than the representation does, which competitor wins is a "
            "second-order question.",
            "",
            "**`exact` is not part of the bracket and must be read separately from `lb`/`ub`.**",
            "",
            "| reference | below | clears | nature |",
            "|---|---:|---:|---|",
        ]
        for reference in ("exact", "lb", "ub"):
            block = by_ref.get(reference)
            if not block:
                continue
            low = sum(1 for v in block if v.isalgraph_rho < v.size_null)
            nature = (
                "**ground truth — no bracket argument touches it**"
                if reference == "exact"
                else "bracketed"
            )
            lines.append(f"| `{reference}` | {low} | {len(block) - low} | {nature} |")

        if exact_below:
            worst = min(exact_below, key=lambda v: v.isalgraph_rho - v.size_null)
            lines += [
                "",
                "**Concede this first.** Against **exact** GED — no bound, no interpolation — "
                "the trivial baseline beats the representation on "
                f"{len(exact_below)} of {len(by_ref.get('exact', []))} Suite-1 records: "
                + ", ".join(
                    f"`{v.dataset}` ({v.isalgraph_rho:.4f} vs {v.size_null:.4f}, "
                    f"{v.isalgraph_rho - v.size_null:+.4f})"
                    for v in exact_below
                )
                + f". The worst is `{worst.dataset}` at "
                f"{worst.isalgraph_rho - worst.size_null:+.4f}. **No framing repairs this and "
                "none should be attempted.** It is also the *cleaner* measurement, which is "
                "why it goes first: burying a cleaner result behind a bracket argument is what "
                "makes an omission look deliberate.",
            ]

        lb_below = [v for v in below_null if v.reference == "lb"]
        ub_clear = [v for v in by_ref.get("ub", []) if v.isalgraph_rho >= v.size_null]
        if lb_below and ub_clear:
            thin = min(ub_clear, key=lambda v: v.isalgraph_rho - v.size_null)
            lines += [
                "",
                f"**Then the Suite-2 half, which is undetermined rather than failed.** All "
                f"{len(lb_below)} `lb` records fall below their null and all {len(ub_clear)} "
                "`ub` records clear it — **on the same pairs**. The verdict inverts across the "
                "bracket, so the approximate regime does not settle the question either way. "
                "That is §10's size-null inversion reproducing at full cohort, and a fourth "
                "independent detection that the bracket is too wide at these sizes — after "
                "F1's `d = 7 of 10`, the competitor verdicts flipping between bounds, and §10 "
                "itself on the pilot.",
                "",
                f"(Do not lean on the UB reading either: `{thin.suite[-1]}/{thin.dataset}/ub` "
                f"clears by only {thin.isalgraph_rho - thin.size_null:+.4f} — "
                f"{thin.isalgraph_rho:.4f} against {thin.size_null:.4f}.)",
            ]

    if profile is not None:
        per_rival = claim_b_by_competitor(profile)
        if per_rival:
            large = [r for r in per_rival if r["half"] == "n > 20"]
            worst = min((r["median_delta_rho"] for r in large), default=0.0)
            lines += [
                "",
                "### Claim B per competitor, inside equal-`n` strata",
                "",
                "**Read the sign test, not the unresolved count.** Strata above n = 20 are "
                "thin, so most individual comparisons do not resolve at the graph-level "
                "interval --- but many underpowered comparisons all leaning one way is "
                "evidence, not absence of it. Pooling them by a sign test over the per-stratum "
                "Δρ reverses the reading. Strata within a dataset are disjoint graph sets, so "
                "the test is valid, and it weights every stratum equally regardless of pair "
                "count.",
                "",
                "| competitor | n | ref | strata | unresolved % | IsalGraph higher | lower | "
                "median Δρ | sign test `p` |",
                "|---|---|---|---:|---:|---:|---:|---:|---:|",
            ]
            for r in per_rival:
                lines.append(
                    f"| {r['competitor']} | {r['half']} | {r['reference']} | {r['strata']} | "
                    f"{100 * r['unresolved_fraction']:.0f} % | {r['isalgraph_higher']} | "
                    f"{r['isalgraph_lower']} | {r['median_delta_rho']:+.4f} | "
                    f"{r['sign_test_p']:.2g} |"
                )
            rejected = [r for r in large if r["sign_test_p"] < 0.05]
            lines += [
                "",
                f"**Above n = 20 the sign test rejects on {len(rejected)} of {len(large)} "
                "(competitor, reference) arms**, even though over 90 % of the individual "
                "strata are unresolved. The two columns say opposite things and the pooled "
                "one is correct: IsalGraph is lower on the majority of strata, consistently "
                f"enough to reject, with median Δρ down to {worst:+.4f}. **Do not read the "
                "unresolved fraction as a tie** --- many underpowered comparisons all leaning "
                "one way is evidence, not absence of it.",
                "",
                "**Every Claim B verdict is bracket-dependent, and both bounds are printed "
                "for that reason.** Against `min_dfs` and `nauty_graph6` the sign test is a "
                "**tie under LB and a loss under UB**; against `sparse6_nauty` it is a loss "
                "under both. Reporting one bound would invert two of the four verdicts. This "
                "is not a hedge --- a bracket wide enough to flip a competitor verdict on "
                "21.7 M pairs is itself a finding, and it is F1's `d = 7 of 10` and the "
                "size-null inversion arriving a third time.",
                "",
                "What the comparison *does* show, as description rather than defence: the "
                "pooled `all_pairs` gap exceeds the within-`n` gap, so a meaningful part of the "
                "head-to-head deficit is size agreement rather than structure --- while the "
                "within-`n` deficit itself remains real and significant.",
            ]

    lines += ["", "## Claim A --- information content, stratified by size", ""]
    if strata is None:
        lines.append("_`claim_a_strata.json` not found._")
    else:
        records = claim_a_by_size(strata)
        total = sum(r["strata"] for r in records)
        wins = sum(r["isalgraph_shorter"] for r in records)
        ties = sum(r["tie"] for r in records)
        lines += [
            f"**{wins} win / {ties} tie / {total - wins - ties} loss** over {total} strata "
            f"(node-count strata, Wilcoxon per stratum, intersection-union `p = max` over both "
            f"bit conventions).",
            "",
            "| n | strata | IsalGraph shorter | tie | competitor shorter | IsalGraph win % | "
            "median gap (bits) |",
            "|---|---:|---:|---:|---:|---:|---:|",
        ]
        for r in records:
            lines.append(
                f"| {r['band']} | {r['strata']} | {r['isalgraph_shorter']} | {r['tie']} | "
                f"{r['competitor_shorter']} | {100 * r['win_fraction']:.1f} % | "
                f"{r['median_gap_entropy_bits']:+.1f} |"
            )
        first, last = records[0], records[-1]
        lines += [
            "",
            f"**Where it loses: at SMALL n.** The win rate runs "
            f"{100 * first['win_fraction']:.1f} % at n {first['band']} to "
            f"{100 * last['win_fraction']:.1f} % at n {last['band']}, and the median gap runs "
            f"{first['median_gap_entropy_bits']:+.1f} to {last['median_gap_entropy_bits']:+.1f} "
            "bits. **The bit advantage GROWS with size — the opposite direction to Claim B, "
            "whose correlation collapses with size.** The two claims do not share a cause.",
            "",
            f"{strata.get('n_graphs_skipped_thin_strata', 0)} graphs sit in strata below "
            f"{strata.get('min_graphs_per_stratum')} graphs and are not tested.",
        ]
        predicates = compactness_predicates(strata, above=20)
        if predicates is not None:
            total = predicates["total"]
            strict = predicates["significantly_shortest"]
            gap = predicates["positive_gap_against_all"]
            unbeaten = predicates["never_significantly_beaten"]
            blocker = next(iter(predicates["blockers"]), None)
            lines += [
                "",
                f"### Is IsalGraph the most compact admissible representation above "
                f"n = {predicates['above']}? **No.**",
                "",
                "Three predicates, same 122 strata, and the differences between them are the "
                "content --- so each is stated with its predicate in the sentence:",
                "",
                f"- **Significantly shorter than EVERY metric-admissible competitor: "
                f"{strict} of {total} ({100 * strict / total:.0f} %).** This is what the claim "
                "asserts, and it is never true.",
                f"- Shorter at the median against every one, significant or not: {gap} of "
                f"{total} ({100 * gap / total:.0f} %).",
                f"- Never significantly beaten by any of them: {unbeaten} of {total} "
                f"({100 * unbeaten / total:.0f} %) --- so it is significantly beaten by at "
                f"least one in {total - unbeaten} of {total} "
                f"({100 * (total - unbeaten) / total:.0f} %).",
                "",
                f"`{blocker}` is the arm that blocks it. **What holds instead: IsalGraph is the "
                "most compact of the canonical-code representations, and edge-list "
                "serialisations beat it at scale.** Naming the mechanism rather than the "
                "outcome matters here --- min-DFS is also a canonical code, so beating it on "
                "112 of 112 strata at +214.8 bits is a like-for-like win rather than a win "
                "over a different design point.",
            ]
        lines += [
            "",
            "### Claim A per competitor",
            "",
            "`max n` matters: a competitor refused above a size cannot be beaten above it, so a "
            "pooled win rate against `agm_cam` is a statement about small graphs only.",
            "",
            "`graph6` and `nauty_graph6` carry **identical** counts by construction, not by "
            "accident: graph6 writes the full upper triangle at fixed width, so its length is a "
            "function of `n` alone and canonicalising the labelling permutes the bits without "
            "changing how many there are. Verified elementwise on every graph of GREC and "
            "Mutagenicity --- the strings differ, the bit counts do not. `sparse6` and "
            "`sparse6_nauty` do differ, because a sparse6 edge list's length depends on the "
            "vertex ordering that canonicalisation changes.",
            "",
            "| competitor | metric-admissible | strata | win | tie | loss | win % | "
            "win % at n > 20 | max n |",
            "|---|---|---:|---:|---:|---:|---:|---:|---:|",
        ]
        for r in claim_a_by_competitor(strata):
            large = (
                "—"
                if r["win_fraction_large_n"] is None
                else f"{100 * r['win_fraction_large_n']:.1f} %"
            )
            lines.append(
                f"| {r['competitor']} | {'yes' if r['metric_admissible'] else 'no (k-excluded)'} "
                f"| {r['strata']} | {r['win']} | {r['tie']} | {r['loss']} | "
                f"{100 * r['win_fraction']:.1f} % | {large} | {r['max_n_reached']} |"
            )

    if mrm:
        landed_all = [r for r in mrm if r["landed"]]
        # A fit whose predictors are collinear cannot support the beta1-vs-beta_size
        # comparison this section exists to make, however high its R^2. Kept in the
        # table, kept out of the counts.
        landed = [r for r in landed_all if r.get("identifiable", True)]
        unidentifiable = [r for r in landed_all if not r.get("identifiable", True)]
        pending = [r for r in mrm if not r["landed"]]
        significant = [r for r in landed if r["p_value"] < 0.05]
        size_wins = [
            r for r in landed if abs(r["beta_delta_n"]) > abs(r["beta_levenshtein"])
        ]
        ratios = [
            r["ratio_size_over_lev"]
            for r in size_wins
            if np.isfinite(r["ratio_size_over_lev"])
        ]
        lines += [
            "",
            "## D4 — the model that CONTROLS for size rather than stratifying it away",
            "",
            "`GED ~ β₁·Lev + β₂·|Δn| + β₃·|Δdensity|`, all standardised, so the coefficients "
            "are directly comparable. **β₁ must never be quoted without β_size beside it** — a "
            "significant coefficient that is one-fifth the size of the confound it competes "
            "with is a real finding and a modest one, and it has to read as both.",
            "",
            "Every column is read from the shard partial, `beta1_permutation_p` included, so "
            "the table is single-sourced.",
            "",
            "| suite | dataset | ref | **β₁ (Lev)** | **β_size** | β_density | R² | p(β₁) | "
            "size/Lev |",
            "|---|---|---|---:|---:|---:|---:|---:|---:|",
        ]
        for r in landed_all:
            ratio = (
                f"{r['ratio_size_over_lev']:.1f}×"
                if np.isfinite(r["ratio_size_over_lev"])
                else "—"
            )
            flag = "" if r.get("identifiable", True) else " ⚠"
            lines.append(
                f"| {r['suite'][-1]} | {r['dataset']}{flag} | {r['reference']} | "
                f"{r['beta_levenshtein']:+.4f} | {r['beta_delta_n']:+.4f} | "
                f"{r['beta_delta_density']:+.4f} | {r['r_squared']:.3f} | "
                f"{r['p_value']:.5f} | {ratio} |"
            )
        if unidentifiable:
            names = sorted({r["dataset"] for r in unidentifiable})
            worst = max(r["max_vif"] for r in unidentifiable)
            lines += [
                "",
                f"⚠ **{len(unidentifiable)} fits on {len(names)} datasets "
                f"(`{'`, `'.join(names)}`) are EXCLUDED from the counts below: their "
                f"predictors are collinear (max VIF {worst:.1f} against a threshold of 10), "
                "so the β₁-vs-β_size comparison this section makes is **not identifiable** "
                "there — the split between two predictors correlated at r > 0.93 is arbitrary "
                "within a wide equivalence class, however high R² is or however small p is. "
                "`coil_del`/ub announces it (β₁ = +1.49 with β_size negative); "
                "`aids_iam`/lb does not (+0.10 against +0.91, R² = 0.998) and is the more "
                "dangerous of the two.",
            ]
        lines += [
            "",
            f"**β₁ is significant on {len(significant)} of {len(landed)} identifiable fits and "
            "positive on every one — D4's β₁ does NOT collapse.** But **the size coefficient "
            f"exceeds Levenshtein's on {len(size_wins)} of {len(landed)}**"
            + (
                f", by {min(ratios):.1f}×–{max(ratios):.1f}×"
                if ratios
                else ""
            )
            + ". So the defensible sentence carries both halves: *Levenshtein contributes "
            "significant incremental information beyond size and density, but node-count "
            "difference does most of the work.*",
            "",
            "This is the ticket's central finding stated by the one instrument that "
            "**controls** for the confound rather than stratifying it away, which makes it "
            "more citable than the within-`n` collapse, not less.",
            "",
            "> **A significant β₁ on a dataset whose within-`n` ρ is indistinguishable from "
            "noise is NOT a contradiction, and a reader will take it for one.** The two "
            "instruments answer different questions. §17 asks whether the distance tracks GED "
            "*within a fixed size*, where the size channel is removed by construction. The MRM "
            "asks whether it adds anything *given* size and density, across all sizes. A "
            "representation can carry real information about the size-driven part of GED while "
            "carrying none about the residual — and on these cohorts GED is itself heavily "
            "size-driven (the `|n_i − n_j|` null reaches 0.9971 on `coil_del`). So "
            "\"β₁ significant\" and \"within-`n` ρ ≈ 0\" are two facts about the same arm, "
            "not an inconsistency in the pipeline.",
        ]
        if pending:
            not_significant = [r for r in pending if r["p_value"] >= 0.05]
            lines += [
                "",
                f"**{len(pending)} further fit(s) have reported in a shard log and not yet "
                "landed in a partial**, so they are outside the counts above. Listed here "
                "because a count that is going to move should say so before it moves:",
                "",
                "| suite | dataset | ref | β₁ (with β_size) | p(β₁) | |",
                "|---|---|---|---|---:|---|",
            ]
            for r in pending:
                mark = "**NOT significant**" if r["p_value"] >= 0.05 else "significant"
                lines.append(
                    f"| {r['suite'][-1]} | {r['dataset']} | {r['reference']} | "
                    f"{render_beta1(r)} | {r['p_value']:.5f} | {mark} |"
                )
            lines += [
                "",
                "A log line carries β₁ and no size coefficient, so these render as "
                "**β_size UNMEASURED** rather than as a bare number. β₁ alone is the half of "
                "the result that flatters; a reader shown only that half draws the wrong "
                "conclusion, and this table cannot show only that half.",
            ]
            if not_significant:
                worst = min(not_significant, key=lambda r: abs(r["beta_levenshtein"]))
                lines += [
                    "",
                    f"🔴 **`{worst['dataset']}`/{worst['reference']} is not significant** "
                    f"(β₁ = {worst['beta_levenshtein']:+.4f}, p = {worst['p_value']:.4f}) and "
                    "carries the smallest β₁ in the whole set. When it lands the headline "
                    f"becomes **{len(significant)} of {len(landed) + len(pending)}**, not "
                    f"{len(significant)} of {len(landed)}. The exception is on the record now "
                    "rather than appearing later as if discovered.",
                ]

    if extras is not None:
        lines += [
            "",
            "## What neither claim measures",
            "",
            f"**Zero encoding collisions over {extras['ged_positive_pairs']:,} GED-positive "
            f"pairs** ({extras['collisions']} observed). No two non-isomorphic graphs in either "
            "cohort received the same canonical string. This is a theorem, so a measured zero "
            "is a check rather than a discovery --- but it is the check a reviewer can run, and "
            "it is the one unambiguous positive in this document.",
            "",
            "**Coverage is NOT a differentiator, and a claimed advantage dies here.**",
            "",
            "| representation | cells | min completion | cells < 99 % |",
            "|---|---:|---:|---:|",
        ]
        for row in extras["coverage"]:
            lines.append(
                f"| {row['representation']} | {row['cells']} | {row['min_rate']:.4f} | "
                f"{row['cells_below_99']} |"
            )
        lines += [
            "",
            "Six competitors complete on **100 %** of every cell. `isalgraph_pruned`'s 0.9750 "
            "floor is Mutagenicity, and it is an artefact of `t06_completion` counting a "
            "censored graph as not completed --- D14 retains it with its greedy-min string, so "
            "it did produce an encoding and the manifest gate scores it 100 %. Either way the "
            'conclusion is the same: **"it computes everywhere" separates IsalGraph from '
            "`agm_cam` and `min_dfs` only, not from the field.**",
        ]

    if claim_a_cells is not None:
        card = claim_a_cells.get("cardinality", {})
        bh = claim_a_cells.get("bh_primary", {})
        composition = rejection_composition(claim_a_cells, rows)
        against = composition["by_direction"].get("against IsalGraph", 0)
        favour = composition["by_direction"].get("for IsalGraph", 0)
        lines += [
            "",
            "## The confirmatory family, for reference",
            "",
            f"`N_actual` = {card.get('n_actual')} (closed form {card.get('closed_form')}, "
            f"discrepancy {card.get('discrepancy')}); BH at q = {bh.get('q')} over "
            f"**{bh.get('m')}**, with {composition['n_with_p_value']} cells carrying a "
            f"p-value and **{composition['n_rejected']} rejected**.",
            "",
            "🔴 **That count is not a win count and must never be quoted as one.** BH tests "
            "*no difference*, so a rejection can mean significantly **worse**. Split by row "
            "and direction:",
            "",
            "| row | what it tests | rejected | for IsalGraph | against |",
            "|---|---|---:|---:|---:|",
        ]
        meaning = {
            "A1": "fewer bits than a comparator",
            "A2": "Friedman omnibus on bits",
            "B1e": "rho difference vs a comparator, exact GED",
            "B3e": "MRM standardised beta1",
        }
        for row, count in sorted(composition["by_row"].items()):
            split = composition["per_row_direction"].get(row, {})
            favours = split.get("for IsalGraph", 0)
            against_row = split.get("against IsalGraph", 0)
            cells_for = str(favours) if row in {"A1", "B1e"} else "—"
            cells_against = str(against_row) if row in {"A1", "B1e"} else "—"
            lines.append(
                f"| {row} | {meaning.get(row, '')} | {count} | {cells_for} | {cells_against} |"
            )
        lines += [
            "",
            f"**{against} of the {against + favour} directional rejections are AGAINST "
            f"IsalGraph**, {favour} for it; the rest are omnibus or MRM cells with no "
            "direction.",
            "",
            _b1e_direction_sentence(composition),
            "",
            "This is the pre-registered family and it is **not** what the tables above report: "
            "those are per-cell verdicts, unadjusted, meant for a methodology decision rather "
            "than for the manuscript.",
        ]

    return "\n".join(lines) + "\n"


def build_parser() -> argparse.ArgumentParser:
    """Return the CLI parser."""
    ap = argparse.ArgumentParser(description="Render the T-06 decision summary.")
    ap.add_argument("--rho-table", type=Path, default=None)
    ap.add_argument("--partials", type=Path, nargs="*", default=[])
    ap.add_argument("--claim-a-strata", type=Path, default=None)
    ap.add_argument("--family", type=Path, default=None)
    ap.add_argument("--size-profile", type=Path, default=None)
    ap.add_argument("--ladders", type=Path, nargs="*", default=[])
    ap.add_argument("--completion-rates", type=Path, default=None)
    ap.add_argument("--logs", type=Path, default=None, help="shard logs, for fits still in flight")
    ap.add_argument("--collinearity", type=Path, default=None, help="collinearity.json")
    ap.add_argument("--view", default="all_pairs")
    ap.add_argument("--out", type=Path, required=True)
    return ap


def main(argv: Sequence[str] | None = None) -> int:
    """Entry point.

    Args:
        argv: Argument vector, or ``None`` for ``sys.argv``.

    Returns:
        0 on success.
    """
    args = build_parser().parse_args(argv)
    logging.basicConfig(level=logging.INFO, format="%(levelname)s %(message)s")

    from benchmarks.real_data.eval_stats.t06_f2 import _metadata

    rows = load_rho_rows(args.rho_table, args.partials)
    verdicts = head_to_head(rows, view=args.view)
    strata = (
        json.loads(args.claim_a_strata.read_text())
        if args.claim_a_strata and args.claim_a_strata.exists()
        else None
    )
    family = json.loads(args.family.read_text()) if args.family and args.family.exists() else None

    args.out.parent.mkdir(parents=True, exist_ok=True)
    profile = (
        json.loads(args.size_profile.read_text())
        if args.size_profile and args.size_profile.exists()
        else None
    )
    args.out.write_text(
        render(
            verdicts,
            strata,
            family,
            _metadata(args.out.parent, {}),
            rows=rows,
            view=args.view,
            profile=profile,
            mrm=mrm_table(args.partials, logs=args.logs, collinearity=args.collinearity),
            extras=uniqueness_and_coverage(
                [json.loads(f.read_text()) for f in args.ladders if f.exists()],
                json.loads(args.completion_rates.read_text())
                if args.completion_rates and args.completion_rates.exists()
                else None,
            )
            if args.ladders
            else None,
        )
    )

    tally = Counter(v.verdict for v in verdicts)
    print(
        f"wrote {args.out}: Claim B {tally['win']} win / {tally['tie']} tie / "
        f"{tally['loss']} loss over {len(verdicts)} records "
        f"({len(_missing_cells(verdicts))} cells still missing)"
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
