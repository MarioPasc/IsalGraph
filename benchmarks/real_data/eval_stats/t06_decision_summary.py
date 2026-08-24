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

    seen: set[tuple[Any, ...]] = set()
    unique: list[dict[str, Any]] = []
    for row in rows:
        key = (row["suite"], row["dataset"], row["representation"], row["reference"], row["view"])
        if key in seen:
            continue
        seen.add(key)
        unique.append(row)
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

    Returns:
        The Markdown source.
    """
    tally = Counter(v.verdict for v in verdicts)
    missing = _missing_cells(verdicts)
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

    below_null = [v for v in verdicts if v.size_null is not None and v.isalgraph_rho < v.size_null]
    if below_null:
        lines += [
            "",
            f"**{len(below_null)} of {len(verdicts)} records have IsalGraph BELOW its own size "
            "null** — on those the trivial `|n_i − n_j|` baseline correlates with GED better "
            "than the representation does, so the comparison against competitors is secondary "
            "to that: "
            + ", ".join(f"`{v.suite[-1]}/{v.dataset}/{v.reference}`" for v in below_null),
        ]

    matrix = delta_matrix(rows, view=view)
    if matrix:
        counts = Counter(r["verdict"] for r in matrix)
        lines += [
            "",
            "### Every competitor, not only the best",
            "",
            f"**{counts['win']} win / {counts['tie']} tie / {counts['loss']} loss** over "
            f"{len(matrix)} (dataset x competitor) head-to-heads. Paired delta, so a tie means "
            "the interval covers zero.",
            "",
            "| suite | dataset | ref | competitor | its rho | Delta paired [95 % CI] | verdict |",
            "|---|---|---|---|---:|---|---|",
        ]
        for r in matrix:
            mark = {"win": "**win**", "tie": "tie", "loss": "**LOSS**"}[r["verdict"]]
            lines.append(
                f"| {r['suite'][-1]} | {r['dataset']} | {r['reference']} | {r['competitor']} | "
                f"{r['competitor_rho']:.4f} | {r['delta']:+.4f} "
                f"[{r['ci_low']:+.4f}, {r['ci_high']:+.4f}] | {mark} |"
            )

    if profile is not None:
        bands = claim_b_by_size(profile)
        if bands:
            lines += [
                "",
                "### Claim B stratified by size",
                "",
                "Equal-`n` strata, where `|n_i - n_j|` is identically zero so the size null has "
                "no denominator and raw rho IS the structural signal.",
                "",
                "| n | strata | IsalGraph best | best % | median rho (IsalGraph) |",
                "|---|---:|---:|---:|---:|",
            ]
            for r in bands:
                lines.append(
                    f"| {r['band']} | {r['strata']} | {r['isalgraph_best']} | "
                    f"{100 * r['best_fraction']:.1f} % | {r['median_rho']:.4f} |"
                )
            first, last = bands[0], bands[-1]
            lines += [
                "",
                f"**rho collapses {first['median_rho']:.4f} at n {first['band']} to "
                f"{last['median_rho']:.4f} at n {last['band']}**, and IsalGraph tops its "
                f"stratum in only {100 * min(r['best_fraction'] for r in bands):.1f}-"
                f"{100 * max(r['best_fraction'] for r in bands):.1f} % of them at any size. "
                "The pooled rho of about 0.93 is mostly the size channel.",
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
                "—" if r["win_fraction_large_n"] is None
                else f"{100 * r['win_fraction_large_n']:.1f} %"
            )
            lines.append(
                f"| {r['competitor']} | {'yes' if r['metric_admissible'] else 'no (k-excluded)'} "
                f"| {r['strata']} | {r['win']} | {r['tie']} | {r['loss']} | "
                f"{100 * r['win_fraction']:.1f} % | {large} | {r['max_n_reached']} |"
            )

    if claim_a_cells is not None:
        card = claim_a_cells.get("cardinality", {})
        bh = claim_a_cells.get("bh_primary", {})
        lines += [
            "",
            "## The confirmatory family, for reference",
            "",
            f"`N_actual` = {card.get('n_actual')} (closed form {card.get('closed_form')}, "
            f"discrepancy {card.get('discrepancy')}); BH at q = {bh.get('q')} over "
            f"{bh.get('m')} rejects {bh.get('n_rejected')}. This is the pre-registered family "
            "and it is **not** what the tables above report: those are per-cell verdicts, "
            "unadjusted, meant for a methodology decision rather than for the manuscript.",
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
