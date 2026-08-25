"""The two comparison tables, emitted as booktabs LaTeX from the artifacts.

**Table 1 --- properties.** The AE.3 side-by-side comparison the Area Editor
asked for in their own voice, on R1.2b's five axes: uniqueness,
expressiveness, efficiency, scalability and downstream learning. Every cell is
transcribed from a measurement, never asserted: the capability flags come from
each backend's declared ``Capability`` set, ``psi`` and the collision counts
from the T-04a E1/E2 annex, metric admissibility from T-04a's F1--F4 selection
rule, and ``max n`` and the completion floor from T-06's own encoding cells.
The downstream column reads *not evaluated* for every row, which is the honest
answer and is what R1.2b asked to see printed.

**Table 2 --- head to head.** One row per competitor: what the compactness
comparison returns above ``n = 20``, and what the correlation comparison
returns inside equal-``n`` strata under each GED reference. Both bracket ends
are printed on every Claim B row. That is not a hedge: LB and UB disagree on
two of four competitors, and reporting one bound inverts two verdicts
(``T-06-FRAMING.md`` 9.6).

**Two things this module will not do.**

It will not drop a competitor to improve a row. ``sparse6_nauty`` is more
compact *and* better correlated than the instruction string above ``n = 20``,
under both bounds, and omitting the one representation that dominates us is
the most checkable dishonesty available in this paper. It is printed, in bold,
with its own footnote.

It will not print a Claim B verdict without saying which reference produced
it, and it will not print a Claim A win rate against a competitor without that
competitor's ``max n`` beside it -- a comparator that refuses above a size
cannot be beaten above it, so a pooled win rate against ``agm_cam`` is a
statement about graphs of twelve nodes or fewer.
"""

from __future__ import annotations

import argparse
import logging
import statistics
from collections import defaultdict
from pathlib import Path
from typing import Any, Final

from scipy import stats

from benchmarks.real_data.eval_t06_figures import data, design

LOGGER: Final = logging.getLogger(__name__)

#: Per-stratum sign-test level for the Claim B head-to-head.
SIGN_TEST_ALPHA: Final[float] = 0.05

#: Relabelling sensitivity ``psi`` under the primary distance, measured over
#: eleven draws in the T-04a E1 annex. Exactly zero for every canonical
#: representation; the three excluded ones are given as their measured range.
#: Source: ``competitors.md`` 9.4.
PSI_MEASURED: Final[dict[str, str]] = {
    "isalgraph_pruned": "0",
    "isalgraph_canonical": "0",
    "min_dfs": "0",
    "agm_cam": "0",
    "nauty_graph6": "0",
    "sparse6_nauty": "0",
    "wl_subtree": "0",
    "adjacency": "0.07--0.74",
    "graph6": "0.32--1.00",
    "sparse6": "0.54--1.15",
    "size_null": "0",
}

#: Measured collision behaviour. The six complete invariants returned zero
#: collisions and their zero set was identical to the VF2-certified isomorphic
#: set; WL returned 45 false isomorphism certificates in 183,016 comparisons.
#: Source: ``competitors.md`` 9.4 E2.
COLLISIONS_MEASURED: Final[dict[str, str]] = {
    "wl_subtree": r"$2.5\times10^{-4}$",
}

#: What governs encoding cost, where T-06 measured it rather than assumed it.
COST_DRIVER: Final[dict[str, str]] = {
    "isalgraph_pruned": r"$|\mathrm{Aut}(G)|$",
    "isalgraph_canonical": r"$|\mathrm{Aut}(G)|$",
    "min_dfs": "search budget",
    "agm_cam": "$n$ (scope $\\leq 12$)",
}

#: Completion floor over the fifteen T-06 cells, under D14's reading in which
#: a censored graph does carry an encoding. Source: ``REPORT.md``.
COMPLETION_FLOOR: Final[dict[str, float]] = {
    "adjacency": 1.0,
    "graph6": 1.0,
    "isalgraph_canonical": 1.0,
    "isalgraph_pruned": 1.0,
    "nauty_graph6": 1.0,
    "size_null": 1.0,
    "sparse6": 1.0,
    "sparse6_nauty": 1.0,
    "wl_subtree": 1.0,
    "min_dfs": 0.9478,
    "agm_cam": 0.0615,
}

#: The categorical property no serialisation has: the encoding is a program
#: that constructs the graph, executable prefix by prefix. It is where the
#: contribution actually lives and it is not adjudicated by rho or by bits.
EXECUTABLE: Final[frozenset[str]] = frozenset({"isalgraph_pruned", "isalgraph_canonical"})

_YES: Final[str] = r"\checkmark"
_NO: Final[str] = r"--"


def _mark(value: bool) -> str:
    """Return the tick or dash for a boolean cell."""
    return _YES if value else _NO


def _bold_if_majority(win_pct: float) -> str:
    """Render a win percentage, bold when it exceeds half the strata.

    Args:
        win_pct: Percentage of strata the reference arm won.

    Returns:
        The formatted cell.
    """
    return rf"\textbf{{{win_pct:.0f}}}" if win_pct > 50 else f"{win_pct:.0f}"


def sign_test(
    profile: dict[str, Any],
    competitor: str,
    reference: str,
    lo: int,
    hi: int,
) -> dict[str, Any] | None:
    """Sign test over per-stratum rho differences against one competitor.

    Strata within a dataset are disjoint graph sets, so the test is valid, and
    it weights every stratum equally regardless of pair count. Pooling is what
    turns many underpowered per-stratum comparisons into evidence; counting how
    many individually resolve is the wrong summary and reverses the reading
    (``T-06-FRAMING.md`` 8).

    Args:
        profile: Parsed ``size_profile.json``.
        competitor: Backend to compare the reference arm against.
        reference: ``exact``, ``lb`` or ``ub``.
        lo: Smallest node count in the band, inclusive.
        hi: Largest node count in the band, inclusive.

    Returns:
        ``strata``, ``higher``, ``lower``, ``median`` and ``p``, or ``None``
        when the band holds no comparable stratum.
    """
    cells: dict[tuple[str, str, int], dict[str, float]] = defaultdict(dict)
    for row in profile["rows"]:
        if row["rho"] is None or row.get("arm", "primary") != "primary":
            continue
        cells[(row["dataset"], row["reference"], int(row["n"]))][row["representation"]] = float(
            row["rho"]
        )
    deltas: list[float] = []
    for (_, ref, n), by_rep in cells.items():
        if ref != reference or not lo <= n <= hi:
            continue
        if design.REFERENCE_KEY in by_rep and competitor in by_rep:
            deltas.append(by_rep[design.REFERENCE_KEY] - by_rep[competitor])
    if not deltas:
        return None
    higher = sum(1 for d in deltas if d > 0)
    lower = sum(1 for d in deltas if d < 0)
    resolved = higher + lower
    p = float(stats.binomtest(higher, resolved, 0.5).pvalue) if resolved else 1.0
    return {
        "strata": len(deltas),
        "higher": higher,
        "lower": lower,
        "median": statistics.median(deltas),
        "p": p,
    }


def _p(value: float) -> str:
    """Render a p-value in LaTeX, never in Python's ``e`` notation.

    Args:
        value: The p-value.

    Returns:
        Math-mode text.
    """
    if value >= 1e-3:
        return f"$p={value:.3g}$"
    exponent = f"{value:.0e}".split("e")
    return rf"$p={exponent[0]}\times10^{{{int(exponent[1])}}}$"


def verdict(result: dict[str, Any] | None, alpha: float = SIGN_TEST_ALPHA) -> str:
    """Render a sign-test result as a verdict cell.

    Args:
        result: Output of :func:`sign_test`, or ``None``.
        alpha: Significance level.

    Returns:
        ``win``, ``loss`` or ``tie`` in LaTeX, with the p-value; ``n/a`` when
        the comparison does not exist.
    """
    if result is None:
        return r"n/a"
    rendered = _p(float(result["p"]))
    if result["p"] >= alpha:
        return rf"tie \tiny{{{rendered}}}"
    won = result["higher"] > result["lower"]
    word = r"\textbf{win}" if won else "loss"
    return rf"{word} \tiny{{{rendered}}}"


def claim_a_row(
    strata: dict[str, Any],
    competitor: str,
    lo: int,
    hi: int,
    *,
    rule: str = "iut",
) -> dict[str, Any]:
    """Summarise the Claim A comparison against one competitor in a size band.

    Args:
        strata: Parsed ``claim_a_strata.json``.
        competitor: Backend name.
        lo: Smallest node count, inclusive.
        hi: Largest node count, inclusive.
        rule: ``iut`` for the pre-registered intersection--union verdict, or
            ``entropy`` for the marginal Wilcoxon on the entropy bound alone.
            ``competitors.md`` 5 requires both conventions to be reported for
            every method, so the marginal is not a new analysis -- it is the
            half of the locked pair that the conjunction absorbs.

    Returns:
        ``strata``, ``win``, ``loss``, ``win_pct`` and ``median_gap``.

    Raises:
        ValueError: If *rule* is neither ``iut`` nor ``entropy``.
    """
    if rule not in ("iut", "entropy"):
        raise ValueError(f"rule must be 'iut' or 'entropy', got {rule!r}")
    alpha = float(strata.get("alpha", 0.05))
    rows = [
        r for r in strata["rows"] if r["representation"] == competitor and lo <= int(r["n"]) <= hi
    ]

    def call(row: dict[str, Any]) -> str:
        if rule == "iut":
            return str(row["verdict"])
        p, gap = float(row["p_entropy"]), float(row["median_gap_entropy"])
        if p < alpha and gap > 0:
            return "isalgraph_shorter"
        if p < alpha and gap < 0:
            return "competitor_shorter"
        return "tie"

    win = sum(1 for r in rows if call(r) == "isalgraph_shorter")
    loss = sum(1 for r in rows if call(r) == "competitor_shorter")
    gaps = [float(r["median_gap_entropy"]) for r in rows if r["median_gap_entropy"] is not None]
    return {
        "strata": len(rows),
        "win": win,
        "loss": loss,
        "win_pct": 100.0 * win / len(rows) if rows else float("nan"),
        "median_gap": statistics.median(gaps) if gaps else float("nan"),
    }


def properties_table() -> str:
    """Emit table 1: the AE.3 property comparison.

    Returns:
        A complete ``table*`` environment.
    """
    lines = [
        r"\begin{table*}[t]",
        r"\centering",
        r"\caption{Side-by-side comparison of graph representations on R1.2's five axes. "
        r"Every cell is measured, not asserted: $\psi$ is relabelling sensitivity under each "
        r"representation's own primary distance (T-04a~E1, eleven draws); the collision column is "
        r"the false-isomorphism-certificate rate over 183{,}016 comparisons (T-04a~E2); "
        r"\emph{metric} is whether any candidate distance passed the axioms, invariance and "
        r"non-degeneracy filters F1--F4, and the three representations that fail do so at "
        r"$1/50$ relabellings; \emph{bits} is whether a message length is defined at all. "
        r"$n_{\max}$ is the largest graph the backend encoded in our cohorts and "
        r"\emph{compl.} its completion floor over fifteen dataset cells. "
        r"The downstream-learning axis is not evaluated in this work for any representation.}",
        r"\label{tab:representation-properties}",
        r"\footnotesize",
        r"\begin{tabular}{@{}llccccccrrl@{}}",
        r"\toprule",
        r" & & \multicolumn{3}{c}{uniqueness} & \multicolumn{3}{c}{expressiveness} & "
        r"\multicolumn{2}{c}{scalability} & downstream \\",
        r"\cmidrule(lr){3-5}\cmidrule(lr){6-8}\cmidrule(lr){9-10}",
        r"representation & family & $\psi$ & complete & collis. & "
        r"revers. & disconn. & metric & $n_{\max}$ & compl. & learning \\",
        r"\midrule",
    ]
    for rep in design.REPRESENTATIONS:
        collisions = COLLISIONS_MEASURED.get(rep.key, "0" if rep.complete else _NO)
        floor = COMPLETION_FLOOR.get(rep.key)
        name = rf"\textbf{{{rep.tex}}}" if rep.is_ours else rep.tex
        lines.append(
            " & ".join(
                [
                    name,
                    rep.family.value,
                    PSI_MEASURED.get(rep.key, "?"),
                    _mark(rep.complete),
                    collisions,
                    _mark(rep.reversible),
                    _mark(rep.handles_disconnected),
                    _mark(rep.metric_admissible),
                    str(rep.max_n) if rep.max_n else _NO,
                    f"{floor:.3f}" if floor is not None else _NO,
                    "not evaluated",
                ]
            )
            + r" \\"
        )
    lines += [
        r"\midrule",
        r"\multicolumn{11}{@{}p{\textwidth}@{}}{\footnotesize "
        r"\emph{Efficiency}, R1.2's third axis, is the whole of "
        r"Table~\ref{tab:representation-headtohead} and Fig.~\ref{fig:information-content} and is "
        r"not compressible into a column. "
        r"The instruction string is additionally an \emph{executable} encoding: every prefix is a "
        r"valid program constructing a subgraph, so the representation is generative as well as "
        r"descriptive. No other row has this property, and it is not adjudicated by either "
        r"experiment.} \\",
        r"\bottomrule",
        r"\end{tabular}",
        r"\end{table*}",
    ]
    return "\n".join(lines)


def head_to_head_table(
    strata: dict[str, Any],
    profile: dict[str, Any],
    *,
    scope_n: int = design.CLAIM_A_SCOPE_N,
) -> str:
    """Emit table 2: the measured head-to-head on both claims.

    Args:
        strata: Parsed ``claim_a_strata.json``.
        profile: Parsed ``size_profile.json``.
        scope_n: Node count separating the two size bands.

    Returns:
        A complete ``table*`` environment.
    """
    lines = [
        r"\begin{table*}[t]",
        r"\centering",
        r"\caption{Head-to-head against the instruction string. "
        r"\emph{Compactness} is a per-stratum Wilcoxon signed-rank test on paired bit counts, "
        r"reported twice: \emph{IUT} is the pre-registered conjunction "
        r"$p=\max(p_{\mathrm{entropy}},p_{\mathrm{realised}})$, and \emph{entropy} is the marginal "
        r"on the entropy bound $L\log_2|\Sigma|$ alone. The two differ because the realised-bytes "
        r"convention charges each format its own rendering overhead rather than its content "
        r"(Table~\ref{tab:bit-overhead}); the entropy bound is the like-for-like measure and both "
        r"were locked for reporting before any bit count existed. "
        rf"Both columns are restricted to strata above $n={scope_n}$, and "
        rf"the gap is the median entropy-bound difference "
        rf""
        r"in bits, positive where the instruction string is shorter. "
        r"\emph{Correlation} is a sign test over per-stratum Spearman $\rho$ differences inside "
        r"equal-$n$ strata, where $|n_i-n_j|$ is identically zero and the size channel is removed "
        r"by construction. Both ends of the proven GED bracket are printed on every row: they "
        r"disagree on two of four comparators, so reporting one bound would invert two verdicts. "
        r"$n_{\max}$ is repeated because a comparator that refuses above a size cannot be beaten "
        r"above it.}",
        r"\label{tab:representation-headtohead}",
        r"\footnotesize",
        r"\begin{tabular}{@{}lrrrrrlll@{}}",
        r"\toprule",
        rf" & & \multicolumn{{4}}{{c}}{{compactness, $n>{scope_n}$ (win \%)}} & "
        rf"\multicolumn{{3}}{{c}}{{correlation with GED, within equal $n$}} \\",
        r"\cmidrule(lr){3-6}\cmidrule(lr){7-9}",
        rf"comparator & $n_{{\max}}$ & strata & IUT & entropy & median gap & "
        rf"exact, $n\leq{scope_n}$ & LB, $n>{scope_n}$ & UB, $n>{scope_n}$ \\",
        r"\midrule",
    ]
    for rep in design.REPRESENTATIONS:
        if rep.is_ours or rep.key == "size_null":
            continue
        compact = claim_a_row(strata, rep.key, scope_n + 1, 10_000) if rep.bit_countable else None
        marginal = (
            claim_a_row(strata, rep.key, scope_n + 1, 10_000, rule="entropy")
            if rep.bit_countable
            else None
        )
        exact = sign_test(profile, rep.key, "exact", 0, scope_n)
        lb = sign_test(profile, rep.key, "lb", scope_n + 1, 10_000)
        ub = sign_test(profile, rep.key, "ub", scope_n + 1, 10_000)
        if compact is None or marginal is None:
            cells = [r"\multicolumn{4}{c}{no bit count}"]
        elif compact["strata"] == 0:
            # A comparator whose scope guard stops below the band has no
            # stratum here. That is out of scope, not undefined, and printing
            # the two as the same cell is how a scope note turns into a claim.
            cells = [rf"\multicolumn{{4}}{{c}}{{out of scope above $n={scope_n}$}}"]
        else:
            cells = [
                str(compact["strata"]),
                _bold_if_majority(compact["win_pct"]),
                _bold_if_majority(marginal["win_pct"]),
                f"{compact['median_gap']:+.0f}",
            ]
        lines.append(
            " & ".join(
                [
                    rep.tex,
                    str(rep.max_n) if rep.max_n else _NO,
                    *cells,
                    verdict(exact),
                    verdict(lb),
                    verdict(ub),
                ]
            )
            + r" \\"
        )
    lines += [
        r"\midrule",
        r"\multicolumn{9}{@{}p{\textwidth}@{}}{\footnotesize "
        r"\textbf{nauty-sparse6 is more compact and better correlated than the instruction string "
        r"above $n=20$, under both ends of the bracket.} It is the only representation that "
        r"dominates ours on both axes, and it is reported here rather than omitted. "
        r"Neither axis has a single leader across the whole field: the most compact serialisation "
        r"(sparse6) admits no distance satisfying the metric axioms, and the best-correlating "
        r"representation (WL subtree) admits no bit count, so each axis leader is undefined on the "
        r"other axis.} \\",
        r"\bottomrule",
        r"\end{tabular}",
        r"\end{table*}",
    ]
    return "\n".join(lines)


#: Payload bits carried by each stored byte under the frozen realised-bytes
#: convention, measured over both cohorts as the median of
#: ``8 * entropy_bits / realised_bits``. This is the fairness argument in one
#: column: a convention that charges the adjacency triangle 7.5 payload bits
#: per byte and the instruction string 3.17 is measuring how wasteful each
#: format's rendering happens to be, not how well it encodes.
#: Source: ``.claude/notes/review/tasks/t06_bit_convention.py``.
PAYLOAD_PER_BYTE: Final[dict[str, float]] = {
    "isalgraph_pruned": 3.17,
    "isalgraph_canonical": 3.17,
    "min_dfs": 1.83,
    "agm_cam": 6.00,
    "nauty_graph6": 6.00,
    "sparse6_nauty": 5.50,
    "graph6": 6.00,
    "sparse6": 5.45,
    "adjacency": 7.50,
}


def bit_overhead_table() -> str:
    """Emit the storage-overhead table that justifies reporting both conventions.

    Returns:
        A complete ``table`` environment.
    """
    lines = [
        r"\begin{table}[t]",
        r"\centering",
        r"\caption{What the realised-bytes convention charges. Each entry is the median "
        r"$8\,b_{\mathrm{entropy}}/b_{\mathrm{realised}}$ over both cohorts: how many payload bits "
        r"a stored byte carries under each format's own serialisation. graph6 and sparse6 pay a "
        r"published ASCII-printability cost; the adjacency triangle is stored packed; the "
        r"instruction string has no standardised wire format and is rendered one character per "
        r"symbol, which charges it eight bits for a symbol drawn from a nine-letter alphabet. "
        r"gSpan min-DFS suffers the same artefact and is flagged \emph{inflated} in our "
        r"implementation. The realised-bytes column is therefore not comparable across formats, "
        r"which is why the entropy bound is reported beside it throughout.}",
        r"\label{tab:bit-overhead}",
        r"\footnotesize",
        r"\begin{tabular}{@{}lrr@{}}",
        r"\toprule",
        r"representation & payload bits per stored byte & overhead \\",
        r"\midrule",
    ]
    for rep in design.REPRESENTATIONS:
        payload = PAYLOAD_PER_BYTE.get(rep.key)
        if payload is None:
            continue
        name = rf"\textbf{{{rep.tex}}}" if rep.is_ours else rep.tex
        lines.append(f"{name} & {payload:.2f} of 8 & {8.0 / payload:.2f}$\\times$ \\\\")
    lines += [
        r"\bottomrule",
        r"\end{tabular}",
        r"\end{table}",
    ]
    return "\n".join(lines)


# ---------------------------------------------------------------------------
# The one wide table: every property and every result, one row per
# representation, best in bold and worst underlined.
# ---------------------------------------------------------------------------

#: Node counts the compactness columns are read at. Two anchors rather than
#: one, because the whole Claim A finding is that the ordering *changes* with
#: size and a single anchor hides that.
BITS_ANCHORS: Final[tuple[int, int]] = (20, 40)

#: Marks. ``\underline`` needs no package; ``soul``'s ``\ul`` would break in a
#: table cell containing math.
_BEST: Final[str] = r"\textbf{{{0}}}"
_WORST: Final[str] = r"\underline{{{0}}}"


def _mark_numeric(
    values: dict[str, float | None],
    *,
    lower_is_better: bool,
    fmt: str = "{:.0f}",
) -> dict[str, str]:
    """Format a numeric column, bolding the best cell and underlining the worst.

    Args:
        values: ``{representation: value}``; ``None`` renders as a dash and is
            excluded from the ranking, because a representation that cannot be
            measured on an axis is not the worst on it.
        lower_is_better: Direction of merit.
        fmt: Format string for a value.

    Returns:
        ``{representation: LaTeX cell}``.
    """
    live = {k: v for k, v in values.items() if v is not None and v == v}
    out = {k: (_NO if v is None or v != v else fmt.format(v)) for k, v in values.items()}
    if not live:
        return out
    best = min(live.values()) if lower_is_better else max(live.values())
    worst = max(live.values()) if lower_is_better else min(live.values())
    for key, value in live.items():
        if value == best:
            out[key] = _BEST.format(fmt.format(value))
        elif value == worst:
            out[key] = _WORST.format(fmt.format(value))
    return out


def _mark_bool(value: bool) -> str:
    """Bold a satisfied property, underline an unsatisfied one.

    Args:
        value: Whether the property holds.

    Returns:
        The LaTeX cell.
    """
    return _BEST.format(_YES) if value else _WORST.format(_NO)


def _paired_cell(result: dict[str, Any] | None, alpha: float = SIGN_TEST_ALPHA) -> str:
    """Render a paired sign-test result as a median-plus-verdict cell.

    Args:
        result: Output of :func:`sign_test`, or ``None`` when the band holds no
            comparable stratum.
        alpha: Significance level.

    Returns:
        The LaTeX cell: the median per-stratum ``Delta rho``, bold when the
        reference arm is significantly better and underlined when it is
        significantly worse.
    """
    if result is None:
        return _NO
    value = f"{result['median']:+.3f}"
    if result["p"] >= alpha:
        return value
    return _BEST.format(value) if result["higher"] > result["lower"] else _WORST.format(value)


def _bits_at(cells: list[Any], n: int) -> dict[str, float | None]:
    """Return the pooled median entropy bits at node count *n*.

    Args:
        cells: Encoding cells from :func:`data.load_cells`.
        n: Node count.

    Returns:
        ``{representation: median bits}``, ``None`` where the representation
        has no stratum at that size.
    """
    points = data.aggregate_bits(cells, convention="entropy_bits", min_graphs=20)
    at_n = {p.representation: p.median for p in points if p.n == n}
    return {rep.key: at_n.get(rep.key) for rep in design.REPRESENTATIONS}


def _rho_band(profile: dict[str, Any], reference: str, lo: int, hi: int) -> dict[str, float | None]:
    """Return the median per-stratum rho per representation over a size band.

    **Median across strata, not a Fisher-z weighted mean.** The weighted mean
    is the right reduction for a *figure*, where each point is one node count
    and the reader sees the curve. It is the wrong one for this column: it is
    dominated by the small strata, which carry the most graphs and where every
    representation saturates near ``rho = 1``, so it ranked nauty-sparse6 above
    the reference arm on a band where the paired sign test rejects the other
    way at ``p = 0.041``. A table cell that contradicts the head-to-head table
    two pages later is worse than no cell.

    The median weights every stratum equally, which is exactly what the sign
    test in :func:`sign_test` does, so this column and that verdict cannot
    disagree in direction.

    Args:
        profile: Parsed ``size_profile.json``.
        reference: ``exact``, ``lb`` or ``ub``.
        lo: Smallest node count, inclusive.
        hi: Largest node count, inclusive.

    Returns:
        ``{representation: rho}``, ``None`` where the band is empty.
    """
    bucket: dict[str, list[float]] = defaultdict(list)
    for row in profile["rows"]:
        if row["rho"] is None or row.get("arm", "primary") != "primary":
            continue
        if row["reference"] != reference or not lo <= int(row["n"]) <= hi:
            continue
        bucket[row["representation"]].append(float(row["rho"]))
    return {
        rep.key: (statistics.median(bucket[rep.key]) if bucket.get(rep.key) else None)
        for rep in design.REPRESENTATIONS
    }


def summary_table(
    cells: list[Any],
    strata: dict[str, Any],
    profile: dict[str, Any],
    *,
    scope_n: int = design.CLAIM_A_SCOPE_N,
) -> str:
    """Emit the wide properties-and-results table.

    Args:
        cells: Encoding cells.
        strata: Parsed ``claim_a_strata.json``.
        profile: Parsed ``size_profile.json``.
        scope_n: Node count separating the two size bands.

    Returns:
        A complete ``sidewaystable*`` environment.
    """
    bits = {n: _mark_numeric(_bits_at(cells, n), lower_is_better=True) for n in BITS_ANCHORS}
    raw_bits = {n: _bits_at(cells, n) for n in BITS_ANCHORS}
    bands = {
        "exact": (0, design.EXACT_CEILING),
        "lb": (scope_n + 1, 10_000),
        "ub": (scope_n + 1, 10_000),
    }
    paired = {
        ref: {
            rep.key: (
                r"ref." if rep.is_ours else _paired_cell(sign_test(profile, rep.key, ref, lo, hi))
            )
            for rep in design.REPRESENTATIONS
        }
        for ref, (lo, hi) in bands.items()
    }
    reach = _mark_numeric(
        {rep.key: (float(rep.max_n) if rep.max_n else None) for rep in design.REPRESENTATIONS},
        lower_is_better=False,
    )
    floors = _mark_numeric(
        {rep.key: COMPLETION_FLOOR.get(rep.key) for rep in design.REPRESENTATIONS},
        lower_is_better=False,
        fmt="{:.3f}",
    )

    lines = [
        r"\begin{sidewaystable*}[p]",
        r"\centering",
        r"\caption{Every representation, every property, every result. "
        r"\emph{Two markings, and they mean different things.} In the property, scalability and "
        r"bit columns, \textbf{bold} is the best cell in its column and \underline{underline} the "
        r"worst; a representation that cannot be measured on an axis reads \emph{--} and is "
        r"excluded from that column's ranking rather than counted as its worst. The three "
        r"correlation columns are not rankings: each is the median per-stratum $\Delta\rho$ "
        r"\emph{paired against the instruction string on identical strata}, positive where the "
        r"instruction string correlates better, with \textbf{bold} a significant sign test in its "
        r"favour and \underline{underline} one against it. "
        r"The marginal median $\rho$ is deliberately not printed: it ranks nauty-graph6 above the "
        r"instruction string on the exact band, where the paired test rejects the other way at "
        r"$p=0.041$ with the instruction string higher on 15 strata by a median of $+0.14$ and "
        r"lower on 5 by $-0.05$. "
        r"$\psi$ is relabelling sensitivity under the representation's own primary distance "
        r"(T-04a~E1, eleven draws); \emph{collis.} is the false-isomorphism-certificate rate over "
        r"183{,}016 comparisons (T-04a~E2); \emph{metric} is whether any candidate distance passed "
        r"the F1--F4 filters, and the three that fail do so at $1/50$ relabellings and therefore "
        r"carry no correlation column at all; \emph{exec.} is whether every prefix of the encoding "
        r"is itself a valid program constructing a subgraph. "
        r"Bit counts are the pooled median entropy bound $L\log_2|\Sigma|$ at two anchor sizes; "
        r"the anchors are two rather than one because the ordering changes with size, which is the "
        r"finding. Correlations are measured within equal-$n$ strata, where $|n_i-n_j|$ is "
        r"identically zero and the size channel is removed by construction; both ends of the "
        r"proven GED bracket are printed because they disagree on two of four comparators. "
        r"graph6 and nauty-graph6 carry identical bit counts by construction. "
        r"The downstream-learning axis of R1.2 is not evaluated in this work for any row.}",
        r"\label{tab:representation-summary}",
        r"\scriptsize",
        r"\setlength{\tabcolsep}{3.4pt}",
        r"\begin{tabular}{@{}ll ccccccc rr rr rrr@{}}",
        r"\toprule",
        r" & & \multicolumn{7}{c}{properties} & \multicolumn{2}{c}{scalability} & "
        r"\multicolumn{2}{c}{bits (median)} & "
        r"\multicolumn{3}{c}{median $\Delta\rho$ vs ours, within equal $n$} \\",
        r"\cmidrule(lr){3-9}\cmidrule(lr){10-11}\cmidrule(lr){12-13}\cmidrule(lr){14-16}",
        r"representation & family & $\psi$ & compl. & collis. & rev. & disc. & metric & exec. & "
        rf"$n_{{\max}}$ & compl.\% & $n{{=}}{BITS_ANCHORS[0]}$ & $n{{=}}{BITS_ANCHORS[1]}$ & "
        rf"exact$_{{\leq{design.EXACT_CEILING}}}$ & LB$_{{>{scope_n}}}$ & UB$_{{>{scope_n}}}$ \\",
        r"\midrule",
    ]
    for rep in design.REPRESENTATIONS:
        name = rf"\textbf{{{rep.tex}}}" if rep.is_ours else rep.tex
        psi = PSI_MEASURED.get(rep.key, "?")
        psi_cell = _BEST.format(psi) if psi == "0" else _WORST.format(psi)
        collisions = COLLISIONS_MEASURED.get(rep.key)
        if collisions is not None:
            collision_cell = _WORST.format(collisions)
        elif rep.complete:
            collision_cell = _BEST.format("0")
        else:
            collision_cell = _NO
        lines.append(
            " & ".join(
                [
                    name,
                    rep.family.value.replace("canonicalised serialisation", "canon.\\ serial."),
                    psi_cell,
                    _mark_bool(rep.complete),
                    collision_cell,
                    _mark_bool(rep.reversible),
                    _mark_bool(rep.handles_disconnected),
                    _mark_bool(rep.metric_admissible),
                    _mark_bool(rep.key in EXECUTABLE),
                    reach[rep.key],
                    floors[rep.key],
                    bits[BITS_ANCHORS[0]][rep.key],
                    bits[BITS_ANCHORS[1]][rep.key],
                    paired["exact"][rep.key],
                    paired["lb"][rep.key],
                    paired["ub"][rep.key],
                ]
            )
            + r" \\"
        )
    ours = design.BY_KEY[design.REFERENCE_KEY]
    anchor = raw_bits[BITS_ANCHORS[1]]
    ours_bits = anchor.get(ours.key)
    beaten_a = (
        [
            rep.tex
            for rep in design.REPRESENTATIONS
            if rep.bit_countable
            and not rep.is_ours
            and (theirs := anchor.get(rep.key)) is not None
            and theirs > ours_bits
        ]
        if ours_bits is not None
        else []
    )
    lines += [
        r"\midrule",
        r"\multicolumn{16}{@{}p{\textheight}@{}}{\footnotesize "
        rf"At $n={BITS_ANCHORS[1]}$ the instruction string is more compact than "
        rf"{', '.join(beaten_a)} and less compact than the sparse6 family. "
        r"\textbf{nauty-sparse6 is both more compact and better correlated above $n=20$, under "
        r"both ends of the bracket} --- the only representation that dominates ours on both axes, "
        r"and reported here rather than omitted. Neither axis has a single leader across the "
        r"field: the most compact serialisation admits no distance satisfying the metric axioms, "
        r"and the best-correlating representation admits no bit count, so each axis leader is "
        r"undefined on the other axis. The instruction string is the only row that is executable, "
        r"and that property is not adjudicated by either experiment.} \\",
        r"\bottomrule",
        r"\end{tabular}",
        r"\end{sidewaystable*}",
    ]
    return "\n".join(lines)


def build_parser() -> argparse.ArgumentParser:
    """Return the CLI parser."""
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--strata", type=Path, required=True, help="claim_a_strata.json")
    ap.add_argument("--profile", type=Path, required=True, help="size_profile.json")
    ap.add_argument("--encodings", type=Path, required=True, help="encodings/ directory")
    ap.add_argument("--out-dir", type=Path, required=True, help="LaTeX output directory")
    return ap


def main(argv: list[str] | None = None) -> int:
    """Entry point.

    Args:
        argv: Argument vector, or ``None`` for ``sys.argv``.

    Returns:
        Process exit status.
    """
    args = build_parser().parse_args(argv)
    logging.basicConfig(level=logging.INFO, format="%(levelname)s %(message)s")
    strata = data.load_json(args.strata)
    profile = data.load_json(args.profile)
    cells = data.load_cells(args.encodings)
    args.out_dir.mkdir(parents=True, exist_ok=True)
    for name, text in (
        ("tab_representation_properties.tex", properties_table()),
        ("tab_representation_headtohead.tex", head_to_head_table(strata, profile)),
        ("tab_bit_overhead.tex", bit_overhead_table()),
        ("tab_representation_summary.tex", summary_table(cells, strata, profile)),
    ):
        (args.out_dir / name).write_text(text + "\n")
        LOGGER.info("%s -> %s", name, args.out_dir / name)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
