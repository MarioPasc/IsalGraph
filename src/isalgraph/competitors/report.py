"""Audit-path reporter for the T-04a metric-feasibility grid.

Reads the grid JSON (contracts section 2 and 3) and, optionally, the F5 JSON
(contracts section 5), and writes the four artifacts the Pattern Recognition
revision needs:

``supplementary_grid.csv``
    One row per attempted (representation x distance) cell, every cell, with
    every ``Cell`` field as a column.  A cell that failed is a result, so the
    table reports it rather than omitting it.
``selection.md``
    Per representation: the selected primary distance, or ``none admissible``
    together with the failing criterion of every candidate it had.
``k.json``
    The pre-registration bookkeeping: ``k``, its membership, the separate
    partial-computability list, and the resulting family size.
``f5_table.md``
    Per dataset and view: Spearman rho, its bootstrap CI, the size null and the
    explicit difference ``rho - rho_null``.  Suite-2 lower and upper bounds stay
    in separate columns and are never interpolated into a midpoint.

This module reads JSON and deliberately imports neither
``isalgraph.competitors.grid`` nor ``isalgraph.competitors.f5``.  It is the
audit path: it recomputes the selection from the measured cells under the frozen
rule and reports any disagreement with the grid's own verdict, which a shared
import would let one bug hide.  Standard library only.
"""

from __future__ import annotations

import argparse
import csv
import json
import logging
import math
from collections.abc import Mapping, Sequence
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any

_LOG = logging.getLogger(__name__)

JsonDict = dict[str, Any]

#: preregistration.md section 4.1's Claim-B comparator set, frozen, and defined
#: here and nowhere else.  ``sparse6_nauty`` is a T-04 addition outside the
#: frozen family, ``isalgraph_canonical``/``isalgraph_pruned`` are the reference
#: arm being compared against, and ``size_null`` is a baseline.  None of the four
#: is a comparator, and counting any of them into ``k`` would silently move the
#: paper's FDR threshold.
CLAIM_B_COMPARATORS: tuple[str, ...] = (
    "graph6",
    "sparse6",
    "nauty_graph6",
    "adjacency",
    "agm_cam",
    "min_dfs",
    "wl_subtree",
)

#: preregistration.md section 5's family size before any exclusion.
N_MAX: int = 182

#: Claim-B rows a representation loses when it has no admissible distance at all.
ROWS_PER_REPRESENTATION: int = 15

#: Claim-B rows charged per dataset-set, by suite: 5 for Suite 1, 10 for Suite 2.
ROWS_PER_SUITE: dict[str, int] = {"suite1": 5, "suite2": 10}

#: The suites a representation can lose independently.  ``f0`` also carries an
#: ``overall`` key, which is a summary and never a suite.
SUITES: tuple[str, ...] = ("suite1", "suite2")

#: competitors.md section 3.4's F4 thresholds, verbatim.
F4_ZERO_MASS_LIMIT: float = 0.5
F4_CV_FLOOR: float = 1e-6

#: Every ``Cell`` field of contracts section 3, in the dataclass's own order.
#: This is the CSV header and it is stable.
CELL_COLUMNS: tuple[str, ...] = (
    "backend",
    "metric",
    "applicable",
    "reason",
    "candidate",
    "f1_defined_frac",
    "f1_n_pairs",
    "f2_declared_pseudometric",
    "f2_violations",
    "f3_invariant",
    "f3_skipped",
    "f4_zero_mass",
    "f4_coeff_variation",
    "f6_ms_per_pair",
    "f6_ms_per_pair_large",
    "f6_over_advisory_limit",
    "passes_selection",
    "excluded_because",
)

#: The number of cells the protocol expects: 11 representations x 6 distances.
EXPECTED_CELLS: int = 66

#: GED reference arms, in report order.  Suite 1 carries ``exact`` alone; Suite 2
#: carries ``lb`` and ``ub`` as two separate columns (approx_ged.md section 4).
REFERENCE_ORDER: tuple[str, ...] = ("exact", "lb", "ub")

#: F5 views, in report order.
VIEW_ORDER: tuple[str, ...] = ("all_pairs", "equal_n")

#: Row key of the size null inside an F5 view.
SIZE_NULL: str = "size_null"


class ReportError(Exception):
    """A malformed or unreadable input to the reporter."""


# --------------------------------------------------------------------------- #
# scalar coercion -- every input field is JSON and may legitimately be null
# --------------------------------------------------------------------------- #


def _as_float(value: object) -> float | None:
    """The value as a float, or ``None`` when it is absent or not numeric."""
    if value is None or isinstance(value, bool):
        return None
    if isinstance(value, int | float):
        return float(value)
    return None


def _as_int(value: object) -> int | None:
    """The value as an int, or ``None`` when it is absent or not numeric."""
    if value is None or isinstance(value, bool):
        return None
    if isinstance(value, int):
        return value
    if isinstance(value, float) and value.is_integer():
        return int(value)
    return None


def _as_mapping(value: object) -> Mapping[str, Any]:
    """The value as a mapping, or an empty mapping when it is absent."""
    if isinstance(value, Mapping):
        return value
    return {}


def _as_sequence(value: object) -> list[Any]:
    """The value as a list, or an empty list when it is absent."""
    if isinstance(value, list):
        return value
    return []


def _parse_ratio(value: object) -> tuple[int, int] | None:
    """Parse an ``"k/attempted"`` invariant string into its two integers."""
    if not isinstance(value, str) or "/" not in value:
        return None
    head, _, tail = value.partition("/")
    try:
        return int(head.strip()), int(tail.strip())
    except ValueError:
        return None


# --------------------------------------------------------------------------- #
# the frozen selection rule, recomputed from the cells
# --------------------------------------------------------------------------- #


def failing_criteria(cell: Mapping[str, Any]) -> list[str]:
    """Every reason this cell cannot be a primary distance.

    Implements competitors.md section 3.4 as frozen in the T-04a design note:
    a candidate cell qualifies with F1 at 100 %, no observed F2 violation, the F3
    invariant on every attempted graph, F4 zero-mass at most 0.5 and F4
    coefficient of variation at least 1e-6.  F6 is the tie-break and never a
    gate, so it appears in no criterion here.

    Args:
        cell: One ``Cell`` record, as written into the grid JSON.

    Returns:
        The failing criteria, in evaluation order.  An empty list means the cell
        qualifies.
    """
    fails: list[str] = []

    if not bool(cell.get("applicable", True)):
        reason = cell.get("reason") or "no reason recorded"
        return [f"not applicable: {reason}"]

    if not bool(cell.get("candidate", True)):
        excluded = cell.get("excluded_because") or "not a candidate distance"
        return [str(excluded)]

    f1 = _as_float(cell.get("f1_defined_frac"))
    if f1 is None:
        fails.append("F1 not measured")
    elif f1 < 1.0:
        fails.append(f"F1 = {f1:.4f} < 1.0")

    violations = cell.get("f2_violations")
    if violations is None:
        fails.append("F2 not measured")
    else:
        table = _as_mapping(violations)
        observed = {str(k): _as_int(v) or 0 for k, v in table.items()}
        hit = {k: v for k, v in observed.items() if v > 0}
        if hit:
            detail = ", ".join(f"{k}={hit[k]}" for k in sorted(hit))
            fails.append(f"F2 violated ({detail})")

    ratio = _parse_ratio(cell.get("f3_invariant"))
    if ratio is None:
        fails.append("F3 not measured")
    else:
        held, attempted = ratio
        if attempted <= 0:
            fails.append("F3 attempted on 0 graphs")
        elif held < attempted:
            fails.append(f"F3 invariant on only {held}/{attempted} graphs")

    zero_mass = _as_float(cell.get("f4_zero_mass"))
    if zero_mass is None:
        fails.append("F4 zero-mass not measured")
    elif zero_mass > F4_ZERO_MASS_LIMIT:
        fails.append(f"F4 zero-mass {zero_mass:.4f} > {F4_ZERO_MASS_LIMIT}")

    coeff = _as_float(cell.get("f4_coeff_variation"))
    if coeff is None:
        fails.append("F4 coefficient of variation not measured")
    elif coeff < F4_CV_FLOOR:
        fails.append(f"F4 coefficient of variation {coeff:.3g} < {F4_CV_FLOOR:g}")

    return fails


def recompute_primary(cells: Sequence[Mapping[str, Any]]) -> str | None:
    """The primary distance the frozen rule selects from these cells.

    Args:
        cells: Every cell of one representation.

    Returns:
        The winning metric name, or ``None`` when no cell qualifies.  Ties on
        F6 break on the metric name, and an unmeasured F6 sorts last.
    """
    qualifying = [cell for cell in cells if not failing_criteria(cell)]
    if not qualifying:
        return None

    def sort_key(cell: Mapping[str, Any]) -> tuple[float, str]:
        ms = _as_float(cell.get("f6_ms_per_pair"))
        return (math.inf if ms is None else ms, str(cell.get("metric", "")))

    return str(min(qualifying, key=sort_key).get("metric"))


def cells_by_backend(grid: Mapping[str, Any]) -> dict[str, list[Mapping[str, Any]]]:
    """Group the grid's cells by representation, preserving their written order."""
    grouped: dict[str, list[Mapping[str, Any]]] = {}
    for raw in _as_sequence(grid.get("cells")):
        if not isinstance(raw, Mapping):
            continue
        grouped.setdefault(str(raw.get("backend")), []).append(raw)
    return grouped


# --------------------------------------------------------------------------- #
# k, and the separate partial-computability term
# --------------------------------------------------------------------------- #


@dataclass(frozen=True, slots=True)
class PartialEntry:
    """One comparator that has an admissible distance but loses one suite."""

    representation: str
    suite: str
    rows_lost: int
    reason: str


def _dominant_error(errors: object) -> str:
    """The most frequent exception type, ties broken by name."""
    table = _as_mapping(errors)
    if not table:
        return "no exception type recorded"
    ranked = sorted(table.items(), key=lambda kv: (-(_as_int(kv[1]) or 0), str(kv[0])))
    return str(ranked[0][0])


def primary_distances(grid: Mapping[str, Any]) -> dict[str, str | None]:
    """The grid's selected primary distance per representation, nulls included."""
    table = _as_mapping(grid.get("primary_distance"))
    return {str(k): (None if v is None else str(v)) for k, v in table.items()}


def compute_k_members(primary: Mapping[str, str | None]) -> list[str]:
    """The comparators with no admissible distance on any suite.

    A comparator absent from ``primary_distance`` altogether is treated as
    having no admissible distance: the grid never produced one for it, so no
    Claim-B row can be run.  ``selection.md`` flags the absence explicitly rather
    than letting it pass as a measured null.
    """
    return [name for name in CLAIM_B_COMPARATORS if primary.get(name) is None]


def compute_partial(
    grid: Mapping[str, Any], primary: Mapping[str, str | None]
) -> list[PartialEntry]:
    """Comparators that have an admissible distance but lose one suite's rows.

    Read from the grid's ``f0`` block: a suite with ``frac < 1.0`` cannot carry a
    printed row for that suite (design note section 3.3).  A comparator already
    counted into ``k`` is skipped, so no representation is charged twice.

    A suite with ``attempted == 0`` is *not* charged: the sample contained no
    graph of that suite, which is a property of the draw and not a statement
    about the representation.
    """
    f0 = _as_mapping(grid.get("f0"))
    entries: list[PartialEntry] = []
    for name in CLAIM_B_COMPARATORS:
        if primary.get(name) is None:
            continue
        block = _as_mapping(f0.get(name))
        for suite in SUITES:
            stats = block.get(suite)
            if not isinstance(stats, Mapping):
                continue
            attempted = _as_int(stats.get("attempted"))
            if attempted is not None and attempted <= 0:
                continue
            frac = _as_float(stats.get("frac"))
            if frac is None or frac >= 1.0:
                continue
            entries.append(
                PartialEntry(
                    representation=name,
                    suite=suite,
                    rows_lost=ROWS_PER_SUITE[suite],
                    reason=f"F0 = {frac:.2f} on {suite} ({_dominant_error(stats.get('errors'))})",
                )
            )
    return entries


def build_k_payload(grid: Mapping[str, Any]) -> JsonDict:
    """The ``k.json`` payload, exactly the schema of contracts section 6."""
    primary = primary_distances(grid)
    members = compute_k_members(primary)
    partial = compute_partial(grid, primary)
    k = len(members)
    rows_lost = sum(entry.rows_lost for entry in partial)
    n_actual = N_MAX - ROWS_PER_REPRESENTATION * k - rows_lost
    formula = (
        f"N_actual(F2) = {N_MAX} - {ROWS_PER_REPRESENTATION}k - 8d - p "
        f"= {N_MAX} - {ROWS_PER_REPRESENTATION}*{k} - 8d - {rows_lost} = {n_actual} - 8d"
    )
    return {
        "k": k,
        "k_members": members,
        "comparator_set": list(CLAIM_B_COMPARATORS),
        "partial": [asdict(entry) for entry in partial],
        "n_actual_f2_before_d": n_actual,
        "formula": formula,
    }


# --------------------------------------------------------------------------- #
# artifact 1 -- supplementary_grid.csv
# --------------------------------------------------------------------------- #


def _csv_value(value: object) -> str:
    """Render one cell field for the supplementary CSV."""
    if value is None:
        return ""
    if isinstance(value, bool):
        return "true" if value else "false"
    if isinstance(value, int | float | str):
        return str(value)
    return json.dumps(value, sort_keys=True, separators=(",", ":"))


def write_supplementary_csv(grid: Mapping[str, Any], path: Path) -> int:
    """Write one row per cell, every cell, with a stable header.

    Returns:
        The number of data rows written.
    """
    rows = [raw for raw in _as_sequence(grid.get("cells")) if isinstance(raw, Mapping)]
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.writer(handle)
        writer.writerow(CELL_COLUMNS)
        for raw in rows:
            writer.writerow([_csv_value(raw.get(column)) for column in CELL_COLUMNS])
    if len(rows) != EXPECTED_CELLS:
        _LOG.warning(
            "supplementary_grid.csv holds %d cells, the protocol expects %d",
            len(rows),
            EXPECTED_CELLS,
        )
    return len(rows)


# --------------------------------------------------------------------------- #
# artifact 2 -- selection.md
# --------------------------------------------------------------------------- #


def _md_escape(text: str) -> str:
    return text.replace("|", r"\|").replace("\n", " ")


def _md_table(header: Sequence[str], rows: Sequence[Sequence[str]]) -> list[str]:
    lines = ["| " + " | ".join(header) + " |", "|" + "|".join("---" for _ in header) + "|"]
    lines.extend("| " + " | ".join(row) + " |" for row in rows)
    return lines


def _selection_summary_rows(
    grid: Mapping[str, Any],
    grouped: Mapping[str, list[Mapping[str, Any]]],
    primary: Mapping[str, str | None],
) -> tuple[list[list[str]], list[str]]:
    """The per-representation summary rows, plus the disagreements found."""
    reasons = _as_mapping(grid.get("selection_reason"))
    backends = [str(b) for b in _as_sequence(grid.get("backends"))]
    for name in list(grouped) + list(primary):
        if name not in backends:
            backends.append(name)

    rows: list[list[str]] = []
    disagreements: list[str] = []
    for name in backends:
        cells = grouped.get(name, [])
        stated = primary.get(name)
        recomputed = recompute_primary(cells) if cells else None
        if name not in primary:
            stated_text = "*absent from grid*"
        elif stated is None:
            stated_text = "**none admissible**"
        else:
            stated_text = f"`{stated}`"
        recomputed_text = f"`{recomputed}`" if recomputed else "none admissible"
        if not cells:
            agreement = "not checkable (no cells)"
        elif stated == recomputed and name in primary:
            agreement = "agree"
        else:
            agreement = "**DISAGREE**"
            disagreements.append(
                f"`{name}`: grid says {stated_text}, recomputation from the cells says "
                f"{recomputed_text}"
            )
        rows.append(
            [
                f"`{name}`",
                "yes" if name in CLAIM_B_COMPARATORS else "no",
                stated_text,
                recomputed_text,
                agreement,
                _md_escape(str(reasons.get(name, ""))),
            ]
        )
    return rows, disagreements


def render_selection_md(grid: Mapping[str, Any], grid_path: Path) -> str:
    """The human-readable selection table."""
    grouped = cells_by_backend(grid)
    primary = primary_distances(grid)
    payload = build_k_payload(grid)

    lines: list[str] = [
        "# T-04a -- primary distance selection",
        "",
        f"Source grid: `{grid_path}`",
        "",
        f"Cells reported: {sum(len(v) for v in grouped.values())} "
        f"(protocol expects {EXPECTED_CELLS}). "
        "Every attempted cell is printed, a failure included: a cell that fails is a result.",
        "",
        "The *Recomputed* column re-derives the primary distance from the measured cells under "
        "competitors.md section 3.4, independently of the grid's own verdict. This file is the "
        "audit path, so a disagreement is printed rather than reconciled.",
        "",
        "## Selected primary distance",
        "",
    ]
    rows, disagreements = _selection_summary_rows(grid, grouped, primary)
    lines.extend(
        _md_table(
            [
                "Representation",
                "Claim-B comparator",
                "Primary distance",
                "Recomputed",
                "Agreement",
                "Grid's reason",
            ],
            rows,
        )
    )
    lines.append("")

    if disagreements:
        lines.extend(["### Disagreements with the grid's own selection", ""])
        lines.extend(f"- {item}" for item in disagreements)
        lines.append("")
    else:
        lines.extend(["The recomputation agrees with the grid on every representation.", ""])

    lines.extend(["## Why each excluded representation was excluded", ""])
    excluded = [name for name in grouped if primary.get(name) is None]
    if not excluded:
        lines.extend(["Every representation has an admissible distance.", ""])
    for name in excluded:
        lines.extend([f"### `{name}` -- none admissible", ""])
        rows_x = [
            [
                f"`{cell.get('metric')}`",
                "yes" if bool(cell.get("candidate", True)) else "no",
                _md_escape("; ".join(failing_criteria(cell)) or "passes every criterion"),
            ]
            for cell in grouped[name]
        ]
        lines.extend(_md_table(["Distance", "Candidate", "Failing criterion"], rows_x))
        lines.append("")

    lines.extend(
        [
            "## Family size",
            "",
            f"- `k` = {payload['k']}"
            f" -- comparators with no admissible distance: "
            f"{', '.join(f'`{m}`' for m in payload['k_members']) or 'none'}",
            f"- Claim-B comparator set (frozen, 7 members): "
            f"{', '.join(f'`{m}`' for m in CLAIM_B_COMPARATORS)}",
            f"- Rows lost to partial computability: "
            f"{sum(int(e['rows_lost']) for e in payload['partial'])}",
            f"- `{payload['formula']}`",
            "",
            "A representation with **no admissible distance** loses "
            f"{ROWS_PER_REPRESENTATION} Claim-B rows and keeps its Claim-A rows, because a bit "
            "count needs no distance. A representation that is merely **not computable on one "
            "suite** loses only that suite's rows "
            f"({ROWS_PER_SUITE['suite1']} for Suite 1, {ROWS_PER_SUITE['suite2']} for Suite 2). "
            "The two are charged separately.",
            "",
        ]
    )

    if payload["partial"]:
        lines.extend(["### Partial computability", ""])
        lines.extend(
            _md_table(
                ["Representation", "Suite", "Rows lost", "Reason"],
                [
                    [
                        f"`{entry['representation']}`",
                        str(entry["suite"]),
                        str(entry["rows_lost"]),
                        _md_escape(str(entry["reason"])),
                    ]
                    for entry in payload["partial"]
                ],
            )
        )
        lines.append("")

    return "\n".join(lines) + "\n"


# --------------------------------------------------------------------------- #
# artifact 4 -- f5_table.md
# --------------------------------------------------------------------------- #


def _fmt_rho(value: float | None, *, best: bool) -> str:
    if value is None:
        return "--"
    text = f"{value:.3f}"
    return f"**{text}**" if best else text


def _fmt_ci(ci: object) -> str:
    values = _as_sequence(ci)
    if len(values) != 2:
        return "--"
    low, high = _as_float(values[0]), _as_float(values[1])
    if low is None or high is None:
        return "--"
    return f"[{low:.3f}, {high:.3f}]"


def _group_f5_by_dataset(f5: Mapping[str, Any]) -> dict[str, dict[str, Mapping[str, Any]]]:
    """``{dataset: {reference: record}}``, in the order the records were written."""
    grouped: dict[str, dict[str, Mapping[str, Any]]] = {}
    for key, raw in _as_mapping(f5.get("results")).items():
        if not isinstance(raw, Mapping):
            continue
        dataset = str(raw.get("dataset") or str(key).split("::")[0])
        reference = str(raw.get("reference") or (str(key).split("::") + ["exact"])[1])
        grouped.setdefault(dataset, {})[reference] = raw
    return grouped


def _ordered(names: Sequence[str], preferred: Sequence[str]) -> list[str]:
    ordered = [name for name in preferred if name in names]
    ordered.extend(name for name in names if name not in preferred)
    return ordered


def _row_order(view_tables: Sequence[Mapping[str, Any]]) -> list[str]:
    """Representation rows in first-appearance order, size null last."""
    order: list[str] = []
    for table in view_tables:
        for name in table:
            if name not in order:
                order.append(str(name))
    body = [name for name in order if name != SIZE_NULL]
    if SIZE_NULL in order:
        body.append(SIZE_NULL)
    return body


def _dataset_section(dataset: str, records: Mapping[str, Mapping[str, Any]]) -> list[str]:
    references = _ordered(list(records), REFERENCE_ORDER)
    suite = str(next(iter(records.values())).get("suite", "?"))
    lines = [f"### `{dataset}` ({suite})", ""]

    view_names: list[str] = []
    for record in records.values():
        for view in _as_mapping(record.get("views")):
            if view not in view_names:
                view_names.append(str(view))
    view_names = _ordered(view_names, VIEW_ORDER)

    for view in view_names:
        tables = {
            ref: _as_mapping(_as_mapping(records[ref].get("views")).get(view)) for ref in references
        }
        rows_order = _row_order(list(tables.values()))
        if not rows_order:
            continue

        header = ["Representation", "Distance"]
        for ref in references:
            header.extend([f"rho ({ref})", f"95 % CI ({ref})", f"rho - rho_null ({ref})"])

        best: dict[str, float] = {}
        for ref in references:
            values = [
                rho
                for name in rows_order
                if (rho := _as_float(_as_mapping(tables[ref].get(name)).get("rho"))) is not None
            ]
            if values:
                best[ref] = max(values)

        body: list[list[str]] = []
        absences: dict[str, str] = {}
        for name in rows_order:
            metric_names = {
                str(_as_mapping(tables[ref].get(name)).get("metric"))
                for ref in references
                if _as_mapping(tables[ref].get(name)).get("metric") is not None
            }
            distance = f"`{sorted(metric_names)[0]}`" if metric_names else "--"
            if name == SIZE_NULL:
                # The pipes of |n1 - n2| would split the markdown cell.
                distance = "`abs(n1 - n2)` (size null)"
            row = [f"`{name}`", distance]
            for ref in references:
                entry = _as_mapping(tables[ref].get(name))
                rho = _as_float(entry.get("rho"))
                null_rho = _as_float(_as_mapping(tables[ref].get(SIZE_NULL)).get("rho"))
                if rho is None:
                    reason = entry.get("reason")
                    if reason:
                        absences[name] = str(reason)
                    row.extend(["--", "--", "--"])
                    continue
                delta = "--" if null_rho is None else f"{rho - null_rho:+.3f}"
                row.extend(
                    [
                        _fmt_rho(rho, best=best.get(ref) is not None and rho >= best[ref]),
                        _fmt_ci(entry.get("ci")),
                        delta,
                    ]
                )
            body.append(row)

        lines.extend([f"**View: `{view}`**", ""])
        lines.extend(_md_table(header, body))
        if absences:
            grouped_reasons: dict[str, list[str]] = {}
            for name, reason in absences.items():
                grouped_reasons.setdefault(reason, []).append(name)
            lines.append("")
            lines.extend(
                f"`--` for {', '.join(f'`{n}`' for n in names)}: {_md_escape(reason)}."
                for reason, names in grouped_reasons.items()
            )

        pair_counts = []
        for ref in references:
            counts = {
                n
                for name in rows_order
                if (n := _as_int(_as_mapping(tables[ref].get(name)).get("n_pairs"))) is not None
            }
            if counts:
                pair_counts.append(f"{ref}: {', '.join(str(c) for c in sorted(counts))}")
        if pair_counts:
            lines.extend(["", f"Pairs -- {'; '.join(pair_counts)}."])
        lines.append("")

    unencodable = {
        f"`{name}` ({ref}): {count}"
        for ref, record in records.items()
        for name, raw in _as_mapping(record.get("n_unencodable")).items()
        if (count := _as_int(raw)) is not None and count > 0
    }
    if unencodable:
        lines.extend([f"Un-encodable graphs -- {', '.join(sorted(unencodable))}.", ""])
    return lines


def render_f5_table_md(f5: Mapping[str, Any] | None, f5_path: Path | None) -> str:
    """The F5 correlation table, or a stated absence when no F5 JSON was given."""
    header = [
        "# T-04a -- F5 correlation against graph edit distance",
        "",
        "**Descriptive. F5 is not an input to distance selection** and was computed after the "
        "selection was written to disk.",
        "",
    ]
    if f5 is None:
        header.extend(
            [
                "## The F5 table is absent",
                "",
                "This run was invoked without `--f5`, so no correlation was reported. The "
                "grid-derived artifacts (`supplementary_grid.csv`, `selection.md`, `k.json`) are "
                "complete and unaffected: F5 is descriptive and never feeds selection. Re-run "
                "with `--f5 <f5.json>` to fill this table.",
                "",
            ]
        )
        return "\n".join(header) + "\n"

    header.extend(
        [
            f"Source: `{f5_path}`",
            "",
            f"Bootstrap resamples: {f5.get('bootstrap_resamples', '?')} "
            f"(graph-level percentile CI, seed {f5.get('seed', '?')}).",
            "",
            "Suite-2 datasets report the lower and the upper GED bound as **two separate "
            "columns**. They are never averaged into a midpoint (approx_ged.md section 4).",
            "",
            "**Bold** marks the best rho in that column, the size null included -- a null that "
            "wins its column is the finding. `rho - rho_null` ranks identically to rho within a "
            "column, since the null is one number per column.",
            "",
        ]
    )

    lines = list(header)
    grouped = _group_f5_by_dataset(f5)
    if not grouped:
        lines.extend(["No records in the F5 JSON.", ""])
        return "\n".join(lines) + "\n"
    for dataset, records in grouped.items():
        lines.extend(_dataset_section(dataset, records))
    return "\n".join(lines) + "\n"


# --------------------------------------------------------------------------- #
# driver
# --------------------------------------------------------------------------- #


def load_json(path: Path) -> JsonDict:
    """Load a JSON object from disk.

    Raises:
        ReportError: The file is missing, malformed, or not a JSON object.
    """
    try:
        text = path.read_text(encoding="utf-8")
    except OSError as exc:
        raise ReportError(f"cannot read {path}: {exc}") from exc
    try:
        payload = json.loads(text)
    except json.JSONDecodeError as exc:
        raise ReportError(f"{path} is not valid JSON: {exc}") from exc
    if not isinstance(payload, dict):
        raise ReportError(f"{path} holds a {type(payload).__name__}, expected a JSON object")
    return payload


def write_report(grid_path: Path, f5_path: Path | None, out_dir: Path) -> JsonDict:
    """Write the four artifacts and return the ``k.json`` payload."""
    grid = load_json(grid_path)
    f5 = load_json(f5_path) if f5_path is not None else None
    out_dir.mkdir(parents=True, exist_ok=True)

    n_cells = write_supplementary_csv(grid, out_dir / "supplementary_grid.csv")
    (out_dir / "selection.md").write_text(
        render_selection_md(grid, grid_path.resolve()), encoding="utf-8"
    )
    payload = build_k_payload(grid)
    (out_dir / "k.json").write_text(
        json.dumps(payload, indent=2, sort_keys=False) + "\n", encoding="utf-8"
    )
    (out_dir / "f5_table.md").write_text(
        render_f5_table_md(f5, None if f5_path is None else f5_path.resolve()), encoding="utf-8"
    )

    _LOG.info("supplementary_grid.csv: %d cells", n_cells)
    _LOG.info(
        "k = %d (%s); rows lost to partial computability = %d; N_actual(F2) before d = %d",
        payload["k"],
        ", ".join(payload["k_members"]) or "none",
        sum(int(entry["rows_lost"]) for entry in payload["partial"]),
        payload["n_actual_f2_before_d"],
    )
    if f5 is None:
        _LOG.info("f5_table.md: the F5 table is absent -- no --f5 was given")
    return payload


def main(argv: Sequence[str] | None = None) -> int:
    """CLI entry point."""
    parser = argparse.ArgumentParser(
        prog="python -m isalgraph.competitors.report",
        description="Report the T-04a metric-feasibility grid: supplementary table, "
        "selection table, k.json and the F5 table.",
    )
    parser.add_argument("--grid", required=True, type=Path, help="path to the grid JSON")
    parser.add_argument(
        "--f5", type=Path, default=None, help="path to the F5 JSON; optional and descriptive"
    )
    parser.add_argument("--out-dir", required=True, type=Path, help="directory for the artifacts")
    args = parser.parse_args(argv)

    logging.basicConfig(level=logging.INFO, format="%(message)s")
    try:
        write_report(args.grid, args.f5, args.out_dir)
    except ReportError as exc:
        parser.error(str(exc))
    return 0


if __name__ == "__main__":  # pragma: no cover
    raise SystemExit(main())
