"""Readers and statistics for the frozen T-13 records.  No plotting.

This module reads two frozen formats and nothing else: the ``t13.1``
measurement shards written by
:mod:`benchmarks.real_data.eval_t13_complexity.measure`, and the ``t13c.1``
counter rows written by
:mod:`benchmarks.real_data.eval_t13_complexity.counters`.  Both are validated
against their own schema module rather than against a transcription of it, so a
field renamed upstream is an error here and not a column of nulls.

**Nothing is dropped.**  ``schema`` is explicit that the censored cells are the
high-``|Aut|`` cells -- exactly where the cost law lives -- so a reader that
filtered them would delete the result it was written to show.  Every row loaded
is kept and every summary states, in its own name, which rows it used.

**Censoring is the reason this module is bigger than its T-06 counterpart.**
``status`` is one of ``ok`` / ``censored`` / ``unsupported`` / ``error``, and
``seconds`` on a censored row is an *observation time*, not a completion: the
true completion time is greater than it.  The two censoring mechanisms are not
interchangeable either.  A ``wallclock_kill`` sits at the full 300 s budget; a
``max_projections`` cap on ``min_dfs`` can fire in four milliseconds.  Pooling
either kind with completions produces the specific wrong sentence *"min-DFS is
fast"* when the data says *"min-DFS did not finish"*.  So:

* :func:`completions_only_median_seconds` is named for what it does and
  silently uses no censored row;
* :func:`km_median_seconds` is the Kaplan--Meier estimate (Kaplan and Meier,
  *JASA* 53(282):457--481, 1958), which uses the censored rows as the
  right-censored observations they are and reports when the median is not
  reached at all;
* :func:`completion_rate` is reported beside every one of them, because a
  median over a cohort that finished 20 % of the time is not a runtime.

**Ladder rungs are ordered by their ``params`` index, never by ``log10_aut``.**
Ordering by ``log10_aut`` would make the rung order a function of a variable
that appears on the x axis of the primary figure and in the correlation the
primary table reports, which is circular.  ``params`` carries ``swaps=`` for
``symmetry_ladder`` and ``rung=`` for ``spider_ladder``; those are the design's
own indices and they are what :func:`ladders` sorts on.
"""

from __future__ import annotations

import glob as globlib
import json
import logging
import math
import random
import statistics
from dataclasses import dataclass
from dataclasses import fields as dataclass_fields
from pathlib import Path
from typing import TYPE_CHECKING, Any, Final

from benchmarks.real_data.eval_t13_complexity import schema
from benchmarks.real_data.eval_t13_complexity.instrumented import OperationCounts

if TYPE_CHECKING:  # pragma: no cover - typing only
    from collections.abc import Iterable, Mapping, Sequence

LOGGER: Final = logging.getLogger(__name__)

#: The counter schema this module reads.
COUNTER_SCHEMA_VERSION: Final[str] = "t13c.1"

#: Provenance fields ``counters._row`` writes before the counts.
COUNTER_PROVENANCE: Final[tuple[str, ...]] = (
    "schema_version",
    "source",
    "family",
    "n_target",
    "replicate",
    "dataset",
    "graph_index",
    "n",
    "m",
    "encoder",
)

#: The count fields, taken from the dataclass rather than transcribed.
COUNTER_COUNTS: Final[tuple[str, ...]] = tuple(f.name for f in dataclass_fields(OperationCounts))

#: The whole ``t13c.1`` row, in write order.
COUNTER_FIELDS: Final[tuple[str, ...]] = COUNTER_PROVENANCE + COUNTER_COUNTS + ("parity_ok",)

#: The arm every figure reads unless told otherwise: the shipped engine.
DEFAULT_ARM: Final[str] = "default"

#: The two ladder families, and the ``params`` key each one's rung index lives
#: under.  ``families.ladder_span`` uses exactly this mapping, so a ladder is
#: addressed the same way in the analysis as in the construction.
RUNG_PARAM: Final[dict[str, str]] = {
    "symmetry_ladder": "swaps",
    "spider_ladder": "rung",
}

#: Statuses that are an observation of a duration at all.  ``unsupported`` is a
#: property of the representation and ``error`` is a fault; neither bounds a
#: runtime, so neither may enter a time summary in any form.
OBSERVED_STATUSES: Final[tuple[str, ...]] = ("ok", "censored")


class DataError(ValueError):
    """A record file cannot be read as the analysis needs it."""


class MixedBuildError(DataError):
    """Two shards were produced by different engine builds.

    Not a warning.  ``measure`` aborts a shard whose build differs from the
    campaign's for the same reason: two shards from different builds cannot be
    pooled, and a timing whose build hash is unknown is unprovenanced -- the
    defect that forced T-06's headline rates to be retracted.
    """


class NotALadderError(DataError):
    """A row was addressed as a ladder rung but belongs to no ladder family."""


class LadderIntegrityError(DataError):
    """A ladder's rungs do not hold the invariants the design guarantees.

    ``symmetry_ladder`` swaps are rejected at construction if they change the
    degree sequence (``families.py``), and a ``spider_ladder`` rung redistributes
    leg lengths antisymmetrically, so ``n``, ``m`` and the degree sequence are
    fixed by construction on both.  A ladder in hand that violates that is not a
    ladder, and the whole contrast -- *"|Aut| moved and nothing else did"* --
    would be false for it.
    """


class ParityError(DataError):
    """A counter row's instrumented mirror did not reproduce the reference."""


# ---------------------------------------------------------------------------
# Containers
# ---------------------------------------------------------------------------


@dataclass(frozen=True, slots=True)
class Records:
    """Every measurement row of one campaign, with the shard headers kept.

    Attributes:
        rows: Data rows, in file order across the sorted shard list.
        headers: One entry per header line encountered.  A shard file is
            append-only across a requeue, so a file may carry several; they are
            all kept rather than collapsed.
        build_hash: The single engine build every shard declares.
        run_ids: Every ``run_id`` seen, sorted.
        paths: The shard files read, in order.
    """

    rows: tuple[Mapping[str, Any], ...]
    headers: tuple[Mapping[str, Any], ...]
    build_hash: str
    run_ids: tuple[str, ...]
    paths: tuple[Path, ...]

    @property
    def representations(self) -> tuple[str, ...]:
        """Backend keys present in the data, sorted."""
        return tuple(sorted({str(r["representation"]) for r in self.rows}))

    @property
    def arms(self) -> tuple[str, ...]:
        """Arms present in the data, sorted."""
        return tuple(sorted({str(r["arm"]) for r in self.rows}))

    def with_arm(self, arm: str) -> tuple[Mapping[str, Any], ...]:
        """Return the rows of one arm.

        Args:
            arm: One of ``schema.ARMS``.

        Returns:
            The matching rows, in file order.

        Raises:
            DataError: If *arm* is outside ``schema.ARMS``.
        """
        if arm not in schema.ARMS:
            raise DataError(f"arm={arm!r} is outside {list(schema.ARMS)}")
        return tuple(r for r in self.rows if r["arm"] == arm)


@dataclass(frozen=True, slots=True)
class LadderGraph:
    """One graph of one ladder: a rung, possibly one of several replicates.

    A ``symmetry_ladder`` rung is a *random* search for asymmetry, so several
    replicates of the same rung reach different ``|Aut|``.  They are distinct
    graphs and are kept distinct; only the ordering key is shared.

    Attributes:
        rung: The design's own rung index, read from ``params``.
        replicate: Replicate index within the rung.
        graph_key: Stable address of the graph, as ``measure`` spells it.
        n: Order, constant across the ladder.
        m: Size, constant across the ladder.
        max_degree: Largest degree, constant across the ladder.
        log10_aut: ``log10|Aut(G)|``, or ``None`` when the shard ran without
            the symmetry toolkit.
        rows: The measurement rows for this graph under one arm, one per
            representation.
    """

    rung: int
    replicate: int
    graph_key: str
    n: int
    m: int
    max_degree: int
    log10_aut: float | None
    rows: tuple[Mapping[str, Any], ...]

    def row(self, representation: str) -> Mapping[str, Any] | None:
        """Return this graph's row for *representation*, or ``None``.

        Args:
            representation: Backend key.

        Returns:
            The row, or ``None`` when the arm produced none.
        """
        for row in self.rows:
            if row["representation"] == representation:
                return row
        return None


@dataclass(frozen=True, slots=True)
class Ladder:
    """One ladder: ``n``, ``m`` and the degree sequence fixed, ``|Aut|`` falling.

    Attributes:
        family: ``symmetry_ladder`` or ``spider_ladder``.
        n: Order, shared by every graph.
        base: ``complete_bipartite`` / ``hypercube`` for a symmetry ladder,
            ``spider_k<k>`` for a spider ladder.
        graphs: Ordered by ``(rung, replicate)`` -- by the design's index,
            never by ``log10_aut``.
    """

    family: str
    n: int
    base: str
    graphs: tuple[LadderGraph, ...]

    @property
    def key(self) -> tuple[str, int, str]:
        """The ``(family, n, base)`` triple this ladder is grouped on."""
        return (self.family, self.n, self.base)

    @property
    def title(self) -> str:
        """A short panel title.

        Deliberately short, and carrying no underscore: a panel title is set at
        8 pt across a third of the IEEE text width, where the full grouping key
        overruns the panel and collides with the panel letter, and matplotlib's
        mathtext reads a bare ``_`` as a subscript.
        """
        if self.family == "spider_ladder":
            return f"spider, $k={self.base.removeprefix('spider_k')}$, $n={self.n}$"
        return f"{self.base.replace('_', ' ')}, $n={self.n}$"

    @property
    def rungs(self) -> tuple[int, ...]:
        """Distinct rung indices, ascending."""
        return tuple(sorted({g.rung for g in self.graphs}))

    @property
    def m(self) -> int:
        """Size, shared by every graph (checked by :func:`ladders`)."""
        return self.graphs[0].m

    @property
    def max_degree(self) -> int:
        """Largest degree, shared by every graph."""
        return self.graphs[0].max_degree

    @property
    def aut_span(self) -> float | None:
        """Range of ``log10|Aut|`` spanned, or ``None`` when unmeasured."""
        values = [g.log10_aut for g in self.graphs if g.log10_aut is not None]
        return max(values) - min(values) if len(values) >= 2 else None

    def series(self, representation: str) -> tuple[tuple[LadderGraph, Mapping[str, Any]], ...]:
        """Return this ladder's ``(graph, row)`` pairs for one representation.

        Args:
            representation: Backend key.

        Returns:
            Pairs in ladder order, omitting graphs the arm produced no row for.
        """
        out: list[tuple[LadderGraph, Mapping[str, Any]]] = []
        for graph in self.graphs:
            row = graph.row(representation)
            if row is not None:
                out.append((graph, row))
        return tuple(out)


@dataclass(frozen=True, slots=True)
class TimeSummary:
    """Everything a caller needs to state a runtime honestly.

    Attributes:
        n_observations: Rows whose ``status`` is ``ok`` or ``censored``.
        n_completed: ``ok`` rows.
        n_censored: ``censored`` rows.
        n_unsupported: ``unsupported`` rows, excluded from every estimate.
        n_error: ``error`` rows, excluded from every estimate.
        completion_rate: ``n_completed / n_observations``, or ``None`` when
            nothing was observed.
        completions_only_median: Median ``seconds`` over ``ok`` rows alone.
            Biased low whenever anything was censored, which is why it is
            named for its rule rather than called *the* median.
        km_median: Kaplan--Meier median completion time, or ``None`` when the
            survival curve never reaches 0.5.
        km_median_reached: Whether :attr:`km_median` is an estimate (``True``)
            or the median is simply not identified from this data (``False``).
        max_observed: Largest observation time of any kind, which is the
            lower bound to quote when the median is not reached.
        censoring_kinds: Count per ``error_kind`` over the censored rows.  A
            wall-clock kill and a projection cap are different processes and a
            reader must be able to tell which produced the censoring.
    """

    n_observations: int
    n_completed: int
    n_censored: int
    n_unsupported: int
    n_error: int
    completion_rate: float | None
    completions_only_median: float | None
    km_median: float | None
    km_median_reached: bool
    max_observed: float | None
    censoring_kinds: tuple[tuple[str, int], ...]


@dataclass(frozen=True, slots=True)
class SignTest:
    """An exact two-sided sign test.

    Attributes:
        n_positive: Differences above zero.
        n_negative: Differences below zero.
        n_ties: Differences exactly zero, dropped from the test.
        p_value: Exact two-sided binomial p under ``p = 1/2``.
    """

    n_positive: int
    n_negative: int
    n_ties: int
    p_value: float


@dataclass(frozen=True, slots=True)
class PowerLawFit:
    """An ordinary least-squares fit of ``log T`` on ``log n``.

    Attributes:
        alpha: Fitted exponent of ``T ~ n^alpha``.
        intercept: Fitted intercept in log space.
        n_points: Completed observations the fit used.
        n_distinct_n: Distinct node counts among them.
        ci_low: Lower percentile-bootstrap bound on :attr:`alpha`.
        ci_high: Upper bound.
    """

    alpha: float
    intercept: float
    n_points: int
    n_distinct_n: int
    ci_low: float
    ci_high: float


@dataclass(frozen=True, slots=True)
class GraphResolution:
    """One graph's partition-resolution record.

    Attributes:
        graph_key: Stable address of the graph.
        family: Constructed family, or ``None`` for a cohort graph.
        dataset: Cohort name, or ``None`` for a constructed graph.
        n: Order.
        n_orbits: Automorphism orbits -- the invariance ceiling of
            Proposition 1.
        n_wl_classes: Stable 1-WL colour classes.
        n_triplet_classes: Classes of the incumbent pruning key.
        log10_aut: ``log10|Aut(G)|``.
        wl_equals_orbits: Whether 1-WL resolves exactly the orbits.
        triplet_equals_orbits: Likewise for the triplet key.
    """

    graph_key: str
    family: str | None
    dataset: str | None
    n: int
    n_orbits: int
    n_wl_classes: int
    n_triplet_classes: int
    log10_aut: float | None
    wl_equals_orbits: bool | None
    triplet_equals_orbits: bool | None


@dataclass(frozen=True, slots=True)
class CounterRecords:
    """Every ``t13c.1`` counter row read.

    Attributes:
        rows: The rows, in file order across the sorted file list.
        paths: The files read.
    """

    rows: tuple[Mapping[str, Any], ...]
    paths: tuple[Path, ...]

    @property
    def encoders(self) -> tuple[str, ...]:
        """Encoder names present, sorted."""
        return tuple(sorted({str(r["encoder"]) for r in self.rows}))


# ---------------------------------------------------------------------------
# Loading
# ---------------------------------------------------------------------------


def resolve_paths(patterns: Iterable[str | Path]) -> tuple[Path, ...]:
    """Expand shell globs into a sorted, de-duplicated file list.

    Args:
        patterns: Globs or literal paths.  A directory expands to the
            ``*.jsonl`` files directly inside it.

    Returns:
        Existing files, sorted by path.

    Raises:
        FileNotFoundError: If nothing matched.  An empty match must never
            produce an empty figure that regenerates without error.
    """
    found: set[Path] = set()
    for pattern in patterns:
        text = str(pattern)
        candidate = Path(text)
        if candidate.is_dir():
            found.update(p for p in candidate.glob("*.jsonl") if p.is_file())
            continue
        matches = [Path(p) for p in globlib.glob(text)]
        found.update(p for p in matches if p.is_file())
    if not found:
        raise FileNotFoundError(f"no record files matched {[str(p) for p in patterns]}")
    return tuple(sorted(found))


def load(paths: Iterable[str | Path]) -> Records:
    """Read one or many ``t13.1`` shards, validating every row.

    Args:
        paths: Globs, directories or literal shard paths.

    Returns:
        Every row of every shard, with the headers kept.

    Raises:
        FileNotFoundError: If nothing matched.
        DataError: On a shard with no header, an unknown schema version, or a
            file with no rows at all.
        MixedBuildError: If the shards do not all declare one ``build_hash``.
        schema.SchemaError: On any row that does not match the frozen field
            set or its domains -- propagated, never caught.
    """
    files = resolve_paths(paths)
    rows: list[Mapping[str, Any]] = []
    headers: list[Mapping[str, Any]] = []
    hashes: dict[str, list[str]] = {}
    run_ids: set[str] = set()

    for path in files:
        file_rows, file_headers = _read_shard(path)
        if not file_headers:
            raise DataError(f"{path} carries no header line; it cannot be provenanced")
        for header in file_headers:
            build_hash = str(header.get("build_info", {}).get("build_hash", ""))
            if not build_hash:
                raise DataError(f"{path} header carries no build_info.build_hash")
            hashes.setdefault(build_hash, []).append(str(path))
            run_ids.add(str(header["run_id"]))
        headers.extend(file_headers)
        rows.extend(file_rows)

    if len(hashes) != 1:
        detail = "; ".join(f"{h}: {sorted(set(files_))}" for h, files_ in sorted(hashes.items()))
        raise MixedBuildError(
            f"the shards declare {len(hashes)} different engine builds and cannot be "
            f"pooled -- a timing whose build is unknown is unprovenanced. {detail}"
        )
    if not rows:
        raise DataError(f"{[str(p) for p in files]} carry headers but no measurement rows")

    build_hash = next(iter(hashes))
    LOGGER.info(
        "loaded %d rows from %d shard(s), build %s, run_id(s) %s",
        len(rows),
        len(files),
        build_hash,
        sorted(run_ids),
    )
    return Records(
        rows=tuple(rows),
        headers=tuple(headers),
        build_hash=build_hash,
        run_ids=tuple(sorted(run_ids)),
        paths=files,
    )


def _read_shard(path: Path) -> tuple[list[Mapping[str, Any]], list[Mapping[str, Any]]]:
    """Split one shard file into rows and headers, validating each row.

    Args:
        path: The shard file.

    Returns:
        ``(rows, headers)``.

    Raises:
        DataError: On a malformed line or an unknown header schema version.
        schema.SchemaError: On a row that fails ``schema.validate_mapping``.
    """
    rows: list[Mapping[str, Any]] = []
    headers: list[Mapping[str, Any]] = []
    with path.open(encoding="utf-8") as handle:
        for lineno, line in enumerate(handle, start=1):
            text = line.strip()
            if not text:
                continue
            try:
                obj = json.loads(text)
            except json.JSONDecodeError as exc:
                raise DataError(f"{path}:{lineno} is not valid JSON: {exc}") from exc
            if obj.get("record_kind") == "header":
                version = obj.get("schema_version")
                if version != schema.SCHEMA_VERSION:
                    raise DataError(
                        f"{path}:{lineno} header declares schema {version!r}, "
                        f"this analysis reads {schema.SCHEMA_VERSION!r}"
                    )
                headers.append(obj)
                continue
            schema.validate_mapping(obj)
            rows.append(obj)
    return rows, headers


def load_counters(paths: Iterable[str | Path], *, strict_parity: bool = True) -> CounterRecords:
    """Read one or many ``t13c.1`` counter files.

    Args:
        paths: Globs, directories or literal paths.
        strict_parity: Reject any row whose ``parity_ok`` is not ``True``.  A
            counter row is only a measurement of the frozen algorithm if the
            instrumented mirror reproduced the reference string byte for byte;
            plotting one that did not is plotting a number from unverified
            code.

    Returns:
        Every row of every file.

    Raises:
        FileNotFoundError: If nothing matched.
        DataError: On a malformed line, a wrong schema version, or a row whose
            field set is not exactly :data:`COUNTER_FIELDS`.
        ParityError: On a ``parity_ok`` failure when *strict_parity*.
    """
    files = resolve_paths(paths)
    rows: list[Mapping[str, Any]] = []
    for path in files:
        with path.open(encoding="utf-8") as handle:
            for lineno, line in enumerate(handle, start=1):
                text = line.strip()
                if not text:
                    continue
                try:
                    obj = json.loads(text)
                except json.JSONDecodeError as exc:
                    raise DataError(f"{path}:{lineno} is not valid JSON: {exc}") from exc
                _validate_counter_row(obj, where=f"{path}:{lineno}")
                if strict_parity and obj["parity_ok"] is not True:
                    raise ParityError(
                        f"{path}:{lineno} has parity_ok={obj['parity_ok']!r}: the "
                        f"instrumented mirror did not reproduce the frozen reference, "
                        f"so its counts do not describe the shipped algorithm"
                    )
                rows.append(obj)
    if not rows:
        raise DataError(f"{[str(p) for p in files]} carry no counter rows")
    LOGGER.info("loaded %d counter rows from %d file(s)", len(rows), len(files))
    return CounterRecords(rows=tuple(rows), paths=files)


def _validate_counter_row(obj: Mapping[str, Any], *, where: str) -> None:
    """Raise unless *obj* is exactly a ``t13c.1`` row.

    Args:
        obj: Candidate row.
        where: ``path:lineno``, for the message.

    Raises:
        DataError: On a missing field, an extra field, or a wrong version.
    """
    keys = set(obj)
    expected = set(COUNTER_FIELDS)
    missing = sorted(expected - keys)
    extra = sorted(keys - expected)
    if missing or extra:
        raise DataError(
            f"{where} does not match counter schema {COUNTER_SCHEMA_VERSION}: "
            f"missing={missing} extra={extra}"
        )
    if obj["schema_version"] != COUNTER_SCHEMA_VERSION:
        raise DataError(
            f"{where} declares schema {obj['schema_version']!r}, "
            f"this analysis reads {COUNTER_SCHEMA_VERSION!r}"
        )


# ---------------------------------------------------------------------------
# Addressing
# ---------------------------------------------------------------------------


def parse_params(text: str | None) -> dict[str, str]:
    """Parse a rendered ``params`` string into a mapping.

    Args:
        text: ``"base=hypercube,swaps=3"``, or ``None`` for a cohort row.

    Returns:
        The parsed parameters, empty for ``None`` or an empty string.

    Raises:
        DataError: On a fragment with no ``=``.
    """
    if not text:
        return {}
    out: dict[str, str] = {}
    for fragment in text.split(","):
        if "=" not in fragment:
            raise DataError(f"params fragment {fragment!r} has no '='; whole value {text!r}")
        name, _, value = fragment.partition("=")
        out[name] = value
    return out


def graph_identity(row: Mapping[str, Any]) -> str:
    """Return the stable address of the graph a row measured.

    The seven-field prefix of the unit key ``measure`` shards on, so a graph is
    addressed identically in the analysis and in the campaign.

    Args:
        row: A validated measurement row.

    Returns:
        The address.
    """
    return (
        f"{row['source']}|{row['family']}|{row['n_target']}|{row['replicate']}"
        f"|{row['params']}|{row['dataset']}|{row['graph_index']}"
    )


def rung_index(row: Mapping[str, Any]) -> int:
    """Return a ladder row's rung index, read from ``params``.

    **Never derived from** ``log10_aut``.  The rung index is the design's
    independent variable; ``log10_aut`` is what the ladder moves, and ordering
    by it would make the abscissa of the primary figure decide the order of the
    correlation computed on it.

    Args:
        row: A validated measurement row.

    Returns:
        ``swaps`` for a ``symmetry_ladder`` row, ``rung`` for a
        ``spider_ladder`` row.

    Raises:
        NotALadderError: If the row's family is neither ladder family.
        DataError: If ``params`` does not carry the expected key or it is not
            an integer.
    """
    family = row["family"]
    key = RUNG_PARAM.get(str(family))
    if key is None:
        raise NotALadderError(
            f"family={family!r} is not a ladder family; expected one of {sorted(RUNG_PARAM)}"
        )
    params = parse_params(row["params"])
    if key not in params:
        raise DataError(
            f"a {family} row carries params={row['params']!r} with no {key!r}: "
            f"without it the rungs are indistinguishable and the ladder contrast "
            f"cannot be computed"
        )
    try:
        return int(params[key])
    except ValueError as exc:
        raise DataError(f"{key}={params[key]!r} is not an integer in {row['params']!r}") from exc


def ladder_key(row: Mapping[str, Any]) -> tuple[str, int, str]:
    """Return the ``(family, n, base)`` triple a ladder row groups under.

    Mirrors ``families.ladder_span``: a symmetry ladder is keyed by its base
    name, a spider ladder by its leg count.

    Args:
        row: A validated measurement row.

    Returns:
        The grouping triple.

    Raises:
        NotALadderError: If the row's family is neither ladder family.
        DataError: If ``params`` lacks the key the family needs.
    """
    family = str(row["family"])
    params = parse_params(row["params"])
    if family == "symmetry_ladder":
        if "base" not in params:
            raise DataError(f"symmetry_ladder row carries no base in params={row['params']!r}")
        return (family, int(row["n"]), params["base"])
    if family == "spider_ladder":
        if "legs" not in params:
            raise DataError(f"spider_ladder row carries no legs in params={row['params']!r}")
        return (family, int(row["n"]), f"spider_k{params['legs']}")
    raise NotALadderError(
        f"family={family!r} is not a ladder family; expected one of {sorted(RUNG_PARAM)}"
    )


def ladders(records: Records, *, arm: str = DEFAULT_ARM) -> tuple[Ladder, ...]:
    """Group a campaign's rows into ladders, rungs in design order.

    Args:
        records: The loaded campaign.
        arm: Which engine arm to read.  Defaults to the shipped engine.

    Returns:
        Ladders sorted by ``(family, n, base)``, each with its graphs ordered
        by ``(rung, replicate)``.

    Raises:
        DataError: If *arm* is outside ``schema.ARMS``.
        LadderIntegrityError: If a ladder's ``n``, ``m`` or ``max_degree``
            varies across its graphs.  The ladder's premise is that only
            ``|Aut|`` moved; a ladder that fails it cannot carry the contrast.
    """
    grouped: dict[tuple[str, int, str], dict[str, list[Mapping[str, Any]]]] = {}
    for row in records.with_arm(arm):
        if str(row["family"]) not in RUNG_PARAM:
            continue
        key = ladder_key(row)
        grouped.setdefault(key, {}).setdefault(graph_identity(row), []).append(row)

    out: list[Ladder] = []
    for key in sorted(grouped):
        family, n, base = key
        graphs: list[LadderGraph] = []
        for graph_key, rows in grouped[key].items():
            head = rows[0]
            graphs.append(
                LadderGraph(
                    rung=rung_index(head),
                    replicate=int(head["replicate"] or 0),
                    graph_key=graph_key,
                    n=int(head["n"]),
                    m=int(head["m"]),
                    max_degree=int(head["max_degree"]),
                    log10_aut=_opt_float(head["log10_aut"]),
                    rows=tuple(rows),
                )
            )
        graphs.sort(key=lambda g: (g.rung, g.replicate, g.graph_key))
        _check_ladder_invariants(key, graphs)
        out.append(Ladder(family=family, n=n, base=base, graphs=tuple(graphs)))
    return tuple(out)


def _check_ladder_invariants(key: tuple[str, int, str], graphs: Sequence[LadderGraph]) -> None:
    """Raise unless every graph of one ladder shares ``n``, ``m`` and ``Delta``.

    Args:
        key: The ladder's grouping triple, for the message.
        graphs: The ladder's graphs.

    Raises:
        LadderIntegrityError: On any variation.
    """
    for name, values in (
        ("n", {g.n for g in graphs}),
        ("m", {g.m for g in graphs}),
        ("max_degree", {g.max_degree for g in graphs}),
    ):
        if len(values) > 1:
            raise LadderIntegrityError(
                f"ladder {key} varies in {name}: {sorted(values)}. The ladder holds n, m "
                f"and the degree sequence fixed by construction, and the whole contrast "
                f"is that only |Aut| moved"
            )


def resolutions(records: Records) -> tuple[GraphResolution, ...]:
    """Return one partition-resolution record per distinct graph.

    Rows repeat the nine symmetry fields once per ``(representation, arm)``,
    so they are collapsed here.  A graph whose shard ran without the symmetry
    toolkit carries nulls and is skipped, with a count logged: a null is a
    missing measurement, not a zero.

    Args:
        records: The loaded campaign.

    Returns:
        Resolution records, sorted by ``(n, graph_key)``.

    Raises:
        DataError: If two rows of one graph disagree on a symmetry field.
    """
    seen: dict[str, GraphResolution] = {}
    skipped = 0
    for row in records.rows:
        key = graph_identity(row)
        needed = (row["n_orbits"], row["n_wl_classes"], row["n_triplet_classes"])
        if any(value is None for value in needed):
            if key not in seen:
                skipped += 1
            continue
        record = GraphResolution(
            graph_key=key,
            family=row["family"],
            dataset=row["dataset"],
            n=int(row["n"]),
            n_orbits=int(row["n_orbits"]),
            n_wl_classes=int(row["n_wl_classes"]),
            n_triplet_classes=int(row["n_triplet_classes"]),
            log10_aut=_opt_float(row["log10_aut"]),
            wl_equals_orbits=row["wl_equals_orbits"],
            triplet_equals_orbits=row["triplet_equals_orbits"],
        )
        previous = seen.get(key)
        if previous is not None and previous != record:
            raise DataError(
                f"graph {key} carries two different symmetry records: "
                f"{previous} vs {record}. Two shards of different provenance were pooled"
            )
        seen[key] = record
    if skipped:
        LOGGER.warning(
            "%d graph(s) carry null symmetry fields and are excluded from the "
            "resolution figure; their shard ran with symmetry_available=false",
            skipped,
        )
    return tuple(sorted(seen.values(), key=lambda r: (r.n, r.graph_key)))


def _opt_float(value: Any) -> float | None:
    """Return *value* as a float, or ``None`` when it is null."""
    return None if value is None else float(value)


# ---------------------------------------------------------------------------
# Censoring-aware summaries
# ---------------------------------------------------------------------------


def is_completed(row: Mapping[str, Any]) -> bool:
    """Return whether *row* timed a completed encoding."""
    return bool(row["status"] == "ok")


def is_censored(row: Mapping[str, Any]) -> bool:
    """Return whether *row* is a right-censored observation."""
    return bool(row["status"] == "censored")


def is_observation(row: Mapping[str, Any]) -> bool:
    """Return whether *row* observes a duration at all.

    ``unsupported`` and ``error`` do not: the first is a property of the
    representation and the second is a fault, and neither bounds a runtime.
    """
    return bool(row["status"] in OBSERVED_STATUSES)


def completion_rate(rows: Iterable[Mapping[str, Any]]) -> float | None:
    """Return the fraction of observations that completed.

    Reported beside every median in this module.  A median over a cohort that
    finished a fifth of the time is not a runtime, and the only way a reader
    can tell is if the rate is printed next to it.

    Args:
        rows: Measurement rows.

    Returns:
        ``n_completed / n_observations``, or ``None`` when nothing was
        observed.
    """
    observed = [r for r in rows if is_observation(r)]
    if not observed:
        return None
    return sum(1 for r in observed if is_completed(r)) / len(observed)


def completions_only_median_seconds(rows: Iterable[Mapping[str, Any]]) -> float | None:
    """Return the median ``seconds`` over completed rows **only**.

    Named for its rule.  It is biased low whenever anything was censored,
    because every censored row is an observation whose true value lies above
    its recorded ``seconds`` and is simply discarded here.  Quote it beside
    :func:`completion_rate`, never alone.

    Args:
        rows: Measurement rows.

    Returns:
        The median, or ``None`` when nothing completed.
    """
    values = [float(r["seconds"]) for r in rows if is_completed(r)]
    return statistics.median(values) if values else None


def km_median_seconds(rows: Iterable[Mapping[str, Any]]) -> tuple[float | None, bool]:
    """Return the Kaplan--Meier median completion time.

    The Kaplan--Meier product-limit estimator (Kaplan and Meier, *JASA*
    53(282):457--481, 1958) treats a ``censored`` row as an observation that
    the completion time exceeds its ``seconds`` -- which is exactly what
    ``schema`` says it is -- and estimates

    ``S(t) = prod_{t_i <= t} (1 - d_i / r_i)``

    over the distinct completion times ``t_i``, with ``d_i`` completions at
    ``t_i`` and ``r_i`` units still at risk just before it.  The median is
    ``min{t : S(t) <= 1/2}``.

    Args:
        rows: Measurement rows.  ``unsupported`` and ``error`` rows are
            excluded; they observe no duration.

    Returns:
        ``(median, reached)``.  When *reached* is ``False`` the median is
        ``None`` and is **not identified** by this data -- more than half the
        units were still running when observation stopped.  Report
        :attr:`TimeSummary.max_observed` as a lower bound instead of
        substituting the completions-only median, which would be the bias this
        estimator exists to avoid.
    """
    events = sorted((float(r["seconds"]), is_completed(r)) for r in rows if is_observation(r))
    at_risk = len(events)
    if at_risk == 0:
        return (None, False)

    survival = 1.0
    index = 0
    while index < len(events):
        time = events[index][0]
        tied = [e for e in events[index:] if e[0] == time]
        completions = sum(1 for _, completed in tied if completed)
        if completions and at_risk > 0:
            survival *= 1.0 - completions / at_risk
            if survival <= 0.5:
                return (time, True)
        at_risk -= len(tied)
        index += len(tied)
    return (None, False)


def summarise_times(rows: Iterable[Mapping[str, Any]]) -> TimeSummary:
    """Return every censoring-aware statistic for one set of rows.

    Args:
        rows: Measurement rows for one ``(representation, cell)``.

    Returns:
        The summary.  Every field is derived from the ``status`` domain, and
        no field pools a censored row with a completed one.
    """
    materialised = list(rows)
    observed = [r for r in materialised if is_observation(r)]
    completed = [r for r in observed if is_completed(r)]
    censored = [r for r in observed if is_censored(r)]
    kinds: dict[str, int] = {}
    for row in censored:
        kind = str(row["error_kind"])
        kinds[kind] = kinds.get(kind, 0) + 1
    km, reached = km_median_seconds(observed)
    return TimeSummary(
        n_observations=len(observed),
        n_completed=len(completed),
        n_censored=len(censored),
        n_unsupported=sum(1 for r in materialised if r["status"] == "unsupported"),
        n_error=sum(1 for r in materialised if r["status"] == "error"),
        completion_rate=(len(completed) / len(observed)) if observed else None,
        completions_only_median=(
            statistics.median([float(r["seconds"]) for r in completed]) if completed else None
        ),
        km_median=km,
        km_median_reached=reached,
        max_observed=max((float(r["seconds"]) for r in observed), default=None),
        censoring_kinds=tuple(sorted(kinds.items())),
    )


# ---------------------------------------------------------------------------
# Plain-Python statistics
# ---------------------------------------------------------------------------


def _ranks(values: Sequence[float]) -> list[float]:
    """Return mid-ranks, averaging over ties."""
    order = sorted(range(len(values)), key=lambda i: values[i])
    out = [0.0] * len(values)
    i = 0
    while i < len(order):
        j = i
        while j + 1 < len(order) and values[order[j + 1]] == values[order[i]]:
            j += 1
        mid = (i + j) / 2.0 + 1.0
        for k in range(i, j + 1):
            out[order[k]] = mid
        i = j + 1
    return out


def spearman(xs: Sequence[float], ys: Sequence[float]) -> float | None:
    """Return Spearman's rho, tie-corrected, in plain Python.

    Computed as the Pearson correlation of mid-ranks, which is the definition
    that stays correct in the presence of ties -- the ``6 sum d^2`` shortcut
    does not, and ladder rungs tie routinely.

    Args:
        xs: First variable.
        ys: Second variable, same length.

    Returns:
        Rho in ``[-1, 1]``, or ``None`` when fewer than three pairs are
        available or either variable is constant (rho is undefined then, and
        returning 0.0 would read as "no relationship").

    Raises:
        ValueError: If the two sequences differ in length.
    """
    if len(xs) != len(ys):
        raise ValueError(f"spearman needs equal lengths, got {len(xs)} and {len(ys)}")
    if len(xs) < 3:
        return None
    rx = _ranks(xs)
    ry = _ranks(ys)
    mx = statistics.fmean(rx)
    my = statistics.fmean(ry)
    num = sum((a - mx) * (b - my) for a, b in zip(rx, ry))
    den_x = math.sqrt(sum((a - mx) ** 2 for a in rx))
    den_y = math.sqrt(sum((b - my) ** 2 for b in ry))
    if den_x == 0.0 or den_y == 0.0:
        return None
    return num / (den_x * den_y)


def sign_test(deltas: Sequence[float]) -> SignTest:
    """Return the exact two-sided sign test of *deltas* against zero.

    Args:
        deltas: Paired differences.  Exact zeros are dropped, which is the
            conventional treatment and is reported in
            :attr:`SignTest.n_ties`.

    Returns:
        The test.  With every difference tied the p-value is ``1.0``.
    """
    pos = sum(1 for d in deltas if d > 0)
    neg = sum(1 for d in deltas if d < 0)
    ties = sum(1 for d in deltas if d == 0)
    n = pos + neg
    if n == 0:
        return SignTest(n_positive=pos, n_negative=neg, n_ties=ties, p_value=1.0)
    k = min(pos, neg)
    tail = sum(math.comb(n, i) for i in range(k + 1))
    p = min(1.0, 2.0 * tail / (2.0**n))
    return SignTest(n_positive=pos, n_negative=neg, n_ties=ties, p_value=p)


def fit_power_law_completions_only(
    rows: Iterable[Mapping[str, Any]],
    *,
    n_boot: int = 2000,
    seed: int = 13,
    alpha_level: float = 0.05,
) -> PowerLawFit | None:
    """Fit ``T ~ n^alpha`` over **completed rows only**, with a bootstrap CI.

    Named for its rule.  A censored row carries no completion time, so it
    cannot enter a least-squares fit at all; the honest consequence is that
    the exponent describes only the graphs this implementation finished, and
    ``T-13-design.md`` 2.2 leg (iii) is explicit that such an exponent is *a
    property of the cohort, not of the algorithm*.  Report it with the
    completion rate beside it.

    Args:
        rows: Measurement rows for one representation.
        n_boot: Percentile-bootstrap resamples.
        seed: Seed of the single ``random.Random`` stream, so the CI is
            reproducible.
        alpha_level: Two-sided level; ``0.05`` gives a 95 % interval.

    Returns:
        The fit, or ``None`` when fewer than three completed rows or fewer
        than two distinct node counts are available -- with one distinct ``n``
        the slope is not identified.
    """
    points = [
        (math.log(float(r["n"])), math.log(float(r["seconds"])))
        for r in rows
        if is_completed(r) and int(r["n"]) > 0 and float(r["seconds"]) > 0.0
    ]
    distinct = {p[0] for p in points}
    if len(points) < 3 or len(distinct) < 2:
        return None

    slope, intercept = _ols(points)
    rng = random.Random(seed)
    slopes: list[float] = []
    for _ in range(n_boot):
        sample = [points[rng.randrange(len(points))] for _ in range(len(points))]
        if len({p[0] for p in sample}) < 2:
            continue
        slopes.append(_ols(sample)[0])
    if not slopes:
        return None
    slopes.sort()
    lo = slopes[max(0, int(math.floor((alpha_level / 2.0) * len(slopes))))]
    hi = slopes[min(len(slopes) - 1, int(math.ceil((1.0 - alpha_level / 2.0) * len(slopes))) - 1)]
    return PowerLawFit(
        alpha=slope,
        intercept=intercept,
        n_points=len(points),
        n_distinct_n=len(distinct),
        ci_low=lo,
        ci_high=hi,
    )


def _ols(points: Sequence[tuple[float, float]]) -> tuple[float, float]:
    """Return the ``(slope, intercept)`` of an ordinary least-squares fit."""
    mx = statistics.fmean(p[0] for p in points)
    my = statistics.fmean(p[1] for p in points)
    sxx = sum((p[0] - mx) ** 2 for p in points)
    sxy = sum((p[0] - mx) * (p[1] - my) for p in points)
    slope = sxy / sxx if sxx else 0.0
    return (slope, my - slope * mx)


__all__ = [
    "COUNTER_COUNTS",
    "COUNTER_FIELDS",
    "COUNTER_PROVENANCE",
    "COUNTER_SCHEMA_VERSION",
    "DEFAULT_ARM",
    "OBSERVED_STATUSES",
    "RUNG_PARAM",
    "CounterRecords",
    "DataError",
    "GraphResolution",
    "Ladder",
    "LadderGraph",
    "LadderIntegrityError",
    "MixedBuildError",
    "NotALadderError",
    "ParityError",
    "PowerLawFit",
    "Records",
    "SignTest",
    "TimeSummary",
    "completion_rate",
    "completions_only_median_seconds",
    "fit_power_law_completions_only",
    "graph_identity",
    "is_censored",
    "is_completed",
    "is_observation",
    "km_median_seconds",
    "ladder_key",
    "ladders",
    "load",
    "load_counters",
    "parse_params",
    "resolutions",
    "resolve_paths",
    "rung_index",
    "sign_test",
    "spearman",
    "summarise_times",
]
