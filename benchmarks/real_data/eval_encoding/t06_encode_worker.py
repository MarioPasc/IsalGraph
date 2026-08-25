"""Encode graphs under one representation, one JSON line each, in a killable process.

This module exists so that the per-graph budget can be enforced by **killing a
process**. A Python signal-based timeout does not work here: CPython runs a
signal handler only between bytecode instructions, so ``SIGALRM`` stays queued
for the whole duration of a native call and the C++ encoder runs to completion
with the alarm pending. A previous attempt hung for 25 minutes on one graph with
the budget silently not applying, and it presented as a hang rather than an
error. Nothing in this file or its driver may use ``signal.setitimer`` or
``signal.alarm``; a test asserts their absence.

The worker streams: it loads the cohort once and prints one JSON record per
graph, flushing each line. The parent reads lines with a per-line deadline and
kills the child when one is late, which gives a hard per-graph wall clock while
paying the interpreter start-up cost once per chunk instead of once per graph.

The worker never decides D14. It reports what happened -- ``ok``, or ``error``
with the exception class name -- and the driver applies the fallback policy.
Keeping the policy in one place is what stops a censored graph being silently
dropped in one code path and retained in another.
"""

from __future__ import annotations

import argparse
import json
import logging
import sys
import time
from collections.abc import Mapping, Sequence
from dataclasses import asdict, dataclass, replace
from pathlib import Path
from typing import TYPE_CHECKING, Any

from benchmarks.real_data.eval_encoding.t06_cohort import Cohort, load_cohort

if TYPE_CHECKING:  # pragma: no cover - typing only
    import networkx as nx

    from isalgraph.competitors.base import Encoding

LOGGER = logging.getLogger(__name__)

#: Sigma = {N, n, P, p, V, v, C, c, W}, the IsalGraph alphabet. Duplicated from
#: ``competitors.backends.isalgraph_ref`` rather than imported so the D14
#: fallback, which does not go through a backend, cannot drift from it
#: silently; a test asserts the two agree.
ISALGRAPH_ALPHABET_SIZE = 9

#: Representations whose encode is an IsalGraph string and therefore have a
#: fallback under D14.
ISALGRAPH_ARMS: tuple[str, ...] = (
    "isalgraph_pruned",
    "isalgraph_canonical",
    "isalgraph_exhaustive",
)

#: D14 fallback cascade, per representation. Each entry is tried in order and
#: the first that produces a string wins; the last tier must be unbudgeted so
#: that "a censored graph is retained, never dropped" is actually true.
#:
#: ``isalgraph_exhaustive`` prefers ``pruned`` over ``greedy``: the pruned form
#: is still a *canonical* form, so a substituted row stays inside the
#: completeness theorem, whereas a greedy-min row does not. Pruned has its own
#: ceiling -- T-06 measured 24/400 on Mutagenicity and 4/400 on Protein at a 2 s
#: budget -- so ``greedy`` still closes the cascade.
#:
#: The keys mirror ``ReprBackend.fallback_variant``; :func:`fallback_cascade`
#: reads the backend and falls back to this table, and a test asserts the two
#: agree rather than trusting either alone.
FALLBACK_CASCADE: dict[str, tuple[str, ...]] = {
    "isalgraph_exhaustive": ("pruned", "greedy"),
}

#: Cascade for any representation not named above.
DEFAULT_CASCADE: tuple[str, ...] = ("greedy",)

MODES: tuple[str, ...] = ("primary", "fallback")

#: ``CONTRACTS.md`` §3.1. ASCII unit separator, used for the representations
#: whose symbols are not single characters.
UNIT_SEP = "\x1f"

#: Representation -> separator, **frozen per representation rather than derived
#: per graph.** ``size_null``'s single symbol is ``str(n)``, one character below
#: n = 10 and two above it, so a per-graph rule would give one file two
#: separators and the consumer no way to split it. Everything not listed uses
#: ``""``, and the assertions in :func:`_join_symbols` check that choice on
#: every encode rather than trusting this table.
SYMBOL_SEP: dict[str, str] = {
    "min_dfs": UNIT_SEP,
    "size_null": UNIT_SEP,
    "wl_subtree": UNIT_SEP,
}

#: Exception class names that mean the 300 s wall clock ran out. Only these,
#: plus the parent's ``Killed``, make a graph *censored* under D14.
WALL_CLOCK_ERRORS = frozenset({"CanonicalizationTimeoutError", "Killed"})

#: Exception class names that mean a backend hit its own frozen internal cap --
#: AGM's branch-and-bound ceiling or min-DFS's projection (memory) ceiling.
#: These are **not** wall-clock failures and ``completion_rates.json`` reports
#: them in a separate column, because a rate conflating the two is not
#: interpretable.
INTERNAL_CAP_ERRORS = frozenset({"AGMBudgetExceeded", "MinDfsBudgetExceeded", "BudgetExceeded"})

SCOPE_ERRORS = frozenset({"SuiteScopeError"})
UNAVAILABLE_ERRORS = frozenset({"BackendUnavailableError", "ImportError", "ModuleNotFoundError"})


class WorkerError(RuntimeError):
    """Raised when an encoding cannot be rendered under the §3.1 convention."""


def symbol_sep(representation: str) -> str:
    """Separator for *representation*, per :data:`SYMBOL_SEP`."""
    return SYMBOL_SEP.get(representation, "")


def error_family(error_kind: str) -> str:
    """Group an exception class name into a reportable failure family.

    Args:
        error_kind: The ``error_kind`` field, i.e. an exception class name,
            ``"Killed"``, or ``""``.

    Returns:
        ``ok``, ``wall_clock``, ``internal_cap``, ``scope``, ``unavailable`` or
        ``other``.
    """
    if not error_kind:
        return "ok"
    if error_kind in WALL_CLOCK_ERRORS:
        return "wall_clock"
    if error_kind in INTERNAL_CAP_ERRORS:
        return "internal_cap"
    if error_kind in SCOPE_ERRORS:
        return "scope"
    if error_kind in UNAVAILABLE_ERRORS:
        return "unavailable"
    return "other"


@dataclass(frozen=True, slots=True)
class EncodeRecord:
    """One graph's outcome under one representation.

    Attributes:
        index: Position in cohort order.
        graph_id: The cohort identifier, carried so a merge can be checked.
        status: ``ok`` or ``error``. The driver rewrites this to ``censored``
            when it substitutes a greedy-min string under D14.
        error_kind: The exception class name when ``status == "error"``, else
            ``''``. The driver writes ``"Killed"`` for a graph it killed.
        encoding: ``symbol_sep.join(symbols)``; ``''`` on failure.
        length: The **symbol count**, i.e. the unit of edit.
        entropy_bits: ``L log2 |Sigma|``; ``None`` when undefined.
        realised_bits: Format-defined byte length x 8; ``None`` when undefined.
        fallback_used: Always ``False`` here. The driver sets it.
        seconds: Wall clock for this encode; ``-1`` when the parent killed it.
        message: Diagnostic text for a failure.
    """

    index: int
    graph_id: str
    status: str
    error_kind: str
    encoding: str
    length: int
    entropy_bits: float | None
    realised_bits: float | None
    fallback_used: bool
    seconds: float
    message: str


def _join_symbols(symbols: Sequence[Any], sep: str) -> str:
    """Render a symbol sequence under the §3.1 convention, asserting it holds.

    Args:
        symbols: The comparison units.
        sep: ``""`` or :data:`UNIT_SEP`.

    Returns:
        The joined string, whose ``split(sep)`` (or characters, when ``sep`` is
        empty) recovers exactly *symbols*.

    Raises:
        WorkerError: If a symbol contains the separator, or if ``sep`` is empty
            and some symbol is not a single character. Either would make the
            consumer's split silently disagree with ``length``.
    """
    rendered = [str(item) for item in symbols]
    if sep:
        offenders = [item for item in rendered if sep in item]
        if offenders:
            raise WorkerError(f"symbol separator occurs inside {len(offenders)} symbol(s)")
        return sep.join(rendered)
    wide = [item for item in rendered if len(item) != 1]
    if wide:
        raise WorkerError(
            f"{len(wide)} symbol(s) are not single characters but the separator is empty; "
            f"add this representation to SYMBOL_SEP"
        )
    return "".join(rendered)


def _bit_fields(encoding: Encoding) -> tuple[float | None, float | None]:
    """Both bit conventions, or ``(None, None)`` when the backend has neither.

    Args:
        encoding: The encoding to measure.

    Returns:
        ``(entropy_bits, realised_bits)``. ``None`` means ``BitCountUndefined``
        was raised, which is the correct answer for ``wl_subtree`` and
        ``size_null`` and must never be replaced by a zero.
    """
    from isalgraph.competitors import bits
    from isalgraph.errors import BitCountUndefined

    try:
        count = bits.count(encoding)
    except BitCountUndefined:
        return None, None
    return float(count.entropy_bits), float(count.realised_bits)


def _budget_for(representation: str, budget_s: float, overrides: Mapping[str, int | None]) -> Any:
    """Build the in-process :class:`Budget` for *representation*.

    The IsalGraph arms get the wall clock, because the C++ engine can enforce
    one and a clean ``CanonicalizationTimeoutError`` is a better record than a
    kill. Everything else keeps its frozen internal cap unless overridden: those
    caps are part of each backend's frozen T-04 specification, and min-DFS's is
    a *memory* guard whose first Suite-2 run was OOM-killed. The 300 s wall
    still applies to every backend through the parent's kill.

    Args:
        representation: Backend name.
        budget_s: Per-graph wall clock in seconds.
        overrides: ``{"search_nodes": ..., "max_projections": ...}``; ``None``
            means "leave the backend default alone". A run that sets either is
            a labelled sensitivity arm, never the primary reading.

    Returns:
        A ``Budget``, or ``None`` to accept the backend's frozen default.
    """
    from isalgraph.competitors.base import Budget

    if representation in ISALGRAPH_ARMS:
        return Budget(timeout_s=budget_s)
    fields = {key: value for key, value in overrides.items() if value is not None}
    return Budget(**fields) if fields else None


def _wl_symbols(features: Mapping[str, int]) -> tuple[str, ...]:
    """Render a WL feature multiset as sorted ``colour:count`` symbols.

    ``wl_subtree`` is a ``VectorBackend``: it has no ``encode`` and no bit
    count. The schema still needs a string, so the multiset becomes a symbol
    sequence under the same §3.1 convention as every other representation, which
    the consumer splits back into the multiset the kernel needs. It is a
    carrier, not a serialisation of the graph, which is why its bit fields stay
    ``None``.

    Args:
        features: Colour -> count. Colours are ``blake2b`` hex digests, so
            neither ``:`` nor the unit separator can occur in one.

    Returns:
        Sorted ``"<colour>:<count>"`` symbols.
    """
    return tuple(f"{colour}:{count}" for colour, count in sorted(features.items()))


def _record(
    index: int,
    graph_id: str,
    symbols: Sequence[Any],
    sep: str,
    bit_fields: tuple[float | None, float | None],
    started: float,
    *,
    fallback_used: bool = False,
) -> EncodeRecord:
    """Assemble a successful record from a symbol sequence."""
    return EncodeRecord(
        index=index,
        graph_id=graph_id,
        status="ok",
        error_kind="",
        encoding=_join_symbols(symbols, sep),
        length=len(symbols),
        entropy_bits=bit_fields[0],
        realised_bits=bit_fields[1],
        fallback_used=fallback_used,
        seconds=time.perf_counter() - started,
        message="",
    )


def _encode_vector(name: str, graph: nx.Graph, index: int, graph_id: str) -> EncodeRecord:
    """Encode under the one ``VectorBackend``."""
    from isalgraph.competitors.registry import get_vector_backend

    started = time.perf_counter()
    symbols = _wl_symbols(get_vector_backend(name).features(graph))
    return _record(index, graph_id, symbols, symbol_sep(name), (None, None), started)


def _encode_repr(
    name: str, graph: nx.Graph, index: int, graph_id: str, budget: Any
) -> EncodeRecord:
    """Encode under a serialisation backend."""
    from isalgraph.competitors.registry import get_repr_backend

    started = time.perf_counter()
    encoding = get_repr_backend(name).encode(graph, budget=budget)
    return _record(
        index, graph_id, encoding.symbols, symbol_sep(name), _bit_fields(encoding), started
    )


def greedy_min_string(graph: nx.Graph) -> str:
    """The D14 fallback string: greedy G2S minimised over starting nodes.

    A graph whose canonical encoding times out is retained with this string and
    flagged, never dropped. The graphs that time out are exactly those with the
    largest automorphism groups, so dropping them would delete the hardest cases
    and leave the paper reporting on a cohort those cases were removed from.

    Args:
        graph: The graph.

    Returns:
        The lexicographically smallest shortest greedy string.
    """
    from isalgraph import GreedyMinG2S, SparseGraph

    n = graph.number_of_nodes()
    sparse = SparseGraph(max(n, 1), False)
    for _ in range(n):
        sparse.add_node()
    for u, v in graph.edges():
        sparse.add_edge(int(u), int(v))
    return GreedyMinG2S().encode(sparse)


def pruned_fallback_string(graph: nx.Graph, timeout_s: float | None) -> str:
    """The pruned canonical string, as a D14 substitute for the exhaustive arm.

    Preferred over :func:`greedy_min_string` wherever it is affordable because
    it is still a **canonical** form: a row substituted with it remains inside
    the completeness theorem, whereas a greedy-min row does not.

    Args:
        graph: The graph.
        timeout_s: Wall clock, or ``None`` to run unbounded.

    Returns:
        ``pruned_canonical_string`` of *graph*.

    Raises:
        CanonicalizationTimeoutError: when *timeout_s* runs out. The caller
            cascades to the next tier; it never returns a partial string.
    """
    from isalgraph import pruned_canonical_string
    from isalgraph.competitors.backends.isalgraph_ref import to_sparse_graph

    return pruned_canonical_string(to_sparse_graph(graph), timeout_s=timeout_s)


def fallback_cascade(representation: str) -> tuple[str, ...]:
    """The ordered D14 substitute tiers for *representation*.

    Args:
        representation: Backend name.

    Returns:
        Tier names from :data:`FALLBACK_CASCADE`, or :data:`DEFAULT_CASCADE`.
        The last tier is always ``"greedy"``, which is unbudgeted and always
        terminates, so the cascade cannot end without a string.
    """
    return FALLBACK_CASCADE.get(representation, DEFAULT_CASCADE)


def _fallback_text(representation: str, graph: nx.Graph, budget: Any) -> tuple[str, str]:
    """Walk the cascade until a tier produces a string.

    Args:
        representation: Backend name the fallback stands in for.
        graph: The graph.
        budget: The in-process ``Budget``, or ``None``. Only ``pruned`` reads
            it; ``greedy`` is deliberately unbudgeted.

    Returns:
        ``(text, tier)``, where *tier* names the tier that produced *text*.

    Raises:
        Exception: only if **every** tier fails, which ``greedy`` closing the
            cascade makes unreachable in practice. Propagating is correct: the
            driver then leaves the row an ``error`` rather than inventing one.
    """
    timeout_s = getattr(budget, "timeout_s", None)
    tiers = fallback_cascade(representation)
    last: Exception | None = None
    for tier in tiers:
        try:
            if tier == "pruned":
                return pruned_fallback_string(graph, timeout_s), tier
            return greedy_min_string(graph), tier
        except Exception as exc:  # noqa: BLE001 - a failed tier is a datum
            last = exc
            LOGGER.warning("fallback tier %r failed for %s: %s", tier, representation, exc)
    raise last if last is not None else WorkerError("empty fallback cascade")


def _encode_fallback(
    name: str, graph: nx.Graph, index: int, graph_id: str, budget: Any = None
) -> EncodeRecord:
    """Produce the D14 substitute record.

    Args:
        name: Backend name the fallback stands in for.
        graph: The graph.
        index: Position in cohort order.
        graph_id: Cohort identifier.
        budget: The in-process budget, read by the ``pruned`` tier only.

    Returns:
        A record with ``status == "ok"``; the driver stamps it ``censored``.
        ``message`` names the tier that produced the string, because a
        pruned-tier row and a greedy-tier row are not the same kind of datum
        and a rate that conflates them is not interpretable.
    """
    from isalgraph.competitors.base import Encoding

    started = time.perf_counter()
    text, tier = _fallback_text(name, graph, budget)
    encoding = Encoding(
        backend=name,
        symbols=tuple(text),
        alphabet_size=ISALGRAPH_ALPHABET_SIZE,
        n_nodes=graph.number_of_nodes(),
        n_edges=graph.number_of_edges(),
        text=text,
    )
    record = _record(
        index,
        graph_id,
        encoding.symbols,
        symbol_sep(name),
        _bit_fields(encoding),
        started,
        fallback_used=True,
    )
    return replace(record, message=f"fallback_tier={tier}")


def encode_one(
    cohort: Cohort, index: int, representation: str, *, mode: str, budget: Any
) -> EncodeRecord:
    """Encode a single graph, converting any exception into a record.

    Args:
        cohort: The loaded cohort.
        index: Position in cohort order.
        representation: Backend name.
        mode: ``"primary"`` or ``"fallback"``.
        budget: In-process budget, or ``None``.

    Returns:
        The record. This function does not raise: every failure is a datum.
    """
    graph_id = str(cohort.graph_ids[index])
    started = time.perf_counter()
    try:
        return _dispatch(cohort, index, representation, mode, budget, graph_id)
    except Exception as exc:  # noqa: BLE001 - every failure is a datum
        return EncodeRecord(
            index=index,
            graph_id=graph_id,
            status="error",
            error_kind=type(exc).__name__,
            encoding="",
            length=-1,
            entropy_bits=None,
            realised_bits=None,
            fallback_used=False,
            seconds=time.perf_counter() - started,
            message=f"{type(exc).__name__}: {exc}",
        )


def _dispatch(
    cohort: Cohort, index: int, representation: str, mode: str, budget: Any, graph_id: str
) -> EncodeRecord:
    """Route one graph to the fallback, vector or serialisation path."""
    graph = cohort.to_networkx(index)
    if mode == "fallback":
        return _encode_fallback(representation, graph, index, graph_id, budget)
    if representation == "wl_subtree":
        return _encode_vector(representation, graph, index, graph_id)
    return _encode_repr(representation, graph, index, graph_id, budget)


def _read_indices(path: Path | None, n_graphs: int) -> list[int]:
    """Resolve the index list, defaulting to the whole cohort.

    Args:
        path: JSON file holding a list of ints, or ``None``.
        n_graphs: Cohort size.

    Returns:
        Indices to process, in order.
    """
    if path is None:
        return list(range(n_graphs))
    return [int(value) for value in json.loads(path.read_text())]


def _build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Encode cohort graphs, one JSON line each.")
    parser.add_argument("--suite", required=True, choices=("suite1", "suite2"))
    parser.add_argument("--dataset", required=True)
    parser.add_argument("--representation", required=True)
    parser.add_argument("--mode", default="primary", choices=MODES)
    parser.add_argument("--budget-s", type=float, default=300.0)
    parser.add_argument("--cohort-root", type=Path, default=None)
    parser.add_argument("--indices-file", type=Path, default=None)
    parser.add_argument("--limit", type=int, default=None)
    parser.add_argument("--agm-search-nodes", type=int, default=None)
    parser.add_argument("--min-dfs-max-projections", type=int, default=None)
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    """CLI entry point: stream one JSON record per graph to stdout.

    Args:
        argv: Argument vector; ``None`` reads ``sys.argv``.

    Returns:
        Process exit status.
    """
    args = _build_parser().parse_args(argv)
    cohort = load_cohort(args.suite, args.dataset, root=args.cohort_root, limit=args.limit)
    budget = _budget_for(
        args.representation,
        args.budget_s,
        {
            "search_nodes": args.agm_search_nodes,
            "max_projections": args.min_dfs_max_projections,
        },
    )
    for index in _read_indices(args.indices_file, len(cohort)):
        record = encode_one(cohort, index, args.representation, mode=args.mode, budget=budget)
        sys.stdout.write(json.dumps(asdict(record)) + "\n")
        sys.stdout.flush()
    return 0


if __name__ == "__main__":  # pragma: no cover - CLI
    raise SystemExit(main())
