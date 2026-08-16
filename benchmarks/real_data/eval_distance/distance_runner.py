"""Compute one contiguous row band of a pairwise distance matrix.

Reads a CONTRACTS §3 encodings file, rebuilds one
:class:`~isalgraph.competitors.base.Encoding` per graph, and writes a row-band
shard (and, for a single-chunk run, the dense CONTRACTS §4 file as well).

**The encodings file is the only input.**  ``node_counts`` and ``edge_counts``
travel inside it precisely so this track never opens a cohort file, which is
what keeps its ownership disjoint from the encoding track's.

Two code paths::

    fast     rapidfuzz.process.cdist        levenshtein, levenshtein_char, hamming
    vector   numpy broadcast                size_null (consumes "order")
    generic  per-pair is_defined/distance   everything else that reads symbols/text

Measured on this workstation, 200 real graphs, Levenshtein over the whole
``200 x 200`` cell block (**sizing input for the orchestrator, not a published
timing**): ``rapidfuzz.process.cdist`` runs at 6.1 M cells/s single-threaded on
graph6 strings of median 60 / max 599 symbols and 32.5 M cells/s on IsalGraph
strings of median 4, against 0.86-0.95 M pairs/s for a Python double loop
through the metric protocol and 1.3 M pairs/s for
``isalgraph.core.backends.levenshtein``.  cdist is 6-38x faster and is the
implementation chosen; the two loops are the implementations rejected.

Usage::

    python -m benchmarks.eval_distance.distance_runner \\
        --encodings encodings/suite2/linux__isalgraph_pruned.npz \\
        --metric levenshtein --out distances/suite2 \\
        [--chunk-index 3 --n-chunks 8] [--jobs 1] [--symbol-sep ' ']
"""

from __future__ import annotations

import argparse
import json
import logging
import sys
from collections.abc import Sequence
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np

from benchmarks.eval_distance.bands import RowBand, band_for
from benchmarks.eval_distance.gates import assert_dense, degenerate_zero_fraction
from benchmarks.eval_distance.schema import (
    EncodingsFile,
    MetricUnsupportedError,
    SchemaError,
    build_metadata,
    load_encodings,
    shard_path,
    write_dense,
    write_shard,
)

logger = logging.getLogger(__name__)

#: The frozen separator of CONTRACTS §3.1.  ASCII unit separator, chosen
#: because it occurs in no symbol rendering the pool produces.
UNIT_SEPARATOR = "\x1f"

#: Last-resort reconstruction rule, used only when the encodings file declares
#: no ``symbol_sep``.  ``min_dfs`` is the one backend whose symbols are not
#: single characters: splitting ``'0-1 1-2 2-0'`` per character charges four
#: edits for one deleted DFS tuple, on the comparator ``competitors.md`` §2
#: calls the single most important one.  ``--symbol-sep`` and the file's own
#: ``symbol_sep`` metadata key both take precedence.
SYMBOL_SEPARATORS: dict[str, str] = {"min_dfs": UNIT_SEPARATOR}

#: Metrics ``rapidfuzz.process.cdist`` can evaluate in C over the whole band.
#: Value is ``(what it reads, whether definedness depends on equal length)``.
_CDIST_METRICS: dict[str, tuple[str, bool]] = {
    "levenshtein": ("symbols", False),
    "levenshtein_char": ("text", False),
    "hamming": ("symbols", True),
}

#: What a metric may read out of a CONTRACTS §3 file.  ``frame`` and
#: ``features`` are not in the schema, so ``padded_hamming`` and ``kernel``
#: are refused rather than approximated.
_SUPPORTED_CONSUMES = frozenset({"symbols", "text", "order"})


@dataclass(frozen=True, slots=True)
class RunnerConfig:
    """Everything one band computation needs.

    Attributes:
        encodings: path to the CONTRACTS §3 input.
        metric: registered metric name.
        out_dir: directory receiving the shard and, for a single chunk, the
            dense file.
        chunk_index: this task's index.
        n_chunks: total task count.
        jobs: threads handed to ``rapidfuzz.process.cdist``.  Defaults to 1;
            raising it is the orchestrator's call, not a worker's.
        symbol_sep: explicit symbol separator, or ``None`` to resolve it.
        suite: ``suite1``/``suite2`` override when the input metadata lacks it.
        on_length_mismatch: ``"raise"`` (default) or ``"warn"`` when the
            rebuilt symbol counts disagree with the file's ``length`` column.
            See :func:`_assert_symbol_counts` for when ``"warn"`` is defensible
            and when it is not.
    """

    encodings: Path
    metric: str
    out_dir: Path
    chunk_index: int = 0
    n_chunks: int = 1
    jobs: int = 1
    symbol_sep: str | None = None
    suite: str | None = None
    on_length_mismatch: str = "raise"


@dataclass(frozen=True, slots=True)
class RebuiltEncodings:
    """CONTRACTS §3 arrays rebuilt into the objects a metric consumes.

    Attributes:
        encodings: one per graph, in cohort order.  Rows flagged in
            *invalid* carry a placeholder that is never compared.
        symbols: the symbol tuples, hoisted for the cdist fast path.
        texts: the flat strings, hoisted for ``levenshtein_char``.
        lengths: symbol counts.
        invalid: ``True`` where the graph has no usable encoding.
        separator: the separator that was applied.
        length_agrees: whether the rebuilt symbol counts match the file's
            ``length`` column on every valid row.  ``False`` can only be
            reached under ``on_length_mismatch="warn"``, and the flag is
            written into the output metadata so the file carries its own
            caveat.
    """

    encodings: list[Any]
    symbols: list[tuple[str, ...]]
    texts: list[str]
    lengths: np.ndarray
    invalid: np.ndarray
    separator: str
    length_agrees: bool


def resolve_symbol_separator(
    representation: str, metadata: dict[str, Any], override: str | None
) -> str:
    """Decide how to split a stored encoding string into symbols.

    Precedence: an explicit *override*, then the encodings file's own
    ``symbol_sep`` metadata key (CONTRACTS §3.1, the contractual source), then
    :data:`SYMBOL_SEPARATORS`, then one symbol per character.

    Args:
        representation: backend name.
        metadata: the encodings file's CONTRACTS §5 block.
        override: value of ``--symbol-sep``, or ``None``.

    Returns:
        The separator; ``''`` means one symbol per character.
    """
    if override is not None:
        return override
    for key in ("symbol_sep", "symbol_separator"):
        declared = metadata.get(key)
        if isinstance(declared, str):
            return declared
    return SYMBOL_SEPARATORS.get(representation, "")


def _split_symbols(text: str, separator: str) -> tuple[str, ...]:
    """Split *text* into symbols under *separator*."""
    if separator == "":
        return tuple(text)
    if text == "":
        return ()
    return tuple(text.split(separator))


def _alphabet_size(entropy_bits: float, n_symbols: int) -> int:
    """Recover ``|Sigma|`` from ``entropy_bits = L log2 |Sigma|``.

    No metric reads ``alphabet_size``; it is reconstructed only so the
    rebuilt :class:`Encoding` is not carrying a fabricated constant.

    Returns:
        The rounded alphabet size, or 0 when it is not recoverable.
    """
    if n_symbols <= 0 or not np.isfinite(entropy_bits) or entropy_bits <= 0.0:
        return 0
    exponent = float(entropy_bits) / float(n_symbols)
    if exponent > 64.0:
        return 0
    return int(round(2.0**exponent))


def rebuild_encodings(
    source: EncodingsFile,
    representation: str,
    separator: str,
    on_length_mismatch: str = "raise",
) -> RebuiltEncodings:
    """Rebuild one :class:`Encoding` per graph from the stored arrays.

    A row is marked invalid when ``status == "error"`` or ``length < 0``: an
    encoding that does not exist must not be compared, and comparing it
    against ``''`` would return the other string's length, which is a number
    rather than an error.  Note that an empty string is **not** by itself a
    fault -- a one-node graph legitimately encodes to zero symbols.

    **The rebuilt symbol count is checked against the file's ``length``
    column on every valid row.**  CONTRACTS §3.1 makes ``length`` the symbol
    count, so a disagreement means the separator is wrong -- which for
    ``min_dfs`` is exactly the fourfold edit-count error the convention
    exists to prevent, and it would otherwise produce a plausible number.

    Args:
        source: the loaded encodings file.
        representation: backend name, carried into ``Encoding.backend``.
        separator: as returned by :func:`resolve_symbol_separator`.
        on_length_mismatch: ``"raise"`` or ``"warn"``.

    Returns:
        The rebuilt encodings and the bookkeeping the runner needs.

    Raises:
        SchemaError: when the rebuilt symbol counts disagree with ``length``
            and *on_length_mismatch* is ``"raise"``.
    """
    from isalgraph.competitors.base import Encoding

    texts = [str(value) for value in source.encoding]
    symbols = [_split_symbols(text, separator) for text in texts]
    lengths = np.array([len(item) for item in symbols], dtype=np.int64)
    invalid = (source.status == "error") | (np.asarray(source.length, dtype=np.int64) < 0)
    declared = np.asarray(source.length, dtype=np.int64)
    agrees = _assert_symbol_counts(
        lengths, declared, ~invalid, separator, source.path, on_length_mismatch
    )
    encodings = [
        Encoding(
            backend=representation,
            symbols=symbols[i],
            alphabet_size=_alphabet_size(float(source.entropy_bits[i]), len(symbols[i])),
            n_nodes=int(source.node_counts[i]),
            n_edges=int(source.edge_counts[i]),
            text=texts[i],
        )
        for i in range(source.n_graphs)
    ]
    return RebuiltEncodings(
        encodings=encodings,
        symbols=symbols,
        texts=texts,
        lengths=lengths,
        invalid=invalid,
        separator=separator,
        length_agrees=agrees,
    )


def _assert_symbol_counts(
    rebuilt: np.ndarray,
    declared: np.ndarray,
    valid: np.ndarray,
    separator: str,
    path: Path,
    on_mismatch: str,
) -> bool:
    """Check the rebuilt symbol counts against the file's ``length`` column.

    ``length`` is CONTRACTS §3.1's ground truth for sequence length, so a
    disagreement means the split is wrong -- for ``min_dfs`` that is exactly
    the fourfold edit-count error the convention exists to prevent, and it
    produces a plausible number rather than an error.

    ``"warn"`` exists for one measured case: ``sparse6`` and
    ``sparse6_nauty`` carry the ``':'`` format marker in
    :attr:`Encoding.text` but **not** in :attr:`Encoding.symbols`, so a
    producer that follows §3.1 literally and stores ``text`` when
    ``symbol_sep == ""`` emits ``len(encoding) == length + 1`` on every row.
    A constant marker present in both operands shifts no edit distance and no
    equal-length test, so tolerating it is sound *there*.  It is not sound in
    general, which is why it is opt-in and why the flag is written into the
    output metadata.

    Returns:
        Whether the counts agree everywhere.

    Raises:
        SchemaError: on a disagreement when *on_mismatch* is ``"raise"``, or
            when *on_mismatch* is neither ``"raise"`` nor ``"warn"``.
    """
    if on_mismatch not in ("raise", "warn"):
        raise SchemaError(f"on_length_mismatch must be 'raise' or 'warn', got {on_mismatch!r}")
    disagree = valid & (rebuilt != declared)
    if not bool(disagree.any()):
        return True
    first = int(np.flatnonzero(disagree)[0])
    message = (
        f"{path}: splitting on {separator!r} gives {int(rebuilt[first])} symbols for row "
        f"{first} but the file declares length {int(declared[first])} "
        f"({int(disagree.sum())} of {int(valid.sum())} valid rows disagree)"
    )
    if on_mismatch == "raise":
        raise SchemaError(
            f"{message}. CONTRACTS §3.1 makes `length` the symbol count; a mismatch means "
            f"the separator is wrong. Pass --on-length-mismatch warn only once you know why"
        )
    logger.warning("%s; continuing under --on-length-mismatch warn", message)
    return False


def _check_metric_supported(metric_name: str, separator: str) -> Any:
    """Instantiate *metric_name* and refuse one this schema cannot feed.

    Two refusals, both because CONTRACTS §3 carries a joined symbol sequence
    and nothing else:

    * ``consumes`` of ``frame`` or ``features`` -- ``padded_hamming`` needs a
      positional frame and ``kernel`` a fitted feature multiset.  Both would
      have to be re-derived from the cohort, which this track does not read.
    * ``consumes == "text"`` while ``separator != ""`` -- the file stores
      ``symbol_sep.join(symbols)``, not ``Encoding.text``, so a
      character-level distance computed here would be over a rendering the
      backend never emitted.

    Raises:
        MetricUnsupportedError: in either case, with the reason.
    """
    from isalgraph.competitors import get_metric

    metric = get_metric(metric_name)
    if metric.consumes not in _SUPPORTED_CONSUMES:
        raise MetricUnsupportedError(
            f"metric {metric_name!r} consumes {metric.consumes!r}, which the CONTRACTS §3 "
            f"encodings schema does not carry (it stores a joined symbol sequence, node "
            f"counts and edge counts only). Supported: {sorted(_SUPPORTED_CONSUMES)}"
        )
    if metric.consumes == "text" and separator != "":
        raise MetricUnsupportedError(
            f"metric {metric_name!r} reads Encoding.text, but this file joins its symbols "
            f"with {separator!r} (CONTRACTS §3.1), so the stored string is not the text the "
            f"backend emitted. A character-level distance over it would measure the "
            f"separator, not the rendering"
        )
    return metric


def _cdist_band(
    metric_name: str, rebuilt: RebuiltEncodings, band: RowBand, jobs: int
) -> tuple[np.ndarray, np.ndarray]:
    """Compute a band with ``rapidfuzz.process.cdist``.

    Returns:
        ``(distance_band, defined_band)`` before invalid rows are masked.
    """
    from rapidfuzz import process
    from rapidfuzz.distance import Hamming, Levenshtein

    reads, needs_equal_length = _CDIST_METRICS[metric_name]
    choices: Sequence[Any] = rebuilt.texts if reads == "text" else rebuilt.symbols
    scorer = Hamming.distance if metric_name == "hamming" else Levenshtein.distance
    raw = process.cdist(
        choices[band.start : band.stop],
        choices,
        scorer=scorer,
        dtype=np.int64,
        workers=jobs,
    )
    distance = np.asarray(raw, dtype=np.float64)
    if needs_equal_length:
        lengths = rebuilt.lengths
        defined = lengths[band.start : band.stop, None] == lengths[None, :]
        distance = np.where(defined, distance, np.nan)
    else:
        defined = np.ones(distance.shape, dtype=bool)
    return distance, np.asarray(defined, dtype=bool)


def _order_band(rebuilt: RebuiltEncodings, band: RowBand) -> tuple[np.ndarray, np.ndarray]:
    """Compute a band for a metric that reads only the node count."""
    counts = np.array([enc.n_nodes for enc in rebuilt.encodings], dtype=np.int64)
    distance = np.abs(counts[band.start : band.stop, None] - counts[None, :]).astype(np.float64)
    return distance, np.ones(distance.shape, dtype=bool)


def _generic_band(
    metric: Any, rebuilt: RebuiltEncodings, band: RowBand
) -> tuple[np.ndarray, np.ndarray]:
    """Compute a band pair by pair through the metric protocol.

    ``is_defined`` is honoured per pair: where it is false the cell is ``nan``
    and the mask is false, never ``0.0``, which a consumer would read as
    "identical".
    """
    n = len(rebuilt.encodings)
    distance = np.full((band.height, n), np.nan, dtype=np.float64)
    defined = np.zeros((band.height, n), dtype=bool)
    encodings = rebuilt.encodings
    for local in range(band.height):
        row = encodings[band.start + local]
        _fill_row(metric, row, encodings, distance[local], defined[local])
    return distance, defined


def _fill_row(
    metric: Any, row: Any, encodings: list[Any], out: np.ndarray, mask: np.ndarray
) -> None:
    """Fill one row of a generic band in place."""
    for j, other in enumerate(encodings):
        if not metric.is_defined(row, other):
            continue
        out[j] = float(metric.distance(row, other))
        mask[j] = True


def compute_band(
    metric_name: str, rebuilt: RebuiltEncodings, band: RowBand, jobs: int = 1
) -> tuple[np.ndarray, np.ndarray]:
    """Compute rows ``[band.start, band.stop)`` over all columns.

    Invalid rows and columns are masked out afterwards, and the diagonal is
    restored to ``0.0`` with ``defined_mask`` true: a graph is at distance 0
    from itself whether or not it encoded, and CONTRACTS §4 requires a zero
    diagonal on every matrix.

    Args:
        metric_name: registered metric name.
        rebuilt: output of :func:`rebuild_encodings`.
        band: the rows this task owns.
        jobs: threads for the cdist fast path.

    Returns:
        ``(distance_band float64, defined_band bool)``, both
        ``(band.height, G)``.
    """
    metric = _check_metric_supported(metric_name, rebuilt.separator)
    n = len(rebuilt.encodings)
    if band.height == 0:
        return np.zeros((0, n), dtype=np.float64), np.zeros((0, n), dtype=bool)
    if metric_name in _CDIST_METRICS:
        distance, defined = _cdist_band(metric_name, rebuilt, band, jobs)
    elif metric.consumes == "order":
        distance, defined = _order_band(rebuilt, band)
    else:
        distance, defined = _generic_band(metric, rebuilt, band)
    invalid = rebuilt.invalid
    local_invalid = invalid[band.start : band.stop]
    distance[local_invalid, :] = np.nan
    defined[local_invalid, :] = False
    distance[:, invalid] = np.nan
    defined[:, invalid] = False
    rows = np.arange(band.start, band.stop)
    distance[np.arange(band.height), rows] = 0.0
    defined[np.arange(band.height), rows] = True
    return distance, defined


def parse_basename(encodings_path: Path) -> tuple[str, str]:
    """Split ``{dataset}__{representation}.npz`` into its two parts.

    Raises:
        SchemaError: when the filename does not carry the ``__`` separator.
    """
    stem = encodings_path.stem
    if "__" not in stem:
        raise SchemaError(
            f"{encodings_path} does not follow '{{dataset}}__{{representation}}.npz'; "
            f"the dataset and representation cannot be recovered from {stem!r}"
        )
    dataset, _, representation = stem.partition("__")
    return dataset, representation


def _resolve_identity(source: EncodingsFile, path: Path, suite: str | None) -> tuple[str, str, str]:
    """Return ``(suite, dataset, representation)`` from metadata, then filename."""
    file_dataset, file_repr = parse_basename(path)
    meta = source.metadata
    resolved_suite = suite or str(meta.get("suite") or "unknown")
    dataset = str(meta.get("dataset") or file_dataset)
    representation = str(meta.get("representation") or file_repr)
    return resolved_suite, dataset, representation


def merge_bands(
    bands: Sequence[tuple[RowBand, np.ndarray, np.ndarray]], n_graphs: int
) -> tuple[np.ndarray, np.ndarray]:
    """Assemble row bands into a dense ``(G, G)`` pair.

    Args:
        bands: ``(band, distance_band, defined_band)`` in any order.
        n_graphs: cohort size.

    Returns:
        ``(distance_matrix, defined_mask)``.
    """
    distance = np.full((n_graphs, n_graphs), np.nan, dtype=np.float64)
    defined = np.zeros((n_graphs, n_graphs), dtype=bool)
    for band, values, mask in bands:
        distance[band.start : band.stop, :] = values
        defined[band.start : band.stop, :] = mask
    return distance, defined


def run(config: RunnerConfig) -> Path:
    """Compute this task's band and write it.

    A single-chunk run also writes the dense CONTRACTS §4 file, through the
    same :func:`merge_bands` used by ``distance_merge``, so there is one
    assembly path rather than two.

    Args:
        config: what to compute and where to put it.

    Returns:
        The shard path that was written.
    """
    source = load_encodings(config.encodings)
    suite, dataset, representation = _resolve_identity(source, config.encodings, config.suite)
    separator = resolve_symbol_separator(representation, source.metadata, config.symbol_sep)
    rebuilt = rebuild_encodings(
        source, representation, separator, on_length_mismatch=config.on_length_mismatch
    )
    band = band_for(source.n_graphs, config.n_chunks, config.chunk_index)
    logger.info(
        "%s/%s [%s] metric=%s chunk %d/%d -> rows [%d, %d) of %d, separator=%r",
        suite,
        dataset,
        representation,
        config.metric,
        config.chunk_index,
        config.n_chunks,
        band.start,
        band.stop,
        source.n_graphs,
        separator,
    )
    values, mask = compute_band(config.metric, rebuilt, band, jobs=config.jobs)
    basename = f"{dataset}__{representation}__{config.metric}"
    metadata = build_metadata(
        suite=suite,
        dataset=dataset,
        representation=representation,
        metric=config.metric,
        n_graphs=source.n_graphs,
        extra={
            "chunk_index": config.chunk_index,
            "n_chunks": config.n_chunks,
            "row_start": band.start,
            "row_stop": band.stop,
            "symbol_sep": separator,
            "symbol_length_matches_npz_length": rebuilt.length_agrees,
            "n_invalid_rows": int(rebuilt.invalid.sum()),
            "encodings_source": str(config.encodings),
            "jobs": config.jobs,
        },
    )
    out = shard_path(config.out_dir, basename, config.chunk_index)
    write_shard(
        out,
        distance_band=values,
        defined_band=mask,
        row_start=band.start,
        row_stop=band.stop,
        n_graphs=source.n_graphs,
        graph_ids=source.graph_ids,
        node_counts=source.node_counts,
        metadata=metadata,
    )
    if config.n_chunks == 1:
        _write_dense_from_single_band(config, source, basename, band, values, mask, metadata)
    return out


def _write_dense_from_single_band(
    config: RunnerConfig,
    source: EncodingsFile,
    basename: str,
    band: RowBand,
    values: np.ndarray,
    mask: np.ndarray,
    metadata: dict[str, Any],
) -> None:
    """Assemble and gate the dense file for an unsharded run."""
    distance, defined = merge_bands([(band, values, mask)], source.n_graphs)
    report = assert_dense(distance, defined)
    degenerate_zero_fraction(report)
    dense_meta = dict(metadata)
    dense_meta.update({"chunk_index": None, "n_chunks": 1, "row_start": 0, "row_stop": band.stop})
    write_dense(
        config.out_dir / f"{basename}.npz",
        distance_matrix=distance,
        graph_ids=source.graph_ids,
        node_counts=source.node_counts,
        defined_mask=defined,
        metadata=dense_meta,
    )


def build_parser() -> argparse.ArgumentParser:
    """Return the CLI parser."""
    parser = argparse.ArgumentParser(
        prog="distance_runner",
        description="Compute one contiguous row band of a pairwise distance matrix.",
    )
    parser.add_argument("--encodings", required=True, type=Path, help="CONTRACTS §3 .npz")
    parser.add_argument("--metric", required=True, help="registered metric name")
    parser.add_argument("--out", required=True, type=Path, help="output directory")
    parser.add_argument("--chunk-index", type=int, default=0)
    parser.add_argument("--n-chunks", type=int, default=1)
    parser.add_argument(
        "--jobs",
        type=int,
        default=1,
        help="threads for rapidfuzz.process.cdist; 1 by default, -1 uses every core",
    )
    parser.add_argument(
        "--symbol-sep",
        default=None,
        help="separator splitting a stored encoding into symbols; '' means per character",
    )
    parser.add_argument("--suite", default=None, choices=("suite1", "suite2"))
    parser.add_argument(
        "--on-length-mismatch",
        default="raise",
        choices=("raise", "warn"),
        help="what to do when the rebuilt symbol count disagrees with the file's `length`",
    )
    parser.add_argument("--log-level", default="INFO")
    return parser


def main(argv: list[str] | None = None) -> int:
    """CLI entry point.

    Returns:
        0 on success, 1 on a schema or metric fault.
    """
    args = build_parser().parse_args(argv)
    logging.basicConfig(
        level=getattr(logging, str(args.log_level).upper(), logging.INFO),
        format="%(asctime)s %(levelname)s %(name)s: %(message)s",
    )
    config = RunnerConfig(
        encodings=args.encodings,
        metric=args.metric,
        out_dir=args.out,
        chunk_index=args.chunk_index,
        n_chunks=args.n_chunks,
        jobs=args.jobs,
        symbol_sep=args.symbol_sep,
        suite=args.suite,
        on_length_mismatch=args.on_length_mismatch,
    )
    try:
        out = run(config)
    except (SchemaError, MetricUnsupportedError) as exc:
        logger.error("%s: %s", type(exc).__name__, exc)
        return 1
    print(json.dumps({"shard": str(out)}))
    return 0


if __name__ == "__main__":
    sys.exit(main())
