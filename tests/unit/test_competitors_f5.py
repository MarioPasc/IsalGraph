"""F5: the graph-level bootstrap, the bounds loader, and the two views.

Every fixture here is **synthetic**.  The real cohort lives on an external
drive, so a suite that needed it would be a suite that silently stops running
the moment the drive is unmounted -- which is exactly when a regression would
get in.  The tests that genuinely need the drive are marked
``integration`` and skip cleanly.

Three of these tests exist because of a failure mode that produces a
plausible number rather than an error:

- ``inf`` is how T-05 censors an unfinished pair.  ``np.isnan(inf)`` is
  ``False``, so a filter written with ``isnan`` passes every censored pair
  through and ``inf <= x`` then compares ``False`` without raising.
- A GED of exactly 0 is **correct** for isomorphic graphs -- 28 % of IAM
  Letter LOW pairs are certified at it.  A per-pair ``value > 0`` guard would
  discard the pairs the encoding gets most obviously right.
- The bootstrap draws must be **shared** across representations within one
  dataset and view.  Independent draws would still produce intervals; they
  would just be intervals that cannot be compared to each other.
"""

from __future__ import annotations

import json
import os
from collections.abc import Iterator
from pathlib import Path
from typing import TYPE_CHECKING, Any

import pytest

if TYPE_CHECKING:
    import numpy.typing as npt

np = pytest.importorskip("numpy")
pytest.importorskip("scipy")
pytest.importorskip("networkx")

from isalgraph.competitors import datasets, f5, ged_reference, registry  # noqa: E402
from isalgraph.competitors.bootstrap import (  # noqa: E402
    BootstrapError,
    graph_bootstrap_ci,
    make_resample_index,
)
from isalgraph.competitors.f5 import SIZE_NULL  # noqa: E402
from isalgraph.competitors.ged_reference import GEDReferenceError, load_bounds  # noqa: E402
from isalgraph.errors import (  # noqa: E402
    BudgetExceeded,
    CanonicalizationTimeoutError,
    CompetitorError,
    IsalGraphError,
    SuiteScopeError,
)

# ----------------------------------------------------------------------
# Synthetic cohort + reference, written under a temporary cohort root
# ----------------------------------------------------------------------

#: Node counts of the synthetic cohort.  Three equal-``n`` pairs out of
#: fifteen, so the equal-``n`` view is a strict, checkable subset.
NODE_COUNTS = (3, 3, 4, 4, 5, 5)


def _write_cohort(root: str, subdir: str, dataset: str) -> None:
    """A six-graph path-and-cycle cohort in the exporter's CSR layout."""
    edges_per_graph = [
        [(0, 1), (1, 2)],
        [(0, 1), (1, 2), (0, 2)],
        [(0, 1), (1, 2), (2, 3)],
        [(0, 1), (1, 2), (2, 3), (0, 3)],
        [(0, 1), (1, 2), (2, 3), (3, 4)],
        [(0, 1), (1, 2), (2, 3), (3, 4), (0, 4)],
    ]
    offsets = [0]
    src: list[int] = []
    dst: list[int] = []
    for pairs in edges_per_graph:
        for a, b in pairs:
            src.append(a)
            dst.append(b)
        offsets.append(len(src))
    directory = os.path.join(root, subdir)
    os.makedirs(directory, exist_ok=True)
    np.savez(
        os.path.join(directory, f"{dataset}.npz"),
        n_nodes=np.asarray(NODE_COUNTS, dtype=np.int32),
        edge_offsets=np.asarray(offsets, dtype=np.int64),
        edges=np.asarray([src, dst], dtype=np.int64),
        graph_ids=np.asarray([f"g{i}" for i in range(len(NODE_COUNTS))]),
    )


def _reference_matrix() -> npt.NDArray[Any]:
    """A symmetric, non-degenerate GED matrix holding one legitimate zero."""
    n = len(NODE_COUNTS)
    matrix = np.zeros((n, n), dtype=np.float64)
    for i in range(n):
        for j in range(n):
            if i != j:
                matrix[i, j] = abs(NODE_COUNTS[i] - NODE_COUNTS[j]) + abs(i - j)
    # An isomorphic-pair stand-in: GED is legitimately 0 off the diagonal.
    matrix[0, 1] = matrix[1, 0] = 0.0
    return matrix


def _write_exact(root: str, dataset: str) -> None:
    directory = os.path.join(root, ged_reference.GED_SUBDIR)
    os.makedirs(directory, exist_ok=True)
    n = len(NODE_COUNTS)
    np.savez(
        os.path.join(directory, f"{dataset}.npz"),
        ged_matrix=_reference_matrix(),
        certified_mask=np.ones((n, n), dtype=bool),
        graph_ids=np.asarray([f"g{i}" for i in range(n)]),
    )


def _write_bounds(root: str, dataset: str, *, censor: bool = False) -> None:
    """Both sides of the bracket.  ``censor`` puts one ``inf`` in each."""
    lb = _reference_matrix()
    ub = lb * 2.0
    np.fill_diagonal(ub, 0.0)
    if censor:
        lb[2, 4] = lb[4, 2] = np.inf
        ub[2, 4] = ub[4, 2] = np.inf
    n = len(NODE_COUNTS)
    ids = np.asarray([f"g{i}" for i in range(n)])
    for which, values in (("lb", lb), ("ub", ub)):
        directory = os.path.join(root, ged_reference.BOUNDS_SUBDIR, ged_reference.BOUNDS_DIR[which])
        os.makedirs(directory, exist_ok=True)
        np.savez(
            os.path.join(directory, f"{dataset}.npz"),
            lb_matrix=lb,
            ub_matrix=ub,
            ged_matrix=values,
            certified_mask=np.zeros((n, n), dtype=bool),
            seconds_matrix=np.zeros((n, n), dtype=np.float32),
            node_counts=np.asarray(NODE_COUNTS, dtype=np.int32),
            edge_counts=np.asarray([2, 3, 3, 4, 4, 5], dtype=np.int32),
            graph_ids=ids,
            labels=np.asarray(["a"] * n),
            metadata=json.dumps({"dataset": dataset, "which": which, "method": "BRANCH_FAST"}),
        )


def _clear_caches() -> None:
    datasets.load.cache_clear()
    ged_reference.load_ged.cache_clear()
    ged_reference.load_bounds.cache_clear()


@pytest.fixture
def cohort_root(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> Iterator[str]:
    """A complete synthetic root: one Suite-1 cohort and one Suite-2 cohort."""
    root = str(tmp_path / "data")
    _write_cohort(root, "exported", "linux")
    _write_exact(root, "linux")
    _write_cohort(root, "exported_suite2", "grec")
    _write_bounds(root, "grec")
    monkeypatch.setenv(datasets.ENV_ROOT, root)
    _clear_caches()
    yield root
    _clear_caches()


@pytest.fixture
def grid_file(tmp_path: Path) -> str:
    """A grid JSON in which one representation has no admissible distance."""
    path = tmp_path / "grid.json"
    path.write_text(
        json.dumps(
            {
                "protocol": "T-04a",
                "primary_distance": {"adjacency": "levenshtein", "min_dfs": None},
            }
        ),
        encoding="utf-8",
    )
    return str(path)


# ----------------------------------------------------------------------
# bootstrap.py -- determinism and the paired-resample property
# ----------------------------------------------------------------------


def test_resample_index_is_a_pure_function_of_its_arguments() -> None:
    first = make_resample_index(12, resamples=40, seed=42)
    second = make_resample_index(12, resamples=40, seed=42)
    assert np.array_equal(first.draws, second.draws)
    assert first.draws.shape == (40, 12)


def test_a_different_seed_gives_different_draws() -> None:
    a = make_resample_index(12, resamples=40, seed=42)
    b = make_resample_index(12, resamples=40, seed=43)
    assert not np.array_equal(a.draws, b.draws)


def test_the_interval_is_deterministic_for_a_fixed_index() -> None:
    index = make_resample_index(10, resamples=100, seed=42)
    pairs = [(a, b) for a in range(10) for b in range(a + 1, 10)]
    x = [float(a * 3 + b) for a, b in pairs]
    y = [float(a + b * 2) for a, b in pairs]
    assert graph_bootstrap_ci(x, y, pairs, index) == graph_bootstrap_ci(x, y, pairs, index)


def test_ci_is_none_below_three_pairs() -> None:
    index = make_resample_index(4, resamples=20, seed=42)
    assert graph_bootstrap_ci([1.0, 2.0], [1.0, 2.0], [(0, 1), (2, 3)], index) is None


def test_a_resampled_self_pair_contributes_nothing() -> None:
    """A graph drawn twice must not pair with itself.

    With two graphs and one pair, a replicate that draws ``[0, 0]`` induces
    no pair at all -- the lookup diagonal is never written.  Fabricating a
    distance-0/GED-0 point there would bias rho upward exactly where the
    distribution is tightest.
    """
    index = make_resample_index(2, resamples=50, seed=42)
    assert graph_bootstrap_ci([1.0], [1.0], [(0, 1)], index) is None


def test_pair_index_outside_the_draw_raises() -> None:
    index = make_resample_index(3, resamples=5, seed=42)
    with pytest.raises(BootstrapError):
        graph_bootstrap_ci([1.0, 2.0, 3.0], [1.0, 2.0, 3.0], [(0, 1), (1, 2), (0, 7)], index)


def test_length_disagreement_raises() -> None:
    index = make_resample_index(3, resamples=5, seed=42)
    with pytest.raises(BootstrapError):
        graph_bootstrap_ci([1.0, 2.0], [1.0, 2.0, 3.0], [(0, 1), (1, 2), (0, 2)], index)


def test_every_representation_in_a_view_shares_one_resample_index(
    cohort_root: str, grid_file: str, monkeypatch: pytest.MonkeyPatch
) -> None:
    """D7: the draws are paired across representations, not independent.

    Independent draws would still produce intervals.  They would just be
    intervals whose differences carry the noise of two separate resamplings,
    which is the one comparison the paper leads with.
    """
    seen: list[int] = []
    real = f5.graph_bootstrap_ci

    def spy(*args: object, **kwargs: object) -> tuple[float, float] | None:
        seen.append(id(args[3]))
        return real(*args, **kwargs)  # type: ignore[arg-type]

    monkeypatch.setattr(f5, "graph_bootstrap_ci", spy)
    f5.run(grid_file, names=("linux",), n_graphs=6, seed=42, resamples=20)
    assert seen, "no interval was computed at all"
    assert len(set(seen)) == 1, f"{len(set(seen))} distinct resample indices were used"


# ----------------------------------------------------------------------
# ged_reference.load_bounds -- the traps that fail silently
# ----------------------------------------------------------------------


def test_censored_pairs_are_filtered_with_isfinite_not_isnan(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """``inf`` is the censoring value, and ``np.isnan`` does not catch it."""
    assert not np.isnan(np.inf), "the premise of this test"

    root = str(tmp_path / "data")
    _write_cohort(root, "exported_suite2", "grec")
    _write_bounds(root, "grec", censor=True)
    monkeypatch.setenv(datasets.ENV_ROOT, root)
    _clear_caches()

    bounds = load_bounds("grec", "lb")
    indices = tuple(range(len(NODE_COUNTS)))
    pairs = bounds.finite_pairs(indices)
    assert (2, 4) not in pairs
    assert len(pairs) == len(indices) * (len(indices) - 1) // 2 - 1
    assert all(np.isfinite(bounds.values[a, b]) for a, b in pairs)
    _clear_caches()


def test_a_ged_of_exactly_zero_is_retained(cohort_root: str) -> None:
    """GED 0 means isomorphic, not missing.  It must reach the pair set."""
    bounds = load_bounds("grec", "lb")
    indices = tuple(range(len(NODE_COUNTS)))
    assert bounds.values[0, 1] == 0.0
    assert (0, 1) in bounds.finite_pairs(indices)


def test_lb_and_ub_read_different_matrices(cohort_root: str) -> None:
    lb = load_bounds("grec", "lb")
    ub = load_bounds("grec", "ub")
    assert not np.array_equal(lb.values, ub.values)
    assert (lb.values <= ub.values).all()


def test_a_midpoint_cannot_be_requested(cohort_root: str) -> None:
    """There is no third side.  Interpolating the bracket is signed policy."""
    with pytest.raises(GEDReferenceError, match="never interpolated"):
        load_bounds("grec", "mid")


def test_an_all_zero_matrix_aborts(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    """The shape of GEDLIB's silent wrong-accessor failure."""
    root = str(tmp_path / "data")
    _write_cohort(root, "exported_suite2", "grec")
    n = len(NODE_COUNTS)
    directory = os.path.join(root, ged_reference.BOUNDS_SUBDIR, "LB")
    os.makedirs(directory, exist_ok=True)
    np.savez(
        os.path.join(directory, "grec.npz"),
        lb_matrix=np.zeros((n, n), dtype=np.float64),
        ub_matrix=np.zeros((n, n), dtype=np.float64),
        graph_ids=np.asarray([f"g{i}" for i in range(n)]),
    )
    monkeypatch.setenv(datasets.ENV_ROOT, root)
    _clear_caches()
    with pytest.raises(GEDReferenceError, match="exactly zero off the"):
        load_bounds("grec", "lb")
    _clear_caches()


def test_misaligned_graph_ids_abort(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    root = str(tmp_path / "data")
    _write_cohort(root, "exported_suite2", "grec")
    _write_bounds(root, "grec")
    directory = os.path.join(root, ged_reference.BOUNDS_SUBDIR, "LB")
    data = dict(np.load(os.path.join(directory, "grec.npz"), allow_pickle=True))
    data["graph_ids"] = np.asarray([f"h{i}" for i in range(len(NODE_COUNTS))])
    np.savez(os.path.join(directory, "grec.npz"), **data)
    monkeypatch.setenv(datasets.ENV_ROOT, root)
    _clear_caches()
    with pytest.raises(GEDReferenceError, match="misalign"):
        load_bounds("grec", "lb")
    _clear_caches()


# ----------------------------------------------------------------------
# f5 -- the views, the printed absence, and the emitted schema
# ----------------------------------------------------------------------


def test_equal_n_view_holds_exactly_the_equal_order_pairs(cohort_root: str, grid_file: str) -> None:
    """Three of fifteen pairs share an order in the synthetic cohort."""
    payload = f5.run(grid_file, names=("linux",), n_graphs=6, seed=42, resamples=20)
    record = payload["results"]["linux"]
    all_pairs = record["views"]["all_pairs"]["adjacency"]["n_pairs"]
    equal_n = record["views"]["equal_n"]["adjacency"]["n_pairs"]
    assert all_pairs == 15
    assert equal_n == 3
    assert record["n_reference_pairs"] == 15


def test_the_size_null_is_undefined_in_the_equal_n_view(cohort_root: str, grid_file: str) -> None:
    """|n1 - n2| is 0 on every equal-``n`` pair, so its rho has no denominator.

    It must be a printed absence with a stated reason, never a NaN: NaN is
    not valid JSON and a downstream reader would take it for a number.
    """
    payload = f5.run(grid_file, names=("linux",), n_graphs=6, seed=42, resamples=20)
    views = payload["results"]["linux"]["views"]
    assert views["all_pairs"]["size_null"]["rho"] is not None
    null = views["equal_n"]["size_null"]
    assert null["rho"] is None
    assert "constant" in null["reason"]


def test_a_null_primary_distance_is_a_printed_absence(cohort_root: str, grid_file: str) -> None:
    payload = f5.run(grid_file, names=("linux",), n_graphs=6, seed=42, resamples=20)
    for view in f5.VIEWS:
        cell = payload["results"]["linux"]["views"][view]["min_dfs"]
        assert cell["rho"] is None
        assert cell["metric"] is None
        assert cell["reason"] == f5.NO_DISTANCE_REASON
        assert set(cell) == {
            "metric",
            "rho",
            "p",
            "ci",
            "n_pairs",
            "n_undefined",
            "zero_frac",
            "size_null_on_my_pairs",
            "reason",
        }


def test_a_representation_absent_from_the_grid_is_still_printed(
    cohort_root: str, grid_file: str
) -> None:
    """A shrinking grid must not silently shrink the F5 table."""
    payload = f5.run(grid_file, names=("linux",), n_graphs=6, seed=42, resamples=20)
    cell = payload["results"]["linux"]["views"]["all_pairs"]["graph6"]
    assert cell["rho"] is None
    assert cell["reason"] == f5.ABSENT_REASON


def test_suite2_emits_two_distinct_records_and_never_a_midpoint(
    cohort_root: str, grid_file: str
) -> None:
    payload = f5.run(grid_file, names=("grec",), n_graphs=6, seed=42, resamples=50)
    keys = set(payload["results"])
    assert keys == {"grec::lb", "grec::ub"}
    assert not any("::mid" in k for k in keys)
    assert payload["results"]["grec::lb"]["reference"] == "lb"
    assert payload["results"]["grec::ub"]["reference"] == "ub"
    lb = payload["results"]["grec::lb"]["views"]["all_pairs"]["adjacency"]["rho"]
    ub = payload["results"]["grec::ub"]["views"]["all_pairs"]["adjacency"]["rho"]
    assert lb is not None and ub is not None


def test_the_payload_is_json_serialisable_without_nan(
    cohort_root: str, grid_file: str, tmp_path: Path
) -> None:
    """``allow_nan=False`` is the guard; this asserts nothing trips it."""
    payload = f5.run(grid_file, names=("linux", "grec"), n_graphs=6, seed=42, resamples=20)
    text = json.dumps(payload, allow_nan=False)
    assert "NaN" not in text
    assert "Infinity" not in text


def test_the_record_carries_the_contract_fields(cohort_root: str, grid_file: str) -> None:
    payload = f5.run(grid_file, names=("linux",), n_graphs=6, seed=42, resamples=20)
    assert payload["protocol"] == "T-04a-F5"
    assert "not an input to distance selection" in payload["note"]
    assert payload["bootstrap_resamples"] == 20
    assert os.path.isabs(payload["primary_distance_source"])
    record = payload["results"]["linux"]
    for field in ("dataset", "suite", "reference", "n_graphs", "n_unencodable", "views"):
        assert field in record
    assert record["reference"] == "exact"
    assert record["suite"] == "suite1"
    cell = record["views"]["all_pairs"]["adjacency"]
    for field in (
        "metric",
        "rho",
        "p",
        "ci",
        "n_pairs",
        "n_undefined",
        "zero_frac",
        "size_null_on_my_pairs",
        "reason",
    ):
        assert field in cell
    assert cell["ci"] is None or len(cell["ci"]) == 2


def test_unencodable_graphs_are_counted_not_dropped_silently(
    cohort_root: str, grid_file: str
) -> None:
    payload = f5.run(grid_file, names=("linux",), n_graphs=6, seed=42, resamples=20)
    counts = payload["results"]["linux"]["n_unencodable"]
    assert "adjacency" in counts
    assert counts["adjacency"] == 0


def test_a_suite1_only_representation_gets_no_printed_suite2_row(
    cohort_root: str, tmp_path: Path
) -> None:
    """Design criteria 5 and 9, applied to F5.

    ``isalgraph_canonical`` raises above n = 12, so on a Suite-2 cohort it
    encodes only the graphs that happen to be Suite-1-sized.  A rho over
    those is a rho over the easy half of the cohort wearing the Suite-2
    label, and ``base.table_scope_error`` is the frozen rule that forbids
    printing it.  The Suite-1 row is unaffected.
    """
    path = tmp_path / "grid_canonical.json"
    path.write_text(
        json.dumps({"primary_distance": {"isalgraph_canonical": "levenshtein"}}),
        encoding="utf-8",
    )
    payload = f5.run(str(path), names=("linux", "grec"), n_graphs=6, seed=42, resamples=20)

    suite1 = payload["results"]["linux"]["views"]["all_pairs"]["isalgraph_canonical"]
    assert suite1["rho"] is not None

    for key in ("grec::lb", "grec::ub"):
        cell = payload["results"][key]["views"]["all_pairs"]["isalgraph_canonical"]
        assert cell["rho"] is None
        assert "SUITE1_ONLY" in cell["reason"]
        # None, not 0: the representation was never attempted here, and a 0
        # would assert that every graph encoded.
        assert payload["results"][key]["n_unencodable"]["isalgraph_canonical"] is None


def test_a_size_correlated_failure_moves_the_null_and_the_restricted_one_is_used(
    cohort_root: str, grid_file: str, monkeypatch: pytest.MonkeyPatch
) -> None:
    """The null must be recomputed on the pairs the representation kept.

    A backend that fails on the largest graphs leaves a cohort with less
    spread in ``|n1 - n2|``, which makes the remaining pairs *easier for the
    null* while the representation's own rho is untouched -- it was always
    on its own subset. Subtracting the cohort-level null from it is then a
    difference of two correlations over two different samples.

    The direction matters: censoring by a wall clock removes the largest
    graphs, so the bias inflates the apparent margin over the null. On
    Mutagenicity, measured, this is worth +0.118 of margin.

    Here the fixture drops the two 5-node graphs, the largest in the
    synthetic cohort.
    """
    real = registry.get_backend

    def flaky(name: str, **kwargs: object) -> object:
        backend = real(name, **kwargs)
        if name != "adjacency":
            return backend
        inner = backend.encode

        def encode(graph: object, **inner_kwargs: object) -> object:
            if graph.number_of_nodes() == max(NODE_COUNTS):  # type: ignore[attr-defined]
                raise CanonicalizationTimeoutError("synthetic, size-correlated")
            return inner(graph, **inner_kwargs)

        backend.encode = encode  # type: ignore[method-assign]
        return backend

    monkeypatch.setattr(f5, "get_backend", flaky)
    payload = f5.run(grid_file, names=("linux",), n_graphs=6, seed=42, resamples=50)
    row = payload["results"]["linux"]["views"]["all_pairs"]

    cell = row["adjacency"]
    cohort_null = row[SIZE_NULL]["rho"]
    own = cell["size_null_on_my_pairs"]

    # The representation lost the two largest graphs, so its pair set is a
    # strict subset and its null is a different number.
    assert payload["results"]["linux"]["n_unencodable"]["adjacency"] == 2
    assert cell["n_pairs"] < row[SIZE_NULL]["n_pairs"]
    assert own["n_pairs"] == cell["n_pairs"]
    assert own["rho"] != cohort_null

    # Both are present. Neither silently replaces the other.
    assert cohort_null is not None
    assert own["rho"] is not None

    # A representation that lost nothing has the two agreeing exactly, so the
    # extra field is inert wherever there is no censoring.
    intact = row[SIZE_NULL]
    assert intact["size_null_on_my_pairs"]["rho"] == pytest.approx(intact["rho"])
    assert intact["size_null_on_my_pairs"]["n_pairs"] == intact["n_pairs"]


def test_the_grids_null_size_null_entry_does_not_delete_the_null_row(
    cohort_root: str, tmp_path: Path
) -> None:
    """The real grid lists ``size_null`` with a null primary distance.

    It must: CONTRACTS §4 forbids selecting a baseline as any representation's
    primary distance.  But ``size_null`` is also the dedicated baseline row
    every other row is judged against, so treating the grid's entry as an
    ordinary representation overwrites that row with a printed absence and
    deletes the null from the whole table -- silently, since a printed
    absence is a legitimate value everywhere else.
    """
    path = tmp_path / "grid_with_null_baseline.json"
    path.write_text(
        json.dumps({"primary_distance": {"adjacency": "levenshtein", "size_null": None}}),
        encoding="utf-8",
    )
    payload = f5.run(str(path), names=("linux",), n_graphs=6, seed=42, resamples=20)
    null = payload["results"]["linux"]["views"]["all_pairs"]["size_null"]
    assert null["rho"] is not None
    assert null["metric"] == "size_null"
    assert null["n_pairs"] == 15


def test_a_core_engine_timeout_is_a_datum_not_a_crash(
    cohort_root: str, grid_file: str, monkeypatch: pytest.MonkeyPatch
) -> None:
    """`CanonicalizationTimeoutError` must not take the run down.

    It is an ``EncodingError``, so it descends from ``IsalGraphError`` but
    **not** from ``CompetitorError``.  A guard written against
    ``CompetitorError`` catches every competitor backend's own budget
    exception and misses the core engine's wall clock -- which is the one
    that fires on ``isalgraph_pruned``, the paper's own reference arm, on
    Suite-2-sized graphs.  Small-graph demonstration runs never reach it.
    """
    assert issubclass(CanonicalizationTimeoutError, IsalGraphError)
    assert not issubclass(CanonicalizationTimeoutError, CompetitorError)

    real = registry.get_backend
    victim = 2  # one graph of the six, chosen so pairs survive around it

    def flaky(name: str, **kwargs: object) -> object:
        backend = real(name, **kwargs)
        if name != "adjacency":
            return backend
        inner = backend.encode

        def encode(graph: object, **inner_kwargs: object) -> object:
            if graph.number_of_nodes() == NODE_COUNTS[victim] and not getattr(
                encode, "fired", False
            ):
                encode.fired = True  # type: ignore[attr-defined]
                raise CanonicalizationTimeoutError("synthetic wall-clock budget")
            return inner(graph, **inner_kwargs)

        backend.encode = encode  # type: ignore[method-assign]
        return backend

    monkeypatch.setattr(f5, "get_backend", flaky)
    payload = f5.run(grid_file, names=("linux",), n_graphs=6, seed=42, resamples=20)

    record = payload["results"]["linux"]
    assert record["n_unencodable"]["adjacency"] == 1
    # The kind is recorded under its own name, never collapsed into one count.
    assert record["encode_errors"]["adjacency"] == {"CanonicalizationTimeoutError": 1}

    # The excluded graph is gone from this backend's pair set and from it
    # alone: five graphs leave 10 pairs, against the size null's 15.
    cell = record["views"]["all_pairs"]["adjacency"]
    assert cell["rho"] is not None
    assert cell["n_pairs"] == 10
    assert record["views"]["all_pairs"]["size_null"]["n_pairs"] == 15
    assert record["n_reference_pairs"] == 15


def test_encode_draw_catches_the_whole_isalgraph_branch(cohort_root: str) -> None:
    """Every leaf under ``IsalGraphError`` is a datum; a genuine bug is not."""
    draw = f5.draw_for("linux", 6, 42)
    for exc in (
        CanonicalizationTimeoutError("wall clock"),
        SuiteScopeError("declared refusal"),
        BudgetExceeded("internal cap"),
    ):
        _, unencodable, errors = _encode_raising("adjacency", draw, exc)
        assert unencodable == len(draw.indices)
        assert errors == {type(exc).__name__: len(draw.indices)}

    with pytest.raises(TypeError):
        _encode_raising("adjacency", draw, TypeError("a bug in our own code"))


def _encode_raising(
    name: str, draw: f5.Draw, exc: BaseException
) -> tuple[dict[int, object], int, dict[str, int]]:
    """Run ``encode_draw`` with a backend whose ``encode`` always raises *exc*."""
    real = registry.get_backend

    def factory(backend_name: str, **kwargs: object) -> object:
        backend = real(backend_name, **kwargs)

        def encode(graph: object, **inner: object) -> object:
            raise exc

        backend.encode = encode  # type: ignore[method-assign]
        return backend

    monkey = pytest.MonkeyPatch()
    monkey.setattr(f5, "get_backend", factory)
    try:
        return f5.encode_draw(name, draw)
    finally:
        monkey.undo()


# ----------------------------------------------------------------------
# The real cohort.  Skips cleanly when the external drive is absent.
# ----------------------------------------------------------------------


@pytest.mark.integration
def test_real_suite2_bounds_align_with_their_cohort() -> None:
    if "grec" not in datasets.available_datasets():
        pytest.skip("the exported cohorts are not mounted")
    for which in ("lb", "ub"):
        bounds = load_bounds("grec", which)
        assert bounds.values.shape[0] == len(datasets.load("grec"))
        assert bounds.offdiag_zero_fraction < ged_reference.MAX_OFFDIAG_ZERO_FRACTION
