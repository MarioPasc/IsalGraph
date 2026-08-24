"""T-06: the WL subtree comparator's own distance driver.

``wl_subtree`` is a ``VectorBackend`` with no ``encode()``, so no CONTRACTS §3
encodings file exists for it and the encodings -> distance pipeline cannot feed
it.  This module tests the driver that stands in for that path.

Per CONTRACTS §1.2, ``import isalgraph`` resolves to the **main checkout**, not
to this worktree.  **No test here asserts a numeric value the WL backend
produced.**  Assertions are invariants -- symmetry, exact zero diagonal,
non-negativity, the ``graph_ids`` join, the frozen ``h``, invariance under
relabelling, and agreement between our Gram construction and grakel's when
grakel is installed -- or properties of this driver's own code.
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import numpy as np
import pytest

from benchmarks.eval_distance import wl_driver
from benchmarks.eval_distance.schema import (
    DENSE_KEYS,
    METADATA_KEYS,
    SchemaError,
    build_metadata,
    load_dense,
)

COHORT_ROOT = Path(
    "/media/mpascual/Sandisk2TB/research/ISAL/completed/isalgraph/data/source/APPROX_GED"
    "/exported_suite2"
)

requires_cohort = pytest.mark.skipif(
    not COHORT_ROOT.is_dir(), reason="the exported Suite-2 cohort is not mounted"
)


# --------------------------------------------------------------------------
# Fixtures: a CSR cohort archive and a matching §3 reference, both synthetic
# --------------------------------------------------------------------------


def _write_cohort(path: Path, graphs: list[Any], graph_ids: list[str]) -> Path:
    """Write a CSR cohort archive in the exported layout."""
    n_nodes = np.asarray([g.number_of_nodes() for g in graphs], dtype=np.int32)
    offsets = [0]
    sources: list[int] = []
    targets: list[int] = []
    for graph in graphs:
        for u, v in graph.edges():
            sources.append(int(u))
            targets.append(int(v))
        offsets.append(len(sources))
    path.parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(
        path,
        n_nodes=n_nodes,
        edge_offsets=np.asarray(offsets, dtype=np.int64),
        edges=np.asarray([sources, targets], dtype=np.int32).reshape(2, -1),
        graph_ids=np.asarray(graph_ids, dtype="<U16"),
    )
    return path


def _write_reference(path: Path, graph_ids: list[str], dataset: str) -> Path:
    """Write a minimal CONTRACTS §3 file carrying only the join."""
    n = len(graph_ids)
    meta = build_metadata(
        suite="suite2",
        dataset=dataset,
        representation="isalgraph_pruned",
        metric=None,
        n_graphs=n,
        notes="synthesised by tests/unit/test_t06_wl_driver.py",
        extra={"symbol_sep": ""},
    )
    path.parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(
        path,
        graph_ids=np.asarray(graph_ids, dtype="<U16"),
        node_counts=np.ones(n, dtype=np.int32),
        edge_counts=np.ones(n, dtype=np.int32),
        encoding=np.asarray(["V"] * n, dtype=np.str_),
        length=np.ones(n, dtype=np.int32),
        error_kind=np.asarray([""] * n, dtype="<U32"),
        entropy_bits=np.full(n, np.nan, dtype=np.float64),
        realised_bits=np.full(n, np.nan, dtype=np.float64),
        status=np.asarray(["ok"] * n, dtype="<U12"),
        fallback_used=np.zeros(n, dtype=bool),
        seconds=np.zeros(n, dtype=np.float32),
        metadata=np.array(json.dumps(meta)),
    )
    return path


@pytest.fixture
def toy(tmp_path: Path) -> tuple[Path, Path, Path]:
    """A 12-graph synthetic dataset: cohort, reference encodings, out dir."""
    nx = pytest.importorskip("networkx")
    graphs = [
        nx.path_graph(3),
        nx.path_graph(4),
        nx.cycle_graph(4),
        nx.cycle_graph(5),
        nx.star_graph(3),
        nx.star_graph(4),
        nx.complete_graph(4),
        nx.complete_graph(5),
        nx.petersen_graph(),
        nx.wheel_graph(5),
        nx.ladder_graph(3),
        nx.lollipop_graph(4, 2),
    ]
    ids = [f"g{i:04d}" for i in range(len(graphs))]
    cohort = _write_cohort(tmp_path / "toy.npz", graphs, ids)
    reference = _write_reference(tmp_path / "toy__isalgraph_pruned.npz", ids, "toy")
    return cohort, reference, tmp_path / "out"


# --------------------------------------------------------------------------
# Schema and CONTRACTS §4 conformance
# --------------------------------------------------------------------------


def test_output_conforms_to_the_contracts_distance_schema(toy: tuple[Path, Path, Path]) -> None:
    cohort, reference, out_dir = toy
    out = wl_driver.run(cohort, reference, out_dir)
    assert out.name == "toy__wl_subtree__kernel.npz"
    with np.load(out, allow_pickle=False) as handle:
        assert set(handle.files) == set(DENSE_KEYS)
        assert handle["distance_matrix"].dtype == np.float64
        assert handle["defined_mask"].dtype == np.bool_
    meta = load_dense(out).metadata
    assert set(METADATA_KEYS) <= set(meta)
    assert meta["representation"] == "wl_subtree"
    assert meta["metric"] == "kernel"


def test_the_distance_is_a_symmetric_zero_diagonal_finite_matrix(
    toy: tuple[Path, Path, Path],
) -> None:
    cohort, reference, out_dir = toy
    loaded = load_dense(wl_driver.run(cohort, reference, out_dir))
    matrix = loaded.distance_matrix
    assert np.array_equal(matrix, matrix.T)
    assert np.all(np.diagonal(matrix) == 0.0)
    assert np.isfinite(matrix).all()
    assert (matrix >= 0.0).all()


def test_every_pair_is_defined_because_wl_cannot_fail(toy: tuple[Path, Path, Path]) -> None:
    """WL takes no budget and loses no graph, so it contributes nothing to `c`."""
    cohort, reference, out_dir = toy
    assert load_dense(wl_driver.run(cohort, reference, out_dir)).defined_mask.all()


# --------------------------------------------------------------------------
# The frozen h, and the join
# --------------------------------------------------------------------------


def test_h_is_frozen_at_the_backends_own_wl_rounds(toy: tuple[Path, Path, Path]) -> None:
    """`h` must not be tunable: tuning it on a correlation with GED is the
    failure this guards against. F-14: h = 2 is grakel n_iter = 2."""
    wl = pytest.importorskip("isalgraph.competitors.backends.wl")
    cohort, reference, out_dir = toy
    meta = load_dense(wl_driver.run(cohort, reference, out_dir)).metadata
    assert meta["h"] == wl.WL_ROUNDS
    options = {
        option for action in wl_driver.build_parser()._actions for option in action.option_strings
    }
    assert not options & {"--h", "--rounds", "--n-iter", "--wl-rounds"}


def test_the_graph_ids_join_is_checked_element_wise(toy: tuple[Path, Path, Path]) -> None:
    """A cohort-order drift must fail here, not silently misalign a rho."""
    cohort, reference, out_dir = toy
    scrambled = reference.with_name("scrambled.npz")
    with np.load(reference, allow_pickle=False) as handle:
        arrays = dict(handle)
    ids = arrays["graph_ids"].copy()
    ids[3], ids[4] = ids[4], ids[3]
    arrays["graph_ids"] = ids
    np.savez_compressed(scrambled, **arrays)
    with pytest.raises(SchemaError, match="diverge from the cohort at position 3"):
        wl_driver.run(cohort, scrambled, out_dir)


def test_a_reference_of_the_wrong_length_is_rejected(toy: tuple[Path, Path, Path]) -> None:
    cohort, reference, out_dir = toy
    short = _write_reference(reference.with_name("short.npz"), ["g0000", "g0001"], "toy")
    with pytest.raises(SchemaError, match="graph_ids against the cohort"):
        wl_driver.run(cohort, short, out_dir)


def test_a_non_cohort_archive_is_rejected(tmp_path: Path) -> None:
    bogus = tmp_path / "bogus.npz"
    np.savez_compressed(bogus, something_else=np.zeros(3))
    reference = _write_reference(tmp_path / "r.npz", ["g0000"], "bogus")
    with pytest.raises(SchemaError, match="not a CSR cohort archive"):
        wl_driver.run(bogus, reference, tmp_path / "out")


# --------------------------------------------------------------------------
# Mathematical properties of the kernel distance
# --------------------------------------------------------------------------


def test_kernel_distance_matches_its_definition_on_a_known_gram() -> None:
    """d(i,j) = sqrt(K_ii + K_jj - 2 K_ij) for the linear kernel on counts."""
    counts = np.array([[1.0, 0.0], [0.0, 1.0], [3.0, 4.0]])
    got = wl_driver.kernel_distance_matrix(counts)
    gram = counts @ counts.T
    for i in range(3):
        for j in range(3):
            expected = np.sqrt(max(gram[i, i] + gram[j, j] - 2 * gram[i, j], 0.0))
            assert got[i, j] == pytest.approx(expected, abs=1e-12)


def test_identical_feature_vectors_are_at_distance_exactly_zero() -> None:
    """The pseudometric property, asserted rather than tolerated: WL gives
    non-isomorphic graphs it cannot separate exactly 0.0, and that is a
    property of WL, not an unfilled buffer."""
    counts = np.array([[2.0, 5.0], [2.0, 5.0]])
    assert wl_driver.kernel_distance_matrix(counts)[0, 1] == 0.0


def test_the_distance_is_invariant_under_relabelling(tmp_path: Path) -> None:
    """A representation whose distance moved under relabelling would not be a
    graph invariant at all."""
    nx = pytest.importorskip("networkx")
    rng = np.random.default_rng(11)
    base = [nx.petersen_graph(), nx.cycle_graph(6), nx.complete_graph(5), nx.ladder_graph(4)]
    permuted = []
    for graph in base:
        order = rng.permutation(graph.number_of_nodes())
        permuted.append(nx.relabel_nodes(graph, {n: int(order[n]) for n in graph.nodes()}))
    straight, _ = wl_driver.feature_table(base)
    shuffled, _ = wl_driver.feature_table(permuted)
    np.testing.assert_allclose(
        wl_driver.kernel_distance_matrix(straight),
        wl_driver.kernel_distance_matrix(shuffled),
        rtol=0,
        atol=1e-9,
    )


def test_our_gram_agrees_with_grakel_at_the_same_h(tmp_path: Path) -> None:
    """The cross-check that F-14 is really h = 2 on both sides.

    Skipped when grakel is absent; it is an oracle, not a dependency.
    """
    nx = pytest.importorskip("networkx")
    wl = pytest.importorskip("isalgraph.competitors.backends.wl")
    if not wl.grakel_available():
        pytest.skip("grakel is not installed")
    graphs = [nx.path_graph(4), nx.cycle_graph(5), nx.complete_graph(4), nx.star_graph(4)]
    counts, rounds = wl_driver.feature_table(graphs)
    ours = counts @ counts.T
    theirs = np.asarray(wl.grakel_gram(graphs, h=rounds), dtype=np.float64)
    np.testing.assert_allclose(ours, theirs, rtol=1e-9, atol=1e-9)


# --------------------------------------------------------------------------
# Real cohort
# --------------------------------------------------------------------------


@requires_cohort
def test_runs_on_a_real_cohort_dataset(tmp_path: Path) -> None:
    cohort = COHORT_ROOT / "linux.npz"
    with np.load(cohort, allow_pickle=False) as handle:
        ids = [str(v) for v in handle["graph_ids"][:60]]
        n_nodes = handle["n_nodes"][:60]
        offsets = handle["edge_offsets"][:61]
        edges = handle["edges"][:, : int(offsets[-1])]
    trimmed = tmp_path / "linux.npz"
    np.savez_compressed(
        trimmed,
        n_nodes=n_nodes,
        edge_offsets=offsets - offsets[0],
        edges=edges,
        graph_ids=np.asarray(ids, dtype="<U16"),
    )
    reference = _write_reference(tmp_path / "linux__isalgraph_pruned.npz", ids, "linux")
    loaded = load_dense(wl_driver.run(trimmed, reference, tmp_path / "out", suite="suite2"))
    assert loaded.n_graphs == 60
    assert np.array_equal(loaded.graph_ids, np.asarray(ids, dtype="<U16"))
    assert np.array_equal(loaded.distance_matrix, loaded.distance_matrix.T)
    assert np.all(np.diagonal(loaded.distance_matrix) == 0.0)
