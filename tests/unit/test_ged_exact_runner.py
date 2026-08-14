"""Tests for the chunked, resumable exact-GED runner (T-03, CONTRACT C).

GEDLIB exists only on Picasso, so every test here injects a backend satisfying the
CONTRACT B protocol. That is the point of the design rather than a workaround: the
runner is typed against a ``Protocol`` and resolves concrete backends lazily, so it
can be exercised end to end on a machine with no solver installed.

What the tests are actually protecting:

* **Resume must not corrupt.** A shard is worth hundreds of core-hours. Resuming
  the wrong chunk's checkpoint would produce a plausible-looking shard containing
  another chunk's answers, so the checkpoint carries a fingerprint of its target
  pair set and a mismatch is a hard error.
* **Nothing is dropped and nothing is promoted.** An uncertified result becomes an
  interval-censored row (D11), never a value in ``ged``. Recording a best-so-far
  cost as exact is the defect the design note found in the incumbent.
* **Per-pair wall time is always recorded**, because the D12 censoring analysis is
  reported per stratum and cannot be reconstructed afterwards.
"""

from __future__ import annotations

import importlib.util
import json
import logging
import os
import signal
import subprocess
import sys
import time
from dataclasses import dataclass
from pathlib import Path

import networkx as nx
import numpy as np
import pytest

from benchmarks.eval_setup.ged_exact_runner import (
    EXIT_FAILURE,
    EXIT_OK,
    EXIT_TERMINATED,
    BackendSpec,
    ContractA,
    PairOutcome,
    RunnerError,
    ShardData,
    _backend_options,
    _outcome_from_result,
    _target_pairs,
    build_parser,
    load_contract_a,
    main,
    make_backend,
    reference_stub_backend,
    run_chunk,
)
from benchmarks.eval_setup.ged_pair_index import index_of_pair, n_pairs, pair_from_index

STUB_FACTORY_REF = "benchmarks.eval_setup.ged_exact_runner:reference_stub_backend"
_LOG = logging.getLogger(__name__)


# --------------------------------------------------------------------------- #
# Fixtures: a CONTRACT A file and an injectable backend
# --------------------------------------------------------------------------- #


def _make_graphs(n: int, seed: int = 0) -> list[nx.Graph]:
    """Build ``n`` small connected graphs of 2 to 12 nodes.

    Args:
        n: Number of graphs.
        seed: RNG seed.

    Returns:
        Connected, undirected, unattributed graphs.
    """
    rng = np.random.default_rng(seed)
    graphs: list[nx.Graph] = []
    while len(graphs) < n:
        size = int(rng.integers(2, 13))
        g = nx.gnp_random_graph(size, 0.45, seed=int(rng.integers(0, 2**31)))
        if nx.is_connected(g):
            graphs.append(nx.convert_node_labels_to_integers(g))
    return graphs


def write_contract_a(path: Path, graphs: list[nx.Graph], key: str = "test") -> None:
    """Write a CONTRACT A ``.npz`` from a list of graphs.

    Fabricated from the key table in CONTRACTS section 4 rather than produced by
    ``export_graphs``, which is owned by a peer and is not on this branch.

    Args:
        path: Destination ``.npz``.
        graphs: Graphs to store.
        key: Dataset key recorded in the metadata.
    """
    n = len(graphs)
    n_nodes = np.array([g.number_of_nodes() for g in graphs], dtype=np.int32)
    n_edges = np.array([g.number_of_edges() for g in graphs], dtype=np.int32)
    offsets = np.zeros(n + 1, dtype=np.int64)
    offsets[1:] = np.cumsum(n_edges.astype(np.int64))
    cols: list[tuple[int, int]] = []
    for g in graphs:
        cols.extend(sorted((min(u, v), max(u, v)) for u, v in g.edges()))
    edges = np.array(cols, dtype=np.int32).T if cols else np.zeros((2, 0), dtype=np.int32)
    np.savez_compressed(
        path,
        graph_ids=np.array([f"{key}_{t:04d}" for t in range(n)], dtype=str),
        n_nodes=n_nodes,
        n_edges=n_edges,
        edge_offsets=offsets,
        edges=edges,
        splits=np.array(["train"] * n, dtype=str),
        labels=np.array([""] * n, dtype=str),
        metadata=np.array(
            json.dumps(
                {
                    "dataset": key,
                    "source": "synthetic",
                    "n_kept": n,
                    "n_pairs": n * (n - 1) // 2,
                    "filter": {"min_nodes": 2, "require_connected": True, "n_max": 12},
                    "n_dropped_size": 0,
                    "n_dropped_disconnected": 0,
                    "n_dropped_trivial": 0,
                    "schema_version": 1,
                }
            )
        ),
    )


@dataclass(frozen=True, slots=True)
class _Result:
    """A CONTRACT B shaped result."""

    lb: float
    ub: float
    exact: float | None
    certified: bool
    seconds: float
    timed_out: bool
    method: str


class _CountingBackend:
    """A deterministic backend that also records how many pairs it was asked for."""

    def __init__(self, certify: bool = True) -> None:
        """Store the certification mode and reset the call counter."""
        self.calls: list[tuple[int, int]] = []
        self._certify = certify

    @property
    def name(self) -> str:
        """Backend identity."""
        return "counting"

    def pair(self, g1: nx.Graph, g2: nx.Graph) -> _Result:
        """Return a deterministic bracket derived from the two node counts."""
        n1, n2 = g1.number_of_nodes(), g2.number_of_nodes()
        self.calls.append((n1, n2))
        val = float(abs(n1 - n2) + 1)
        if self._certify:
            return _Result(val, val, val, True, 0.5, False, "counting/certified")
        return _Result(val, val + 3.0, None, False, 0.25, False, "counting/bracket")


@pytest.fixture()
def cohort(tmp_path: Path) -> tuple[Path, ContractA]:
    """A 12-graph CONTRACT A file and its loaded form."""
    path = tmp_path / "test.npz"
    write_contract_a(path, _make_graphs(12, seed=1))
    return path, load_contract_a(path)


# --------------------------------------------------------------------------- #
# CONTRACT A round-trip
# --------------------------------------------------------------------------- #


def test_contract_a_loads_graphs_edge_for_edge(tmp_path: Path) -> None:
    """The CSR encoding round-trips to the same graphs, without importing torch."""
    graphs = _make_graphs(9, seed=4)
    path = tmp_path / "d.npz"
    write_contract_a(path, graphs, key="aids")
    loaded = load_contract_a(path)
    assert loaded.key == "aids"
    assert loaded.n_graphs == 9
    for original, restored in zip(graphs, loaded.graphs, strict=True):
        assert restored.number_of_nodes() == original.number_of_nodes()
        assert set(map(frozenset, restored.edges())) == set(map(frozenset, original.edges()))


def test_contract_a_rejects_inconsistent_csr_offsets(tmp_path: Path) -> None:
    """An offsets/edge-count disagreement is corruption and must not load silently."""
    path = tmp_path / "bad.npz"
    np.savez_compressed(
        path,
        graph_ids=np.array(["a", "b"], dtype=str),
        n_nodes=np.array([3, 3], dtype=np.int32),
        n_edges=np.array([2, 2], dtype=np.int32),
        edge_offsets=np.array([0, 2, 3], dtype=np.int64),
        edges=np.array([[0, 1, 0], [1, 2, 2]], dtype=np.int32),
    )
    with pytest.raises(RunnerError, match="disagree"):
        load_contract_a(path)


def test_contract_a_rejects_a_file_missing_required_keys(tmp_path: Path) -> None:
    """A non-CONTRACT-A file is named as such rather than raising a KeyError."""
    path = tmp_path / "nope.npz"
    np.savez_compressed(path, something=np.arange(3))
    with pytest.raises(RunnerError, match="CONTRACT A"):
        load_contract_a(path)


# --------------------------------------------------------------------------- #
# Chunk targeting
# --------------------------------------------------------------------------- #


def test_chunks_partition_the_whole_triangle(cohort: tuple[Path, ContractA]) -> None:
    """Three chunks of a 12-graph cohort cover all 66 pairs exactly once."""
    _, dataset = cohort
    seen: list[int] = []
    for t in range(3):
        pairs, _ = _target_pairs(dataset, pair_list=None, chunk_index=t, n_chunks=3)
        seen.extend(int(k) for k in pairs)
    assert sorted(seen) == list(range(n_pairs(12)))


def test_a_pair_list_is_split_instead_of_the_triangle(
    cohort: tuple[Path, ContractA], tmp_path: Path
) -> None:
    """With --pair-list the sampled list is what gets chunked, by the same rule."""
    _, dataset = cohort
    listed = np.array([0, 5, 9, 17, 30, 44, 65], dtype=np.int64)
    plist = tmp_path / "pairs.npz"
    np.savez_compressed(plist, pair_index=listed)
    got: list[int] = []
    for t in range(3):
        pairs, _ = _target_pairs(dataset, pair_list=str(plist), chunk_index=t, n_chunks=3)
        got.extend(int(k) for k in pairs)
    assert got == listed.tolist()


def test_a_pair_list_index_beyond_the_triangle_is_rejected(
    cohort: tuple[Path, ContractA], tmp_path: Path
) -> None:
    """An index for a different cohort would silently compute the wrong pairs."""
    _, dataset = cohort
    plist = tmp_path / "pairs.npz"
    np.savez_compressed(plist, pair_index=np.array([0, 10_000], dtype=np.int64))
    with pytest.raises(RunnerError, match="outside"):
        _target_pairs(dataset, pair_list=str(plist), chunk_index=0, n_chunks=1)


# --------------------------------------------------------------------------- #
# D11: censoring, and never promoting an uncertified value
# --------------------------------------------------------------------------- #


def test_a_certified_result_becomes_a_finite_distance() -> None:
    """certified=True with lb == ub == exact is the only path to a value in ged."""
    out = _outcome_from_result(7, _Result(3.0, 3.0, 3.0, True, 1.5, False, "m"), 9.0)
    assert out == PairOutcome(7, 3.0, 3.0, 3.0, True, 1.5, False)


def test_an_uncertified_result_is_interval_censored_not_dropped() -> None:
    """D11: ged is inf, the bracket survives, and the pair stays in the shard."""
    out = _outcome_from_result(7, _Result(2.0, 6.0, None, False, 0.25, False, "m"), 9.0)
    assert out.ged == float("inf")
    assert out.certified is False
    assert (out.lb, out.ub) == (2.0, 6.0)
    assert np.isfinite(out.lb) and np.isfinite(out.ub)


def test_an_uncertified_best_so_far_value_is_never_promoted() -> None:
    """A backend offering `exact` without certification must not set `ged`.

    This is exactly the defect the design note found in
    ``ged_computer.py::compute_ged_pair``: ``nx.graph_edit_distance(timeout=t)``
    returns its best-found cost when the budget expires, and the incumbent recorded
    it as if it were optimal.
    """
    out = _outcome_from_result(1, _Result(2.0, 9.0, 5.0, False, 0.1, True, "m"), 0.1)
    assert out.ged == float("inf")
    assert out.certified is False


@pytest.mark.parametrize(
    "bad",
    [
        _Result(float("inf"), 3.0, None, False, 0.1, False, "m"),
        _Result(1.0, float("inf"), None, False, 0.1, False, "m"),
        _Result(5.0, 2.0, None, False, 0.1, False, "m"),
        _Result(1.0, 3.0, None, True, 0.1, False, "m"),
        _Result(1.0, 3.0, 99.0, True, 0.1, False, "m"),
    ],
)
def test_an_inconsistent_backend_result_raises(bad: _Result) -> None:
    """Non-finite bounds, inverted brackets and uncertifiable claims all raise."""
    with pytest.raises(RunnerError):
        _outcome_from_result(0, bad, 0.1)


def test_a_backend_reporting_no_wall_time_falls_back_to_measured_time() -> None:
    """seconds is mandatory for D12, so an unusable backend value is substituted."""
    out = _outcome_from_result(0, _Result(1.0, 1.0, 1.0, True, float("nan"), False, "m"), 4.25)
    assert out.seconds == 4.25


# --------------------------------------------------------------------------- #
# Running a chunk
# --------------------------------------------------------------------------- #


def _run(
    dataset: ContractA,
    path: Path,
    pairs: np.ndarray,
    backend: object,
    **kwargs: object,
) -> tuple[ShardData, bool, dict[str, object]]:
    """Run one chunk with an injected backend.

    Args:
        dataset: The cohort.
        path: CONTRACT A path.
        pairs: Pair indices to compute.
        backend: The backend instance to inject.
        **kwargs: Overrides forwarded to :func:`run_chunk`.

    Returns:
        Whatever :func:`run_chunk` returns.
    """
    options: dict[str, object] = {
        "checkpoint_path": None,
        "checkpoint_every": 1000,
        "workers": 1,
    }
    options.update(kwargs)
    return run_chunk(
        dataset=dataset,
        pairs=pairs,
        backend_spec=BackendSpec(name="stub"),
        input_path=str(path),
        backend_factory=lambda _spec: backend,
        **options,  # type: ignore[arg-type]
    )


def test_a_chunk_computes_every_pair_it_owns(cohort: tuple[Path, ContractA]) -> None:
    """The shard holds one row per owned pair, keyed by linear index."""
    path, dataset = cohort
    pairs = np.arange(10, 30, dtype=np.int64)
    shard, completed, stats = _run(dataset, path, pairs, _CountingBackend())
    assert completed is True
    assert len(shard) == 20
    assert stats["n_computed"] == 20
    assert np.array_equal(shard.arrays()["pair_index"], pairs)


def test_the_runner_pairs_the_graphs_the_index_names(
    cohort: tuple[Path, ContractA],
) -> None:
    """Row k really is the pair (i, j) with index_of_pair(i, j, N) == k.

    The whole programme rests on this: a transposition here would fill the matrix
    with other pairs' distances while leaving it symmetric and finite.
    """
    path, dataset = cohort
    n = dataset.n_graphs
    backend = _CountingBackend()
    pairs = np.array([0, 1, 11, 20, 64, 65], dtype=np.int64)
    _run(dataset, path, pairs, backend)
    expected = []
    for k in pairs:
        i, j = pair_from_index(int(k), n)
        assert index_of_pair(i, j, n) == int(k)
        expected.append((dataset.graphs[i].number_of_nodes(), dataset.graphs[j].number_of_nodes()))
    assert backend.calls == expected


def test_seconds_are_recorded_for_every_pair(cohort: tuple[Path, ContractA]) -> None:
    """D12 needs per-pair wall time; a shard without it cannot be re-derived."""
    path, dataset = cohort
    shard, _, _ = _run(dataset, path, np.arange(0, 15, dtype=np.int64), _CountingBackend())
    seconds = shard.arrays()["seconds"]
    assert seconds.dtype == np.float32
    assert seconds.size == 15
    assert bool(np.all(seconds > 0))


def test_a_raising_backend_yields_a_flagged_row_rather_than_a_hole(
    cohort: tuple[Path, ContractA],
) -> None:
    """One bad pair must not kill the shard, and must not vanish from it either."""

    class _Exploding:
        @property
        def name(self) -> str:
            return "exploding"

        def pair(self, g1: nx.Graph, g2: nx.Graph) -> _Result:
            raise ValueError("solver blew up")

    path, dataset = cohort
    shard, completed, _ = _run(dataset, path, np.arange(0, 4, dtype=np.int64), _Exploding())
    assert completed is True
    assert len(shard) == 4
    assert shard.n_failed == 4
    assert all(np.isinf(r.ged) and not r.certified for r in shard.rows.values())


# --------------------------------------------------------------------------- #
# Checkpointing and resume
# --------------------------------------------------------------------------- #


def test_the_checkpoint_is_a_single_file_overwritten_in_place(
    cohort: tuple[Path, ContractA], tmp_path: Path
) -> None:
    """fscratch limits file count, so a checkpoint series is not acceptable."""
    path, dataset = cohort
    ckpt_dir = tmp_path / "ckpt"
    ckpt = ckpt_dir / "c.ckpt.npz"
    _run(
        dataset,
        path,
        np.arange(0, 40, dtype=np.int64),
        _CountingBackend(),
        checkpoint_path=ckpt,
        checkpoint_every=5,
    )
    assert ckpt.is_file()
    # Eight flushes happened; exactly one file survives, and no .tmp is left behind.
    assert sorted(p.name for p in ckpt_dir.iterdir()) == ["c.ckpt.npz"]


def test_resume_skips_completed_pairs_and_reproduces_the_shard(
    cohort: tuple[Path, ContractA], tmp_path: Path
) -> None:
    """A resumed run recomputes only what is missing and matches the full run."""
    path, dataset = cohort
    pairs = np.arange(0, 30, dtype=np.int64)
    full, _, _ = _run(dataset, path, pairs, _CountingBackend())

    ckpt = tmp_path / "r.ckpt.npz"
    partial, _, _ = _run(
        dataset, path, pairs[:12], _CountingBackend(), checkpoint_path=ckpt, checkpoint_every=1000
    )
    assert len(partial) == 12

    # The checkpoint belongs to the 12-pair task; the 30-pair task must refuse it.
    with pytest.raises(RunnerError, match="different pair set"):
        _run(dataset, path, pairs, _CountingBackend(), checkpoint_path=ckpt)

    # Resuming the same task reuses everything and computes nothing.
    backend = _CountingBackend()
    resumed, completed, stats = _run(dataset, path, pairs[:12], backend, checkpoint_path=ckpt)
    assert completed is True
    assert stats["n_resumed"] == 12
    assert stats["n_computed"] == 0
    assert backend.calls == []
    for key, value in resumed.arrays().items():
        assert np.array_equal(value, partial.arrays()[key]), key
    assert np.array_equal(resumed.arrays()["ged"], full.arrays()["ged"][:12]), (
        "resumed values must equal the uninterrupted ones"
    )


def test_a_checkpoint_holding_foreign_pairs_is_rejected(
    cohort: tuple[Path, ContractA], tmp_path: Path
) -> None:
    """Resuming chunk 7's checkpoint into chunk 8 would be silent corruption."""
    path, dataset = cohort
    ckpt = tmp_path / "foreign.ckpt.npz"
    np.savez_compressed(
        ckpt,
        pair_index=np.array([61, 62], dtype=np.int64),
        ged=np.array([1.0, 2.0]),
        lb=np.array([1.0, 2.0]),
        ub=np.array([1.0, 2.0]),
        certified=np.array([True, True]),
        seconds=np.array([0.1, 0.1], dtype=np.float32),
    )
    with pytest.raises(RunnerError, match="does not own"):
        _run(
            dataset,
            path,
            np.arange(0, 10, dtype=np.int64),
            _CountingBackend(),
            checkpoint_path=ckpt,
        )


def test_seed_from_reuses_an_earlier_run_without_recomputing(
    cohort: tuple[Path, ContractA], tmp_path: Path
) -> None:
    """Stage 2 seeds itself from stage 1, which is what makes the merge check meaningful."""
    path, dataset = cohort
    stage1_pairs = np.array([2, 4, 6, 8], dtype=np.int64)
    stage1, _, _ = _run(dataset, path, stage1_pairs, _CountingBackend())
    seed_file = tmp_path / "stage1.npz"
    np.savez_compressed(seed_file, **stage1.arrays())

    backend = _CountingBackend()
    shard, _, stats = _run(
        dataset, path, np.arange(0, 10, dtype=np.int64), backend, seed_from=str(seed_file)
    )
    assert stats["n_seeded"] == 4
    assert stats["n_computed"] == 6
    assert len(backend.calls) == 6
    for k in stage1_pairs:
        assert shard.rows[int(k)] == stage1.rows[int(k)]


# --------------------------------------------------------------------------- #
# Backend resolution
# --------------------------------------------------------------------------- #


def test_an_unavailable_backend_raises_instead_of_degrading() -> None:
    """No silent fallback: a matrix computed by the wrong solver is worse than none.

    Written pre-merge, when ``ged_backends`` did not exist and ``make_backend`` was the
    only place that could fail. Post-merge the module resolves and the GEDLIB import is
    deliberately lazy, so the guarantee moved: what must hold is that asking for GEDLIB
    without GEDLIB installed never yields a *usable* backend. Raising at construction and
    raising at the first pair are both acceptable; returning a value is not.
    """
    if importlib.util.find_spec("gklearn") is not None:
        pytest.skip("GEDLIB is installed here, so unavailability cannot be exercised")
    with pytest.raises(Exception) as excinfo:  # noqa: PT011 - backend layer owns its type
        make_backend(BackendSpec(name="gedlib")).pair(nx.path_graph(4), nx.cycle_graph(4))
    assert not isinstance(excinfo.value, AssertionError)


def test_an_unknown_backend_factory_reference_raises() -> None:
    """A malformed or unresolvable factory reference is named precisely."""
    with pytest.raises(RunnerError, match="module:callable"):
        make_backend(BackendSpec(name="stub", factory_ref="not_a_reference"))
    with pytest.raises(RunnerError, match="cannot import"):
        make_backend(BackendSpec(name="stub", factory_ref="no.such.module:fn"))


def test_the_reference_stub_is_deterministic_and_brackets_correctly() -> None:
    """Same pair, same numbers, in any process -- which is what makes resume checkable."""
    backend = reference_stub_backend(BackendSpec(name="stub"))
    g1 = nx.path_graph(5)
    g2 = nx.cycle_graph(4)
    first = backend.pair(g1, g2)
    second = reference_stub_backend(BackendSpec(name="stub")).pair(g1, g2)
    assert (first.lb, first.ub, first.certified, first.seconds) == (
        second.lb,
        second.ub,
        second.certified,
        second.seconds,
    )
    assert first.lb <= first.ub
    assert np.isfinite(first.lb) and np.isfinite(first.ub)
    assert first.seconds > 0


# --------------------------------------------------------------------------- #
# CLI
# --------------------------------------------------------------------------- #


def test_cli_writes_a_shard_with_the_contract_c_keys(tmp_path: Path) -> None:
    """All six data keys plus meta, with the dtypes CONTRACT C specifies."""
    src = tmp_path / "aids.npz"
    write_contract_a(src, _make_graphs(14, seed=8), key="aids")
    out = tmp_path / "aids_c0000.npz"
    code = main(
        [
            "--input",
            str(src),
            "--out",
            str(out),
            "--backend",
            "stub",
            "--backend-factory",
            STUB_FACTORY_REF,
            "--chunk-index",
            "0",
            "--n-chunks",
            "1",
            "--workers",
            "1",
            "--log-level",
            "WARNING",
        ]
    )
    assert code == EXIT_OK
    with np.load(out) as data:
        assert data["pair_index"].dtype == np.int64
        assert data["ged"].dtype == np.float64
        assert data["lb"].dtype == np.float64
        assert data["ub"].dtype == np.float64
        assert data["certified"].dtype == np.bool_
        assert data["seconds"].dtype == np.float32
        assert data["pair_index"].size == n_pairs(14)
        meta = json.loads(str(data["meta"]))
    assert meta["dataset"] == "aids"
    assert meta["n_graphs"] == 14
    assert meta["chunk_index"] == 0
    assert meta["cost_model"] == "unit"
    assert meta["backend_name"] == STUB_FACTORY_REF
    assert "hostname" in meta and "cpu_model" in meta
    assert meta["n_certified"] + meta["n_censored"] == n_pairs(14)


def test_cli_refuses_a_backend_factory_without_the_stub_slot(tmp_path: Path) -> None:
    """The override is restricted to --backend stub so it cannot shadow a real solver."""
    src = tmp_path / "d.npz"
    write_contract_a(src, _make_graphs(4, seed=2))
    code = main(
        [
            "--input",
            str(src),
            "--out",
            str(tmp_path / "o.npz"),
            "--backend",
            "gedlib",
            "--backend-factory",
            STUB_FACTORY_REF,
            "--log-level",
            "CRITICAL",
        ]
    )
    assert code == EXIT_FAILURE


def test_sigterm_flushes_the_checkpoint_and_the_resumed_shard_is_byte_identical(
    tmp_path: Path,
) -> None:
    """Kill a running chunk, resume it, and get exactly the uninterrupted answer.

    This is the property a thousand core-hours rest on. SLURM sends ``SIGTERM`` and
    waits 30 s before ``SIGKILL``; without a handler a wallclock kill discards
    everything since the last checkpoint, and a checkpoint that resumed
    *approximately* would be worse still.

    The comparison covers ``seconds`` as well as the four value arrays, which is
    only meaningful because the reference stub reports a deterministic per-pair
    time rather than a measured one. Against a real solver ``seconds`` is wall time
    and would legitimately differ.
    """
    repo_root = Path(__file__).resolve().parents[2]
    src = tmp_path / "big.npz"
    write_contract_a(src, _make_graphs(250, seed=812), key="big")
    total = n_pairs(250)

    def _cmd(out: Path, ckpt: Path) -> list[str]:
        return [
            sys.executable,
            "-m",
            "benchmarks.real_data.eval_setup.ged_exact_runner",
            "--input",
            str(src),
            "--out",
            str(out),
            "--backend",
            "stub",
            "--backend-factory",
            STUB_FACTORY_REF,
            "--chunk-index",
            "0",
            "--n-chunks",
            "1",
            "--workers",
            "2",
            "--checkpoint-every",
            "200",
            "--checkpoint",
            str(ckpt),
            "--log-level",
            "WARNING",
        ]

    baseline = tmp_path / "baseline.npz"
    done = subprocess.run(
        _cmd(baseline, tmp_path / "baseline.ckpt.npz"), cwd=repo_root, capture_output=True
    )
    assert done.returncode == EXIT_OK, done.stderr.decode()

    # Phase 1: start, wait for the first checkpoint to land, then SIGTERM.
    out = tmp_path / "resumed.npz"
    ckpt = tmp_path / "resumed.ckpt.npz"
    proc = subprocess.Popen(_cmd(out, ckpt), cwd=repo_root, stderr=subprocess.PIPE)
    rows_at_kill = 0
    deadline = time.monotonic() + 120.0
    while time.monotonic() < deadline:
        if ckpt.is_file():
            try:
                with np.load(ckpt) as data:
                    rows_at_kill = int(data["pair_index"].size)
            except (OSError, ValueError):  # pragma: no cover - mid-rename race
                rows_at_kill = 0
            if rows_at_kill >= 3000:
                break
        if proc.poll() is not None:  # pragma: no cover - finished before we looked
            break
        time.sleep(0.01)
    assert rows_at_kill >= 3000, "the run finished before a checkpoint could be observed"
    proc.send_signal(signal.SIGTERM)
    proc.communicate(timeout=120)

    assert proc.returncode == EXIT_TERMINATED, "SIGTERM must exit 143, not be ignored"
    assert not out.exists(), "an incomplete chunk must not write a shard"
    assert not list(tmp_path.glob("*.tmp*")), "the atomic write must leave no temporary file"
    with np.load(ckpt) as data:
        flushed = int(data["pair_index"].size)
    assert 0 < flushed < total, "the checkpoint must hold real, partial progress"

    # Phase 2: rerun the identical command; it resumes rather than restarting.
    again = subprocess.run(_cmd(out, ckpt), cwd=repo_root, capture_output=True)
    assert again.returncode == EXIT_OK, again.stderr.decode()

    # Phase 3: the resumed shard equals the uninterrupted one, array for array.
    with np.load(baseline) as a, np.load(out) as b:
        for key in ("pair_index", "ged", "lb", "ub", "certified", "seconds"):
            assert np.array_equal(a[key], b[key]), f"{key} differs after resume"
            _LOG.info("resume identity: %-11s dtype=%-8s n=%d", key, a[key].dtype, a[key].size)
        meta = json.loads(str(b["meta"]))
    assert meta["n_resumed"] == flushed
    assert meta["n_resumed"] + meta["n_computed"] == total
    _LOG.info(
        "kill/resume over %d pairs: killed at %d checkpointed rows, exit %d, "
        "no shard written; resumed run reused %d and computed %d",
        total,
        flushed,
        EXIT_TERMINATED,
        meta["n_resumed"],
        meta["n_computed"],
    )


def test_cli_chunks_cover_the_triangle_exactly_once(tmp_path: Path) -> None:
    """Three CLI shards of one cohort partition its 66 pairs."""
    src = tmp_path / "d.npz"
    write_contract_a(src, _make_graphs(12, seed=6))
    seen: list[int] = []
    for t in range(3):
        out = tmp_path / f"d_c{t:04d}.npz"
        assert (
            main(
                [
                    "--input",
                    str(src),
                    "--out",
                    str(out),
                    "--backend",
                    "stub",
                    "--backend-factory",
                    STUB_FACTORY_REF,
                    "--chunk-index",
                    str(t),
                    "--n-chunks",
                    "3",
                    "--workers",
                    "1",
                    "--log-level",
                    "WARNING",
                ]
            )
            == EXIT_OK
        )
        with np.load(out) as data:
            seen.extend(int(k) for k in data["pair_index"])
    assert sorted(seen) == list(range(n_pairs(12)))


# --------------------------------------------------------------------------- #
# T-05: the runner expresses the per-role method specification
# --------------------------------------------------------------------------- #


def _parsed(argv: list[str]) -> object:
    """Parse a CLI invocation without running it."""
    return build_parser().parse_args(argv)


_BASE_ARGV = ["--input", "i.npz", "--out", "o.npz", "--backend", "gedlib"]
_DET_START = (
    "--threads 1 --randomness PSEUDO --initialization-method BIPARTITE --initial-solutions 1"
)


class TestMethodSpecificationCli:
    """CONTRACTS §6: additive flags whose defaults are T-03's behaviour."""

    def test_the_defaults_are_exactly_what_t03_ran(self) -> None:
        """A closed ticket's 2,081 core-hours depend on these six values."""
        args = _parsed(_BASE_ARGV)
        assert args.lb_method == "BRANCH_FAST"
        assert args.lb_options == "--threads 1"
        assert args.ub_method == "IPFP"
        assert args.ub_options == "--threads 1"
        assert args.compute == "both"
        assert args.role is None

    def test_the_options_reach_the_backend_verbatim(self) -> None:
        args = _parsed(
            [*_BASE_ARGV, "--ub-method", "BP_BEAM", "--ub-options", _DET_START, "--compute", "ub"]
        )
        opts = _backend_options(args)
        assert opts["ub_method"] == "BP_BEAM"
        assert opts["ub_options"] == _DET_START
        assert opts["compute"] == "ub"

    def test_a_non_gedlib_backend_takes_no_method_options(self) -> None:
        """The stub and networkx constructors are untouched by this change."""
        args = _parsed(["--input", "i.npz", "--out", "o.npz", "--backend", "stub"])
        assert _backend_options(args) == {}

    def test_an_unknown_compute_mode_is_refused_by_the_parser(self) -> None:
        with pytest.raises(SystemExit):
            _parsed([*_BASE_ARGV, "--compute", "upper"])

    def test_the_shard_records_the_role_and_the_options(self, tmp_path: Path) -> None:
        """A shard whose metadata omits the options string does not say what it computed."""
        src = tmp_path / "d.npz"
        write_contract_a(src, _make_graphs(6, seed=3))
        out = tmp_path / "shard.npz"
        assert (
            main(
                [
                    "--input",
                    str(src),
                    "--out",
                    str(out),
                    "--backend",
                    "stub",
                    "--backend-factory",
                    STUB_FACTORY_REF,
                    "--workers",
                    "1",
                    "--role",
                    "ubs",
                    "--ub-method",
                    "BP_BEAM",
                    "--ub-options",
                    _DET_START,
                    "--log-level",
                    "WARNING",
                ]
            )
            == EXIT_OK
        )
        with np.load(out) as data:
            meta = json.loads(str(data["meta"]))
        assert meta["role"] == "ubs"
        assert meta["ub_method"] == "BP_BEAM"
        assert meta["ub_options"] == _DET_START
        assert meta["compute"] == "both"


class TestOneSidedOutcome:
    """A single-role campaign writes a one-sided bracket, and says so."""

    def test_a_lb_only_result_is_accepted_with_an_infinite_upper_end(self) -> None:
        outcome = _outcome_from_result(0, _OneSided(1.0, float("inf"), "lb"), 0.1)
        assert outcome.lb == 1.0 and outcome.ub == float("inf")
        assert outcome.ged == float("inf") and outcome.certified is False

    def test_a_ub_only_result_is_accepted_with_a_negative_infinite_lower_end(self) -> None:
        outcome = _outcome_from_result(0, _OneSided(float("-inf"), 4.0, "ub"), 0.1)
        assert outcome.lb == float("-inf") and outcome.ub == 4.0

    def test_the_unevaluated_end_must_be_exactly_the_sentinel(self) -> None:
        """A plausible number in an unevaluated slot is the failure to prevent."""
        with pytest.raises(RunnerError, match="compute='lb' must leave ub"):
            _outcome_from_result(0, _OneSided(1.0, 9.0, "lb"), 0.1)
        with pytest.raises(RunnerError, match="compute='ub' must leave lb"):
            _outcome_from_result(0, _OneSided(0.0, 9.0, "ub"), 0.1)

    def test_the_two_sided_path_still_rejects_a_non_finite_bracket(self) -> None:
        """T-03's invariant, unchanged on the default path."""
        with pytest.raises(RunnerError, match="non-finite upper bound"):
            _outcome_from_result(0, _OneSided(1.0, float("inf"), "both"), 0.1)


@dataclass(frozen=True, slots=True)
class _OneSided:
    """A CONTRACT B result carrying a compute mode."""

    lb: float
    ub: float
    computed: str
    exact: float | None = None
    certified: bool = False
    seconds: float = 0.5
    timed_out: bool = False
    method: str = "role"


class TestSubsamplePairList:
    """CONTRACTS §5: the sampler's pooled list, keyed by graph index."""

    def test_a_pooled_pair_i_pair_j_list_is_read_and_filtered(self, tmp_path: Path) -> None:
        src = tmp_path / "d.npz"
        write_contract_a(src, _make_graphs(8, seed=2), key="linux")
        dataset = load_contract_a(src)
        listed = tmp_path / "subsample_pairs.npz"
        np.savez_compressed(
            listed,
            dataset_key=np.array(["linux", "linux", "aids_iam"], dtype=str),
            pair_i=np.array([0, 2, 0], dtype=np.int32),
            pair_j=np.array([1, 5, 1], dtype=np.int32),
        )
        pairs, _rng = _target_pairs(dataset, pair_list=str(listed), chunk_index=0, n_chunks=1)
        assert pairs.tolist() == sorted([index_of_pair(0, 1, 8), index_of_pair(2, 5, 8)])

    def test_a_list_naming_no_row_for_this_dataset_is_refused(self, tmp_path: Path) -> None:
        src = tmp_path / "d.npz"
        write_contract_a(src, _make_graphs(8, seed=2), key="linux")
        dataset = load_contract_a(src)
        listed = tmp_path / "s.npz"
        np.savez_compressed(
            listed,
            dataset_key=np.array(["grec"], dtype=str),
            pair_i=np.array([0], dtype=np.int32),
            pair_j=np.array([1], dtype=np.int32),
        )
        with pytest.raises(RunnerError, match="no rows for dataset"):
            _target_pairs(dataset, pair_list=str(listed), chunk_index=0, n_chunks=1)

    def test_a_list_with_neither_schema_is_refused(self, tmp_path: Path) -> None:
        src = tmp_path / "d.npz"
        write_contract_a(src, _make_graphs(8, seed=2))
        dataset = load_contract_a(src)
        listed = tmp_path / "s.npz"
        np.savez_compressed(listed, something_else=np.array([1]))
        with pytest.raises(RunnerError, match="neither a 'pair_index'"):
            _target_pairs(dataset, pair_list=str(listed), chunk_index=0, n_chunks=1)


class TestOrientationRecording:
    """``--record-orientations`` is opt-in and changes nothing when it is off."""

    def _run(self, src: Path, out: Path, *extra: str) -> None:
        assert (
            main(
                [
                    "--input",
                    str(src),
                    "--out",
                    str(out),
                    "--backend",
                    "stub",
                    "--backend-factory",
                    STUB_FACTORY_REF,
                    "--workers",
                    "1",
                    "--log-level",
                    "WARNING",
                    *extra,
                ]
            )
            == EXIT_OK
        )

    def test_the_flag_off_produces_the_six_frozen_arrays_only(self, tmp_path: Path) -> None:
        """T-03's shard schema is untouched by anyone who does not ask for more."""
        src = tmp_path / "d.npz"
        write_contract_a(src, _make_graphs(6, seed=11))
        out = tmp_path / "s.npz"
        self._run(src, out)
        with np.load(out) as data:
            assert set(data.files) == {
                "pair_index",
                "ged",
                "lb",
                "ub",
                "certified",
                "seconds",
                "meta",
            }
            meta = json.loads(str(data["meta"]))
        assert "ub_orientation" not in meta

    def test_the_flag_off_is_byte_identical_to_a_run_without_the_flag_existing(
        self, tmp_path: Path
    ) -> None:
        """Every array and the whole meta, field for field, modulo timestamps.

        This is the regression guard for a closed ticket: a shard written today
        without the flag must be the shard T-03 would have written.
        """
        src = tmp_path / "d.npz"
        write_contract_a(src, _make_graphs(6, seed=12))
        a, b = tmp_path / "a.npz", tmp_path / "b.npz"
        self._run(src, a)
        self._run(src, b)
        with np.load(a) as da, np.load(b) as db:
            assert set(da.files) == set(db.files)
            for name in da.files:
                if name == "meta":
                    continue
                assert np.array_equal(da[name], db[name]), name
            ma = json.loads(str(da["meta"]))
            mb = json.loads(str(db["meta"]))
        volatile = {"started_utc", "ended_utc", "hostname", "cpu_model"}
        assert {k: v for k, v in ma.items() if k not in volatile} == {
            k: v for k, v in mb.items() if k not in volatile
        }

    def test_the_flag_on_adds_exactly_two_arrays(self, tmp_path: Path) -> None:
        src = tmp_path / "d.npz"
        write_contract_a(src, _make_graphs(6, seed=13))
        out = tmp_path / "s.npz"
        self._run(src, out, "--record-orientations")
        with np.load(out) as data:
            assert "ub_fwd" in data and "ub_rev" in data
            assert data["ub_fwd"].shape == data["ub"].shape
            assert data["ub_fwd"].dtype == np.float64

    def test_a_backend_reporting_orientations_reaches_the_shard(self, tmp_path: Path) -> None:
        """The stub reports none, so this uses one that does."""
        src = tmp_path / "d.npz"
        write_contract_a(src, _make_graphs(5, seed=14))
        dataset = load_contract_a(src)

        class _Oriented:
            name = "oriented"
            compute = "both"
            last_ub_orientations = (7.0, 5.0)

            def pair(self, g1: nx.Graph, g2: nx.Graph) -> _Result:
                return _Result(1.0, 5.0, None, False, 0.5, False, "oriented")

        shard, _completed, _stats = run_chunk(
            dataset=dataset,
            pairs=np.arange(3, dtype=np.int64),
            backend_spec=BackendSpec(name="stub"),
            input_path=str(src),
            workers=1,
            checkpoint_path=None,
            checkpoint_every=100,
            backend_factory=lambda _spec: _Oriented(),
        )
        arrays = shard.arrays(record_orientations=True)
        assert arrays["ub_fwd"].tolist() == [7.0, 7.0, 7.0]
        assert arrays["ub_rev"].tolist() == [5.0, 5.0, 5.0]
        stats = shard.orientation_stats()
        assert stats["n_pairs"] == 3
        assert stats["asymmetry_rate"] == 1.0
        assert stats["mean_abs_gap"] == 2.0
        assert stats["reverse_tighter_rate"] == 1.0

    def test_orientation_stats_are_empty_without_recorded_values(self, tmp_path: Path) -> None:
        src = tmp_path / "d.npz"
        write_contract_a(src, _make_graphs(5, seed=15))
        dataset = load_contract_a(src)
        shard, _c, _s = run_chunk(
            dataset=dataset,
            pairs=np.arange(3, dtype=np.int64),
            backend_spec=BackendSpec(name="stub", factory_ref=STUB_FACTORY_REF),
            input_path=str(src),
            workers=1,
            checkpoint_path=None,
            checkpoint_every=100,
        )
        assert shard.orientation_stats() == {}


class TestAccessorProbeAtCampaignInit:
    """A failed probe stops the campaign; it is not swallowed per pair."""

    def test_a_failing_probe_aborts_the_run(self, tmp_path: Path) -> None:
        src = tmp_path / "d.npz"
        write_contract_a(src, _make_graphs(5, seed=7))
        dataset = load_contract_a(src)

        class _Probed:
            name = "probed"

            def probe_accessors(self) -> dict[str, float]:
                raise ValueError("returned 0.00 on P4 vs C4")

            def pair(self, g1: nx.Graph, g2: nx.Graph) -> object:
                raise AssertionError("no pair may be computed after a failed probe")

        with pytest.raises(RunnerError, match="accessor probe failed"):
            run_chunk(
                dataset=dataset,
                pairs=np.arange(3, dtype=np.int64),
                backend_spec=BackendSpec(name="stub"),
                input_path=str(src),
                workers=1,
                checkpoint_path=None,
                checkpoint_every=100,
                backend_factory=lambda _spec: _Probed(),
            )

    def test_a_backend_without_a_probe_is_left_alone(self, tmp_path: Path) -> None:
        """The stub and networkx backends have no accessors to probe."""
        src = tmp_path / "d.npz"
        write_contract_a(src, _make_graphs(5, seed=7))
        dataset = load_contract_a(src)
        shard, completed, _stats = run_chunk(
            dataset=dataset,
            pairs=np.arange(3, dtype=np.int64),
            backend_spec=BackendSpec(name="stub", factory_ref=STUB_FACTORY_REF),
            input_path=str(src),
            workers=1,
            checkpoint_path=None,
            checkpoint_every=100,
        )
        assert completed and len(shard) == 3


# --------------------------------------------------------------------------- #
# T-05: one GEDLIB environment per dataset instead of one per pair
# --------------------------------------------------------------------------- #


class _CohortBackend:
    """A cohort-mode backend that answers only by graph index.

    Refuses ``pair()`` outright. That is the point: a runner that quietly kept
    passing graph objects would still produce a correct shard against a real
    GEDLIB backend, at the per-pair cost the whole change exists to remove, and
    no assertion on the values would notice.
    """

    env_mode = "cohort"
    name = "cohort_fake"

    def __init__(self) -> None:
        """Start with no cohort loaded and an empty call log."""
        self.graphs: list[nx.Graph] | None = None
        self.key: str | None = None
        self.n_loads = 0
        self.index_calls: list[tuple[int, int]] = []
        self.pid_at_load: int | None = None

    def load_cohort(self, graphs: list[nx.Graph], *, key: str = "") -> None:
        """Take the whole cohort, recording the process that did so."""
        import os

        self.graphs = list(graphs)
        self.key = key
        self.n_loads += 1
        self.pid_at_load = os.getpid()

    def pair(self, g1: nx.Graph, g2: nx.Graph) -> _Result:
        """Refuse: a cohort backend must never be addressed by graph object."""
        raise AssertionError("cohort mode must address the backend by graph index")

    def pair_by_index(self, i: int, j: int) -> _Result:
        """Return a bracket derived from the two cohort positions."""
        if self.graphs is None:
            raise AssertionError("pair_by_index before load_cohort")
        self.index_calls.append((i, j))
        val = float(abs(self.graphs[i].number_of_nodes() - self.graphs[j].number_of_nodes()) + 1)
        return _Result(val, val + 2.0, None, False, 0.25, False, "cohort_fake")


class TestCohortEnvMode:
    """``--env-mode cohort`` addresses the backend by graph index, once loaded."""

    def test_the_default_is_per_pair(self) -> None:
        """T-03's behaviour is untouched until the orchestrator flips it."""
        assert _parsed(_BASE_ARGV).env_mode == "per-pair"
        assert _backend_options(_parsed(_BASE_ARGV))["env_mode"] == "per-pair"

    def test_the_flag_reaches_the_backend(self) -> None:
        args = _parsed([*_BASE_ARGV, "--env-mode", "cohort"])
        assert _backend_options(args)["env_mode"] == "cohort"

    def test_an_unknown_env_mode_is_refused_by_the_parser(self) -> None:
        with pytest.raises(SystemExit):
            _parsed([*_BASE_ARGV, "--env-mode", "one-env"])

    def test_a_non_gedlib_backend_still_takes_no_options(self) -> None:
        args = _parsed(["--input", "i.npz", "--out", "o.npz", "--backend", "stub"])
        assert _backend_options(args) == {}

    def test_the_cohort_is_loaded_once_and_pairs_go_by_index(
        self, cohort: tuple[Path, ContractA]
    ) -> None:
        """One load for the chunk, and the indices are the ones the pair index names."""
        path, dataset = cohort
        backend = _CohortBackend()
        pairs = np.array([0, 1, 11, 20, 64, 65], dtype=np.int64)
        shard, completed, _stats = _run(dataset, path, pairs, backend)
        assert completed and len(shard) == 6
        assert backend.n_loads == 1
        assert backend.key == dataset.key
        assert backend.graphs is not None and len(backend.graphs) == dataset.n_graphs
        assert backend.index_calls == [pair_from_index(int(k), dataset.n_graphs) for k in pairs]

    def test_cohort_and_per_pair_produce_the_same_shard(
        self, cohort: tuple[Path, ContractA]
    ) -> None:
        """The acceptance criterion at the runner level, not just the backend's.

        Measured for real against GEDLIB on all 3,916 LINUX pairs: the ``lb``,
        ``ub`` and ``ubs`` roles each matched per-pair mode with max abs diff
        0.0 and an identical sha256, and each matched T-27's recorded census.
        """

        class _ByGraph(_CohortBackend):
            """Same numbers, reached through ``pair()`` on the per-pair path."""

            env_mode = "per-pair"

            def pair(self, g1: nx.Graph, g2: nx.Graph) -> _Result:
                val = float(abs(g1.number_of_nodes() - g2.number_of_nodes()) + 1)
                return _Result(val, val + 2.0, None, False, 0.25, False, "cohort_fake")

        path, dataset = cohort
        pairs = np.arange(0, 66, dtype=np.int64)
        by_index, _, _ = _run(dataset, path, pairs, _CohortBackend())
        by_graph, _, _ = _run(dataset, path, pairs, _ByGraph())
        for key, value in by_index.arrays().items():
            assert np.array_equal(value, by_graph.arrays()[key]), key

    def test_a_cohort_backend_that_cannot_serve_it_raises_rather_than_falls_back(
        self, cohort: tuple[Path, ContractA]
    ) -> None:
        """A silent fallback would run the campaign at 33x its budgeted cost."""

        class _Broken:
            env_mode = "cohort"
            name = "broken"

            def pair(self, g1: nx.Graph, g2: nx.Graph) -> _Result:
                raise AssertionError("must not be reached")

        path, dataset = cohort
        with pytest.raises(RunnerError, match="no load_cohort"):
            _run(dataset, path, np.arange(3, dtype=np.int64), _Broken())

    def test_a_failing_load_cohort_stops_the_run(self, cohort: tuple[Path, ContractA]) -> None:
        class _Failing(_CohortBackend):
            def load_cohort(self, graphs: list[nx.Graph], *, key: str = "") -> None:
                raise ValueError("GEDLIB rejected the cohort")

        path, dataset = cohort
        with pytest.raises(RunnerError, match="load_cohort failed"):
            _run(dataset, path, np.arange(3, dtype=np.int64), _Failing())

    def test_the_shard_records_which_call_sequence_produced_it(self, tmp_path: Path) -> None:
        src = tmp_path / "d.npz"
        write_contract_a(src, _make_graphs(6, seed=21))
        out = tmp_path / "s.npz"
        assert (
            main(
                [
                    "--input",
                    str(src),
                    "--out",
                    str(out),
                    "--backend",
                    "stub",
                    "--backend-factory",
                    STUB_FACTORY_REF,
                    "--workers",
                    "1",
                    "--env-mode",
                    "cohort",
                    "--log-level",
                    "WARNING",
                ]
            )
            == EXIT_OK
        )
        with np.load(out) as data:
            assert set(data.files) == {
                "pair_index",
                "ged",
                "lb",
                "ub",
                "certified",
                "seconds",
                "meta",
            }
            meta = json.loads(str(data["meta"]))
        assert meta["env_mode"] == "cohort"

    def test_the_reference_stub_is_untouched_by_the_flag(self, tmp_path: Path) -> None:
        """A backend that declares no env_mode keeps its per-pair call path.

        The stub, networkx and every injected double must be unaffected, so a
        shard written with the flag set is byte-identical to one written
        without it wherever the backend does not opt in.
        """
        src = tmp_path / "d.npz"
        write_contract_a(src, _make_graphs(7, seed=22))
        outs = []
        for name, extra in (("a", []), ("b", ["--env-mode", "cohort"])):
            out = tmp_path / f"{name}.npz"
            assert (
                main(
                    [
                        "--input",
                        str(src),
                        "--out",
                        str(out),
                        "--backend",
                        "stub",
                        "--backend-factory",
                        STUB_FACTORY_REF,
                        "--workers",
                        "1",
                        "--log-level",
                        "WARNING",
                        *extra,
                    ]
                )
                == EXIT_OK
            )
            outs.append(out)
        with np.load(outs[0]) as a, np.load(outs[1]) as b:
            for key in ("pair_index", "ged", "lb", "ub", "certified", "seconds"):
                assert np.array_equal(a[key], b[key]), key


class TestCohortIsBuiltPerProcess:
    """A GEDLIB environment must never be inherited across ``fork()``."""

    def test_the_parent_never_builds_a_backend_for_a_pool_run(
        self, cohort: tuple[Path, ContractA], monkeypatch: pytest.MonkeyPatch
    ) -> None:
        """Any parent-side construction would be inherited by every child.

        ``ProcessPoolExecutor`` forks on Linux, and GEDLIB state is not
        fork-safe once ``init()`` has run. The guarantee is structural: with
        ``workers > 1`` the parent must reach the pool without ever calling
        ``make_backend``, leaving construction entirely to ``_init_worker``,
        which runs inside the child.
        """
        import benchmarks.eval_setup.ged_exact_runner as mod

        path, dataset = cohort
        built: list[int] = []
        monkeypatch.setattr(mod, "make_backend", lambda spec: built.append(1) or _CohortBackend())
        pooled: list[object] = []
        monkeypatch.setattr(mod, "_run_pool", lambda **kw: pooled.append(kw) or True)

        run_chunk(
            dataset=dataset,
            pairs=np.arange(4, dtype=np.int64),
            backend_spec=BackendSpec(name="gedlib"),
            input_path=str(path),
            workers=4,
            checkpoint_path=None,
            checkpoint_every=100,
        )
        assert built == [], "the parent must not construct a backend before forking"
        assert len(pooled) == 1

    def test_each_worker_loads_its_own_cohort(
        self, cohort: tuple[Path, ContractA], monkeypatch: pytest.MonkeyPatch
    ) -> None:
        """``_init_worker`` runs in the child, so every process gets its own env."""
        import benchmarks.eval_setup.ged_exact_runner as mod

        path, dataset = cohort
        # _init_worker installs SIG_IGN for SIGTERM, which is right in a pool
        # child and wrong in the pytest process.
        monkeypatch.setattr(mod.signal, "signal", lambda *_a: None)
        monkeypatch.setattr(mod, "_WORKER_BY_INDEX", False, raising=False)

        mod._init_worker(
            mod._WorkerSpec(
                input_path=str(path),
                backend=BackendSpec(name="stub", factory_ref=STUB_FACTORY_REF),
            )
        )
        assert mod._WORKER_BY_INDEX is False, "a stub declares no env_mode"

        backend = _CohortBackend()
        monkeypatch.setattr(mod, "make_backend", lambda _spec: backend)
        mod._init_worker(mod._WorkerSpec(input_path=str(path), backend=BackendSpec(name="gedlib")))
        assert mod._WORKER_BY_INDEX is True
        assert backend.n_loads == 1
        assert backend.pid_at_load == os.getpid()
        assert backend.graphs is not None
        assert len(backend.graphs) == dataset.n_graphs
        monkeypatch.setattr(mod, "_WORKER_BY_INDEX", False, raising=False)
