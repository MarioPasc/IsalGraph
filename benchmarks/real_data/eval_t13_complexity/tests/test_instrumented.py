"""Parity and invariant tests for the T-13 instrumented encoder mirror.

The instrumented mirror only earns its counts if it reproduces the frozen
encoder exactly, so every test in this file either checks byte parity against
``isalgraph.core.*`` or checks one of the structural invariants CONTRACTS §4
requires. The large parity sweep (>= 50,000 ``(graph, start)`` pairs) is run
offline and reported in the work log; what runs here is a fast deterministic
subset of the same generator.
"""

from __future__ import annotations

import json
from pathlib import Path

import pytest

from benchmarks.real_data.eval_t13_complexity import counters as cli
from benchmarks.real_data.eval_t13_complexity.instrumented import (
    canonical_counts,
    canonical_detail,
    canonical_profile,
    greedy_counts,
    greedy_detail,
    greedy_min_counts,
    pair_generation_work,
    pruned_counts,
    pruned_detail,
)
from isalgraph.core.canonical import canonical_string
from isalgraph.core.canonical_pruned import pruned_canonical_string
from isalgraph.core.graph_to_string import GraphToString, generate_pairs_sorted_by_sum
from isalgraph.core.sparse_graph import SparseGraph

SEED = 13

# All densities, every order the sweep covers -- used for the greedy arm.
GREEDY_POOL: list[tuple[int, list[tuple[int, int]]]] = list(
    cli.random_connected_graphs(seed=SEED, sizes=tuple(range(2, 11)), per_size=12)
)

# The exhaustive canonical search is super-exponential in the branching factor,
# so its pool is capped in density; the cap is on the cohort, never on the
# encoder.
CANON_POOL: list[tuple[int, list[tuple[int, int]]]] = [
    *cli.random_connected_graphs(seed=SEED, sizes=(2, 3, 4, 5, 6), per_size=10),
    *cli.random_connected_graphs(
        seed=SEED + 1, sizes=(7, 8, 9), per_size=6, p_min=0.1, p_max=0.5, max_edges=11
    ),
]


def _sg(entry: tuple[int, list[tuple[int, int]]]) -> SparseGraph:
    """Materialise a pool entry as a :class:`SparseGraph`."""
    n, edges = entry
    return cli.to_sparse(n, edges)


def _max_degree(graph: SparseGraph) -> int:
    """Largest node degree in *graph*."""
    return max(len(graph.neighbors(v)) for v in range(graph.node_count()))


# ----------------------------------------------------------------------
# Parity -- the deliverable
# ----------------------------------------------------------------------


def test_greedy_parity_and_invariants() -> None:
    """Greedy mirror is byte-identical to ``GraphToString.run`` from every start."""
    pairs = 0
    for entry in GREEDY_POOL:
        graph = _sg(entry)
        m = graph.logical_edge_count()
        for start in range(graph.node_count()):
            reference = GraphToString(graph).run(start)[0]
            string, counts = greedy_counts(graph, start)
            pairs += 1
            assert string == reference
            assert counts.frames == m
            assert counts.pair_trials >= counts.frames
            assert counts.scan_depth_total == counts.pair_trials
            assert counts.pointer_steps >= 0
            assert counts.backtrack_nodes == 0
            assert counts.search_leaves == 0
            assert counts.string_length == len(string)
    assert pairs > 400


def test_canonical_parity_and_invariants() -> None:
    """Exhaustive canonical mirror is byte-identical to ``canonical_string``."""
    for entry in CANON_POOL:
        graph = _sg(entry)
        string, counts, _ = canonical_detail(graph)
        assert string == canonical_string(graph)
        assert counts.backtrack_nodes >= counts.search_leaves >= 1
        assert counts.frames == counts.backtrack_nodes - counts.search_leaves
        assert counts.pair_trials >= counts.frames
        assert counts.pointer_steps >= 0
        assert counts.string_length == len(string)


def test_pruned_parity_and_invariants() -> None:
    """Pruned canonical mirror is byte-identical to ``pruned_canonical_string``."""
    for entry in CANON_POOL:
        graph = _sg(entry)
        string, counts, _ = pruned_detail(graph)
        assert string == pruned_canonical_string(graph)
        assert counts.backtrack_nodes >= counts.search_leaves >= 1
        assert counts.pair_trials >= counts.frames
        assert counts.string_length == len(string)


def test_greedy_min_parity() -> None:
    """``greedy_min_counts`` reproduces the pure-Python ``GreedyMinG2S``."""
    for entry in GREEDY_POOL[:60]:
        graph = _sg(entry)
        string, counts = greedy_min_counts(graph)
        assert string == cli._greedy_min_reference(graph)
        # One greedy encode per start node, each with exactly m frames.
        assert counts.frames == graph.node_count() * graph.logical_edge_count()


def test_directed_parity_exercises_the_c_branch() -> None:
    """A directed cycle plus a chord exercises the ``c`` guard and its counting."""
    graph = SparseGraph(max_nodes=5, directed_graph=True)
    for _ in range(5):
        graph.add_node()
    for u in range(5):
        graph.add_edge(u, (u + 1) % 5)
    graph.add_edge(0, 2)

    string, counts = greedy_counts(graph, 0)
    assert string == GraphToString(graph).run(0)[0]
    assert counts.frames == graph.logical_edge_count()


# ----------------------------------------------------------------------
# Derivation checks (T-13 design note, section 2.1)
# ----------------------------------------------------------------------


def test_frames_equal_edge_count() -> None:
    """A greedy encode has exactly ``m`` frames, whatever the start node."""
    for entry in GREEDY_POOL:
        graph = _sg(entry)
        m = graph.logical_edge_count()
        for start in range(graph.node_count()):
            _, _, frames = greedy_detail(graph, start)
            assert len(frames) == m
            inserts = sum(1 for f in frames if f.opcode in ("V", "v"))
            assert inserts == graph.node_count() - 1


def test_string_length_identity() -> None:
    """``|w| == frames + sum of the displacement each frame actually emits``.

    Note that a ``V`` frame emits only its primary displacement and a ``v``
    frame only its secondary one; summing ``|a| + |b|`` over all frames
    over-counts. See the work log's "Defects found in the brief".
    """
    for entry in GREEDY_POOL:
        graph = _sg(entry)
        for start in range(graph.node_count()):
            string, counts, frames = greedy_detail(graph, start)
            emitted = sum(f.disp_emitted for f in frames)
            assert counts.string_length == counts.frames + emitted
            assert len(string) == counts.frames + emitted


def test_pairs_generated_matches_the_closed_form() -> None:
    """Every frame generates exactly ``(2M + 1)**2`` pairs, with ``M <= n``."""
    for entry in GREEDY_POOL[:40]:
        graph = _sg(entry)
        n = graph.node_count()
        for start in range(n):
            _, _, frames = greedy_detail(graph, start)
            for f in frames:
                assert f.pair_scope <= n
                assert f.pairs_generated == (2 * f.pair_scope + 1) ** 2
                assert len(generate_pairs_sorted_by_sum(f.pair_scope)) == f.pairs_generated


@pytest.mark.parametrize("m", [1, 2, 3, 5, 8, 12])
def test_pair_generation_work_is_p_log_p(m: int) -> None:
    """Sorting the displacement list costs ``Theta(P log P)`` comparisons."""
    n_pairs, comparisons, analytic = pair_generation_work(m)
    assert n_pairs == (2 * m + 1) ** 2
    assert 0.1 * analytic <= comparisons <= 1.0 * analytic


def test_scan_depth_is_far_below_the_worst_case() -> None:
    """Realised scan depth ``D_f`` is a small fraction of ``(2M + 1)**2``."""
    depths: list[float] = []
    for entry in GREEDY_POOL:
        graph = _sg(entry)
        for start in range(graph.node_count()):
            _, _, frames = greedy_detail(graph, start)
            for f in frames:
                assert f.pair_trials <= f.pairs_generated
                depths.append(f.pair_trials / f.pairs_generated)
    assert sum(depths) / len(depths) < 0.10


def test_pruning_never_increases_the_search_tree() -> None:
    """The triplet key can only remove leaves, never add them."""
    strictly_smaller = 0
    for entry in CANON_POOL:
        graph = _sg(entry)
        _, exhaustive, _ = canonical_detail(graph)
        _, pruned, pruned_frames = pruned_detail(graph)
        assert pruned.search_leaves <= exhaustive.search_leaves
        assert pruned.backtrack_nodes <= exhaustive.backtrack_nodes
        prunes = any(f.branch_factor < f.n_cands for f in pruned_frames)
        if prunes:
            assert pruned.search_leaves < exhaustive.search_leaves
            strictly_smaller += 1
    assert strictly_smaller > 0


def test_counting_and_collecting_paths_agree() -> None:
    """The memory-safe ``*_counts`` path produces the same counts as ``*_detail``.

    ``*_counts`` keeps running scalars and retains nothing per frame; ``*_detail``
    materialises a :class:`FrameRecord` per frame. An earlier version always
    collected and was OOM-killed part-way through the parity sweep, so the two
    paths must be kept in step by a test rather than by inspection.
    """
    for entry in CANON_POOL[:40]:
        graph = _sg(entry)
        for detail, counts_fn in (
            (canonical_detail(graph), canonical_counts),
            (pruned_detail(graph), pruned_counts),
        ):
            string_d, counts_d, frames = detail
            string_c, counts_c = counts_fn(graph)
            assert string_c == string_d
            assert counts_c == counts_d
            assert counts_c.frames == len(frames)
            assert counts_c.scan_depth_max == max(f.pair_trials for f in frames)


def test_search_profile_detects_pruning_without_retaining_frames() -> None:
    """``SearchProfile.prunes`` agrees with the frame-level predicate."""
    unpruned_ever_prunes = False
    agreements = 0
    for entry in CANON_POOL:
        graph = _sg(entry)
        _, _, exhaustive_profile = canonical_profile(graph, pruned=False)
        _, _, pruned_profile = canonical_profile(graph, pruned=True)
        _, _, pruned_frames = pruned_detail(graph)

        unpruned_ever_prunes |= exhaustive_profile.prunes
        assert pruned_profile.prunes == any(f.branch_factor < f.n_cands for f in pruned_frames)
        agreements += 1
    # The unpruned arm expands every candidate it sees, by definition.
    assert not unpruned_ever_prunes
    assert agreements == len(CANON_POOL)


def test_search_leaves_respect_the_delta_bound() -> None:
    """``search_leaves <= n * Delta**(n-1)``, the bound of T-13 section 2.1."""
    for entry in CANON_POOL:
        graph = _sg(entry)
        n = graph.node_count()
        delta = _max_degree(graph)
        _, counts, _ = canonical_detail(graph)
        assert counts.search_leaves <= n * delta ** (n - 1)


def test_neighbour_checks_show_the_short_circuit_gap() -> None:
    """Greedy short-circuits its neighbour scan; the canonical arms do not.

    On a star encoded from its hub, the opening ``V`` frame is reached at pair
    ``(0, 0)`` with every leaf still uninserted. ``_find_new_neighbor`` returns
    on the first leaf and charges one check; the canonical comprehension scans
    the whole adjacency set and charges ``Delta``. That is the ``O(deg)``
    versus ``O(Delta * D_f)`` split of T-13 section 2.1, isolated to a single
    frame -- an aggregate ratio over a whole encode does **not** show it,
    because deeper frames run with ``nleft == 0`` and skip the scan entirely.
    """
    degree = 6
    graph = SparseGraph(max_nodes=degree + 1, directed_graph=False)
    for _ in range(degree + 1):
        graph.add_node()
    for leaf in range(1, degree + 1):
        graph.add_edge(0, leaf)

    _, _, greedy_frames = greedy_detail(graph, 0)
    assert greedy_frames[0].opcode == "V"
    assert greedy_frames[0].pair_trials == 1
    assert greedy_frames[0].neighbour_checks == 1

    _, _, canon_frames = canonical_detail(graph)
    root = next(f for f in canon_frames if f.start_node == 0 and f.depth == 0)
    assert root.opcode == "V"
    assert root.pair_trials == 1
    assert root.neighbour_checks == degree
    assert root.n_cands == degree


# ----------------------------------------------------------------------
# CLI
# ----------------------------------------------------------------------


def test_cli_self_test_emits_the_frozen_schema(tmp_path: Path) -> None:
    """``counters.py --self-test`` writes schema ``t13c.1`` with parity everywhere."""
    out = tmp_path / "counts.jsonl"
    status = cli.main(["--self-test", "2", "--out", str(out), "--seed", "13"])
    assert status == 0

    expected = {
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
        "frames",
        "pair_trials",
        "scan_depth_total",
        "scan_depth_max",
        "pointer_steps",
        "neighbour_checks",
        "backtrack_nodes",
        "search_leaves",
        "string_length",
        "parity_ok",
    }
    rows = [json.loads(line) for line in out.read_text(encoding="utf-8").splitlines()]
    assert rows
    seen: set[str] = set()
    for row in rows:
        assert set(row) == expected
        assert row["schema_version"] == "t13c.1"
        assert row["parity_ok"] is True
        assert row["encoder"] in cli.ENCODERS
        seen.add(row["encoder"])
    assert seen == set(cli.DEFAULT_ENCODERS)


def test_encoder_semantics_travel_in_the_data(tmp_path: Path) -> None:
    """``greedy_single`` and ``greedy_min`` are distinct schema values.

    The frame accounting differs between them, so the distinction must be
    readable from the row rather than from how the CLI happened to be invoked.
    """
    spec = tmp_path / "spec.jsonl"
    spec.write_text(
        json.dumps({"n": 5, "edges": [[0, 1], [1, 2], [2, 3], [3, 4], [4, 0]]}) + "\n",
        encoding="utf-8",
    )
    out = tmp_path / "counts.jsonl"
    assert cli.main(["--spec-file", str(spec), "--out", str(out), "--greedy-mode", "both"]) == 0

    rows = {
        row["encoder"]: row
        for row in (json.loads(line) for line in out.read_text(encoding="utf-8").splitlines())
    }
    assert set(rows) == {"greedy_single", "greedy_min", "canonical", "pruned"}
    assert all(row["parity_ok"] for row in rows.values())
    assert rows["greedy_single"]["frames"] == rows["greedy_single"]["m"] == 5
    assert rows["greedy_min"]["frames"] == rows["greedy_min"]["n"] * rows["greedy_min"]["m"] == 25


def test_cli_spec_file_round_trip(tmp_path: Path) -> None:
    """A hand-written spec file is honoured, provenance fields included."""
    spec = tmp_path / "spec.jsonl"
    spec.write_text(
        json.dumps(
            {
                "source": "constructed",
                "family": "path",
                "n_target": 5,
                "replicate": 0,
                "dataset": None,
                "graph_index": 7,
                "n": 5,
                "edges": [[0, 1], [1, 2], [2, 3], [3, 4]],
            }
        )
        + "\n",
        encoding="utf-8",
    )
    out = tmp_path / "counts.jsonl"
    assert cli.main(["--spec-file", str(spec), "--out", str(out)]) == 0

    rows = [json.loads(line) for line in out.read_text(encoding="utf-8").splitlines()]
    assert len(rows) == 3
    for row in rows:
        assert row["family"] == "path"
        assert row["graph_index"] == 7
        assert row["n"] == 5
        assert row["m"] == 4
        assert row["parity_ok"] is True
