"""Validation oracles for the minimum DFS code, and the third-party gate.

**The oracles are the deliverable; the backend is 200 lines around them.**
Without exhaustive validation the backend is unverified graph theory, and
the ``kaviniitm/DFSCode`` episode is the proof: that implementation agreed
with ours on the running example and on every path and cycle, and is wrong
on half of all 6-node graphs.  A single-example check would have adopted it.

Three checks, ported from ``scratch/validate_min_dfs.py``:

``V1``
    The returned code equals the lexicographic minimum over **every valid
    DFS traversal**, enumerated exhaustively.  The comparison uses the
    *general* DFS lexicographic order of Yan & Han Def. 5, written out here
    independently of ``min_dfs.extension_key`` -- an oracle that shares the
    implementation's ordering function proves nothing.
``V2``
    Isomorphism invariance under relabelling.
``V3``
    Complete invariant: two graphs share a code iff they are isomorphic.

Plus the ``kaviniitm`` gate: the acceptance test **any** third-party
minimum-DFS implementation proposed later must pass, **K2 (isomorphism
invariance) before anything else**.  K2 needs no oracle and it is where that
implementation died -- 46 of 90 graphs non-invariant.
"""

from __future__ import annotations

import functools
import itertools
import math
import os
import random
import subprocess
from collections.abc import Callable, Iterator, Sequence

import pytest

pytestmark = pytest.mark.unit

nx = pytest.importorskip("networkx")

from isalgraph.competitors import fixtures  # noqa: E402
from isalgraph.competitors.backends.min_dfs import (  # noqa: E402
    MAX_PROJECTIONS,
    DfsEdge,
    MinDfsBackend,
    code_symbols,
    min_dfs_code,
    render,
    rightmost_path,
)
from isalgraph.competitors.base import Budget, Capability  # noqa: E402
from isalgraph.competitors.registry import get_metric, get_repr_backend  # noqa: E402
from isalgraph.errors import MinDfsBudgetExceeded  # noqa: E402

#: A code reduced to its index pairs, which is what the corpus carries.
Pairs = tuple[tuple[int, int], ...]

#: A candidate implementation under test by the third-party gate.
Candidate = Callable[["nx.Graph"], Pairs]


# ---------------------------------------------------------------------------
# The oracle: the general DFS lexicographic order, written independently
# ---------------------------------------------------------------------------


def edge_lt(e1: DfsEdge, e2: DfsEdge) -> bool:
    """Yan & Han Def. 5: is DFS edge *e1* strictly before *e2*?

    Written out in full rather than delegating to
    :func:`~isalgraph.competitors.backends.min_dfs.extension_key`.  The
    implementation's key is specialised to one extension step; this is the
    general order over arbitrary pairs, and keeping them separate is what
    makes ``V1`` an independent check.

    Args:
        e1: a DFS edge.
        e2: a DFS edge.

    Returns:
        ``True`` if *e1* precedes *e2*.
    """
    i1, j1 = e1[0], e1[1]
    i2, j2 = e2[0], e2[1]
    f1, f2 = i1 < j1, i2 < j2
    if f1 and f2:  # both forward
        if j1 != j2:
            return j1 < j2
        if i1 != i2:
            return i1 > i2
        return (e1[2], e1[3], e1[4]) < (e2[2], e2[3], e2[4])
    if not f1 and not f2:  # both backward
        if i1 != i2:
            return i1 < i2
        if j1 != j2:
            return j1 < j2
        return e1[3] < e2[3]
    if not f1 and f2:  # backward vs forward
        return i1 < j2
    return j1 <= i2  # forward vs backward


def code_lt(c1: Sequence[DfsEdge], c2: Sequence[DfsEdge]) -> bool:
    """Lexicographic order on whole DFS codes.

    Args:
        c1: a DFS code.
        c2: a DFS code.

    Returns:
        ``True`` if *c1* precedes *c2*.
    """
    for a, b in zip(c1, c2, strict=False):
        if a != b:
            return edge_lt(a, b)
    return len(c1) < len(c2)


def all_dfs_codes(graph: nx.Graph) -> list[list[DfsEdge]]:
    """Every valid DFS code of a connected unlabelled graph.

    Exponential; only ever called for ``n <= 6``.

    Args:
        graph: a connected graph with at least one edge.

    Returns:
        Every complete DFS code, one list per traversal.
    """
    m = graph.number_of_edges()
    out: list[list[DfsEdge]] = []

    def rec(
        code: list[DfsEdge],
        v_of: list[int],
        g_of: dict[int, int],
        used: frozenset[frozenset[int]],
    ) -> None:
        if len(code) == m:
            out.append(list(code))
            return
        rmp = rightmost_path(code)
        rm_idx = rmp[-1]
        rm_v = v_of[rm_idx]
        for anc in rmp[:-1]:
            anc_v = v_of[anc]
            edge = frozenset((rm_v, anc_v))
            if graph.has_edge(rm_v, anc_v) and edge not in used:
                rec([*code, (rm_idx, anc, 0, 0, 0)], v_of, g_of, used | {edge})
        new_idx = len(v_of)
        for src in rmp:
            src_v = v_of[src]
            for w in graph.neighbors(src_v):
                if w in g_of:
                    continue
                edge = frozenset((src_v, w))
                if edge in used:
                    continue
                rec(
                    [*code, (src, new_idx, 0, 0, 0)],
                    [*v_of, w],
                    {**g_of, w: new_idx},
                    used | {edge},
                )

    for u, v in graph.edges():
        for a, b in ((u, v), (v, u)):
            rec([(0, 1, 0, 0, 0)], [a, b], {a: 0, b: 1}, frozenset({frozenset((a, b))}))
    return out


def brute_force_min(graph: nx.Graph) -> list[DfsEdge]:
    """The lexicographic minimum over every valid DFS traversal.

    Args:
        graph: a connected graph with at least one edge.

    Returns:
        The minimum DFS code, computed by exhaustive enumeration.
    """
    return min(
        all_dfs_codes(graph),
        key=functools.cmp_to_key(lambda a, b: -1 if code_lt(a, b) else (1 if code_lt(b, a) else 0)),
    )


@functools.cache
def connected_classes(n: int) -> tuple[nx.Graph, ...]:
    """Every connected graph on ``n`` nodes, one per isomorphism class.

    Sourced from ``networkx``'s graph atlas (Read & Wilson, *An Atlas of
    Graphs*, Oxford, 1998), which is an **independent** enumeration: it does
    not use any code under test, so counting distinct min-DFS codes against
    it is a real completeness check rather than a tautology.

    Args:
        n: node count, at most 7.

    Returns:
        One representative per connected isomorphism class.
    """
    if n > 7:
        raise ValueError("the graph atlas stops at n = 7")
    return tuple(g for g in nx.graph_atlas_g() if g.number_of_nodes() == n and nx.is_connected(g))


def as_pairs(code: Sequence[DfsEdge]) -> Pairs:
    """Reduce a DFS code to its index pairs."""
    return tuple((i, j) for i, j, *_ in code)


def ours(graph: nx.Graph) -> Pairs:
    """Our minimum DFS code as index pairs, unbounded."""
    return as_pairs(min_dfs_code(graph, max_projections=None))


# ---------------------------------------------------------------------------
# Criterion 1 -- the running example reproduces exactly
# ---------------------------------------------------------------------------


def test_running_example_reproduces_exactly() -> None:
    """``C4(0,1,2,3) + K3(3,4,5)`` gives the published code, ``m`` tuples long."""
    graph = fixtures.to_networkx(fixtures.RUNNING_EXAMPLE)
    code = min_dfs_code(graph)
    assert render(code) == "0-1 1-2 2-0 2-3 3-4 4-5 5-2"
    assert as_pairs(code) == ((0, 1), (1, 2), (2, 0), (2, 3), (3, 4), (4, 5), (5, 2))
    assert len(code) == graph.number_of_edges() == 7


def test_running_example_minus_edge() -> None:
    """``H = G - (0,3)`` drops exactly one tuple, because ``|code| = m``."""
    graph = fixtures.to_networkx(fixtures.RUNNING_EXAMPLE_MINUS_EDGE)
    code = min_dfs_code(graph)
    assert render(code) == "0-1 1-2 2-0 2-3 3-4 4-5"
    assert len(code) == 6


def test_two_hundred_relabellings_give_one_code() -> None:
    """Criterion 1's invariance half: 200 relabellings, one distinct code."""
    rng = random.Random(7)
    codes = {
        render(min_dfs_code(fixtures.relabelled(fixtures.RUNNING_EXAMPLE, rng))) for _ in range(200)
    }
    assert codes == {"0-1 1-2 2-0 2-3 3-4 4-5 5-2"}


def test_code_length_is_exactly_m() -> None:
    """``|code| = m`` on every connected fixture -- the deterministic length."""
    for name in fixtures.CONNECTED_FIXTURES:
        graph = fixtures.to_networkx(fixtures.ALL_FIXTURES[name])
        assert len(min_dfs_code(graph)) == graph.number_of_edges(), name


# ---------------------------------------------------------------------------
# V1 -- exhaustive brute force
# ---------------------------------------------------------------------------


@pytest.mark.slow
def test_v1_exhaustive_brute_force() -> None:
    """V1: agrees with the minimum over every valid DFS traversal, ``n <= 5``.

    All **30** connected isomorphism classes: 1, 2, 6, 21 at ``n = 2..5``.
    """
    checked = 0
    per_n = {}
    for n in (2, 3, 4, 5):
        graphs = connected_classes(n)
        per_n[n] = len(graphs)
        for graph in graphs:
            assert min_dfs_code(graph, max_projections=None) == brute_force_min(graph), (
                f"n={n} edges={sorted(graph.edges())}"
            )
            checked += 1
    assert per_n == {2: 1, 3: 2, 4: 6, 5: 21}
    assert checked == 30


@pytest.mark.slow
def test_v1_holds_with_the_frozen_budget_in_place() -> None:
    """The cap must not change the answer where it does not fire.

    Design note §9 condition 6: the 50,000 projection budget sits behind a
    published failure rate, so the validation suite is re-run *after* it is
    installed rather than before.
    """
    for n in (2, 3, 4, 5):
        for graph in connected_classes(n):
            assert min_dfs_code(graph, max_projections=MAX_PROJECTIONS) == brute_force_min(graph)


# ---------------------------------------------------------------------------
# V3 -- complete invariant
# ---------------------------------------------------------------------------


@pytest.mark.slow
def test_v3_complete_invariant_no_collisions() -> None:
    """V3: distinct codes 1 / 2 / 6 / 21 / 112 at ``n = 2..6`` (OEIS A001349).

    One code per connected isomorphism class and no collisions, so the
    minimum DFS code is a complete invariant on this window.
    """
    counts: dict[int, int] = {}
    for n in (2, 3, 4, 5, 6):
        graphs = connected_classes(n)
        seen: dict[Pairs, nx.Graph] = {}
        for graph in graphs:
            key = ours(graph)
            if key in seen:
                assert nx.is_isomorphic(graph, seen[key]), (
                    f"V3 collision at n={n}: {sorted(graph.edges())} vs {sorted(seen[key].edges())}"
                )
            seen[key] = graph
        assert len(seen) == len(graphs), f"n={n}: {len(seen)} codes for {len(graphs)} classes"
        counts[n] = len(seen)
    assert counts == {2: 1, 3: 2, 4: 6, 5: 21, 6: 112}


@pytest.mark.slow
def test_reversibility_on_every_class_to_n_6() -> None:
    """``code -> graph`` is isomorphic to the original in every case."""
    backend = MinDfsBackend()
    for n in (2, 3, 4, 5, 6):
        for graph in connected_classes(n):
            code = min_dfs_code(graph, max_projections=None)
            assert nx.is_isomorphic(backend.decode(backend.encode(graph)), graph), sorted(
                graph.edges()
            )
            assert len(code) == graph.number_of_edges()


# ---------------------------------------------------------------------------
# V2 -- isomorphism invariance
# ---------------------------------------------------------------------------


@pytest.mark.slow
def test_v2_isomorphism_invariance_scratch_protocol() -> None:
    """V2: **4,440** relabellings, 0 mismatches, seed 42.

    The protocol is ``scratch/validate_min_dfs.py``'s verbatim -- ``n`` in
    ``5..10``, 40 ``gnp_random_graph(n, 0.35)`` draws each, 30 relabellings
    per connected draw -- because the count 4,440 is a function of the
    ``Random(42)`` draw *stream*, not just of the seed.

    Note that the design note and the evidence file both write the range as
    ``6 <= n <= 10``; the script that produced 4,440 loops over
    ``(5, 6, 7, 8, 9, 10)``.  Reproducing the number requires the script's
    range, so that is what is used here.
    """
    rng = random.Random(42)
    trials = 0
    mismatches = 0
    for n in (5, 6, 7, 8, 9, 10):
        for _ in range(40):
            graph = nx.gnp_random_graph(n, 0.35, seed=rng.randrange(10**9))
            if not nx.is_connected(graph) or graph.number_of_edges() == 0:
                continue
            base = min_dfs_code(graph, max_projections=None)
            for _ in range(30):
                perm = list(graph.nodes())
                rng.shuffle(perm)
                relabelled = nx.relabel_nodes(
                    graph, dict(zip(graph.nodes(), perm, strict=True)), copy=True
                )
                trials += 1
                if min_dfs_code(relabelled, max_projections=None) != base:
                    mismatches += 1
    assert (trials, mismatches) == (4440, 0)


@pytest.mark.slow
def test_v2_invariance_under_a_fresh_insertion_order() -> None:
    """The stronger relabeller: ``fixtures.shuffled_copy``, not ``relabel_nodes``.

    ``nx.relabel_nodes(copy=True)`` **preserves insertion order** (finding
    13), so a harness built on it makes order-dependent formats look
    invariant.  min-DFS is order-independent by construction and passes
    either way, but a harness that cannot fail is worthless, so the
    invariance claim is also checked against the relabeller that can break
    graph6.
    """
    rng = random.Random(42)
    trials = 0
    for n in (6, 7, 8, 9, 10):
        for _ in range(10):
            graph = nx.gnp_random_graph(n, 0.35, seed=rng.randrange(10**9))
            if not nx.is_connected(graph) or graph.number_of_edges() == 0:
                continue
            base = ours(graph)
            for _ in range(20):
                assert ours(fixtures.shuffled_copy(graph, rng)) == base
                trials += 1
    assert trials >= 500


# ---------------------------------------------------------------------------
# The kaviniitm gate -- reusable, K2 first
# ---------------------------------------------------------------------------


def gate_k2_isomorphism_invariance(
    candidate: Candidate,
    *,
    seed: int = 42,
    n_values: Sequence[int] = (6, 7, 8, 9, 10, 12),
    trials_per_n: int = 15,
    copies: int = 12,
) -> tuple[int, int]:
    """**K2, and it runs first.**  Is *candidate* isomorphism-invariant?

    K2 makes no reference to any oracle: a canonical form that returns
    different codes for different labellings of the same graph is not a
    canonical form, and nothing downstream -- no distance, no F3 row, no
    comparison -- can be built on it.  ``kaviniitm/DFSCode`` failed here on
    46 of 90 graphs, one 6-node graph producing 6 distinct codes from 13
    relabellings.

    Args:
        candidate: the implementation under test.
        seed: rng seed.
        n_values: node counts to sample.
        trials_per_n: graphs per node count.
        copies: relabellings per graph, plus the original.

    Returns:
        ``(graphs_tested, graphs_not_invariant)``.
    """
    rng = random.Random(seed)
    tested = 0
    bad = 0
    for n in n_values:
        for _ in range(trials_per_n):
            graph = _random_connected(n, rng.randint(n, min(2 * n, n * (n - 1) // 2)), rng)
            variants = [graph] + [fixtures.shuffled_copy(graph, rng) for _ in range(copies)]
            tested += 1
            if len({candidate(g) for g in variants}) != 1:
                bad += 1
    return tested, bad


def gate_k1_agreement(
    candidate: Candidate, *, n_values: Sequence[int] = (2, 3, 4, 5, 6)
) -> dict[int, tuple[int, int]]:
    """K1: does *candidate* agree with the validated oracle on every class?

    Runs **after** K2.  Args and returns as named.

    Args:
        candidate: the implementation under test.
        n_values: node counts.

    Returns:
        ``{n: (classes, mismatches)}``.
    """
    out: dict[int, tuple[int, int]] = {}
    for n in n_values:
        graphs = connected_classes(n)
        bad = sum(1 for g in graphs if candidate(g) != ours(g))
        out[n] = (len(graphs), bad)
    return out


def _random_connected(n: int, m: int, rng: random.Random) -> nx.Graph:
    """A connected graph on ``n`` nodes with ``m`` edges: spanning tree plus extras."""
    nodes = list(range(n))
    rng.shuffle(nodes)
    graph = nx.Graph()
    graph.add_nodes_from(range(n))
    for k in range(1, n):
        graph.add_edge(nodes[k], nodes[rng.randrange(k)])
    pool = [(u, v) for u, v in itertools.combinations(range(n), 2) if not graph.has_edge(u, v)]
    rng.shuffle(pool)
    for u, v in pool[: max(0, m - (n - 1))]:
        graph.add_edge(u, v)
    return graph


def greedy_no_branch_code(graph: nx.Graph) -> Pairs:
    """A deliberately wrong candidate: greedy extension with **no tie branching**.

    This is what ``LasseRegin/gSpan``'s ``is_canonical`` does internally, and
    the structural reason ``kaviniitm/DFSCode`` fails: for an unlabelled
    graph every label is equal, so *every* step is a tie and a construction
    that keeps one embedding is not guaranteed to reach the minimum.

    It exists so the gate is shown capable of failing.  A gate that cannot
    fail is worthless -- and the reusable lesson from ``kaviniitm`` is
    exactly that it passed every check anyone bothered to run.

    Args:
        graph: a connected graph with at least one edge.

    Returns:
        A valid but not necessarily minimal DFS code, as index pairs.
    """
    from isalgraph.competitors.backends.min_dfs import extension_key

    first = next(iter(graph.edges()))
    v_of = [first[0], first[1]]
    g_of = {first[0]: 0, first[1]: 1}
    used = {frozenset(first)}
    code: list[DfsEdge] = [(0, 1, 0, 0, 0)]
    while len(code) < graph.number_of_edges():
        rmp = rightmost_path(code)
        rm_v = v_of[rmp[-1]]
        options: list[tuple[DfsEdge, int | None]] = []
        for anc in rmp[:-1]:
            edge = frozenset((rm_v, v_of[anc]))
            if graph.has_edge(rm_v, v_of[anc]) and edge not in used:
                options.append(((rmp[-1], anc, 0, 0, 0), None))
        for src in rmp:
            for w in graph.neighbors(v_of[src]):
                if w in g_of or frozenset((v_of[src], w)) in used:
                    continue
                options.append(((src, len(v_of), 0, 0, 0), w))
        tup, new_vertex = min(options, key=lambda pair: extension_key(pair[0]))
        code.append(tup)
        if new_vertex is None:
            used.add(frozenset((rm_v, v_of[tup[1]])))
        else:
            used.add(frozenset((v_of[tup[0]], new_vertex)))
            g_of[new_vertex] = len(v_of)
            v_of.append(new_vertex)
    return as_pairs(code)


#: The archived ``kaviniitm/DFSCode`` verdict, ``scratch/test_kavin.out``.
#: The binary itself is **not** in this repository and is not vendored, so
#: the counts are data.  Every worked counterexample below is re-verified
#: against the oracle, which is the part that carries the argument.
KAVIN_VERDICT = {
    "k1_mismatches": {2: (1, 0), 3: (2, 0), 4: (6, 1), 5: (21, 7), 6: (112, 56)},
    "k2_graphs": 90,
    "k2_not_invariant": 46,
}

#: ``(edges, tool_code, minimum_code)`` -- every K1 counterexample the
#: archived run printed.  The first is the smallest failure that exists:
#: ``K4`` minus edge ``(2,3)``.
KAVIN_COUNTEREXAMPLES: tuple[tuple[Pairs, Pairs, Pairs], ...] = (
    (
        ((0, 1), (0, 2), (0, 3), (1, 2), (1, 3)),
        ((0, 1), (1, 2), (2, 0), (2, 3), (3, 1)),
        ((0, 1), (1, 2), (2, 0), (2, 3), (3, 0)),
    ),
    (
        ((0, 1), (0, 2), (0, 4), (1, 2), (1, 3)),
        ((0, 1), (1, 2), (2, 3), (3, 1), (3, 4)),
        ((0, 1), (1, 2), (2, 0), (2, 3), (1, 4)),
    ),
    (
        ((0, 1), (0, 2), (0, 3), (0, 4), (1, 2), (1, 3)),
        ((0, 1), (1, 2), (2, 0), (2, 3), (2, 4), (4, 0)),
        ((0, 1), (1, 2), (2, 0), (2, 3), (3, 0), (2, 4)),
    ),
    (
        ((0, 1), (0, 2), (0, 3), (0, 4), (1, 2), (1, 3), (1, 4)),
        ((0, 1), (1, 2), (2, 0), (2, 3), (3, 1), (2, 4), (4, 1)),
        ((0, 1), (1, 2), (2, 0), (2, 3), (3, 0), (2, 4), (4, 0)),
    ),
    (
        ((0, 1), (0, 4), (0, 5), (1, 2), (1, 3)),
        ((0, 1), (1, 2), (1, 3), (3, 4), (3, 5)),
        ((0, 1), (1, 2), (2, 3), (2, 4), (1, 5)),
    ),
    (
        ((0, 1), (0, 2), (0, 4), (0, 5), (1, 2), (1, 3)),
        ((0, 1), (1, 2), (2, 3), (3, 1), (3, 4), (3, 5)),
        ((0, 1), (1, 2), (2, 0), (2, 3), (2, 4), (1, 5)),
    ),
    (
        ((0, 1), (0, 2), (0, 3), (0, 4), (0, 5), (1, 2), (1, 3)),
        ((0, 1), (1, 2), (2, 0), (2, 3), (2, 4), (2, 5), (5, 0)),
        ((0, 1), (1, 2), (2, 0), (2, 3), (3, 0), (2, 4), (2, 5)),
    ),
)


def _graph_from_pairs(edges: Pairs) -> nx.Graph:
    graph = nx.Graph()
    graph.add_nodes_from(range(1 + max(max(e) for e in edges)))
    graph.add_edges_from(edges)
    return graph


@pytest.mark.parametrize(("edges", "tool", "minimum"), KAVIN_COUNTEREXAMPLES)
def test_kaviniitm_counterexamples_are_valid_but_not_minimal(
    edges: Pairs, tool: Pairs, minimum: Pairs
) -> None:
    """Each archived ``kaviniitm`` answer is a real DFS code, and is larger.

    That is the whole finding: the tool returns a **valid** DFS code that is
    not the minimum one, which is why it agreed with us on the running
    example and on every path and cycle.  The smallest such failure is
    ``K4`` minus edge ``(2,3)``, where it returns ``<0,1><1,2><2,0><2,3><3,1>``
    and the minimum is ``<0,1><1,2><2,0><2,3><3,0>``: both open with the same
    triangle and the same forward edge, so the codes are decided by the final
    backward edge, and backward edges sort by *increasing* target index, so
    ``(3,0)`` precedes ``(3,1)``.
    """
    graph = _graph_from_pairs(edges)
    assert ours(graph) == minimum
    enumerated = {as_pairs(c) for c in all_dfs_codes(graph)}
    assert tool in enumerated, "the tool's answer is not even a valid DFS code"
    assert minimum in enumerated
    assert code_lt([(i, j, 0, 0, 0) for i, j in minimum], [(i, j, 0, 0, 0) for i, j in tool])
    assert not code_lt([(i, j, 0, 0, 0) for i, j in tool], [(i, j, 0, 0, 0) for i, j in minimum])


def test_kaviniitm_smallest_counterexample_is_k4_minus_an_edge() -> None:
    """The fixture named in the brief: ``K4`` minus edge ``(2,3)``."""
    k4 = nx.complete_graph(4)
    k4.remove_edge(2, 3)
    assert ours(k4) == ((0, 1), (1, 2), (2, 0), (2, 3), (3, 0))
    assert sorted(k4.edges()) == sorted(KAVIN_COUNTEREXAMPLES[0][0])


def test_gate_k2_passes_on_our_implementation() -> None:
    """K2 on the shipped backend: every graph invariant."""
    tested, bad = gate_k2_isomorphism_invariance(ours, trials_per_n=4, copies=6)
    assert bad == 0
    assert tested == 24


def test_gate_k2_detects_a_broken_candidate() -> None:
    """**The gate must be able to fail.**

    ``kaviniitm/DFSCode`` agreed with us on the running example and on every
    path and cycle; a single-example check would have adopted it.  A greedy
    no-branch construction -- the same structural defect -- is rejected by
    K2 here, which is what makes a K2 pass mean something.
    """
    _tested, bad = gate_k2_isomorphism_invariance(greedy_no_branch_code, trials_per_n=4, copies=6)
    assert bad > 0, "K2 accepted a construction with no tie branching"


@pytest.mark.slow
def test_gate_k1_passes_on_our_implementation() -> None:
    """K1 on the shipped backend: zero mismatches against the oracle."""
    report = gate_k1_agreement(ours)
    assert {n: bad for n, (_total, bad) in report.items()} == {2: 0, 3: 0, 4: 0, 5: 0, 6: 0}
    assert {n: total for n, (total, _bad) in report.items()} == {
        2: 1,
        3: 2,
        4: 6,
        5: 21,
        6: 112,
    }


@pytest.mark.slow
def test_gate_k1_detects_a_broken_candidate() -> None:
    """A no-branch construction is wrong on some class with ``n <= 6``."""
    report = gate_k1_agreement(greedy_no_branch_code, n_values=(4, 5, 6))
    assert sum(bad for _total, bad in report.values()) > 0


@pytest.mark.slow
@pytest.mark.skipif(
    not os.environ.get("KAVIN_DFSCODE_BIN"),
    reason=(
        "kaviniitm/DFSCode is not vendored and its binary is not in this "
        "repository; set KAVIN_DFSCODE_BIN to re-run the differential. The "
        "archived verdict lives in KAVIN_VERDICT and every counterexample it "
        "printed is re-verified by test_kaviniitm_counterexamples_*."
    ),
)
def test_kaviniitm_differential_against_the_binary() -> None:
    """Re-run the archived differential when a binary is supplied.

    **K2 first**, then K1: K2 needs no oracle, and it is where that
    implementation died.
    """
    binary = os.environ["KAVIN_DFSCODE_BIN"]

    def candidate(graph: nx.Graph) -> Pairs:
        return _run_kavin(binary, [graph])[0]

    _tested, bad = gate_k2_isomorphism_invariance(candidate)
    assert bad == KAVIN_VERDICT["k2_not_invariant"]
    report = gate_k1_agreement(candidate)
    assert report == KAVIN_VERDICT["k1_mismatches"]


def _run_kavin(binary: str, graphs: Sequence[nx.Graph]) -> list[Pairs]:
    """Feed graphs to the ``kaviniitm/DFSCode`` binary and parse its codes."""
    import re

    payload = "".join(_kavin_input(g, f"g{i}") for i, g in enumerate(graphs))
    proc = subprocess.run(  # noqa: S603 - path comes from the operator's env
        [binary], input=payload, capture_output=True, text=True, check=True
    )
    return [
        tuple((int(a), int(b)) for a, b in re.findall(r"<(\d+),(\d+),[^>]*>", line))
        for line in proc.stdout.strip().split("\n")
        if line.strip()
    ]


def _kavin_input(graph: nx.Graph, gid: str) -> str:
    nodes = list(graph.nodes())
    index = {v: i for i, v in enumerate(nodes)}
    lines = [
        gid,
        str(len(nodes)),
        " ".join("a" for _ in nodes),
        str(graph.number_of_edges()),
    ]
    lines += [f"{index[u]} {index[v]} e" for u, v in graph.edges()]
    return "\n".join(lines) + "\n"


# ---------------------------------------------------------------------------
# The budget -- memory, not time
# ---------------------------------------------------------------------------


def test_budget_default_is_the_frozen_value() -> None:
    """50,000, and it is behind a published failure rate."""
    assert MAX_PROJECTIONS == 50_000


def test_budget_raises_and_returns_no_incumbent() -> None:
    """A budget that runs out raises; it never returns a degraded value."""
    graph = nx.complete_graph(8)
    with pytest.raises(MinDfsBudgetExceeded) as info:
        min_dfs_code(graph, max_projections=4)
    assert "memory" in str(info.value)


def test_budget_does_not_change_the_answer_where_it_does_not_fire() -> None:
    """The cap is inert below it -- checked on every fixture and on ``K5``."""
    graphs = [fixtures.to_networkx(fixtures.ALL_FIXTURES[n]) for n in fixtures.CONNECTED_FIXTURES]
    graphs.append(nx.complete_graph(5))
    for graph in graphs:
        assert min_dfs_code(graph, max_projections=None) == min_dfs_code(
            graph, max_projections=MAX_PROJECTIONS
        )


def test_backend_budget_keyword_is_honoured() -> None:
    """``Budget(max_projections=...)`` reaches the construction."""
    backend = MinDfsBackend()
    with pytest.raises(MinDfsBudgetExceeded):
        backend.encode(nx.complete_graph(8), budget=Budget(max_projections=4))
    # An explicit unbounded budget runs to completion.
    assert backend.encode(nx.complete_graph(8), budget=Budget()).length == 28


# ---------------------------------------------------------------------------
# Conventions: symbols, bits, capabilities, scope
# ---------------------------------------------------------------------------


def test_one_symbol_is_one_dfs_tuple() -> None:
    """The 2x trap: tuple-level charges 1 edit where character level charges 4."""
    backend = get_repr_backend("min_dfs")
    a = backend.encode(fixtures.to_networkx(fixtures.RUNNING_EXAMPLE))
    b = backend.encode(fixtures.to_networkx(fixtures.RUNNING_EXAMPLE_MINUS_EDGE))
    assert a.symbols[-1] == "5-2"
    assert get_metric("levenshtein").distance(a, b) == 1.0
    assert get_metric("levenshtein_char").distance(a, b) == 4.0


def test_text_is_the_character_rendering_and_symbols_are_not() -> None:
    """``text`` exists for figures; it is not the comparison unit."""
    encoding = MinDfsBackend().encode(fixtures.to_networkx(fixtures.RUNNING_EXAMPLE))
    assert encoding.text == "0-1 1-2 2-0 2-3 3-4 4-5 5-2"
    assert len(encoding.symbols) == 7
    assert len(encoding.text) == 27


def test_entropy_bits_are_the_fixed_width_upper_bound() -> None:
    """``m * 2*ceil(log2 n)``, with the realised count flagged inflated."""
    backend = MinDfsBackend()
    for name in fixtures.CONNECTED_FIXTURES:
        graph = fixtures.to_networkx(fixtures.ALL_FIXTURES[name])
        encoding = backend.encode(graph)
        counted = backend.bits(encoding)
        n, m = graph.number_of_nodes(), graph.number_of_edges()
        assert counted.entropy_bits == m * 2 * math.ceil(math.log2(n)), name
        assert counted.inflated is True
        assert counted.realised_bits == 8 * len(encoding.text)


def test_no_positional_frame() -> None:
    """min-DFS is a sequence of index pairs, not a bit vector over fixed cells.

    ``padded_hamming`` is therefore *undefined* here, which is a reported F1
    result rather than an error to work around.
    """
    encoding = MinDfsBackend().encode(fixtures.to_networkx(fixtures.RUNNING_EXAMPLE))
    assert encoding.frame is None
    assert Capability.POSITIONAL_FRAME not in MinDfsBackend.capabilities


def test_declared_capabilities() -> None:
    """Canonical, complete, reversible; not disconnection-tolerant, not Suite-1-only."""
    caps = MinDfsBackend.capabilities
    assert Capability.CANONICAL in caps
    assert Capability.COMPLETE_INVARIANT in caps
    assert Capability.REVERSIBLE in caps
    assert Capability.HANDLES_DISCONNECTED not in caps
    assert Capability.SUITE1_ONLY not in caps
    assert Capability.BASELINE not in caps


def test_disconnected_and_edgeless_graphs_raise() -> None:
    """A documented AE.3 row: AGM, graph6 and sparse6 handle these; we do not."""
    disconnected = fixtures.to_networkx(fixtures.C4_PLUS_K3_DISJOINT)
    with pytest.raises(ValueError, match="disconnected"):
        min_dfs_code(disconnected)
    with pytest.raises(ValueError, match="no edges"):
        min_dfs_code(fixtures.to_networkx(fixtures.EMPTY_3))


def test_k33_and_prism_are_separated() -> None:
    """The completeness witness, min-DFS half: the codes differ.

    ``wl_subtree`` gives these two graphs distance exactly 0 at every ``h``
    (``tests/unit/test_wl_subtree.py``); the minimum DFS code separates
    them, which is the contrast the figure makes.
    """
    k33 = fixtures.to_networkx(fixtures.K33)
    prism = fixtures.to_networkx(fixtures.PRISM)
    assert render(min_dfs_code(k33)) == "0-1 1-2 2-3 3-0 3-4 4-1 4-5 5-0 5-2"
    assert render(min_dfs_code(prism)) == "0-1 1-2 2-0 2-3 3-4 4-0 4-5 5-1 5-3"
    assert not nx.is_isomorphic(k33, prism)


def test_registry_returns_the_backend() -> None:
    """``get_repr_backend('min_dfs')`` resolves through ``_LAZY_MODULES``."""
    backend = get_repr_backend("min_dfs")
    assert backend.name == "min_dfs"
    assert backend.is_available() is True


def test_code_symbols_and_render_agree() -> None:
    """``render`` is exactly the symbols joined by a space."""
    code = min_dfs_code(fixtures.to_networkx(fixtures.RUNNING_EXAMPLE))
    assert render(code) == " ".join(code_symbols(code))


def _iter_connected(n_values: Sequence[int]) -> Iterator[nx.Graph]:
    for n in n_values:
        yield from connected_classes(n)


@pytest.mark.slow
def test_rightmost_path_is_a_path_in_the_code() -> None:
    """Structural invariant: the rightmost path is a root-to-leaf chain."""
    for graph in _iter_connected((4, 5)):
        code = min_dfs_code(graph, max_projections=None)
        for k in range(1, len(code) + 1):
            path = rightmost_path(code[:k])
            assert path[0] == 0
            assert len(set(path)) == len(path)
