"""The AGM brute-force oracle and the measured ceiling.

No package implements "the AGM canonical code of one graph", so the port in
``backends/agm.py`` is ours and **this file is the whole value of it**.  The
oracle is exhaustive rather than sampled: for every test graph the code is
compared against the lexicographic minimum over **all ``n!`` permutations**,
which is the definition itself.

Criterion 3: **327 graphs, 0 mismatches** -- every isomorphism class on
``n <= 6`` (2, 4, 11, 34, 156, including disconnected) plus 120 random
graphs at ``n = 7, 8``, with reversibility on all 327.

The isomorphism classes are enumerated by deduplicating every labelled
graph on ``n`` vertices with ``pynauty.certificate``, which is an
**independent** complete invariant.  Deduplicating with AGM's own code would
make the class counts circular; matching OEIS A000088 (1, 2, 4, 11, 34, 156)
with an outside oracle does not.
"""

from __future__ import annotations

import itertools
import random
from typing import TYPE_CHECKING

import pytest

if TYPE_CHECKING:
    import networkx as nx

from isalgraph.competitors import datasets, fixtures
from isalgraph.competitors.base import Budget
from isalgraph.competitors.registry import get_repr_backend
from isalgraph.errors import AGMBudgetExceeded

pytestmark = pytest.mark.unit

nx = pytest.importorskip("networkx")
pynauty = pytest.importorskip("pynauty")

from isalgraph.competitors.backends.agm import (  # noqa: E402
    SUITE1_NODE_BUDGET,
    SUITE2_NODE_BUDGET,
    _adjacency_sets,
    _code_from_perm,
    _greedy_incumbent,
    agm_canonical_code,
    code_to_graph,
)

#: OEIS A000088, graphs on n unlabelled nodes: n = 2..6.
ISOMORPHISM_CLASS_COUNTS = {2: 2, 3: 4, 4: 11, 5: 34, 6: 156}

#: 120 random graphs at n = 7 and n = 8, 60 each.
RANDOM_ORACLE_N = (7, 8)
RANDOM_ORACLE_PER_N = 60
ORACLE_SEED = 42

#: 207 isomorphism classes + 120 random graphs.
ORACLE_TOTAL = sum(ISOMORPHISM_CLASS_COUNTS.values()) + len(RANDOM_ORACLE_N) * RANDOM_ORACLE_PER_N


def brute_force_code(graph: nx.Graph) -> str:
    """The definition: minimum over every permutation.  ``n <= 8`` only.

    Deliberately unoptimised.  It is the oracle, so it must be obviously
    the definition rather than cleverly equivalent to it.
    """
    adjacency, n = _adjacency_sets(graph)
    return min(_code_from_perm(adjacency, list(p)) for p in itertools.permutations(range(n)))


def brute_force_max_code(graph: nx.Graph) -> str:
    """FFSM's convention, for the one test that pins which mirror we took."""
    adjacency, n = _adjacency_sets(graph)
    return max(_code_from_perm(adjacency, list(p)) for p in itertools.permutations(range(n)))


def _all_labelled_graphs(n: int) -> list[nx.Graph]:
    pairs = [(i, j) for j in range(1, n) for i in range(j)]
    out = []
    for mask in range(1 << len(pairs)):
        graph = nx.Graph()
        graph.add_nodes_from(range(n))
        graph.add_edges_from(pair for k, pair in enumerate(pairs) if mask >> k & 1)
        out.append(graph)
    return out


def isomorphism_classes(n: int) -> list[nx.Graph]:
    """One representative per isomorphism class on ``n`` vertices.

    Deduplicated with ``pynauty.certificate`` -- an outside complete
    invariant -- so the class counts are evidence and not a restatement of
    what AGM already believes.
    """
    seen: dict[bytes, nx.Graph] = {}
    for graph in _all_labelled_graphs(n):
        adjacency = {v: list(graph.neighbors(v)) for v in range(n)}
        cert = pynauty.certificate(pynauty.Graph(n, directed=False, adjacency_dict=adjacency))
        seen.setdefault(cert, graph)
    return list(seen.values())


def random_oracle_graphs() -> list[nx.Graph]:
    rng = random.Random(ORACLE_SEED)
    out: list[nx.Graph] = []
    for n in RANDOM_ORACLE_N:
        for _ in range(RANDOM_ORACLE_PER_N):
            m = rng.randint(0, n * (n - 1) // 2)
            out.append(nx.gnm_random_graph(n, m, seed=rng.randrange(10**9)))
    return out


def oracle_corpus() -> list[nx.Graph]:
    """The 327 graphs criterion 3 names."""
    graphs: list[nx.Graph] = []
    for n in sorted(ISOMORPHISM_CLASS_COUNTS):
        graphs.extend(isomorphism_classes(n))
    graphs.extend(random_oracle_graphs())
    return graphs


# ==========================================================================
# The corpus itself is checked before it is used as evidence
# ==========================================================================


@pytest.mark.slow
@pytest.mark.parametrize(("n", "expected"), sorted(ISOMORPHISM_CLASS_COUNTS.items()))
def test_isomorphism_class_counts_match_oeis_a000088(n: int, expected: int) -> None:
    assert len(isomorphism_classes(n)) == expected


@pytest.mark.slow
def test_oracle_corpus_is_327_graphs() -> None:
    assert len(oracle_corpus()) == ORACLE_TOTAL == 327


@pytest.mark.slow
def test_oracle_corpus_includes_disconnected_graphs() -> None:
    """AGM was designed for disconnected graphs; the corpus must exercise it."""
    corpus = oracle_corpus()
    disconnected = sum(1 for g in corpus if g.number_of_nodes() > 1 and not nx.is_connected(g))
    assert disconnected > 0
    isolated = sum(1 for g in corpus if any(d == 0 for _, d in g.degree()))
    assert isolated > 0


# ==========================================================================
# Criterion 3 -- the oracle
# ==========================================================================


@pytest.mark.slow
def test_agm_agrees_with_brute_force_over_all_permutations() -> None:
    """**327 graphs, 0 mismatches**, against the lex-min over all ``n!``."""
    mismatches: list[tuple[int, str, str]] = []
    checked = 0
    for graph in oracle_corpus():
        code, _expanded = agm_canonical_code(graph)
        reference = brute_force_code(graph)
        checked += 1
        if code != reference:
            mismatches.append((graph.number_of_nodes(), code, reference))
    assert checked == ORACLE_TOTAL
    assert mismatches == []


@pytest.mark.slow
def test_agm_is_reversible_on_the_whole_corpus() -> None:
    """``code + n`` rebuilds an isomorphic graph on all 327."""
    checked = 0
    for graph in oracle_corpus():
        code, _expanded = agm_canonical_code(graph)
        rebuilt = code_to_graph(code, graph.number_of_nodes())
        assert nx.is_isomorphic(graph, rebuilt)
        checked += 1
    assert checked == ORACLE_TOTAL


@pytest.mark.slow
def test_agm_is_a_complete_invariant_on_the_whole_corpus() -> None:
    """Equal codes iff isomorphic, checked against ``pynauty.certificate``.

    Over the 207 isomorphism classes on ``n <= 6`` the codes must be
    pairwise distinct within each ``n``; a collision would refute the
    complete-invariant claim in the AE.3 properties table.
    """
    for n in sorted(ISOMORPHISM_CLASS_COUNTS):
        codes = {agm_canonical_code(g)[0] for g in isomorphism_classes(n)}
        assert len(codes) == ISOMORPHISM_CLASS_COUNTS[n]


@pytest.mark.slow
def test_agm_is_isomorphism_invariant_on_the_whole_corpus() -> None:
    """One code per graph over 20 genuine relabellings each."""
    rng = random.Random(ORACLE_SEED)
    for graph in oracle_corpus():
        codes = {agm_canonical_code(graph)[0]}
        for _ in range(20):
            codes.add(agm_canonical_code(fixtures.shuffled_copy(graph, rng))[0])
        assert len(codes) == 1


# ==========================================================================
# The convention: AGM takes the MINIMUM
# ==========================================================================


def test_agm_takes_the_minimum_not_the_maximum() -> None:
    """AGM takes the minimum; FFSM takes the maximum.  They are mirrors.

    If this ever flips, every published AGM number becomes unreproducible,
    which is why the convention is stated in ``agm.py``'s docstring and
    pinned here.
    """
    graph = fixtures.to_networkx(fixtures.RUNNING_EXAMPLE)
    code, _expanded = agm_canonical_code(graph)
    assert code == brute_force_code(graph) == "000001110011110"
    assert code < brute_force_max_code(graph)


def test_code_length_is_exactly_the_triangle() -> None:
    for name, fixture in fixtures.ALL_FIXTURES.items():
        graph = fixtures.to_networkx(fixture)
        n = graph.number_of_nodes()
        assert len(agm_canonical_code(graph)[0]) == n * (n - 1) // 2, name


def test_code_ones_count_equals_edge_count() -> None:
    for name, fixture in fixtures.ALL_FIXTURES.items():
        graph = fixtures.to_networkx(fixture)
        code, _expanded = agm_canonical_code(graph)
        assert code.count("1") == graph.number_of_edges(), name


def test_prefix_property_holds() -> None:
    """The first ``k(k-1)/2`` bits depend only on the first ``k`` vertices.

    That property is the only reason branch and bound is possible, so it is
    asserted rather than assumed.
    """
    graph = fixtures.to_networkx(fixtures.RUNNING_EXAMPLE)
    adjacency, n = _adjacency_sets(graph)
    perm = [3, 0, 5, 1, 4, 2]
    full = _code_from_perm(adjacency, perm)
    for k in range(1, n + 1):
        assert _code_from_perm(adjacency, perm[:k]) == full[: k * (k - 1) // 2]


# ==========================================================================
# The budget raises; it never returns the incumbent
# ==========================================================================


def test_budget_exhaustion_raises_and_discards_the_incumbent() -> None:
    """A stated ceiling is a result; a silent one is a defect.

    The greedy incumbent is not canonical, would fail F3, and would put a
    non-invariant code into a column headed canonical -- precisely the error
    ``graph6`` is in the pool to expose.
    """
    graph = nx.complete_graph(9)
    with pytest.raises(AGMBudgetExceeded) as excinfo:
        agm_canonical_code(graph, node_budget=50)
    message = str(excinfo.value)
    assert "NOT canonical" in message
    assert not hasattr(excinfo.value, "best")


def test_the_greedy_incumbent_is_not_canonical() -> None:
    """The value the budget path refuses to return is genuinely wrong.

    Without a graph on which the incumbent differs from the canonical code,
    "never return the incumbent" would be a rule with no teeth.
    """
    differing = 0
    rng = random.Random(3)
    for _ in range(40):
        graph = nx.gnm_random_graph(7, rng.randint(6, 15), seed=rng.randrange(10**9))
        adjacency, n = _adjacency_sets(graph)
        if _greedy_incumbent(adjacency, n) != agm_canonical_code(graph)[0]:
            differing += 1
    assert differing > 0


def test_the_greedy_incumbent_is_not_isomorphism_invariant() -> None:
    """It would fail F3, which is the second half of the same argument."""
    graph = fixtures.to_networkx(fixtures.RUNNING_EXAMPLE)
    rng = random.Random(11)
    codes = set()
    for _ in range(20):
        copy = fixtures.shuffled_copy(graph, rng)
        adjacency, n = _adjacency_sets(copy)
        codes.add(_greedy_incumbent(adjacency, n))
    assert len(codes) > 1


def test_backend_budget_default_is_the_frozen_suite_1_value() -> None:
    backend = get_repr_backend("agm_cam")
    graph = nx.complete_graph(9)
    with pytest.raises(AGMBudgetExceeded, match="200000"):
        backend.encode(graph)
    with pytest.raises(AGMBudgetExceeded, match="100000"):
        backend.encode(graph, budget=Budget(search_nodes=SUITE2_NODE_BUDGET))


def test_expanded_node_count_is_reported() -> None:
    """47 search nodes on the running example (``agm.md`` §2)."""
    graph = fixtures.to_networkx(fixtures.RUNNING_EXAMPLE)
    _code, expanded = agm_canonical_code(graph)
    assert expanded == 47


# ==========================================================================
# Criterion 5 -- the measured ceiling, at the frozen budgets
# ==========================================================================


def _exact_rate(graphs: list[nx.Graph], node_budget: int) -> tuple[int, int, list[int]]:
    """Return ``(n_exact, n_total, failed_indices)``.  Failures are kept."""
    exact = 0
    failed: list[int] = []
    for idx, graph in enumerate(graphs):
        try:
            agm_canonical_code(graph, node_budget=node_budget)
        except AGMBudgetExceeded:
            failed.append(idx)
            continue
        exact += 1
    return exact, len(graphs), failed


@pytest.mark.slow
@pytest.mark.parametrize(
    "dataset", ["iam_letter_low", "iam_letter_med", "iam_letter_high", "linux"]
)
def test_ceiling_is_100_percent_on_the_easy_suite_1_datasets(dataset: str) -> None:
    """Letter LOW / MED / HIGH and LINUX: **100 %** at the 200k budget."""
    if dataset not in datasets.available_datasets():
        pytest.skip(f"cohort {dataset!r} not on this workstation")
    cohort = datasets.load(dataset)
    graphs = list(cohort.graphs)
    exact, total, failed = _exact_rate(graphs, SUITE1_NODE_BUDGET)
    assert failed == []
    assert exact == total


@pytest.mark.slow
def test_ceiling_on_aids_is_99_6_percent_with_three_named_failures() -> None:
    """AIDS, 769 graphs, 200k budget: **99.6 %**, 3 failures.

    **The 3 failures are recorded and printed, not dropped** -- they are why
    AGM has no rho column on AIDS.
    """
    if "aids" not in datasets.available_datasets():
        pytest.skip("cohort 'aids' not on this workstation")
    cohort = datasets.load("aids")
    graphs = list(cohort.graphs)
    assert len(graphs) == 769
    exact, total, failed = _exact_rate(graphs, SUITE1_NODE_BUDGET)
    detail = [
        (int(i), graphs[i].number_of_nodes(), graphs[i].number_of_edges(), cohort.graph_ids[i])
        for i in failed
    ]
    assert len(failed) == 3, f"expected 3 AIDS failures, got {len(failed)}: {detail}"
    assert exact / total == pytest.approx(0.996, abs=5e-4)


@pytest.mark.slow
@pytest.mark.parametrize(("dataset", "expected"), [("grec", 0.76), ("aids_iam", 0.82)])
def test_ceiling_on_suite_2_at_the_100k_budget(dataset: str, expected: float) -> None:
    """GREC **76 %** and AIDS-IAM **82 %** on 400 sampled graphs at 100k.

    Reached through :func:`agm_canonical_code` rather than through the
    backend: ``agm_cam`` carries ``SUITE1_ONLY`` and refuses these graphs by
    policy, but the ceiling is a property of the *algorithm* and has to stay
    measurable for the paper to be able to state it.
    """
    if dataset not in datasets.available_datasets():
        pytest.skip(f"cohort {dataset!r} not on this workstation")
    cohort = datasets.load(dataset)
    graphs = [cohort.graphs[i] for i in cohort.sample(400, seed=42)]
    exact, total, failed = _exact_rate(graphs, SUITE2_NODE_BUDGET)
    assert total == 400
    assert exact / total == pytest.approx(expected, abs=0.02), (
        f"{dataset}: {exact}/{total} exact, {len(failed)} budget failures"
    )
