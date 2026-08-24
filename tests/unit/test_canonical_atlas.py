"""Both canonical forms are complete invariants -- on the graph atlas.

**Promoted out of `scratchpad/verify_canonical.json` on 2026-08-23.**
[data](.claude/notes/review/plan/data.md) §6 forbids a decision resting on a
scratchpad artifact, and **F-1 rests on this one**: the choice of reference arm
was made only after the PI required the completeness premise be re-checked
under the C++ engine.  A measurement a frozen decision depends on belongs in
the suite, where it is re-run, not in a JSON file nobody opens again.

**Why the atlas and not a random sample.**  ``networkx.graph_atlas_g``
enumerates exactly **one representative per isomorphism class** for
``n <= 7``.  Two distinct entries sharing a canonical string is therefore a
collision *by construction* -- no isomorphism test is needed, and none is run.
The per-``n`` counts reproduce **OEIS A001349** (connected simple graphs on
``n`` unlabelled nodes), which is what certifies the enumeration is the
complete class set rather than an arbitrary corpus.

What this establishes, within the undirected class (Thm 2.12; the canonical
string does not encode directedness, invariant 6):

* ``canonical_string`` and ``pruned_canonical_string`` are **both** complete
  invariants -- 995 graphs, 995 distinct strings, 0 collisions each.
* Both are invariant under relabelling, at 20 random relabellings per graph.
* **They are nevertheless different functions**, agreeing on only 137 of 995
  classes.  Substituting one for the other changes every Claim A bit count,
  which is why the reference arm could not be chosen for convenience.
"""

from __future__ import annotations

import numpy as np
import pytest

from isalgraph import canonical_string, pruned_canonical_string

nx = pytest.importorskip("networkx")

#: OEIS A001349, connected simple graphs on n unlabelled nodes, 2 <= n <= 7.
A001349 = {2: 1, 3: 2, 4: 6, 5: 21, 6: 112, 7: 853}
N_CLASSES = 995
RELABELLINGS = 20
SEED = 42

FORMS = {"canonical": canonical_string, "pruned": pruned_canonical_string}


@pytest.fixture(scope="module")
def atlas() -> list:
    """Every connected graph on 2 <= n <= 7, one per isomorphism class."""
    from networkx.generators.atlas import graph_atlas_g

    return [
        graph
        for graph in graph_atlas_g()
        if 2 <= graph.number_of_nodes() <= 7 and nx.is_connected(graph)
    ]


def _encode(graph, form: str) -> str:
    """Encode through the production NetworkX adapter."""
    from isalgraph.adapters.networkx_adapter import NetworkXAdapter

    return FORMS[form](NetworkXAdapter().from_external(graph, directed=False))


def test_the_atlas_is_the_complete_isomorphism_class_set(atlas: list) -> None:
    """A001349 reproduced. Without this the collision counts below would be
    over an arbitrary corpus rather than over every class."""
    counts: dict[int, int] = {}
    for graph in atlas:
        counts[graph.number_of_nodes()] = counts.get(graph.number_of_nodes(), 0) + 1
    assert counts == A001349
    assert len(atlas) == N_CLASSES == sum(A001349.values())


@pytest.mark.parametrize("form", sorted(FORMS))
def test_the_canonical_form_is_a_complete_invariant(atlas: list, form: str) -> None:
    """995 classes, 995 distinct strings, 0 collisions.

    Two atlas entries are never isomorphic, so a shared string is a collision
    outright and the assertion needs no isomorphism check.
    """
    strings = [_encode(graph, form) for graph in atlas]
    assert len(strings) == N_CLASSES
    assert len(set(strings)) == N_CLASSES

    seen: dict[str, int] = {}
    collisions: list[tuple[int, int, str]] = []
    for index, string in enumerate(strings):
        if string in seen:
            collisions.append((seen[string], index, string))
        seen[string] = index
    assert collisions == []


@pytest.mark.parametrize("form", sorted(FORMS))
def test_the_canonical_form_is_invariant_under_relabelling(atlas: list, form: str) -> None:
    """20 random relabellings per graph, 0 failures.

    Completeness without invariance would be an artefact of node ordering, so
    this is the other half of the claim, not a nicety.
    """
    rng = np.random.default_rng(SEED)
    failures: list[tuple[int, str, str]] = []
    for index, graph in enumerate(atlas):
        reference = _encode(graph, form)
        nodes = list(graph.nodes())
        for _ in range(RELABELLINGS):
            order = rng.permutation(len(nodes))
            mapping = {node: int(order[i]) for i, node in enumerate(nodes)}
            got = _encode(nx.relabel_nodes(graph, mapping), form)
            if got != reference:
                failures.append((index, reference, got))
    assert failures == []


def test_the_two_canonical_forms_are_different_functions(atlas: list) -> None:
    """Both are complete, and they are still not interchangeable.

    They agree on 137 of 995 classes and the pruned string is longer on 558.
    Substituting one for the other changes every Claim A bit count, which is
    why F-1 had to be settled by measurement rather than by convenience.
    """
    canonical = [_encode(graph, "canonical") for graph in atlas]
    pruned = [_encode(graph, "pruned") for graph in atlas]
    agree = sum(1 for a, b in zip(canonical, pruned, strict=True) if a == b)
    longer = sum(1 for a, b in zip(canonical, pruned, strict=True) if len(b) > len(a))
    assert agree == 137
    assert longer == 558
    assert agree < N_CLASSES  # they are not the same function


def test_a_disconnected_graph_is_outside_this_claim() -> None:
    """The atlas evidence is for connected graphs; the guard is explicit so a
    later reader does not over-read the scope of the theorem."""
    disconnected = nx.Graph()
    disconnected.add_nodes_from(range(4))
    disconnected.add_edge(0, 1)
    disconnected.add_edge(2, 3)
    assert not nx.is_connected(disconnected)
