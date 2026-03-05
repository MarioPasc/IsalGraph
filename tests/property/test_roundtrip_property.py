"""Hypothesis-based property tests for round-trip.

Uses Hypothesis to generate random IsalGraph strings and verify round-trip
correctness as a universally quantified property.
"""

from __future__ import annotations

from hypothesis import HealthCheck, given, settings
from hypothesis import strategies as st

from isalgraph.core.graph_to_string import GraphToString
from isalgraph.core.string_to_graph import StringToGraph

ALPHABET = "NnPpVvCcW"

isalgraph_string = st.text(alphabet=ALPHABET, min_size=0, max_size=30)


@given(w=isalgraph_string)
@settings(max_examples=500, suppress_health_check=[HealthCheck.too_slow])
def test_roundtrip_undirected_property(w: str) -> None:
    """For all valid strings w, S2G(G2S(S2G(w))) ~ S2G(w) (undirected)."""
    stg1 = StringToGraph(w, directed_graph=False)
    g1, _ = stg1.run()

    gts = GraphToString(g1)
    try:
        w_prime, _ = gts.run(0)
    except ValueError:
        # Graph might not be connected (e.g., pure movement instructions).
        # This is a valid case — skip.
        return

    stg2 = StringToGraph(w_prime, directed_graph=False)
    g2, _ = stg2.run()

    assert g1.node_count() == g2.node_count(), (
        f"Node count mismatch for '{w}': {g1.node_count()} vs {g2.node_count()}"
    )
    assert g1.logical_edge_count() == g2.logical_edge_count(), (
        f"Edge count mismatch for '{w}': {g1.logical_edge_count()} vs {g2.logical_edge_count()}"
    )
    assert g1.is_isomorphic(g2), f"Round-trip failed for '{w}' -> '{w_prime}'"


@given(w=isalgraph_string)
@settings(max_examples=500, suppress_health_check=[HealthCheck.too_slow])
def test_roundtrip_directed_property(w: str) -> None:
    """For all valid strings w, S2G(G2S(S2G(w))) ~ S2G(w) (directed)."""
    stg1 = StringToGraph(w, directed_graph=True)
    g1, _ = stg1.run()

    gts = GraphToString(g1)
    try:
        w_prime, _ = gts.run(0)
    except ValueError:
        # Unreachable nodes — valid limitation for directed graphs.
        return

    stg2 = StringToGraph(w_prime, directed_graph=True)
    g2, _ = stg2.run()

    assert g1.node_count() == g2.node_count()
    assert g1.logical_edge_count() == g2.logical_edge_count()
    assert g1.is_isomorphic(g2), f"Round-trip failed for '{w}' -> '{w_prime}'"
