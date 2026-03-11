"""Hypothesis-based property tests for pruned canonical string.

Tests two critical properties:
1. Invariance: isomorphic graphs produce same pruned canonical string
2. Round-trip: S2G(pruned_canonical_string(S2G(w))) ~ S2G(w)
"""

from __future__ import annotations

from hypothesis import HealthCheck, given, settings
from hypothesis import strategies as st

from isalgraph.core.canonical_pruned import pruned_canonical_string
from isalgraph.core.string_to_graph import StringToGraph

ALPHABET = "NnPpVvCcW"

isalgraph_string = st.text(alphabet=ALPHABET, min_size=0, max_size=20)


@given(w=isalgraph_string)
@settings(max_examples=200, suppress_health_check=[HealthCheck.too_slow])
def test_pruned_invariance_property(w: str) -> None:
    """Pruned canonical string is labeling-independent: encoding a graph,
    decoding, and re-encoding produces the same string (idempotent)."""
    stg = StringToGraph(w, directed_graph=False)
    g, _ = stg.run()

    n = g.node_count()
    e = g.logical_edge_count()

    if n == 0 or (n == 1 and e == 0):
        return

    try:
        w_pruned = pruned_canonical_string(g)
    except ValueError:
        return

    # Decode the pruned string and re-encode: must produce the same string
    stg2 = StringToGraph(w_pruned, directed_graph=False)
    g2, _ = stg2.run()

    try:
        w_pruned2 = pruned_canonical_string(g2)
    except ValueError:
        return

    assert w_pruned == w_pruned2, (
        f"Invariance failure: '{w}' -> pruned '{w_pruned}' -> re-encode '{w_pruned2}'"
    )


@given(w=isalgraph_string)
@settings(max_examples=200, suppress_health_check=[HealthCheck.too_slow])
def test_pruned_roundtrip_property(w: str) -> None:
    """For all valid strings w, S2G(pruned_canonical_string(S2G(w))) ~ S2G(w)."""
    stg1 = StringToGraph(w, directed_graph=False)
    g1, _ = stg1.run()

    n = g1.node_count()
    e = g1.logical_edge_count()

    if n == 0 or (n == 1 and e == 0):
        return

    try:
        w_canonical = pruned_canonical_string(g1)
    except ValueError:
        return

    stg2 = StringToGraph(w_canonical, directed_graph=False)
    g2, _ = stg2.run()

    assert g1.node_count() == g2.node_count(), (
        f"Node count mismatch for '{w}': {g1.node_count()} vs {g2.node_count()}"
    )
    assert g1.logical_edge_count() == g2.logical_edge_count(), (
        f"Edge count mismatch for '{w}': {g1.logical_edge_count()} vs {g2.logical_edge_count()}"
    )
    assert g1.is_isomorphic(g2), f"Round-trip failed for '{w}' -> '{w_canonical}'"
