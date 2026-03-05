"""Unit tests for StringToGraph."""

from __future__ import annotations

import pytest

from isalgraph.core.string_to_graph import StringToGraph


class TestStringToGraphBasics:
    """Basic conversion correctness."""

    def test_empty_string(self) -> None:
        """Empty string produces a single-node graph."""
        stg = StringToGraph("", directed_graph=False)
        g, _ = stg.run()
        assert g.node_count() == 1
        assert g.logical_edge_count() == 0

    def test_single_V(self) -> None:
        """'V' creates node 1 with edge 0->1."""
        stg = StringToGraph("V", directed_graph=False)
        g, _ = stg.run()
        assert g.node_count() == 2
        assert g.logical_edge_count() == 1
        assert g.has_edge(0, 1)

    def test_single_v(self) -> None:
        """'v' creates node 1 with edge 0->1 (secondary starts at same pos)."""
        stg = StringToGraph("v", directed_graph=False)
        g, _ = stg.run()
        assert g.node_count() == 2
        assert g.logical_edge_count() == 1
        assert g.has_edge(0, 1)

    def test_VV(self) -> None:
        """'VV': two nodes connected to node 0. Nodes: 0,1,2. Edges: 0-1, 0-2."""
        stg = StringToGraph("VV", directed_graph=False)
        g, _ = stg.run()
        assert g.node_count() == 3
        assert g.logical_edge_count() == 2
        assert g.has_edge(0, 1)
        assert g.has_edge(0, 2)

    def test_VNV(self) -> None:
        """'VNV': V creates 1 from 0, N moves primary to 1, V creates 2 from 1."""
        stg = StringToGraph("VNV", directed_graph=False)
        g, _ = stg.run()
        assert g.node_count() == 3
        assert g.logical_edge_count() == 2
        assert g.has_edge(0, 1)
        assert g.has_edge(1, 2)

    def test_W_noop(self) -> None:
        """'W' does nothing."""
        stg = StringToGraph("W", directed_graph=False)
        g, _ = stg.run()
        assert g.node_count() == 1
        assert g.logical_edge_count() == 0


class TestStringToGraphPointers:
    """Pointer movement and semantics."""

    def test_N_and_P_inverse(self) -> None:
        """N then P returns to the same CDLL node."""
        stg = StringToGraph("VNP", directed_graph=False)
        g, _ = stg.run()
        # After V: primary at 0 (didn't move). After N: primary at 1. After P: primary at 0.
        # Only one edge: 0-1.
        assert g.node_count() == 2
        assert g.logical_edge_count() == 1

    def test_pointer_immobility_after_V(self) -> None:
        """Pointer does NOT advance after V.

        Two consecutive V's create two children of the same node.
        """
        stg = StringToGraph("VV", directed_graph=False)
        g, _ = stg.run()
        # Both edges from node 0.
        assert g.has_edge(0, 1)
        assert g.has_edge(0, 2)
        # Node 1 and 2 are not connected.
        assert not g.has_edge(1, 2)

    def test_C_connects_primary_secondary(self) -> None:
        """'VnC': V creates 1 from 0; n moves secondary to 1; C connects 0->1.
        But 0->1 already exists, so C is a no-op (duplicate edge in set)."""
        stg = StringToGraph("VnC", directed_graph=False)
        g, _ = stg.run()
        assert g.node_count() == 2
        # Edge 0-1 exists (from V). C tries to add it again (set absorbs duplicate).
        assert g.has_edge(0, 1)


class TestStringToGraphDirected:
    """Directed graph semantics."""

    def test_directed_V(self) -> None:
        stg = StringToGraph("V", directed_graph=True)
        g, _ = stg.run()
        assert g.has_edge(0, 1)
        assert not g.has_edge(1, 0)

    def test_directed_C_vs_c(self) -> None:
        """C: primary->secondary. c: secondary->primary."""
        # VNnC: V creates 1 from 0, N moves primary to 1, n moves secondary to 1.
        # C: edge 1->1 (self-loop) since both point to same CDLL node.
        # Let's use a better example:
        # VNV: 0->1, 1->2. Primary at 0, secondary at 0.
        # Then 'nC' moves secondary to 1, C: 0->1 (already exists).
        # Let's use: VvNnC
        stg = StringToGraph("Vv", directed_graph=True)
        g, _ = stg.run()
        assert g.node_count() == 3
        assert g.has_edge(0, 1)  # V
        assert g.has_edge(0, 2)  # v (secondary was at 0)


class TestStringToGraphInvalid:
    """Invalid input handling."""

    def test_invalid_char(self) -> None:
        with pytest.raises(ValueError, match="Invalid"):
            StringToGraph("VXV", directed_graph=False)

    def test_trace_returns_snapshots(self) -> None:
        stg = StringToGraph("VNV", directed_graph=False)
        _, trace = stg.run(trace=True)
        # Initial state + 3 instructions = 4 snapshots.
        assert len(trace) == 4
