"""Integration tests for PyTorch Geometric adapter.

Cross-validates IsalGraph round-trip correctness using PyG Data objects,
verifying structural preservation through SparseGraph's ``is_isomorphic``
and edge/node count invariants.
"""

from __future__ import annotations

import pytest

torch = pytest.importorskip("torch")
pyg = pytest.importorskip("torch_geometric")

from torch_geometric.data import Data  # noqa: E402

from isalgraph.adapters.pyg_adapter import PyGAdapter  # noqa: E402
from isalgraph.core.graph_to_string import GraphToString  # noqa: E402
from isalgraph.core.sparse_graph import SparseGraph  # noqa: E402
from isalgraph.core.string_to_graph import StringToGraph  # noqa: E402


@pytest.fixture()
def adapter() -> PyGAdapter:
    return PyGAdapter()


# ======================================================================
# Helper: build SparseGraph from edge list
# ======================================================================


def _make_sparse_graph(n: int, edges: list[tuple[int, int]], *, directed: bool) -> SparseGraph:
    """Create a SparseGraph from a node count and edge list."""
    sg = SparseGraph(max_nodes=n, directed_graph=directed)
    for _ in range(n):
        sg.add_node()
    for u, v in edges:
        sg.add_edge(u, v)
    return sg


# ======================================================================
# from_external tests
# ======================================================================


class TestFromExternal:
    """Test PyGAdapter.from_external under various Data inputs."""

    def test_with_edges(self, adapter: PyGAdapter) -> None:
        """Data with edges is converted correctly."""
        edge_index = torch.tensor([[0, 1, 2], [1, 2, 0]], dtype=torch.long)
        data = Data(num_nodes=3, edge_index=edge_index)

        sg = adapter.from_external(data, directed=True)
        assert sg.node_count() == 3
        assert sg.logical_edge_count() == 3
        assert 1 in sg.neighbors(0)
        assert 2 in sg.neighbors(1)
        assert 0 in sg.neighbors(2)

    def test_without_edges(self, adapter: PyGAdapter) -> None:
        """Data with no edge_index produces an edgeless graph."""
        data = Data(num_nodes=4)

        sg = adapter.from_external(data, directed=False)
        assert sg.node_count() == 4
        assert sg.logical_edge_count() == 0

    def test_num_nodes_none_raises(self, adapter: PyGAdapter) -> None:
        """Data with num_nodes=None raises ValueError."""
        data = Data()
        # Ensure num_nodes is truly None.
        assert data.num_nodes is None
        with pytest.raises(ValueError, match="no num_nodes"):
            adapter.from_external(data, directed=False)

    def test_directed_preserves_direction(self, adapter: PyGAdapter) -> None:
        """Edges in a directed graph are one-way."""
        edge_index = torch.tensor([[0], [1]], dtype=torch.long)
        data = Data(num_nodes=2, edge_index=edge_index)

        sg = adapter.from_external(data, directed=True)
        assert 1 in sg.neighbors(0)
        assert 0 not in sg.neighbors(1)

    def test_undirected_symmetrises(self, adapter: PyGAdapter) -> None:
        """When directed=False, add_edge symmetrises the edge."""
        edge_index = torch.tensor([[0], [1]], dtype=torch.long)
        data = Data(num_nodes=2, edge_index=edge_index)

        sg = adapter.from_external(data, directed=False)
        assert 1 in sg.neighbors(0)
        assert 0 in sg.neighbors(1)
        assert sg.logical_edge_count() == 1  # single undirected edge

    def test_empty_edge_index_tensor(self, adapter: PyGAdapter) -> None:
        """Data with an explicit but empty edge_index tensor."""
        edge_index = torch.zeros((2, 0), dtype=torch.long)
        data = Data(num_nodes=3, edge_index=edge_index)

        sg = adapter.from_external(data, directed=False)
        assert sg.node_count() == 3
        assert sg.logical_edge_count() == 0


# ======================================================================
# to_external tests
# ======================================================================


class TestToExternal:
    """Test PyGAdapter.to_external under various SparseGraph inputs."""

    def test_directed_triangle(self, adapter: PyGAdapter) -> None:
        """Directed triangle converts to correct edge_index."""
        sg = _make_sparse_graph(3, [(0, 1), (1, 2), (2, 0)], directed=True)
        data = adapter.to_external(sg)

        assert data.num_nodes == 3
        assert data.edge_index is not None
        assert data.edge_index.shape[1] == 3

    def test_undirected_triangle(self, adapter: PyGAdapter) -> None:
        """Undirected triangle: each undirected edge produces two directed entries."""
        sg = _make_sparse_graph(3, [(0, 1), (1, 2), (0, 2)], directed=False)
        data = adapter.to_external(sg)

        assert data.num_nodes == 3
        assert data.edge_index is not None
        # 3 undirected edges -> 6 directed entries in edge_index
        assert data.edge_index.shape[1] == 6

    def test_no_edges_empty_edge_index(self, adapter: PyGAdapter) -> None:
        """Graph with no edges produces a (2, 0) edge_index tensor."""
        sg = _make_sparse_graph(5, [], directed=False)
        data = adapter.to_external(sg)

        assert data.num_nodes == 5
        assert data.edge_index is not None
        assert data.edge_index.shape == (2, 0)

    def test_single_node(self, adapter: PyGAdapter) -> None:
        """Single-node graph has no edges."""
        sg = _make_sparse_graph(1, [], directed=False)
        data = adapter.to_external(sg)

        assert data.num_nodes == 1
        assert data.edge_index is not None
        assert data.edge_index.shape[1] == 0

    def test_edge_index_dtype(self, adapter: PyGAdapter) -> None:
        """Edge index tensor has dtype torch.long."""
        sg = _make_sparse_graph(3, [(0, 1)], directed=True)
        data = adapter.to_external(sg)

        assert data.edge_index is not None
        assert data.edge_index.dtype == torch.long


# ======================================================================
# Round-trip: SparseGraph -> Data -> SparseGraph
# ======================================================================


class TestPyGRoundTrip:
    """Verify that SparseGraph -> Data -> SparseGraph preserves graph structure."""

    def _roundtrip(self, sg_original: SparseGraph, adapter: PyGAdapter, *, directed: bool) -> None:
        """Convert to PyG Data and back, then assert isomorphism."""
        data = adapter.to_external(sg_original)
        sg_recovered = adapter.from_external(data, directed=directed)
        assert sg_original.is_isomorphic(sg_recovered), (
            f"Round-trip failed: "
            f"original {sg_original.node_count()} nodes, "
            f"{sg_original.logical_edge_count()} edges; "
            f"recovered {sg_recovered.node_count()} nodes, "
            f"{sg_recovered.logical_edge_count()} edges"
        )

    def test_triangle_undirected(self, adapter: PyGAdapter) -> None:
        sg = _make_sparse_graph(3, [(0, 1), (1, 2), (0, 2)], directed=False)
        self._roundtrip(sg, adapter, directed=False)

    def test_triangle_directed(self, adapter: PyGAdapter) -> None:
        sg = _make_sparse_graph(3, [(0, 1), (1, 2), (2, 0)], directed=True)
        self._roundtrip(sg, adapter, directed=True)

    def test_path_undirected(self, adapter: PyGAdapter) -> None:
        edges = [(i, i + 1) for i in range(4)]
        sg = _make_sparse_graph(5, edges, directed=False)
        self._roundtrip(sg, adapter, directed=False)

    def test_path_directed(self, adapter: PyGAdapter) -> None:
        edges = [(i, i + 1) for i in range(4)]
        sg = _make_sparse_graph(5, edges, directed=True)
        self._roundtrip(sg, adapter, directed=True)

    def test_star_undirected(self, adapter: PyGAdapter) -> None:
        edges = [(0, i) for i in range(1, 6)]
        sg = _make_sparse_graph(6, edges, directed=False)
        self._roundtrip(sg, adapter, directed=False)

    def test_star_directed(self, adapter: PyGAdapter) -> None:
        edges = [(0, i) for i in range(1, 6)]
        sg = _make_sparse_graph(6, edges, directed=True)
        self._roundtrip(sg, adapter, directed=True)

    def test_complete_k5_undirected(self, adapter: PyGAdapter) -> None:
        edges = [(i, j) for i in range(5) for j in range(i + 1, 5)]
        sg = _make_sparse_graph(5, edges, directed=False)
        self._roundtrip(sg, adapter, directed=False)

    def test_isolated_nodes(self, adapter: PyGAdapter) -> None:
        sg = _make_sparse_graph(4, [], directed=False)
        self._roundtrip(sg, adapter, directed=False)

    def test_single_edge_undirected(self, adapter: PyGAdapter) -> None:
        sg = _make_sparse_graph(2, [(0, 1)], directed=False)
        self._roundtrip(sg, adapter, directed=False)


# ======================================================================
# Full round-trip: SparseGraph -> string -> SparseGraph -> Data -> SparseGraph
# ======================================================================


class TestFullRoundTrip:
    """Full IsalGraph string round-trip cross-validated via PyG adapter."""

    def _full_roundtrip(
        self,
        sg_original: SparseGraph,
        adapter: PyGAdapter,
        *,
        directed: bool,
    ) -> None:
        """S2G -> G2S -> S2G, converting via PyG at each stage."""
        gts = GraphToString(sg_original)
        w, _ = gts.run(0)

        stg = StringToGraph(w, directed_graph=directed)
        sg_from_string, _ = stg.run()

        # Convert through PyG and back.
        data = adapter.to_external(sg_from_string)
        sg_final = adapter.from_external(data, directed=directed)

        assert sg_original.is_isomorphic(sg_final), (
            f"Full round-trip failed: string={w!r}, "
            f"original {sg_original.node_count()} nodes / "
            f"{sg_original.logical_edge_count()} edges, "
            f"final {sg_final.node_count()} nodes / "
            f"{sg_final.logical_edge_count()} edges"
        )

    def test_triangle(self, adapter: PyGAdapter) -> None:
        sg = _make_sparse_graph(3, [(0, 1), (1, 2), (0, 2)], directed=False)
        self._full_roundtrip(sg, adapter, directed=False)

    def test_path_5(self, adapter: PyGAdapter) -> None:
        edges = [(i, i + 1) for i in range(4)]
        sg = _make_sparse_graph(5, edges, directed=False)
        self._full_roundtrip(sg, adapter, directed=False)

    def test_star_5(self, adapter: PyGAdapter) -> None:
        edges = [(0, i) for i in range(1, 6)]
        sg = _make_sparse_graph(6, edges, directed=False)
        self._full_roundtrip(sg, adapter, directed=False)

    def test_directed_triangle(self, adapter: PyGAdapter) -> None:
        sg = _make_sparse_graph(3, [(0, 1), (1, 2), (2, 0)], directed=True)
        self._full_roundtrip(sg, adapter, directed=True)


# ======================================================================
# Cross-validation: known graph structures
# ======================================================================


class TestKnownGraphs:
    """Build known graphs, convert to/from PyG, verify structure."""

    def test_triangle_edges(self, adapter: PyGAdapter) -> None:
        """Triangle: 3 nodes, 3 undirected edges."""
        sg = _make_sparse_graph(3, [(0, 1), (1, 2), (0, 2)], directed=False)
        data = adapter.to_external(sg)

        assert data.num_nodes == 3
        # 3 undirected edges -> 6 directed in edge_index
        assert data.edge_index is not None
        assert data.edge_index.shape[1] == 6

        # Convert back and verify
        sg2 = adapter.from_external(data, directed=False)
        assert sg2.node_count() == 3
        assert sg2.logical_edge_count() == 3

    def test_path_edges(self, adapter: PyGAdapter) -> None:
        """Path P4: 4 nodes, 3 undirected edges."""
        edges = [(0, 1), (1, 2), (2, 3)]
        sg = _make_sparse_graph(4, edges, directed=False)
        data = adapter.to_external(sg)

        assert data.num_nodes == 4
        assert data.edge_index is not None
        assert data.edge_index.shape[1] == 6  # 3 edges * 2 directions

        sg2 = adapter.from_external(data, directed=False)
        assert sg2.node_count() == 4
        assert sg2.logical_edge_count() == 3

    def test_star_edges(self, adapter: PyGAdapter) -> None:
        """Star S4: center node 0 connected to 1,2,3,4."""
        edges = [(0, 1), (0, 2), (0, 3), (0, 4)]
        sg = _make_sparse_graph(5, edges, directed=False)
        data = adapter.to_external(sg)

        assert data.num_nodes == 5
        assert data.edge_index is not None
        assert data.edge_index.shape[1] == 8  # 4 edges * 2 directions

        sg2 = adapter.from_external(data, directed=False)
        assert sg2.node_count() == 5
        assert sg2.logical_edge_count() == 4
        # Center node degree = 4
        assert len(sg2.neighbors(0)) == 4


# ======================================================================
# CUDA tensor handling
# ======================================================================


class TestCudaTensor:
    """Verify that edge_index tensors on GPU are handled correctly."""

    @pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA not available")
    def test_from_external_cuda_edge_index(self, adapter: PyGAdapter) -> None:
        """Data with edge_index on GPU should still convert correctly."""
        edge_index = torch.tensor([[0, 1, 2], [1, 2, 0]], dtype=torch.long, device="cuda")
        data = Data(num_nodes=3, edge_index=edge_index)

        sg = adapter.from_external(data, directed=True)
        assert sg.node_count() == 3
        assert sg.logical_edge_count() == 3
        assert 1 in sg.neighbors(0)
        assert 2 in sg.neighbors(1)
        assert 0 in sg.neighbors(2)

    @pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA not available")
    def test_cuda_roundtrip(self, adapter: PyGAdapter) -> None:
        """SparseGraph -> Data (moved to CUDA) -> SparseGraph preserves structure."""
        sg = _make_sparse_graph(4, [(0, 1), (1, 2), (2, 3), (3, 0)], directed=False)
        data = adapter.to_external(sg)
        # Move edge_index to CUDA
        data.edge_index = data.edge_index.cuda()

        sg2 = adapter.from_external(data, directed=False)
        assert sg.is_isomorphic(sg2)


# ======================================================================
# String-level adapter methods (inherited from GraphAdapter base)
# ======================================================================


class TestStringMethods:
    """Test to_isalgraph_string / from_isalgraph_string via PyG."""

    def test_to_and_from_string(self, adapter: PyGAdapter) -> None:
        """Round-trip through IsalGraph string representation."""
        edge_index = torch.tensor([[0, 1, 1, 2, 2, 0], [1, 0, 2, 1, 0, 2]], dtype=torch.long)
        data_in = Data(num_nodes=3, edge_index=edge_index)

        s = adapter.to_isalgraph_string(data_in, directed=False)
        data_out = adapter.from_isalgraph_string(s, directed=False)

        assert data_out.num_nodes == 3
        # Triangle: 3 undirected edges -> 6 directed in edge_index
        assert data_out.edge_index is not None
        assert data_out.edge_index.shape[1] == 6
