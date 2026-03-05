"""Shared pytest fixtures for IsalGraph test suite."""

from __future__ import annotations

import pytest

from isalgraph.core.cdll import CircularDoublyLinkedList
from isalgraph.core.sparse_graph import SparseGraph


@pytest.fixture()
def small_cdll() -> CircularDoublyLinkedList:
    """A CDLL with capacity 10 and one node (value=0)."""
    cdll = CircularDoublyLinkedList(10)
    cdll.insert_after(-1, 0)
    return cdll


@pytest.fixture()
def triangle_undirected() -> SparseGraph:
    """Undirected triangle: 0--1--2--0."""
    g = SparseGraph(max_nodes=3, directed_graph=False)
    g.add_node()
    g.add_node()
    g.add_node()
    g.add_edge(0, 1)
    g.add_edge(1, 2)
    g.add_edge(2, 0)
    return g


@pytest.fixture()
def path_3_undirected() -> SparseGraph:
    """Undirected path: 0--1--2."""
    g = SparseGraph(max_nodes=3, directed_graph=False)
    g.add_node()
    g.add_node()
    g.add_node()
    g.add_edge(0, 1)
    g.add_edge(1, 2)
    return g


@pytest.fixture()
def triangle_directed() -> SparseGraph:
    """Directed triangle: 0->1->2->0."""
    g = SparseGraph(max_nodes=3, directed_graph=True)
    g.add_node()
    g.add_node()
    g.add_node()
    g.add_edge(0, 1)
    g.add_edge(1, 2)
    g.add_edge(2, 0)
    return g
