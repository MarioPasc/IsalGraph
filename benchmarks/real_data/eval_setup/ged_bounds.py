"""Independent lower and upper bounds on graph edit distance.

This module exists to cross-check GEDLIB. Every GED value the revision reports
above ``n = 12`` comes from GEDLIB's ``BRANCH_FAST`` (lower) and ``IPFP`` /
``BIPARTITE`` (upper); validation gate 2 requires a second, independently
derived implementation to agree with it on a shared sample, so that a
systematic misconfiguration of the library cannot pass unnoticed.

Two bounds are implemented, both under a parametrised cost model:

- ``branch_lower_bound`` -- the BRANCH lower bound of Blumenthal & Gamper,
  *IEEE TKDE* 30(3):503-516, 2018. Node assignments are scored by their branch
  cost (node operation plus half the incident-edge assignment cost) and
  minimised by a linear sum assignment. Halving is what makes the result a
  proven lower bound: each edge operation is incident to two nodes and is
  therefore charged at most twice across the assignment.

- ``bipartite_upper_bound`` -- the Riesen & Bunke bipartite approximation,
  *Image and Vision Computing* 27(7):950-959, 2009. A linear sum assignment on
  star costs proposes a node mapping; the returned value is the **exact cost of
  that mapping**, not the assignment objective. Recomputing the induced cost is
  what makes the result a proven upper bound -- the assignment objective itself
  double-counts edges and is not achievable.

Both reduce to a rectangular assignment problem of size ``(n + m) x (n + m)``
solved with ``scipy.optimize.linear_sum_assignment``.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass
from typing import TypeAlias

import networkx as nx
import numpy as np
from scipy.optimize import linear_sum_assignment

logger = logging.getLogger(__name__)

NodeMapping: TypeAlias = dict[object, object | None]

_INF = float("inf")


class GEDBoundsError(Exception):
    """Raised when a bound cannot be computed or violates its own guarantee."""


@dataclass(frozen=True, slots=True)
class EditCosts:
    """Edit operation costs.

    Attributes
    ----------
    node_ins : float
        Cost of inserting a node.
    node_del : float
        Cost of deleting a node.
    node_sub : float
        Cost of substituting one node for another.
    edge_ins : float
        Cost of inserting an edge.
    edge_del : float
        Cost of deleting an edge.
    edge_sub : float
        Cost of substituting one edge for another.
    """

    node_ins: float = 1.0
    node_del: float = 1.0
    node_sub: float = 0.0
    edge_ins: float = 1.0
    edge_del: float = 1.0
    edge_sub: float = 0.0

    def as_gedlib_constant(self) -> list[float]:
        """Return the cost vector in GEDLIB's ``CONSTANT`` ordering.

        Returns
        -------
        list of float
            ``[node_ins, node_del, node_rel, edge_ins, edge_del, edge_rel]``,
            the argument order ``set_edit_cost('CONSTANT', ...)`` expects.
        """
        return [
            self.node_ins,
            self.node_del,
            self.node_sub,
            self.edge_ins,
            self.edge_del,
            self.edge_sub,
        ]


#: The production cost model: unit node and edge insert/delete, free
#: substitution. This is ``statistics.md`` D6 and GEDLIB ``[1, 1, 0, 1, 1, 0]``.
UNIT_COSTS = EditCosts()

#: GraphEdX's published model: node operations are free. Used *only* by the
#: T-03 agreement gate, never in production -- zero node cost makes GED a
#: pseudometric, which is what D6 exists to avoid.
GRAPHEDX_COSTS = EditCosts(node_ins=0.0, node_del=0.0)


def _incident_edge_assignment_cost(deg_u: int, deg_v: int, costs: EditCosts) -> float:
    """Cost of optimally matching two incident-edge multisets.

    Edges carry no labels here, so every pairing has the same substitution
    cost and the optimum is to substitute ``min(deg_u, deg_v)`` of them and
    delete or insert the remainder.

    Parameters
    ----------
    deg_u : int
        Degree of the node in the first graph.
    deg_v : int
        Degree of the node in the second graph.
    costs : EditCosts
        Edit operation costs.

    Returns
    -------
    float
        The assignment cost.
    """
    shared = min(deg_u, deg_v)
    surplus = deg_u - deg_v
    cost = shared * costs.edge_sub
    if surplus > 0:
        cost += surplus * costs.edge_del
    else:
        cost += (-surplus) * costs.edge_ins
    return cost


def _assignment_matrix(
    g1: nx.Graph,
    g2: nx.Graph,
    costs: EditCosts,
    *,
    halve_edges: bool,
) -> tuple[np.ndarray, list, list]:
    """Build the ``(n + m) x (n + m)`` node assignment cost matrix.

    The layout follows Riesen & Bunke: the top-left block holds substitutions,
    the top-right block deletions (diagonal only), the bottom-left block
    insertions (diagonal only), and the bottom-right block is zero.

    Parameters
    ----------
    g1, g2 : networkx.Graph
        The graphs to compare.
    costs : EditCosts
        Edit operation costs.
    halve_edges : bool
        Halve the incident-edge term, which turns the star cost into the
        BRANCH branch cost and the assignment optimum into a lower bound.

    Returns
    -------
    tuple
        ``(cost_matrix, nodes1, nodes2)``.
    """
    nodes1 = list(g1.nodes())
    nodes2 = list(g2.nodes())
    n, m = len(nodes1), len(nodes2)
    deg1 = [g1.degree(u) for u in nodes1]
    deg2 = [g2.degree(v) for v in nodes2]
    factor = 0.5 if halve_edges else 1.0

    size = n + m
    cost = np.full((size, size), _INF, dtype=np.float64)

    for i in range(n):
        for j in range(m):
            cost[i, j] = costs.node_sub + factor * _incident_edge_assignment_cost(
                deg1[i], deg2[j], costs
            )
    for i in range(n):
        cost[i, m + i] = costs.node_del + factor * deg1[i] * costs.edge_del
    for j in range(m):
        cost[n + j, j] = costs.node_ins + factor * deg2[j] * costs.edge_ins
    cost[n:, m:] = 0.0

    return cost, nodes1, nodes2


def branch_lower_bound(g1: nx.Graph, g2: nx.Graph, costs: EditCosts = UNIT_COSTS) -> float:
    """Compute the BRANCH lower bound on the graph edit distance.

    Parameters
    ----------
    g1, g2 : networkx.Graph
        The graphs to compare.
    costs : EditCosts, optional
        Edit operation costs. Defaults to the unit model.

    Returns
    -------
    float
        A value provably no greater than the true graph edit distance.

    Notes
    -----
    On unlabeled graphs with uniform edge costs the incident-edge assignment
    has the closed form ``|deg(u) - deg(v)|``, so BRANCH and BRANCH-FAST
    coincide. A cross-check against GEDLIB therefore validates the value both
    methods return but cannot distinguish them on this data.
    """
    cost, _, _ = _assignment_matrix(g1, g2, costs, halve_edges=True)
    rows, cols = linear_sum_assignment(cost)
    return float(cost[rows, cols].sum())


def _mapping_from_assignment(cost: np.ndarray, nodes1: list, nodes2: list) -> NodeMapping:
    """Extract a node mapping from a solved assignment.

    Parameters
    ----------
    cost : numpy.ndarray
        The assignment cost matrix.
    nodes1, nodes2 : list
        Node identifiers, in the order the matrix was built from.

    Returns
    -------
    dict
        Maps each node of the first graph to a node of the second, or to
        ``None`` where the assignment deletes it.
    """
    n, m = len(nodes1), len(nodes2)
    rows, cols = linear_sum_assignment(cost)
    mapping: NodeMapping = {}
    for r, c in zip(rows, cols, strict=True):
        if r < n:
            mapping[nodes1[r]] = nodes2[c] if c < m else None
    return mapping


def induced_edit_cost(
    g1: nx.Graph,
    g2: nx.Graph,
    mapping: NodeMapping,
    costs: EditCosts = UNIT_COSTS,
) -> float:
    """Compute the exact cost of the edit path induced by a node mapping.

    Any node mapping induces a complete, achievable edit path, so its cost is
    an upper bound on the graph edit distance regardless of how the mapping
    was obtained.

    Parameters
    ----------
    g1, g2 : networkx.Graph
        The graphs to compare.
    mapping : dict
        Maps nodes of ``g1`` to nodes of ``g2`` or to ``None`` for deletion.
    costs : EditCosts, optional
        Edit operation costs.

    Returns
    -------
    float
        The cost of the induced edit path.

    Raises
    ------
    GEDBoundsError
        If the mapping is not injective on its non-``None`` image.
    """
    image = [v for v in mapping.values() if v is not None]
    if len(set(image)) != len(image):
        raise GEDBoundsError("node mapping is not injective")

    substituted = set(image)
    total = 0.0
    for v in mapping.values():
        total += costs.node_del if v is None else costs.node_sub
    total += costs.node_ins * (g2.number_of_nodes() - len(substituted))

    matched_in_g2: set[tuple] = set()
    for a, b in g1.edges():
        pa, pb = mapping.get(a), mapping.get(b)
        if pa is not None and pb is not None and g2.has_edge(pa, pb):
            total += costs.edge_sub
            matched_in_g2.add((pa, pb) if pa <= pb else (pb, pa))
        else:
            total += costs.edge_del
    for x, y in g2.edges():
        key = (x, y) if x <= y else (y, x)
        if key not in matched_in_g2:
            total += costs.edge_ins

    return float(total)


def _oriented_upper_bound(g1: nx.Graph, g2: nx.Graph, costs: EditCosts) -> float:
    """Compute the bipartite upper bound in one fixed argument order.

    Parameters
    ----------
    g1, g2 : networkx.Graph
        The graphs to compare, in the order the assignment is built from.
    costs : EditCosts
        Edit operation costs.

    Returns
    -------
    float
        The cost of the edit path induced by the assignment.
    """
    cost, nodes1, nodes2 = _assignment_matrix(g1, g2, costs, halve_edges=False)
    mapping = _mapping_from_assignment(cost, nodes1, nodes2)
    return induced_edit_cost(g1, g2, mapping, costs)


def bipartite_upper_bound(
    g1: nx.Graph, g2: nx.Graph, costs: EditCosts = UNIT_COSTS, *, symmetrise: bool = True
) -> float:
    """Compute the Riesen-Bunke bipartite upper bound.

    Parameters
    ----------
    g1, g2 : networkx.Graph
        The graphs to compare.
    costs : EditCosts, optional
        Edit operation costs. Defaults to the unit model.
    symmetrise : bool, optional
        Evaluate both argument orders and return the smaller. Default ``True``.
        Pass ``False`` only to reproduce a single-orientation reference such as
        GEDLIB's ``BIPARTITE``.

    Returns
    -------
    float
        The cost of an achievable edit path, hence an upper bound on the true
        graph edit distance.

    Notes
    -----
    **The underlying heuristic is not symmetric.** The star costs that drive
    the assignment are asymmetric in the two graphs' roles, so swapping the
    arguments can return a different value -- measured differences of 12 vs 14
    and 5 vs 7 on small connected pairs. Both values are valid upper bounds,
    but a pairwise matrix filled in one orientation is not a distance matrix,
    and the Levenshtein correlation would be computed against an asymmetric
    reference.

    Taking the minimum of the two orientations is symmetric, is still an upper
    bound because each orientation is an achievable edit path, and is never
    worse than either. The lower bound needs no such treatment: its cost matrix
    depends only on ``|deg(u) - deg(v)|`` and its assignment optimum is
    invariant under transposition.
    """
    value = _oriented_upper_bound(g1, g2, costs)
    if symmetrise:
        value = min(value, _oriented_upper_bound(g2, g1, costs))
    return value


def ged_bracket(g1: nx.Graph, g2: nx.Graph, costs: EditCosts = UNIT_COSTS) -> tuple[float, float]:
    """Compute a proven bracket ``LB <= GED <= UB``.

    Parameters
    ----------
    g1, g2 : networkx.Graph
        The graphs to compare.
    costs : EditCosts, optional
        Edit operation costs.

    Returns
    -------
    tuple of float
        ``(lower_bound, upper_bound)``. Equality certifies the exact distance.

    Raises
    ------
    GEDBoundsError
        If the lower bound exceeds the upper bound, which indicates a defect
        in one of the two constructions rather than a property of the input.
    """
    lb = branch_lower_bound(g1, g2, costs)
    ub = bipartite_upper_bound(g1, g2, costs)
    if lb > ub + 1e-9:
        raise GEDBoundsError(
            f"lower bound {lb} exceeds upper bound {ub}; one construction is wrong"
        )
    return lb, ub


def exact_ged(
    g1: nx.Graph,
    g2: nx.Graph,
    costs: EditCosts = UNIT_COSTS,
    timeout: float | None = None,
) -> float:
    """Compute exact graph edit distance with NetworkX A*.

    Provided here so that the bracket and the reference it is validated
    against are driven by one cost model object and cannot drift apart.

    Parameters
    ----------
    g1, g2 : networkx.Graph
        The graphs to compare.
    costs : EditCosts, optional
        Edit operation costs.
    timeout : float, optional
        Seconds after which the search returns its best result so far.

    Returns
    -------
    float
        The exact distance, or ``inf`` if the search was cut off before
        finding any complete edit path.
    """
    value = nx.graph_edit_distance(
        g1,
        g2,
        node_subst_cost=lambda a, b: costs.node_sub,
        node_del_cost=lambda a: costs.node_del,
        node_ins_cost=lambda a: costs.node_ins,
        edge_subst_cost=lambda a, b: costs.edge_sub,
        edge_del_cost=lambda a: costs.edge_del,
        edge_ins_cost=lambda a: costs.edge_ins,
        timeout=timeout,
    )
    return float(value) if value is not None else _INF
