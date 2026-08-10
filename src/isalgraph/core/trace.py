"""Algorithm trace recording for IsalGraph.

A :class:`StepSnapshot` captures the virtual-machine state after one
symbol of ``Sigma = {N, n, P, p, V, v, C, c, W}`` has been processed:
which *graph* nodes sit on the CDLL ring (in forward circular order),
which graph nodes the primary and secondary pointers carry, which nodes
and edges are currently materialised, which edge (if any) this step
created, and the prefix of the instruction string emitted so far. An
:class:`AlgorithmTrace` is the ordered sequence of snapshots for one run,
either ``"s2g"`` (produced by :class:`~isalgraph.core.string_to_graph.StringToGraph`)
or ``"g2s"`` (produced by :class:`~isalgraph.core.graph_to_string.GraphToString`).

Three design decisions carry the schema:

1. A snapshot stores *id masks over one final structure*, never a copy of
   the structure. Deep-copying a ``SparseGraph`` per step is ``O(|E|)``
   memory per step; a mask is ``O(|V| + |E|)`` integers total and
   serialises to JSON without custom encoders.
2. The graph is serialised exactly once, in the envelope
   (:attr:`AlgorithmTrace.final_graph`).
3. The envelope carries the version tag ``"isalgraph.trace.v1"`` so that
   readers can reject formats they do not understand.

Two IsalGraph-specific invariants:

* **Snapshots store graph node ids, already resolved from CDLL indices.**
  CDLL indices are not graph node indices (see ``core/README.md``,
  invariant 1); resolving at emission time means the view layer never
  calls ``cdll.get_value`` and therefore cannot conflate the two index
  spaces.
* **:attr:`StepSnapshot.created_edge` is recorded by the emitter**, not
  re-derived by the view. A ``C``/``c`` between already-adjacent nodes is
  a genuine no-op in IsalGraph, so any scheme that attributes edges by
  counting ``V``/``C`` tokens desynchronises on the first such no-op.

Restriction: Python standard library only. No numpy, no matplotlib, no
networkx.
"""

from __future__ import annotations

import json
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

from isalgraph.core.cdll import CircularDoublyLinkedList
from isalgraph.core.sparse_graph import SparseGraph
from isalgraph.types import NodeId

SCHEMA_VERSION: str = "isalgraph.trace.v1"

Edge = tuple[int, int]


# ---------------------------------------------------------------------------
# CDLL helpers
# ---------------------------------------------------------------------------


def cdll_forward_order(
    cdll: CircularDoublyLinkedList,
    anchor: int = 0,
) -> tuple[NodeId, ...]:
    """Return the ring payloads in forward circular order from *anchor*.

    The CDLL carries no distinguished head, so the caller supplies the
    anchor slot. Both converters anchor at CDLL index ``0``: it is the
    first slot the free list hands out, neither converter ever calls
    ``remove``, so slot ``0`` stays live for the whole run and the ring
    keeps a stable rotation across every frame of a step figure.

    Args:
        cdll: The list to walk.
        anchor: CDLL node index to start from. Not a graph node index.

    Returns:
        Tuple of *graph* node ids, resolved via ``get_value``.
    """
    n = cdll.size()
    if n == 0:
        return ()
    out: list[NodeId] = []
    ptr = anchor
    for _ in range(n):
        out.append(int(cdll.get_value(ptr)))
        ptr = cdll.next_node(ptr)
    return tuple(out)


# ---------------------------------------------------------------------------
# Edge helpers
# ---------------------------------------------------------------------------


def normalise_edge(source: int, target: int, *, directed: bool) -> Edge:
    """Return the canonical tuple form of one edge.

    Undirected edges are normalised to ``(min, max)`` so that ``(u, v)``
    and ``(v, u)`` compare equal; directed edges keep their orientation.

    Args:
        source: Source graph node id.
        target: Target graph node id.
        directed: Whether the containing graph is directed.

    Returns:
        The normalised ``(u, v)`` pair.
    """
    if directed or source <= target:
        return (int(source), int(target))
    return (int(target), int(source))


def graph_edges(graph: SparseGraph) -> tuple[Edge, ...]:
    """Return every edge of *graph*, sorted and normalised.

    ``SparseGraph`` exposes adjacency sets rather than an edge iterator,
    so the edge list is materialised by scanning ``neighbors`` over the
    contiguous node range. For undirected graphs each edge is emitted
    once, under the ``u <= v`` orientation.

    Args:
        graph: The graph to enumerate.

    Returns:
        Sorted tuple of ``(u, v)`` pairs.
    """
    directed = graph.directed()
    out: set[Edge] = set()
    for u in range(graph.node_count()):
        for v in graph.neighbors(u):
            out.add(normalise_edge(u, v, directed=directed))
    return tuple(sorted(out))


def graph_to_dict(graph: SparseGraph) -> dict[str, Any]:
    """Serialise *graph* to the plain dict carried by the trace envelope.

    Args:
        graph: The graph to serialise.

    Returns:
        Mapping with keys ``"n_nodes"``, ``"edges"`` and ``"directed"``.
    """
    return {
        "n_nodes": int(graph.node_count()),
        "edges": [[int(u), int(v)] for u, v in graph_edges(graph)],
        "directed": bool(graph.directed()),
    }


# ---------------------------------------------------------------------------
# Data classes
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class StepSnapshot:
    """State of the IsalGraph virtual machine after one instruction.

    Args:
        step_idx: ``0`` for the initial state (no instruction consumed);
            ``i`` for the state immediately after instruction ``i - 1``.
        instruction: The single character of ``Sigma`` that produced this
            state, or ``None`` at ``step_idx == 0``.
        cdll_node_order: Graph node ids in forward circular order from the
            CDLL head. Already resolved; never CDLL indices.
        primary_node: Graph node id under the primary pointer.
        secondary_node: Graph node id under the secondary pointer.
        active_nodes: Sorted graph node ids materialised at this step.
        active_edges: Sorted, normalised edges materialised at this step.
        created_edge: The edge this instruction created, or ``None`` when
            the instruction created none (movement, ``W``, or a ``C``/``c``
            between already-adjacent nodes).
        partial_string: Prefix of the instruction string consumed so far.
    """

    step_idx: int
    instruction: str | None
    cdll_node_order: tuple[NodeId, ...]
    primary_node: NodeId
    secondary_node: NodeId
    active_nodes: tuple[NodeId, ...]
    active_edges: tuple[Edge, ...]
    created_edge: Edge | None
    partial_string: str

    def to_json(self) -> dict[str, Any]:
        """Return the JSON-compatible dict form of this snapshot."""
        return {
            "step_idx": self.step_idx,
            "instruction": self.instruction,
            "cdll_node_order": list(self.cdll_node_order),
            "primary_node": self.primary_node,
            "secondary_node": self.secondary_node,
            "active_nodes": list(self.active_nodes),
            "active_edges": [[u, v] for u, v in self.active_edges],
            "created_edge": (
                None if self.created_edge is None else [self.created_edge[0], self.created_edge[1]]
            ),
            "partial_string": self.partial_string,
        }

    @classmethod
    def from_json(cls, obj: dict[str, Any]) -> StepSnapshot:
        """Rebuild a snapshot from :meth:`to_json` output."""
        raw_edge = obj["created_edge"]
        return cls(
            step_idx=int(obj["step_idx"]),
            instruction=obj["instruction"],
            cdll_node_order=tuple(int(v) for v in obj["cdll_node_order"]),
            primary_node=int(obj["primary_node"]),
            secondary_node=int(obj["secondary_node"]),
            active_nodes=tuple(int(v) for v in obj["active_nodes"]),
            active_edges=tuple((int(e[0]), int(e[1])) for e in obj["active_edges"]),
            created_edge=None if raw_edge is None else (int(raw_edge[0]), int(raw_edge[1])),
            partial_string=str(obj["partial_string"]),
        )


@dataclass(frozen=True)
class AlgorithmTrace:
    """Ordered snapshots for one S2G or G2S run.

    Args:
        direction: ``"s2g"`` or ``"g2s"``. Selects the framing used by the
            visualisation layer; the snapshot content is structurally
            identical in both directions.
        directed: Whether the traced graph is directed.
        final_graph: Serialised final graph, as produced by
            :func:`graph_to_dict`.
        snapshots: The trace itself. Length is ``len(instructions) + 1``.

    Raises:
        ValueError: If *direction* is neither ``"s2g"`` nor ``"g2s"``.
    """

    direction: str
    directed: bool
    final_graph: dict[str, Any]
    snapshots: tuple[StepSnapshot, ...] = field(default_factory=tuple)

    def __post_init__(self) -> None:
        if self.direction not in ("s2g", "g2s"):
            raise ValueError(f"direction must be 's2g' or 'g2s', got {self.direction!r}")

    @property
    def instruction_string(self) -> str:
        """The full instruction string, taken from the last snapshot."""
        if not self.snapshots:
            return ""
        return self.snapshots[-1].partial_string

    def to_json(self) -> dict[str, Any]:
        """Return the JSON-compatible dict form of this trace."""
        return {
            "schema": SCHEMA_VERSION,
            "direction": self.direction,
            "directed": self.directed,
            "final_graph": self.final_graph,
            "snapshots": [s.to_json() for s in self.snapshots],
        }

    @classmethod
    def from_json(cls, obj: dict[str, Any]) -> AlgorithmTrace:
        """Rebuild a trace from :meth:`to_json` output.

        Raises:
            ValueError: If the payload carries an unknown schema tag.
        """
        schema = obj.get("schema", SCHEMA_VERSION)
        if schema != SCHEMA_VERSION:
            raise ValueError(f"unsupported trace schema {schema!r}; expected {SCHEMA_VERSION!r}")
        return cls(
            direction=str(obj["direction"]),
            directed=bool(obj["directed"]),
            final_graph=dict(obj["final_graph"]),
            snapshots=tuple(StepSnapshot.from_json(s) for s in obj["snapshots"]),
        )


# ---------------------------------------------------------------------------
# JSON I/O
# ---------------------------------------------------------------------------


def dump_trace(trace: AlgorithmTrace, path: str | Path) -> None:
    """Write *trace* to *path* as UTF-8 JSON with ``indent=2``."""
    Path(path).write_text(json.dumps(trace.to_json(), indent=2), encoding="utf-8")


def load_trace(path: str | Path) -> AlgorithmTrace:
    """Read an :class:`AlgorithmTrace` previously written by :func:`dump_trace`."""
    obj = json.loads(Path(path).read_text(encoding="utf-8"))
    return AlgorithmTrace.from_json(obj)


__all__ = [
    "SCHEMA_VERSION",
    "AlgorithmTrace",
    "Edge",
    "StepSnapshot",
    "cdll_forward_order",
    "dump_trace",
    "graph_edges",
    "graph_to_dict",
    "load_trace",
    "normalise_edge",
]
