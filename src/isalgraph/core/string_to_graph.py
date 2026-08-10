"""IsalGraph string to SparseGraph converter.

Migration of the original ``stringtograph.py`` with the following bug fix:

* **B6**: The V/v/C/c handlers passed ``self._primary_ptr`` (a CDLL node
  index) directly to ``SparseGraph.add_edge``, which expects *graph* node
  indices.  Fixed to use ``self._cdll.get_value(ptr)`` throughout.

  In the original code this bug was **latent**: the CDLL free-list pops
  indices 0, 1, 2, ... in order and graph nodes are also 0, 1, 2, ...,
  so the two index spaces coincide as long as no CDLL nodes are ever
  removed.  The fix makes the code correct regardless of allocation order.
"""

from __future__ import annotations

from collections import Counter
from copy import deepcopy

from isalgraph.core.cdll import CircularDoublyLinkedList
from isalgraph.core.sparse_graph import SparseGraph
from isalgraph.core.trace import (
    AlgorithmTrace,
    Edge,
    StepSnapshot,
    cdll_forward_order,
    graph_edges,
    graph_to_dict,
    normalise_edge,
)
from isalgraph.types import VALID_INSTRUCTIONS


class StringToGraph:
    """Convert an IsalGraph instruction string into a ``SparseGraph``.

    Args:
        input_string: The instruction string over the alphabet
            ``{N, n, P, p, V, v, C, c, W}``.
        directed_graph: Whether to build a directed graph.

    Raises:
        ValueError: If *input_string* contains characters outside the
            valid alphabet.
    """

    __slots__ = (
        "_input_string",
        "_directed_graph",
        "_max_nodes",
        "_output_graph",
        "_cdll",
        "_primary_ptr",
        "_secondary_ptr",
    )

    def __init__(
        self,
        input_string: str,
        directed_graph: bool | None = None,
        *,
        directed: bool | None = None,
    ) -> None:
        if directed_graph is not None and directed is not None:
            raise TypeError("Cannot specify both 'directed_graph' and 'directed'. Use one.")
        resolved = directed_graph if directed_graph is not None else directed
        if resolved is None:
            raise TypeError("StringToGraph requires 'directed_graph' (or 'directed') argument.")

        if not set(input_string).issubset(VALID_INSTRUCTIONS):
            raise ValueError(f"Invalid IsalGraph string: {input_string!r}")

        self._input_string: str = input_string
        self._directed_graph: bool = resolved

        counter = Counter(self._input_string)
        self._max_nodes: int = 1 + counter.get("V", 0) + counter.get("v", 0)

        self._output_graph = SparseGraph(self._max_nodes, self._directed_graph)
        self._cdll = CircularDoublyLinkedList(self._max_nodes)
        self._primary_ptr: int = -1
        self._secondary_ptr: int = -1

    # ------------------------------------------------------------------
    # Public accessors (useful for trace / debugging)
    # ------------------------------------------------------------------

    @property
    def cdll(self) -> CircularDoublyLinkedList:
        """The CDLL after (or during) conversion."""
        return self._cdll

    @property
    def primary_ptr(self) -> int:
        """Current primary pointer (CDLL node index)."""
        return self._primary_ptr

    @property
    def secondary_ptr(self) -> int:
        """Current secondary pointer (CDLL node index)."""
        return self._secondary_ptr

    # ------------------------------------------------------------------
    # Conversion
    # ------------------------------------------------------------------

    def run(
        self, *, trace: bool = False
    ) -> tuple[
        SparseGraph,
        list[tuple[SparseGraph, CircularDoublyLinkedList, int, int, str]],
    ]:
        """Execute the string-to-graph conversion.

        Args:
            trace: If ``True``, collect deep-copied snapshots after each
                instruction for debugging / visualization.

        Returns:
            A 2-tuple ``(graph, trace_list)``.  *trace_list* is empty when
            *trace* is ``False``.
        """
        # ---- initial state: one node, both pointers on it ----
        initial_graph_node = self._output_graph.add_node()
        initial_cdll_node = self._cdll.insert_after(-1, initial_graph_node)
        self._primary_ptr = initial_cdll_node
        self._secondary_ptr = initial_cdll_node

        graph_trace: list[tuple[SparseGraph, CircularDoublyLinkedList, int, int, str]] = []
        if trace:
            graph_trace.append(
                (
                    deepcopy(self._output_graph),
                    deepcopy(self._cdll),
                    self._primary_ptr,
                    self._secondary_ptr,
                    "",
                )
            )

        # ---- process each instruction ----
        for idx, instruction in enumerate(self._input_string):
            self._execute_instruction(instruction)

            if trace:
                graph_trace.append(
                    (
                        deepcopy(self._output_graph),
                        deepcopy(self._cdll),
                        self._primary_ptr,
                        self._secondary_ptr,
                        self._input_string[: idx + 1],
                    )
                )

        return self._output_graph, graph_trace

    # ------------------------------------------------------------------
    # Structured trace
    # ------------------------------------------------------------------

    def _snapshot(
        self,
        step_idx: int,
        instruction: str | None,
        created_edge: Edge | None,
        partial: str,
    ) -> StepSnapshot:
        """Capture the current VM state as a :class:`StepSnapshot`."""
        directed = self._directed_graph
        return StepSnapshot(
            step_idx=step_idx,
            instruction=instruction,
            cdll_node_order=cdll_forward_order(self._cdll),
            primary_node=self._cdll.get_value(self._primary_ptr),
            secondary_node=self._cdll.get_value(self._secondary_ptr),
            active_nodes=tuple(range(self._output_graph.node_count())),
            active_edges=graph_edges(self._output_graph),
            created_edge=(
                None if created_edge is None else normalise_edge(*created_edge, directed=directed)
            ),
            partial_string=partial,
        )

    def _created_edge_for(self, instruction: str) -> Edge | None:
        """Predict the edge *instruction* is about to create, from live VM state.

        Called immediately *before* :meth:`_execute_instruction`, so the
        pointer reads and the ``has_edge`` probe reflect the pre-step
        state. Recording attribution here rather than re-deriving it from
        the token stream is deliberate: ``C``/``c`` between two already
        adjacent nodes is a genuine no-op in IsalGraph, so any counter
        keyed on ``V``/``C`` occurrences desynchronises the first time the
        string revisits an existing edge.

        Args:
            instruction: The single character about to be executed.

        Returns:
            The ``(source, target)`` pair the step will add, or ``None``
            for movement instructions, ``W``, and no-op ``C``/``c``.
        """
        if instruction in ("N", "P", "n", "p", "W"):
            return None

        primary_gn = self._cdll.get_value(self._primary_ptr)
        secondary_gn = self._cdll.get_value(self._secondary_ptr)

        # V/v always allocate a fresh node, so the edge is always new.
        # The new node takes the next contiguous id.
        new_node = self._output_graph.node_count()
        if instruction == "V":
            return (primary_gn, new_node)
        if instruction == "v":
            return (secondary_gn, new_node)

        source, target = (
            (primary_gn, secondary_gn) if instruction == "C" else (secondary_gn, primary_gn)
        )
        if self._output_graph.has_edge(source, target):
            return None  # genuine no-op: the edge already exists
        return (source, target)

    def run_with_trace(self) -> tuple[SparseGraph, AlgorithmTrace]:
        """Execute the conversion, recording a structured :class:`AlgorithmTrace`.

        Semantically identical to ``run()``: the same instruction dispatch
        drives both, so the returned graph equals the one ``run()``
        produces. The difference is the second element, which carries id
        masks over the final graph rather than deep copies of intermediate
        ones.

        Returns:
            A 2-tuple ``(graph, trace)``. The trace holds
            ``len(input_string) + 1`` snapshots and has direction
            ``"s2g"``.
        """
        initial_graph_node = self._output_graph.add_node()
        initial_cdll_node = self._cdll.insert_after(-1, initial_graph_node)
        self._primary_ptr = initial_cdll_node
        self._secondary_ptr = initial_cdll_node

        snapshots: list[StepSnapshot] = [self._snapshot(0, None, None, "")]

        for idx, instruction in enumerate(self._input_string):
            created = self._created_edge_for(instruction)
            self._execute_instruction(instruction)
            snapshots.append(
                self._snapshot(idx + 1, instruction, created, self._input_string[: idx + 1])
            )

        trace = AlgorithmTrace(
            direction="s2g",
            directed=self._directed_graph,
            final_graph=graph_to_dict(self._output_graph),
            snapshots=tuple(snapshots),
        )
        return self._output_graph, trace

    # ------------------------------------------------------------------
    # Instruction dispatch
    # ------------------------------------------------------------------

    def _execute_instruction(self, instruction: str) -> None:
        """Execute a single IsalGraph instruction, mutating internal state."""
        if instruction == "N":
            self._primary_ptr = self._cdll.next_node(self._primary_ptr)

        elif instruction == "P":
            self._primary_ptr = self._cdll.prev_node(self._primary_ptr)

        elif instruction == "n":
            self._secondary_ptr = self._cdll.next_node(self._secondary_ptr)

        elif instruction == "p":
            self._secondary_ptr = self._cdll.prev_node(self._secondary_ptr)

        elif instruction == "V":
            new_node = self._output_graph.add_node()
            # BUG FIX B6: was add_edge(self._primary_ptr, new_node).
            # self._primary_ptr is a CDLL index, not a graph node.
            primary_graph_node = self._cdll.get_value(self._primary_ptr)
            self._output_graph.add_edge(primary_graph_node, new_node)
            self._cdll.insert_after(self._primary_ptr, new_node)

        elif instruction == "v":
            new_node = self._output_graph.add_node()
            # BUG FIX B6: same fix for secondary pointer.
            secondary_graph_node = self._cdll.get_value(self._secondary_ptr)
            self._output_graph.add_edge(secondary_graph_node, new_node)
            self._cdll.insert_after(self._secondary_ptr, new_node)

        elif instruction == "C":
            # BUG FIX B6: was add_edge(self._primary_ptr, self._secondary_ptr).
            primary_graph_node = self._cdll.get_value(self._primary_ptr)
            secondary_graph_node = self._cdll.get_value(self._secondary_ptr)
            self._output_graph.add_edge(primary_graph_node, secondary_graph_node)

        elif instruction == "c":
            # BUG FIX B6: same fix, reversed direction.
            primary_graph_node = self._cdll.get_value(self._primary_ptr)
            secondary_graph_node = self._cdll.get_value(self._secondary_ptr)
            self._output_graph.add_edge(secondary_graph_node, primary_graph_node)

        elif instruction == "W":
            pass  # no-op
