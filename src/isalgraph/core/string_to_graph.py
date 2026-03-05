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

    def __init__(self, input_string: str, directed_graph: bool) -> None:
        if not set(input_string).issubset(VALID_INSTRUCTIONS):
            raise ValueError(f"Invalid IsalGraph string: {input_string!r}")

        self._input_string: str = input_string
        self._directed_graph: bool = directed_graph

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
