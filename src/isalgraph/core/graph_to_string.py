"""SparseGraph to IsalGraph string converter.

Migration of the original ``graphtostring.py`` with the following bug fixes:

* **B2**: ``generate_pairs_sorted_by_sum`` sorted by ``a + b`` (algebraic
  sum).  Fixed to sort by ``|a| + |b|`` (total displacement cost), which
  is the number of pointer-movement instructions that will be emitted.

* **B3**: The ``while`` loop used ``and`` (terminates when *either*
  node-count or edge-count reaches zero).  Fixed to ``or`` so the loop
  continues until *both* all nodes and all edges have been inserted.

* **B4**: After emitting N/P/n/p instructions the *actual* pointer fields
  (``self._primary_ptr``, ``self._secondary_ptr``) were never updated to
  the tentative positions.  Fixed: pointers are updated at the end of
  each successful operation.

* **B5**: Debug ``print(self._output_string)`` removed from main loop.

* **B7**: ``cdll.insert_after(tentative_primary_output_node, ...)`` passed
  a *graph* node index where a *CDLL* node index is required.  Fixed to
  use the tentative CDLL pointer instead.

* **B8**: The V/v branch did not check whether the candidate input-graph
  neighbor was already present in the output graph (mapped in ``_i2o``).
  It only checked whether the *edge* was present, which caused it to
  call ``add_node()`` for nodes that already existed.  Fixed: V/v now
  filters candidates to those **not yet in** ``_i2o``.
"""

from __future__ import annotations

from copy import deepcopy

from isalgraph.core.cdll import CircularDoublyLinkedList
from isalgraph.core.sparse_graph import SparseGraph


def generate_pairs_sorted_by_sum(m: int) -> list[tuple[int, int]]:
    """Return all integer pairs (a, b) with a, b in [-m, m],
    sorted by |a| + |b| (total displacement cost).

    Within the same cost, pairs are further sorted by (|a|, |b|) for
    determinism, and then by (a, b) lexicographically for tie-breaking.

    Args:
        m: Positive integer defining the range bounds.

    Returns:
        Sorted list of (a, b) tuples.

    Raises:
        ValueError: If *m* is not positive.
    """
    if m <= 0:
        raise ValueError("m must be a positive integer.")

    pairs: list[tuple[int, int]] = [(a, b) for a in range(-m, m + 1) for b in range(-m, m + 1)]
    # BUG FIX B2: was ``pair[0] + pair[1]``.
    pairs.sort(key=lambda pair: (abs(pair[0]) + abs(pair[1]), abs(pair[0]), pair))
    return pairs


class GraphToString:
    """Convert a ``SparseGraph`` into an IsalGraph instruction string.

    The greedy algorithm searches for the least-cost pointer displacement
    that enables either a V/v (node+edge insertion) or C/c (edge-only
    insertion) at each step.

    Args:
        input_graph: The graph to convert.
    """

    __slots__ = (
        "_input_graph",
        "_output_string",
        "_cdll",
        "_primary_ptr",
        "_secondary_ptr",
        "_output_graph",
        "_i2o",
        "_o2i",
    )

    def __init__(self, input_graph: SparseGraph) -> None:
        self._input_graph = input_graph
        self._output_string: str = ""
        self._cdll = CircularDoublyLinkedList(input_graph.max_nodes())
        self._primary_ptr: int = -1
        self._secondary_ptr: int = -1
        self._output_graph = SparseGraph(input_graph.max_nodes(), input_graph.directed())
        self._i2o: dict[int, int] = {}
        self._o2i: dict[int, int] = {}

    # ------------------------------------------------------------------
    # Conversion
    # ------------------------------------------------------------------

    def run(
        self,
        initial_node: int,
        *,
        trace: bool = False,
    ) -> tuple[
        str,
        list[tuple[SparseGraph, CircularDoublyLinkedList, int, int, str]],
    ]:
        """Execute the graph-to-string conversion.

        Args:
            initial_node: Index of the starting node in the *input* graph.
            trace: If ``True``, collect deep-copied snapshots for
                debugging/visualization.

        Returns:
            A 2-tuple ``(instruction_string, trace_list)``.
        """
        if initial_node < 0 or initial_node >= self._input_graph.node_count():
            raise ValueError("Initial node out of range")

        self._check_reachability(initial_node)

        # ---- initial state ----
        new_initial_node = self._output_graph.add_node()
        new_initial_cdll_node = self._cdll.insert_after(-1, new_initial_node)
        self._primary_ptr = new_initial_cdll_node
        self._secondary_ptr = new_initial_cdll_node
        self._i2o[initial_node] = new_initial_node
        self._o2i[new_initial_node] = initial_node

        graph_trace: list[tuple[SparseGraph, CircularDoublyLinkedList, int, int, str]] = []

        num_nodes_to_insert: int = self._input_graph.node_count() - 1
        num_edges_to_insert: int = self._input_graph.logical_edge_count()

        # BUG FIX B3: was ``and``; must continue while nodes OR edges remain.
        while num_nodes_to_insert > 0 or num_edges_to_insert > 0:
            # BUG FIX B5: removed debug print().

            if trace:
                graph_trace.append(
                    (
                        deepcopy(self._output_graph),
                        deepcopy(self._cdll),
                        self._primary_ptr,
                        self._secondary_ptr,
                        self._output_string,
                    )
                )

            current_node_count = self._output_graph.node_count()
            pairs = generate_pairs_sorted_by_sum(current_node_count)

            found = False
            for num_primary_moves, num_secondary_moves in pairs:
                # ---- tentative primary position ----
                tent_pri_ptr = self._move_pointer(self._primary_ptr, num_primary_moves)
                tent_pri_out = self._cdll.get_value(tent_pri_ptr)
                tent_pri_in = self._o2i[tent_pri_out]

                # -- V: insert new node via primary? --
                if num_nodes_to_insert > 0:
                    candidate = self._find_new_neighbor(tent_pri_in)
                    if candidate is not None:
                        new_out = self._output_graph.add_node()
                        num_nodes_to_insert -= 1
                        self._i2o[candidate] = new_out
                        self._o2i[new_out] = candidate
                        self._output_graph.add_edge(tent_pri_out, new_out)
                        num_edges_to_insert -= 1
                        # BUG FIX B7: was insert_after(tent_pri_out, ...)
                        # which is a *graph* node index; must be CDLL index.
                        self._cdll.insert_after(tent_pri_ptr, new_out)
                        self._emit_primary_moves(num_primary_moves)
                        self._output_string += "V"
                        # BUG FIX B4: update actual pointer.
                        self._primary_ptr = tent_pri_ptr
                        found = True
                        break

                # ---- tentative secondary position ----
                tent_sec_ptr = self._move_pointer(self._secondary_ptr, num_secondary_moves)
                tent_sec_out = self._cdll.get_value(tent_sec_ptr)
                tent_sec_in = self._o2i[tent_sec_out]

                # -- v: insert new node via secondary? --
                if num_nodes_to_insert > 0:
                    candidate = self._find_new_neighbor(tent_sec_in)
                    if candidate is not None:
                        new_out = self._output_graph.add_node()
                        num_nodes_to_insert -= 1
                        self._i2o[candidate] = new_out
                        self._o2i[new_out] = candidate
                        self._output_graph.add_edge(tent_sec_out, new_out)
                        num_edges_to_insert -= 1
                        # BUG FIX B7: same fix for secondary.
                        self._cdll.insert_after(tent_sec_ptr, new_out)
                        self._emit_secondary_moves(num_secondary_moves)
                        self._output_string += "v"
                        # BUG FIX B4: update actual pointer.
                        self._secondary_ptr = tent_sec_ptr
                        found = True
                        break

                # -- C: edge primary -> secondary? --
                if tent_sec_in in self._input_graph.neighbors(
                    tent_pri_in
                ) and tent_sec_out not in self._output_graph.neighbors(tent_pri_out):
                    self._output_graph.add_edge(tent_pri_out, tent_sec_out)
                    num_edges_to_insert -= 1
                    self._emit_primary_moves(num_primary_moves)
                    self._emit_secondary_moves(num_secondary_moves)
                    self._output_string += "C"
                    # BUG FIX B4: update both pointers.
                    self._primary_ptr = tent_pri_ptr
                    self._secondary_ptr = tent_sec_ptr
                    found = True
                    break

                # -- c: edge secondary -> primary? (directed only) --
                if (
                    self._input_graph.directed()
                    and tent_pri_in in self._input_graph.neighbors(tent_sec_in)
                    and tent_pri_out not in self._output_graph.neighbors(tent_sec_out)
                ):
                    self._output_graph.add_edge(tent_sec_out, tent_pri_out)
                    num_edges_to_insert -= 1
                    self._emit_primary_moves(num_primary_moves)
                    self._emit_secondary_moves(num_secondary_moves)
                    self._output_string += "c"
                    # BUG FIX B4: update both pointers.
                    self._primary_ptr = tent_pri_ptr
                    self._secondary_ptr = tent_sec_ptr
                    found = True
                    break

            if not found:
                raise RuntimeError(
                    "GraphToString: no valid operation found. "
                    f"Remaining: {num_nodes_to_insert} nodes, "
                    f"{num_edges_to_insert} edges. "
                    "This indicates an algorithmic error."
                )

        if trace:
            graph_trace.append(
                (
                    deepcopy(self._output_graph),
                    deepcopy(self._cdll),
                    self._primary_ptr,
                    self._secondary_ptr,
                    self._output_string,
                )
            )

        return self._output_string, graph_trace

    # ------------------------------------------------------------------
    # Internal helpers
    # ------------------------------------------------------------------

    def _move_pointer(self, ptr: int, steps: int) -> int:
        """Walk *ptr* through the CDLL by *steps* (positive = next, negative = prev)."""
        if steps >= 0:
            for _ in range(steps):
                ptr = self._cdll.next_node(ptr)
        else:
            for _ in range(-steps):
                ptr = self._cdll.prev_node(ptr)
        return ptr

    def _find_new_neighbor(self, input_node: int) -> int | None:
        """Find a neighbor of *input_node* in the input graph that has
        **not yet been added** to the output graph.

        BUG FIX B8: the original code checked whether the *edge* existed
        in the output graph rather than whether the *node* had been
        created.  This caused duplicate node creation for nodes that
        existed but lacked a specific edge.

        Returns:
            An input-graph node ID, or ``None`` if all neighbors are
            already in the output graph.
        """
        for neighbor in self._input_graph.neighbors(input_node):
            if neighbor not in self._i2o:
                return neighbor
        return None

    def _emit_primary_moves(self, steps: int) -> None:
        """Append N or P instructions for *steps* primary pointer movements."""
        if steps >= 0:
            self._output_string += "N" * steps
        else:
            self._output_string += "P" * (-steps)

    def _emit_secondary_moves(self, steps: int) -> None:
        """Append n or p instructions for *steps* secondary pointer movements."""
        if steps >= 0:
            self._output_string += "n" * steps
        else:
            self._output_string += "p" * (-steps)

    def _check_reachability(self, initial_node: int) -> None:
        """Verify all nodes are reachable from *initial_node*.

        For undirected graphs, the graph must be connected.
        For directed graphs, all nodes must be reachable via outgoing
        edges from *initial_node* (the V/v instructions only create
        edges ``existing -> new``).

        Raises:
            ValueError: If unreachable nodes are detected.
        """
        n = self._input_graph.node_count()
        if n <= 1:
            return

        visited: set[int] = set()
        stack: list[int] = [initial_node]
        while stack:
            node = stack.pop()
            if node in visited:
                continue
            visited.add(node)
            for neighbor in self._input_graph.neighbors(node):
                if neighbor not in visited:
                    stack.append(neighbor)

        if len(visited) != n:
            unreachable = set(range(n)) - visited
            raise ValueError(
                f"GraphToString requires all nodes to be reachable from "
                f"initial_node={initial_node} via outgoing edges. "
                f"Unreachable nodes: {unreachable}. "
                f"For directed graphs, ensure all nodes are reachable "
                f"from the start node. For undirected graphs, ensure "
                f"the graph is connected."
            )
