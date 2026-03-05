from typing import List, Set


class SparseGraph:
    """
    A high-performance representation of a large, sparse graph using
    adjacency sets stored in a preallocated list.

    This data structure is optimized for:
    1. Fast edge existence checks
    2. Fast edge insertion
    3. Fast node insertion

    Nodes are identified by contiguous integer IDs starting from 0.
    The maximum number of nodes must be known at construction time.

    The graph can be used as either directed or undirected depending
    on how edges are inserted.
    """

    __slots__ = ("_adjacency", "_max_nodes", "_node_count", "_edge_count", "_directed_graph")

    def __init__(self, max_nodes: int, directed_graph: bool) -> None:
        """
        Initialize the graph with a fixed maximum number of nodes.

        :param max_nodes:
            The maximum number of nodes the graph can contain.
            This value is used to preallocate internal storage
            for maximum performance.
        :type max_nodes: int
        """
        # Store the maximum number of nodes
        self._max_nodes: int = max_nodes

        # Store the graph type
        self._directed_graph: bool = directed_graph

        # Preallocate the adjacency list as a list of empty sets
        # Each index corresponds to a node ID
        self._adjacency: List[Set[int]] = [
            set() for _ in range(max_nodes)
        ]

        # Track the number of nodes currently in use
        self._node_count: int = 0

        # Track the number of edges currently in use
        self._edge_count: int = 1

    def directed(self) -> bool:
        """

        :return:
        Whether the graph is directed or undirected.
        """
        return self._directed_graph

    def add_node(self) -> int:
        """
        Add a new node to the graph.

        Nodes are assigned sequential integer IDs starting from 0.

        :return:
            The integer ID of the newly added node.
        :rtype: int

        :raises RuntimeError:
            If the maximum number of nodes has already been reached.
        """
        # Check whether we have exceeded the allowed number of nodes
        if self._node_count >= self._max_nodes:
            raise RuntimeError(f"Maximum number of nodes reached: {self._max_nodes}")

        # Assign the next available node ID
        node_id: int = self._node_count

        # Increment the active node counter
        self._node_count += 1

        # Return the newly created node ID
        return node_id

    def add_edge(self, source: int, target: int) -> None:
        """
        Add a directed edge from ``source`` to ``target``.

        For undirected graphs, call this method twice:
        ``add_edge(u, v)`` and ``add_edge(v, u)``.

        :param source:
            The source node ID.
        :type source: int
        :param target:
            The target node ID.
        :type target: int

        :raises IndexError:
            If either node ID is invalid.
        """
        # Validate that the source node exists
        if source < 0 or source >= self._node_count:
            raise IndexError(f"Invalid source node ID: {source}")

        # Validate that the target node exists
        if target < 0 or target >= self._node_count:
            raise IndexError(f"Invalid target node ID: {target}")

        # Insert the edge into the adjacency set
        # This operation is O(1) on average
        self._adjacency[source].add(target)
        # Increment the active edge counter
        self._edge_count += 1

        # Add the symmetric edge for undirected graphs
        if not self._directed_graph:
            self._adjacency[target].add(source)
            # Increment the active edge counter
            self._edge_count += 1

    def has_edge(self, source: int, target: int) -> bool:
        """
        Test whether a directed edge from ``source`` to ``target`` exists.

        This operation is optimized for speed and runs in O(1) average time.

        :param source:
            The source node ID.
        :type source: int
        :param target:
            The target node ID.
        :type target: int

        :return:
            ``True`` if the edge exists, ``False`` otherwise.
        :rtype: bool

        :raises IndexError:
            If either node ID is invalid.
        """
        # Validate the source node ID
        if source < 0 or source >= self._node_count:
            raise IndexError(f"Invalid source node ID: {source}")

        # Validate the target node ID
        if target < 0 or target >= self._node_count:
            raise IndexError(f"Invalid target node ID: {target}")

        # Perform a constant-time membership test
        return target in self._adjacency[source]

    def neighbors(self, node: int) -> Set[int]:
        """
        Return the set of neighbors for a given node.

        The returned set should be treated as read-only
        to avoid corrupting the graph structure.

        :param node:
            The node ID whose neighbors are requested.
        :type node: int

        :return:
            A set containing the neighboring node IDs.
        :rtype: Set[int]

        :raises IndexError:
            If the node ID is invalid.
        """
        # Validate the node ID
        if node < 0 or node >= self._node_count:
            raise IndexError(f"Invalid node ID: {node}")

        # Return the adjacency set for the node
        return self._adjacency[node]

    def node_count(self) -> int:
        """
        Return the number of nodes currently in the graph.

        :return:
            The number of active nodes.
        :rtype: int
        """
        # Return the current node count
        return self._node_count

    def edge_count(self) -> int:
        """
        Return the number of edges currently in the graph.

        :return:
            The number of active edges.
        :rtype: int
        """
        # Return the current edge count
        return self._edge_count

    def max_nodes(self) -> int:
        """
        Return the maximum number of nodes supported by the graph.

        :return:
            The maximum node capacity.
        :rtype: int
        """
        # Return the preconfigured maximum number of nodes
        return self._max_nodes

    def is_isomorphic(self, other: "SparseGraph") -> bool:
        """
        Check whether this graph is structurally isomorphic to another graph.
        If not, print a diagnostic message describing the difference.
        """
        if not isinstance(other, SparseGraph):
            print("Graphs are not isomorphic: other object is not a SparseGraph.")
            return False

        if self._directed_graph != other._directed_graph:
            print(
                "Graphs are not isomorphic: one graph is directed and the other is not."
            )
            return False

        if self._node_count != other._node_count:
            print(
                f"Graphs are not isomorphic: different number of nodes "
                f"({self._node_count} vs {other._node_count})."
            )
            return False

        if self._edge_count != other._edge_count:
            print(
                f"Graphs are not isomorphic: different number of edges "
                f"({self._edge_count} vs {other._edge_count})."
            )
            return False

        n = self._node_count

        # Compute degrees
        self_degrees = [len(self._adjacency[u]) for u in range(n)]
        other_degrees = [len(other._adjacency[u]) for u in range(n)]

        # Degree multiset check
        if sorted(self_degrees) != sorted(other_degrees):
            print("Graphs are not isomorphic: degree distributions differ.")
            print("Self degrees :", self_degrees)
            print("Other degrees:", other_degrees)
            return False

        # Order nodes by degree (high → low) for pruning
        self_nodes = sorted(range(n), key=lambda u: self_degrees[u], reverse=True)
        other_nodes = sorted(range(n), key=lambda u: other_degrees[u], reverse=True)

        mapping = {}
        used = set()

        def backtrack(i: int) -> bool:
            if i == n:
                return True

            u = self_nodes[i]

            for v in other_nodes:
                if v in used:
                    continue

                if self_degrees[u] != other_degrees[v]:
                    continue

                # Check consistency with already-mapped nodes
                for u2, v2 in mapping.items():
                    # Edge u -> u2 consistency
                    if (u2 in self._adjacency[u]) != (v2 in other._adjacency[v]):
                        print(
                            "Graphs are not isomorphic: adjacency mismatch.\n"
                            f"Self: edge ({u} -> {u2}) = {u2 in self._adjacency[u]}\n"
                            f"Other: edge ({v} -> {v2}) = {v2 in other._adjacency[v]}"
                        )
                        return False

                    # For directed graphs, also check reverse direction
                    if self._directed_graph:
                        if (u in self._adjacency[u2]) != (v in other._adjacency[v2]):
                            print(
                                "Graphs are not isomorphic: reverse adjacency mismatch.\n"
                                f"Self: edge ({u2} -> {u}) = {u in self._adjacency[u2]}\n"
                                f"Other: edge ({v2} -> {v}) = {v in other._adjacency[v2]}"
                            )
                            return False

                # Extend mapping
                mapping[u] = v
                used.add(v)

                if backtrack(i + 1):
                    return True

                # Backtrack
                del mapping[u]
                used.remove(v)

            # No valid match for this node
            print(
                "Graphs are not isomorphic: no valid mapping found for node.\n"
                f"Self node {u} (degree {self_degrees[u]}) "
                f"cannot be matched with any remaining node in other graph."
            )
            return False

        return backtrack(0)


    def __str__(self) -> str:
        """
        Return a string representation of the graph.
        :return:
        A string representation of the graph.
        :rtype: str
        """
        return f"{self.__class__.__name__}(max_nodes={self._max_nodes}, directed_graph={self._directed_graph}, node_count={self._node_count}), adjacency={self._adjacency})"