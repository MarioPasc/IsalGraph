from copy import deepcopy
from typing import List, Set, Tuple

from sparsegraph import SparseGraph
from collections import Counter
from circulardoublylinkedlist import CircularDoublyLinkedList


def generate_pairs_sorted_by_sum(m: int) -> List[Tuple[int, int]]:
    """
    Generate all integer pairs (a, b) such that a, b ∈ {-M, ..., 0, ..., M},
    and return them sorted in increasing order of a + b.

    Args:
        m (int): A positive integer defining the range bounds.

    Returns:
        List[Tuple[int, int]]: A list of two-element integer tuples (a, b),
        sorted by the value of a + b in increasing order.

    Raises:
        ValueError: If M is not a positive integer.
    """
    # Check that the input M is a positive integer.
    if m <= 0:
        # Raise an error if the input constraint is violated.
        raise ValueError("M must be a positive integer.")

    # Initialize an empty list to store the resulting pairs.
    pairs: List[Tuple[int, int]] = []

    # Iterate over all possible values of a in the range [-M, M].
    for a in range(-m, m + 1):
        # Iterate over all possible values of b in the range [-M, M].
        for b in range(-m, m + 1):
            # Append the current pair (a, b) to the list.
            pairs.append((a, b))

    # Sort the list of pairs by the sum a + b in increasing order.
    pairs.sort(key=lambda pair: pair[0] + pair[1])

    # Return the sorted list of pairs.
    return pairs


class GraphToString:
    """
    IsalGraph conversion algorithm from SparseGraph to string.

    """

    def __init__(self, input_graph: SparseGraph) -> None:
        """

        :param input_graph: The input SparseGraph to be converted.
        """

        self._input_graph = input_graph

        # Create the output string
        self._output_string = ""

        # Create the circular doubly linked list
        self._cdll = CircularDoublyLinkedList(input_graph.max_nodes())

        # Create the pointers
        self._primary_ptr: int = -1
        self._secondary_ptr: int = -1

        # Create the output graph
        self._output_graph = SparseGraph(input_graph.max_nodes(), input_graph.directed())

        # Dictionary to convert input graph node indices to output graph node indices
        self._i2o = dict()
        # Dictionary to convert output graph node indices to input graph node indices
        self._o2i = dict()

    def run(self, initial_node: int, trace: bool = False) -> Tuple[
        str, List[Tuple[SparseGraph, CircularDoublyLinkedList, int, int, str]]]:
        """
        Run the conversion from SparseGraph to string.

        :param initial_node: The initial node in the input graph to start the conversion from.
        :param trace: Whether the conversion trace must be returned
        :return: The converted string, with the optional conversion trace.
        """

        if initial_node < 0 or initial_node >= self._input_graph.max_nodes():
            raise ValueError("Initial node out of range")

        # Create the initial node of the output graph
        new_initial_node = self._output_graph.add_node()
        # self._output_graph.add_edge(self._primary_ptr, new_initial_node)
        new_initial_cdll_node = self._cdll.insert_after(-1, new_initial_node)
        self._primary_ptr = new_initial_cdll_node
        self._secondary_ptr = new_initial_cdll_node
        self._i2o[initial_node] = new_initial_node
        self._o2i[new_initial_node] = initial_node

        # Initialize the conversion trace
        graph_trace = []

        # Compute the required number of node and edge insertion operations
        num_nodes_to_insert : int = self._input_graph.node_count() - 1
        if self._input_graph.directed():
            num_edges_to_insert: int = self._input_graph.edge_count()
        else:
            num_edges_to_insert: int = self._input_graph.edge_count() // 2

        # Insert more instructions while the output graph is not complete
        while num_nodes_to_insert > 0 and num_edges_to_insert > 0:
            print(self._output_string)
            if trace:
                graph_trace.append( (deepcopy(self._output_graph), deepcopy(self._cdll), self._primary_ptr, self._secondary_ptr, self._output_string) )
            # Find the current number of nodes in the output graph
            current_number_nodes = self._output_graph.node_count()
            # Generate the set of all possible pairs of pointer displacements (a, b)
            # where each are signed integers (positive is next, negative is previous)
            # for both pointers (a for the displacement of primary pointer,
            # b for the displacement of the secondary pointer).
            pairs_list = generate_pairs_sorted_by_sum(current_number_nodes)
            for (num_primary_moves, num_secondary_moves) in pairs_list:

                # Compute the final position of the primary pointer for the
                # considered number of primary pointer moves
                tentative_primary_ptr : int = self._primary_ptr
                if num_primary_moves >=0 :
                    for ndx_move in range(num_primary_moves):
                        tentative_primary_ptr = self._cdll.next_node(tentative_primary_ptr)
                else:
                    for ndx_move in range(-num_primary_moves):
                        tentative_primary_ptr = self._cdll.prev_node(tentative_primary_ptr)
                tentative_primary_output_node = self._cdll.get_value(tentative_primary_ptr)
                tentative_primary_input_node = self._o2i[tentative_primary_output_node]
                # The neighbors of the node pointed by the primary pointer after the moves
                # in the input graph
                inserted_primary_neighbors = {self._o2i[output_neighbor] for output_neighbor in
                 self._output_graph.neighbors(tentative_primary_output_node)}
                # The nodes in the input graph that are neighbors of
                # the node pointed by the primary pointer after the moves
                # in the input graph but not in the output graph
                not_inserted_primary_neighbors = self._input_graph.neighbors(tentative_primary_input_node) - inserted_primary_neighbors
                # Check whether an input neighbor is not yet in the output graph
                if bool(not_inserted_primary_neighbors):
                    # Insert a new node in the output graph
                    new_input_node = not_inserted_primary_neighbors.pop()
                    new_output_node = self._output_graph.add_node()
                    num_nodes_to_insert -= 1
                    self._i2o[new_input_node] = new_output_node
                    self._o2i[new_output_node] = new_input_node
                    # Insert a new edge to the new node in the output graph
                    self._output_graph.add_edge(tentative_primary_output_node, new_output_node)
                    num_edges_to_insert -= 1
                    # Insert the new node into the circular doubly linked list of output nodes
                    self._cdll.insert_after(tentative_primary_output_node, new_output_node)
                    # Update the output string
                    if num_primary_moves >= 0:
                        for ndx_move in range(num_primary_moves):
                            self._output_string += "N"
                    else:
                        for ndx_move in range(-num_primary_moves):
                            self._output_string += "P"
                    self._output_string += "V"
                    # Go to the next instruction
                    break


                # Compute the final position of the secondary pointer for the
                # considered number of secondary pointer moves
                tentative_secondary_ptr : int = self._secondary_ptr
                if num_secondary_moves >=0 :
                    for ndx_move in range(num_secondary_moves):
                        tentative_secondary_ptr = self._cdll.next_node(tentative_secondary_ptr)
                else:
                    for ndx_move in range(-num_secondary_moves):
                        tentative_secondary_ptr = self._cdll.prev_node(tentative_secondary_ptr)
                tentative_secondary_output_node = self._cdll.get_value(tentative_secondary_ptr)
                tentative_secondary_input_node = self._o2i[tentative_secondary_output_node]
                # The neighbors of the node pointed by the secondary pointer after the moves
                # in the input graph
                inserted_secondary_neighbors = {self._o2i[output_neighbor] for output_neighbor in
                 self._output_graph.neighbors(tentative_secondary_output_node)}
                # The nodes in the input graph that are neighbors of
                # the node pointed by the secondary pointer after the moves
                # in the input graph but not in the output graph
                not_inserted_secondary_neighbors = self._input_graph.neighbors(tentative_secondary_input_node) - inserted_secondary_neighbors
                # Check whether an input neighbor is not yet in the output graph
                if bool(not_inserted_secondary_neighbors):
                    # Insert a new node in the output graph
                    new_input_node = not_inserted_secondary_neighbors.pop()
                    new_output_node = self._output_graph.add_node()
                    num_nodes_to_insert -= 1
                    self._i2o[new_input_node] = new_output_node
                    self._o2i[new_output_node] = new_input_node
                    # Insert a new edge to the new node in the output graph
                    self._output_graph.add_edge(tentative_secondary_output_node, new_output_node)
                    num_edges_to_insert -= 1
                    # Insert the new node into the circular doubly linked list of output nodes
                    self._cdll.insert_after(tentative_secondary_output_node, new_output_node)
                    # Update the output string
                    if num_secondary_moves >= 0:
                        for ndx_move in range(num_secondary_moves):
                            self._output_string += "n"
                    else:
                        for ndx_move in range(-num_secondary_moves):
                            self._output_string += "p"
                    self._output_string += "v"
                    # Go to the next instruction
                    break

                # Check whether there is an edge between the tentative positions of the primary
                # and secondary pointers which has not been inserted into the output graph yet.
                if (tentative_secondary_input_node in self._input_graph.neighbors(tentative_primary_input_node) and
                        tentative_secondary_output_node not in self._output_graph.neighbors(tentative_primary_output_node)):
                    # Insert a new edge to the new node in the output graph
                    self._output_graph.add_edge(tentative_primary_output_node, tentative_secondary_output_node)
                    num_edges_to_insert -= 1
                    if num_primary_moves >= 0:
                        for ndx_move in range(num_primary_moves):
                            self._output_string += "N"
                    else:
                        for ndx_move in range(-num_primary_moves):
                            self._output_string += "P"
                    if num_secondary_moves >= 0:
                        for ndx_move in range(num_secondary_moves):
                            self._output_string += "n"
                    else:
                        for ndx_move in range(-num_secondary_moves):
                            self._output_string += "p"
                    self._output_string += "C"
                    # Go to the next instruction
                    break

                # Check whether there is an edge between the tentative positions of the secondary
                # and primary pointers which has not been inserted into the output graph yet.
                # For undirected graphs this case is already managed by the C instruction.
                if self._input_graph.directed():
                    if (tentative_primary_input_node in self._input_graph.neighbors(tentative_secondary_input_node) and
                            tentative_primary_output_node not in self._output_graph.neighbors(tentative_secondary_output_node)):
                        # Insert a new edge to the new node in the output graph
                        self._output_graph.add_edge(tentative_secondary_output_node, tentative_primary_output_node)
                        num_edges_to_insert -= 1
                        if num_primary_moves >= 0:
                            for ndx_move in range(num_primary_moves):
                                self._output_string += "N"
                        else:
                            for ndx_move in range(-num_primary_moves):
                                self._output_string += "P"
                        if num_secondary_moves >= 0:
                            for ndx_move in range(num_secondary_moves):
                                self._output_string += "n"
                        else:
                            for ndx_move in range(-num_secondary_moves):
                                self._output_string += "p"
                        self._output_string += "c"
                        # Go to the next instruction
                        break

        if trace:
            graph_trace.append(
                (deepcopy(self._output_graph), deepcopy(self._cdll), self._primary_ptr, self._secondary_ptr, self._output_string))

        return self._output_string, graph_trace



