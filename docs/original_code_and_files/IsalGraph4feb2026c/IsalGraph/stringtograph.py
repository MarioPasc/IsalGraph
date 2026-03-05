from copy import deepcopy
from typing import List, Set, Tuple

from sparsegraph import SparseGraph
from collections import Counter
from circulardoublylinkedlist import CircularDoublyLinkedList


class StringToGraph:
    """
    IsalGraph conversion algorithm from string to SparseGraph.



    """

    def __init__(self, input_string: str, directed_graph: bool) -> None:
        """
        Initialize the converter with the input string.

        :param input_string:
            The input string to be converted.
        :type input_string: str

        :raises RuntimeError:
            If the input is not a valid IsalGraph string.
        """

        valid_instructions = {'N', 'n', 'P', 'p', 'c', 'C', 'v', 'V', 'W'}

        if not set(input_string).issubset(valid_instructions):
            raise RuntimeError('Invalid IsalGraph string')

        # Store the input string
        self.input_string: str = input_string


        # Store the graph type
        self._directed_graph: bool = directed_graph

        # Count the character occurrences in the input string
        counter = Counter(self.input_string)

        # Store the maximum number of nodes of the output graph by counting vertex insertion instructions
        self._max_nodes: int = 1 + counter['v'] + counter['V']

        # Create the output graph
        self._output_graph = SparseGraph(self._max_nodes, self._directed_graph)

        # Create the circular doubly linked list
        self._cdll = CircularDoublyLinkedList(self._max_nodes)

        # Create the pointers
        self._primary_ptr: int = -1
        self._secondary_ptr: int = -1

    def cdll(self) -> CircularDoublyLinkedList:
        """
        Return the circular doubly-linked list of the input string.

        :return: The circular doubly-linked list.
        :rtype: CircularDoublyLinkedList
        """
        return self._cdll

    def primary_ptr(self) -> int:
        """
        Return the primary pointer of the input string.
        :return: The primary pointer of the input string.
        :rtype: int
        """
        return self._primary_ptr

    def secondary_ptr(self) -> int:
        """
        Return the secondary pointer of the input string.
        :return: The secondary pointer of the input string.
        :rtype: int
        """
        return self._secondary_ptr

    def run(self, trace: bool = False) -> Tuple[SparseGraph, List[Tuple[SparseGraph, CircularDoublyLinkedList, int, int, str]]]:
        """
        Run the conversion from string to SparseGraph.

        :param trace:
            Whether to generate a trace of the intermediate SparseGraph instances.

        :return:
            The output SparseGraph, with optional conversion trace.
        :rtype: SparseGraph
        """

        # Set up the initial state of the graph
        initial_graph_node = self._output_graph.add_node()
        initial_cdll_node = self._cdll.insert_after(-1, initial_graph_node)
        self._primary_ptr = initial_cdll_node
        self._secondary_ptr = initial_cdll_node
        if trace:
            graph_trace = [(deepcopy(self._output_graph), deepcopy(self._cdll), self._primary_ptr, self._secondary_ptr, "")]
        else:
            graph_trace = []


        # Iterate for all characters (instructions) of the input string

        for ndx_instruction, instruction in enumerate(self.input_string):
            if instruction == 'N':
                self._primary_ptr = self._cdll.next_node(self._primary_ptr)
            elif instruction == 'n':
                self._secondary_ptr = self._cdll.next_node(self._secondary_ptr)
            elif instruction == 'P':
                self._primary_ptr = self._cdll.prev_node(self._primary_ptr)
            elif instruction == 'p':
                self._secondary_ptr = self._cdll.prev_node(self._secondary_ptr)
            elif instruction == 'V':
                new_node = self._output_graph.add_node()
                self._output_graph.add_edge(self._primary_ptr, new_node)
                self._cdll.insert_after(self._primary_ptr, new_node)
            elif instruction == 'v':
                new_node = self._output_graph.add_node()
                self._output_graph.add_edge(self._secondary_ptr, new_node)
                self._cdll.insert_after(self._secondary_ptr, new_node)
            elif instruction == 'C':
                self._output_graph.add_edge(self._primary_ptr, self._secondary_ptr)
            elif instruction == 'c':
                self._output_graph.add_edge(self._secondary_ptr, self._primary_ptr)
            elif instruction == 'W':
                pass
            else:
                raise RuntimeError('Invalid IsalGraph string')

            if trace:
                graph_trace.append((deepcopy(self._output_graph), deepcopy(self._cdll), self._primary_ptr, self._secondary_ptr, self.input_string[:ndx_instruction+1]))

        return self._output_graph, graph_trace


