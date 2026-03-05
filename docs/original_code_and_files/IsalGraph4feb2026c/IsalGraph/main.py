from sparsegraph import SparseGraph
from circulardoublylinkedlist import CircularDoublyLinkedList
from stringtograph import StringToGraph
from plotgraphlist import trace_to_pptx
from graphtostring import GraphToString

def basic_tests() -> None:
    """
    Sample usage of the SparseGraph class.

    This function demonstrates:
    - Graph creation
    - Node insertion
    - Edge insertion
    - Edge existence queries
    """
    # Define the maximum number of nodes in advance
    max_nodes: int = 10
    directed: bool = False

    # Create the graph
    graph: SparseGraph = SparseGraph(max_nodes, directed)

    # Add some nodes to the graph
    node_a: int = graph.add_node()
    node_b: int = graph.add_node()
    node_c: int = graph.add_node()

    # Add directed edges
    graph.add_edge(node_a, node_b)
    graph.add_edge(node_b, node_c)

    # Check for edge existence
    print("Edge A -> B exists:", graph.has_edge(node_a, node_b))
    print("Edge B -> A exists:", graph.has_edge(node_b, node_a))
    print("Edge B -> C exists:", graph.has_edge(node_b, node_c))

    # Iterate over neighbors
    print("Neighbors of node B:", graph.neighbors(node_b))

    # Display node counts
    print("Current node count:", graph.node_count())
    print("Maximum node capacity:", graph.max_nodes())


    """
    Demonstrate usage of the CircularDoublyLinkedList.
    """
    # Create a list with capacity for 10 nodes
    cdll = CircularDoublyLinkedList(capacity=10)

    # Insert the first node (list is empty, so node argument is ignored)
    head = cdll.insert_after(node=0, value=10)

    # Insert additional nodes
    n1 = cdll.insert_after(head, 20)
    n2 = cdll.insert_after(n1, 30)
    n3 = cdll.insert_after(n2, 40)

    # Traverse the list once starting from head
    current = head
    print("List contents:")
    for _ in range(cdll.size()):
        print(cdll.get_value(current))
        current = cdll.next_node(current)

    # Remove one node
    cdll.remove(n2)

    # Traverse again after removal
    print("\nAfter removal:")
    current = head
    for _ in range(cdll.size()):
        print(cdll.get_value(current))
        current = cdll.next_node(current)

def main() -> None:
    # basic_tests()

    my_string = "vvc"
    directed = False

    stg = StringToGraph(my_string, directed)
    output_graph, graph_trace_stg = stg.run(trace = True)


    for g in graph_trace_stg:
        print(g)

    trace_to_pptx(graph_trace_stg, "./graph_trace_stg.pptx")

    gts = GraphToString(output_graph)
    initial_node = 0
    output_string, graph_trace_gts = gts.run(initial_node, trace = True)

    print(output_string)

    for g in graph_trace_gts:
        print(g)

    trace_to_pptx(graph_trace_gts, "./graph_trace_gts.pptx")

    isomorphism_check = graph_trace_stg[-1][0].is_isomorphic(graph_trace_gts[-1][0])



# Execute the sample program if this file is run directly
if __name__ == "__main__":
    main()
