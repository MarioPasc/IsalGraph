#include <isalgraph/string_to_graph.hpp>

#include <isalgraph/cdll.hpp>
#include <isalgraph/errors.hpp>
#include <isalgraph/sparse_graph.hpp>

namespace isalgraph {

StringToGraphResult string_to_graph(const std::string& instructions, bool directed) {
    int32_t inserts = 0;
    for (const char ch : instructions) {
        if (ch == 'V' || ch == 'v') ++inserts;
    }
    const int32_t max_nodes = 1 + inserts;

    SparseGraph og(max_nodes, directed);
    Cdll cdll(max_nodes);

    StringToGraphResult out;
    out.max_nodes = max_nodes;
    out.directed = directed;
    out.edges.reserve(static_cast<size_t>(instructions.size()));

    // Initial state: one node, both pointers on it.
    const int32_t initial_graph_node = og.add_node();
    const int32_t initial_cdll_node = cdll.insert_after(-1, initial_graph_node);
    int32_t pri = initial_cdll_node;
    int32_t sec = initial_cdll_node;

    auto emit_edge = [&](int32_t s, int32_t t) {
        // Record before add_edge so the log mirrors call order exactly, and
        // record unconditionally: the reference calls add_edge regardless of
        // whether the edge already exists, and replaying a duplicate on the
        // Python side is likewise a no-op.
        out.edges.emplace_back(s, t);
        og.add_edge(s, t);
    };

    for (const char instruction : instructions) {
        switch (instruction) {
            case 'N':
                pri = cdll.next_node(pri);
                break;
            case 'P':
                pri = cdll.prev_node(pri);
                break;
            case 'n':
                sec = cdll.next_node(sec);
                break;
            case 'p':
                sec = cdll.prev_node(sec);
                break;
            case 'V': {
                const int32_t new_node = og.add_node();
                // INVARIANT 1: pri is a CDLL index; the graph node is its payload.
                emit_edge(cdll.get_value(pri), new_node);
                // INVARIANT 4: pri does NOT advance past the inserted node.
                cdll.insert_after(pri, new_node);
                break;
            }
            case 'v': {
                const int32_t new_node = og.add_node();
                emit_edge(cdll.get_value(sec), new_node);
                cdll.insert_after(sec, new_node);
                break;
            }
            case 'C':
                emit_edge(cdll.get_value(pri), cdll.get_value(sec));
                break;
            case 'c':
                emit_edge(cdll.get_value(sec), cdll.get_value(pri));
                break;
            case 'W':
                break;  // no-op
            default:
                // Unreachable: the alphabet is validated Python-side before
                // marshalling so the ValueError message can carry a Python repr.
                throw InvalidStringError(std::string("Invalid IsalGraph instruction: ") +
                                         instruction);
        }
    }

    out.node_count = og.node_count();
    return out;
}

}  // namespace isalgraph
