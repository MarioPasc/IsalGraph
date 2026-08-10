#pragma once
// Greedy graph-to-string encoder.
//
// Port of src/isalgraph/core/graph_to_string.py (GraphToString.run).
//
// At each step the encoder scans displacement pairs in increasing cost order
// and takes the first that enables an operation, with priority V > v > C > c.
// This path is ORDER-DEPENDENT: `_find_new_neighbor` returns the first
// uninserted neighbour in CPython set-iteration order, which is why
// InputGraph preserves that order verbatim (see sparse_graph.hpp).
//
// Reachability and initial-node range checks are performed on the PYTHON
// side before marshalling: the reference's error message embeds a Python
// `set` repr of the unreachable nodes, and reproducing CPython's set repr
// ordering in C++ would be fragile for no gain (the check is O(V+E) against
// an encoder that is far more expensive).

#include <string>

#include <isalgraph/sparse_graph.hpp>

namespace isalgraph {

/// Greedy encode @p ig starting from input-graph node @p initial_node.
std::string graph_to_string(const InputGraph& ig, int32_t initial_node);

}  // namespace isalgraph
