#pragma once
// IsalGraph instruction-string virtual machine.
//
// Port of src/isalgraph/core/string_to_graph.py (StringToGraph.run).
//
// Alphabet and semantics (README.md 2.6):
//
//   N / P  move the primary pointer next / prev in the CDLL
//   n / p  move the secondary pointer next / prev in the CDLL
//   V      new node + edge from the primary's GRAPH node; insert into the
//          CDLL after the primary
//   v      new node + edge from the secondary's GRAPH node; insert into the
//          CDLL after the secondary
//   C      edge primary -> secondary (graph nodes)
//   c      edge secondary -> primary (graph nodes)
//   W      no-op
//
// INVARIANT 4 (pointer immobility).  V and v do NOT advance the pointer that
// triggered the insertion.  The new CDLL node is spliced in after it and the
// pointer stays put.
//
// Alphabet validation happens on the PYTHON side of the FFI, before
// marshalling, so that the ValueError message -- which embeds a Python repr
// of the offending string -- is byte-identical to the reference's.

#include <cstdint>
#include <string>
#include <utility>
#include <vector>

namespace isalgraph {

struct StringToGraphResult {
    int32_t node_count = 0;
    int32_t max_nodes = 0;
    bool directed = false;
    /// Edges in the exact chronological order add_edge() was called, so the
    /// Python side can replay them and obtain a SparseGraph whose adjacency
    /// sets are bit-identical to the reference's -- including CPython set
    /// slot layout, which depends on insertion order under collision.
    std::vector<std::pair<int32_t, int32_t>> edges;
};

/// Execute @p instructions. Assumes the alphabet has already been validated.
StringToGraphResult string_to_graph(const std::string& instructions, bool directed);

}  // namespace isalgraph
