#pragma once
// Two graph representations, split by what the algorithms actually need.
//
// * InputGraph  -- immutable, marshalled from Python.  Provides ORDERED
//                  adjacency plus O(1)/O(log d) membership.
// * SparseGraph -- mutable output graph.  Membership + counts only; it is
//                  never iterated, so no ordering contract is needed.
//
// WHY InputGraph KEEPS AN ORDER (the single design decision that makes
// byte-exact greedy parity possible):
//
//   GraphToString._find_new_neighbor returns THE FIRST uninserted neighbour
//   obtained by iterating a Python `set[int]`.  CPython set iteration order
//   for small ints is slot order, i & (table_size - 1), not ascending value
//   order: for {2, 9} with table size 8, 9 occupies slot 1 and 2 occupies
//   slot 2, so Python yields 9 then 2.  A std::set<int32_t> would yield 2
//   then 9 and produce a different -- equally valid, but different -- greedy
//   string.  The Python marshalling layer therefore hands us each adjacency
//   as `list(graph.neighbors(u))`, i.e. in CPython's own iteration order, and
//   we store it verbatim.  "First neighbour" in C++ is then by construction
//   "first neighbour" in Python.
//
//   Consequences: the canonical and pruned-canonical searches branch over
//   ALL candidates and take a minimum, so they are order-independent and are
//   byte-exact regardless.  The greedy path is order-dependent and relies on
//   the above.  Nothing whose iteration order can reach an output may be a
//   std::unordered_set.
//
// INVARIANT 3.  add_edge(source, target): both arguments are GRAPH node
// indices, never CDLL indices.

#include <cstddef>
#include <cstdint>
#include <vector>

namespace isalgraph {

/// Above this node count the dense adjacency matrix is replaced by sorted
/// per-node vectors.  2048 nodes -> 4 MiB, which is the largest allocation
/// worth paying for eagerly.  Both algorithms in this package become
/// intractable long before this bound for other reasons.
inline constexpr int32_t kDenseMatrixMaxNodes = 2048;

/// Immutable input graph with Python-order adjacency.
class InputGraph {
public:
    /// @param logical_edge_count Taken from the Python graph's own counter,
    ///        never derived from the adjacency lengths.  A self-loop in an
    ///        UNDIRECTED graph occupies one adjacency slot but increments
    ///        SparseGraph._edge_count twice, so sum(len(adj))/2 would
    ///        undercount it and the encoder would stop one edge early.
    InputGraph(int32_t node_count, int32_t max_nodes, bool directed, int32_t logical_edge_count,
               std::vector<std::vector<int32_t>> ordered_adjacency);

    [[nodiscard]] int32_t node_count() const noexcept { return node_count_; }
    [[nodiscard]] int32_t max_nodes() const noexcept { return max_nodes_; }
    [[nodiscard]] bool directed() const noexcept { return directed_; }
    [[nodiscard]] int32_t logical_edge_count() const noexcept { return logical_edge_count_; }

    /// Adjacency of @p u in CPython set-iteration order.
    [[nodiscard]] const std::vector<int32_t>& neighbors(int32_t u) const noexcept {
        return adj_[static_cast<std::size_t>(u)];
    }

    [[nodiscard]] bool has_edge(int32_t s, int32_t t) const noexcept;

private:
    int32_t node_count_ = 0;
    int32_t max_nodes_ = 0;
    bool directed_ = false;
    int32_t logical_edge_count_ = 0;
    std::vector<std::vector<int32_t>> adj_;     // ordered, load-bearing
    std::vector<uint8_t> dense_;                // node_count^2, or empty
    std::vector<std::vector<int32_t>> sorted_;  // fallback membership index
};

/// Mutable output graph: add_node/add_edge with exact undo, membership tests.
///
/// undo_edge / undo_node reproduce canonical.py's `_undo_edge` / `_undo_node`
/// private helpers, which reach into SparseGraph's internals to backtrack
/// without deep copies.
class SparseGraph {
public:
    SparseGraph(int32_t max_nodes, bool directed);

    [[nodiscard]] bool directed() const noexcept { return directed_; }
    [[nodiscard]] int32_t node_count() const noexcept { return node_count_; }
    [[nodiscard]] int32_t edge_count() const noexcept { return edge_count_; }
    [[nodiscard]] int32_t max_nodes() const noexcept { return max_nodes_; }
    [[nodiscard]] int32_t logical_edge_count() const noexcept {
        return directed_ ? edge_count_ : edge_count_ / 2;
    }

    [[nodiscard]] bool has_edge(int32_t s, int32_t t) const noexcept;

    /// Append a node; throws CapacityError past max_nodes.
    int32_t add_node();
    void add_edge(int32_t s, int32_t t) noexcept;

    /// Exact inverse of add_edge (canonical.py::_undo_edge).
    void undo_edge(int32_t s, int32_t t) noexcept;
    /// Exact inverse of add_node (canonical.py::_undo_node): drop the last
    /// node and clear its out-adjacency row.
    void undo_node() noexcept;

private:
    void set_bit(int32_t s, int32_t t) noexcept;
    void clear_bit(int32_t s, int32_t t) noexcept;

    int32_t max_nodes_ = 0;
    bool directed_ = false;
    int32_t node_count_ = 0;
    int32_t edge_count_ = 0;
    bool dense_mode_ = true;
    std::vector<uint8_t> dense_;                // max_nodes^2 when dense_mode_
    std::vector<std::vector<int32_t>> sorted_;  // sorted adjacency otherwise
};

}  // namespace isalgraph
