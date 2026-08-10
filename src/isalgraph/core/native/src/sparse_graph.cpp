#include <isalgraph/sparse_graph.hpp>

#include <algorithm>
#include <string>

#include <isalgraph/errors.hpp>

namespace isalgraph {

// ---------------------------------------------------------------------------
// InputGraph
// ---------------------------------------------------------------------------

InputGraph::InputGraph(int32_t node_count, int32_t max_nodes, bool directed,
                       int32_t logical_edge_count,
                       std::vector<std::vector<int32_t>> ordered_adjacency)
    : node_count_(node_count),
      max_nodes_(max_nodes),
      directed_(directed),
      logical_edge_count_(logical_edge_count),
      adj_(std::move(ordered_adjacency)) {
    const int64_t n = node_count_;
    if (n > 0 && n <= kDenseMatrixMaxNodes) {
        dense_.assign(static_cast<size_t>(n * n), 0u);
        for (int32_t u = 0; u < node_count_; ++u) {
            for (const int32_t v : adj_[static_cast<size_t>(u)]) {
                dense_[static_cast<size_t>(u) * static_cast<size_t>(n) + static_cast<size_t>(v)] = 1u;
            }
        }
    } else {
        sorted_ = adj_;
        for (auto& row : sorted_) std::sort(row.begin(), row.end());
    }
}

bool InputGraph::has_edge(int32_t s, int32_t t) const noexcept {
    if (!dense_.empty()) {
        return dense_[static_cast<size_t>(s) * static_cast<size_t>(node_count_) +
                      static_cast<size_t>(t)] != 0u;
    }
    const auto& row = sorted_[static_cast<size_t>(s)];
    return std::binary_search(row.begin(), row.end(), t);
}

// ---------------------------------------------------------------------------
// SparseGraph (mutable output graph)
// ---------------------------------------------------------------------------

SparseGraph::SparseGraph(int32_t max_nodes, bool directed)
    : max_nodes_(max_nodes), directed_(directed) {
    const int64_t m = max_nodes_;
    dense_mode_ = (m > 0 && m <= kDenseMatrixMaxNodes);
    if (dense_mode_) {
        dense_.assign(static_cast<size_t>(m * m), 0u);
    } else {
        sorted_.resize(static_cast<size_t>(std::max<int32_t>(max_nodes_, 0)));
    }
}

void SparseGraph::set_bit(int32_t s, int32_t t) noexcept {
    if (dense_mode_) {
        dense_[static_cast<size_t>(s) * static_cast<size_t>(max_nodes_) +
               static_cast<size_t>(t)] = 1u;
        return;
    }
    auto& row = sorted_[static_cast<size_t>(s)];
    row.insert(std::lower_bound(row.begin(), row.end(), t), t);
}

void SparseGraph::clear_bit(int32_t s, int32_t t) noexcept {
    if (dense_mode_) {
        dense_[static_cast<size_t>(s) * static_cast<size_t>(max_nodes_) +
               static_cast<size_t>(t)] = 0u;
        return;
    }
    auto& row = sorted_[static_cast<size_t>(s)];
    const auto it = std::lower_bound(row.begin(), row.end(), t);
    if (it != row.end() && *it == t) row.erase(it);
}

bool SparseGraph::has_edge(int32_t s, int32_t t) const noexcept {
    if (dense_mode_) {
        return dense_[static_cast<size_t>(s) * static_cast<size_t>(max_nodes_) +
                      static_cast<size_t>(t)] != 0u;
    }
    const auto& row = sorted_[static_cast<size_t>(s)];
    return std::binary_search(row.begin(), row.end(), t);
}

int32_t SparseGraph::add_node() {
    if (node_count_ >= max_nodes_) {
        throw CapacityError("Maximum number of nodes reached: " + std::to_string(max_nodes_));
    }
    return node_count_++;
}

void SparseGraph::add_edge(int32_t s, int32_t t) noexcept {
    // Mirrors SparseGraph.add_edge: a duplicate insertion is a no-op and does
    // not touch the edge count.  Bounds checking is the caller's job here --
    // every call site in this package derives its arguments from live CDLL
    // payloads, and the Python-side wrappers validate user input first.
    if (has_edge(s, t)) return;
    set_bit(s, t);
    ++edge_count_;
    if (!directed_) {
        set_bit(t, s);
        ++edge_count_;
    }
}

void SparseGraph::undo_edge(int32_t s, int32_t t) noexcept {
    // canonical.py::_undo_edge -- unconditional discard + decrement, matching
    // the reference exactly (it does not re-check membership either).
    clear_bit(s, t);
    --edge_count_;
    if (!directed_) {
        clear_bit(t, s);
        --edge_count_;
    }
}

void SparseGraph::undo_node() noexcept {
    // canonical.py::_undo_node -- drop the last node, then clear its
    // out-adjacency row.  The row is already empty in every reachable state
    // (the only edge touching a freshly added node is removed by undo_edge
    // immediately before), so this is belt-and-braces in both languages.
    --node_count_;
    if (dense_mode_) {
        const size_t base = static_cast<size_t>(node_count_) * static_cast<size_t>(max_nodes_);
        std::fill_n(dense_.begin() + static_cast<std::ptrdiff_t>(base),
                    static_cast<size_t>(max_nodes_), static_cast<uint8_t>(0u));
    } else {
        sorted_[static_cast<size_t>(node_count_)].clear();
    }
}

}  // namespace isalgraph
