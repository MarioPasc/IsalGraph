#pragma once
// Array-backed circular doubly linked list.
//
// Port of src/isalgraph/core/cdll.py (CircularDoublyLinkedList).
//
// INVARIANT 1 (silent-corruption class).  CDLL indices are NOT graph node
// indices.  A pointer held by the string<->graph converters is a CDLL index;
// the graph node it denotes is `get_value(ptr)`.  The two index spaces
// coincide only for as long as no CDLL node is ever removed -- which is true
// for StringToGraph and greedy GraphToString but FALSE inside the canonical
// backtracking search, where `remove()` recycles indices through the free
// list.  Never conflate them.
//
// INVARIANT 2.  `insert_after(cdll_index, graph_node_payload)`: the first
// argument is a CDLL index, the second is the payload.
//
// Free-list ordering is load-bearing.  The Python reference initialises the
// free list to `list(range(capacity - 1, -1, -1))` and allocates with
// `.pop()`, so the first allocations yield 0, 1, 2, ... in that order and a
// removed index is the next one handed back (LIFO).  This C++ port
// reproduces that exactly: any deviation changes which CDLL slot a node
// lands in and, through `get_value`, can change the emitted string.

#include <cstddef>
#include <cstdint>
#include <vector>

namespace isalgraph {

class Cdll {
public:
    explicit Cdll(int32_t capacity);

    [[nodiscard]] int32_t size() const noexcept { return size_; }
    [[nodiscard]] int32_t capacity() const noexcept { return capacity_; }

    [[nodiscard]] int32_t get_value(int32_t node) const noexcept { return data_[static_cast<std::size_t>(node)]; }
    void set_value(int32_t node, int32_t value) noexcept { data_[static_cast<std::size_t>(node)] = value; }

    [[nodiscard]] int32_t next_node(int32_t node) const noexcept { return next_[static_cast<std::size_t>(node)]; }
    [[nodiscard]] int32_t prev_node(int32_t node) const noexcept { return prev_[static_cast<std::size_t>(node)]; }

    /// Insert a new node carrying @p value after CDLL index @p node.
    /// When the list is empty @p node is ignored (the Python reference is
    /// called with -1 in that case) and the new node points at itself.
    /// Throws CapacityError when the free list is exhausted.
    int32_t insert_after(int32_t node, int32_t value);

    /// Unlink @p node and return its index to the free list.
    void remove(int32_t node) noexcept;

    /// Walk @p ptr by @p steps (positive = next, negative = prev).
    [[nodiscard]] int32_t walk(int32_t ptr, int32_t steps) const noexcept {
        if (steps > 0) {
            for (int32_t i = 0; i < steps; ++i) ptr = next_[static_cast<std::size_t>(ptr)];
        } else {
            for (int32_t i = 0; i < -steps; ++i) ptr = prev_[static_cast<std::size_t>(ptr)];
        }
        return ptr;
    }

private:
    std::vector<int32_t> next_;
    std::vector<int32_t> prev_;
    std::vector<int32_t> data_;
    std::vector<int32_t> free_;  // stack; back() is the next allocation
    int32_t size_ = 0;
    int32_t capacity_ = 0;
};

}  // namespace isalgraph
