#include <isalgraph/cdll.hpp>

#include <isalgraph/errors.hpp>

namespace isalgraph {

Cdll::Cdll(int32_t capacity)
    : next_(static_cast<size_t>(capacity), -1),
      prev_(static_cast<size_t>(capacity), -1),
      data_(static_cast<size_t>(capacity), 0),
      capacity_(capacity) {
    // The reference builds `list(range(capacity - 1, -1, -1))` and allocates
    // with `.pop()`, so the first allocation must yield 0.  Descending
    // initialisation + pop_back() reproduces that exactly, including the LIFO
    // reuse of an index that remove() has just released.
    free_.reserve(static_cast<size_t>(capacity));
    for (int32_t i = capacity - 1; i >= 0; --i) free_.push_back(i);
}

int32_t Cdll::insert_after(int32_t node, int32_t value) {
    if (free_.empty()) {
        throw CapacityError("CircularDoublyLinkedList is full");
    }
    const int32_t new_node = free_.back();
    free_.pop_back();
    data_[static_cast<size_t>(new_node)] = value;

    if (size_ == 0) {
        // `node` is ignored when the list is empty (the reference is called
        // with -1 here); the new node becomes the sole, self-linked element.
        next_[static_cast<size_t>(new_node)] = new_node;
        prev_[static_cast<size_t>(new_node)] = new_node;
    } else {
        const int32_t next_of_node = next_[static_cast<size_t>(node)];
        next_[static_cast<size_t>(node)] = new_node;
        prev_[static_cast<size_t>(new_node)] = node;
        next_[static_cast<size_t>(new_node)] = next_of_node;
        prev_[static_cast<size_t>(next_of_node)] = new_node;
    }

    ++size_;
    return new_node;
}

void Cdll::remove(int32_t node) noexcept {
    if (size_ == 0) return;

    if (size_ == 1) {
        free_.push_back(node);
        size_ = 0;
        return;
    }

    const int32_t prev_of_node = prev_[static_cast<size_t>(node)];
    const int32_t next_of_node = next_[static_cast<size_t>(node)];
    next_[static_cast<size_t>(prev_of_node)] = next_of_node;
    prev_[static_cast<size_t>(next_of_node)] = prev_of_node;
    free_.push_back(node);
    --size_;
}

}  // namespace isalgraph
