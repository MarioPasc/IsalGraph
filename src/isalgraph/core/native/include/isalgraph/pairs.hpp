#pragma once
// Displacement-pair enumeration, memoised.
//
// Port of graph_to_string.py::generate_pairs_sorted_by_sum.
//
// INVARIANT 4.  The sort key is (|a| + |b|, |a|, (a, b)) -- all three
// components, in that order.  |a| + |b| is the number of pointer-movement
// instructions that will be emitted, so it is the real cost; sorting by the
// algebraic sum a + b is historical bug B2 and produces different (longer)
// strings.
//
// MEMOISATION.  The Python reference calls generate_pairs_sorted_by_sum at
// EVERY recursion frame of the canonical search, rebuilding and re-sorting
// (2m+1)^2 pairs -- Theta(m^2 log m) per frame -- even though the result
// depends only on m.  The cache below is keyed on m alone and is
// thread_local so that a parallel per-start-node search stays race-free.

#include <cstdint>
#include <vector>

namespace isalgraph {

struct Pair {
    int32_t a;
    int32_t b;
};

/// All (a, b) with a, b in [-m, m], sorted by (|a|+|b|, |a|, a, b).
/// The reference is returned from a thread-local cache and stays valid for
/// the lifetime of the thread; callers must not mutate it.
const std::vector<Pair>& pairs_sorted_by_cost(int32_t m);

/// Number of distinct m values currently memoised on this thread.
/// Exposed only so the test suite can prove the cache is live.
std::size_t pairs_cache_size() noexcept;

}  // namespace isalgraph
