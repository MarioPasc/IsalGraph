#include <isalgraph/pairs.hpp>

#include <algorithm>
#include <cstdlib>
#include <stdexcept>
#include <unordered_map>

namespace isalgraph {

static bool g_pairs_memo = true;

namespace {

std::vector<Pair> build_pairs(int32_t m) {
    std::vector<Pair> pairs;
    pairs.reserve(static_cast<size_t>(2 * m + 1) * static_cast<size_t>(2 * m + 1));
    for (int32_t a = -m; a <= m; ++a) {
        for (int32_t b = -m; b <= m; ++b) pairs.push_back(Pair{a, b});
    }
    // Key (|a| + |b|, |a|, a, b): total displacement cost first, then |a| to
    // break cost ties deterministically, then the raw pair lexicographically.
    // The full key admits no ties, so stability of the sort is irrelevant and
    // the ordering is identical to CPython's list.sort on the same key.
    std::sort(pairs.begin(), pairs.end(), [](const Pair& x, const Pair& y) {
        const int32_t ax = std::abs(x.a);
        const int32_t bx = std::abs(x.b);
        const int32_t ay = std::abs(y.a);
        const int32_t by = std::abs(y.b);
        if (ax + bx != ay + by) return ax + bx < ay + by;
        if (ax != ay) return ax < ay;
        if (x.a != y.a) return x.a < y.a;
        return x.b < y.b;
    });
    return pairs;
}

std::unordered_map<int32_t, std::vector<Pair>>& cache() {
    // thread_local: a parallel per-start-node search must not share this.
    static thread_local std::unordered_map<int32_t, std::vector<Pair>> c;
    return c;
}

}  // namespace

const std::vector<Pair>& pairs_sorted_by_cost(int32_t m) {
    // std::invalid_argument, not an isalgraph type: nanobind maps it to a
    // plain Python ValueError, which is exactly what the reference raises
    // here. This is an internal precondition, unreachable from user input.
    if (m <= 0) throw std::invalid_argument("m must be a positive integer.");

    if (!g_pairs_memo) {
        // A/B path: rebuild every call, as the reference does at every frame.
        // A thread_local scratch keeps the reference return type valid.
        static thread_local std::vector<Pair> scratch;
        scratch = build_pairs(m);
        return scratch;
    }

    auto& c = cache();
    const auto it = c.find(m);
    if (it != c.end()) return it->second;
    return c.emplace(m, build_pairs(m)).first->second;
}

std::size_t pairs_cache_size() noexcept { return cache().size(); }

void set_pairs_memo(bool on) noexcept { g_pairs_memo = on; }
bool pairs_memo() noexcept { return g_pairs_memo; }

}  // namespace isalgraph
