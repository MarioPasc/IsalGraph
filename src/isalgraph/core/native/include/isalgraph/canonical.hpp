#pragma once
// Canonical and triplet-pruned canonical IsalGraph strings.
//
// Ports of src/isalgraph/core/canonical.py and canonical_pruned.py.
//
//   w*_G = lexmin { w in argmin_{v in V} |G2S(G, v)| }
//
// The search is the exhaustive backtracking variant of the greedy encoder:
// at each V/v branch point it explores EVERY uninserted neighbour and takes
// the minimum by (length, lexicographic order); at C/c it commits to the
// first applicable displacement pair.  Because every branch point minimises
// over the whole candidate set, the result does not depend on the order in
// which candidates are visited -- so this path is byte-exact against the
// reference regardless of CPython set-iteration order.
//
// The pruned variant additionally filters V/v candidates to those attaining
// the maximum structural triplet (|N_1|, |N_2|, |N_3|).  Triplets are BFS
// distance counts on the INPUT graph and are automorphism-invariant, so the
// filter preserves the complete-invariant property while collapsing the
// branching factor.  max() over a candidate set is likewise order-independent.

#include <chrono>
#include <cstdint>
#include <string>
#include <vector>

#include <isalgraph/sparse_graph.hpp>

namespace isalgraph {

struct Triplet {
    int32_t d1 = 0;
    int32_t d2 = 0;
    int32_t d3 = 0;

    [[nodiscard]] bool operator<(const Triplet& o) const noexcept {
        if (d1 != o.d1) return d1 < o.d1;
        if (d2 != o.d2) return d2 < o.d2;
        return d3 < o.d3;
    }
    [[nodiscard]] bool operator==(const Triplet& o) const noexcept {
        return d1 == o.d1 && d2 == o.d2 && d3 == o.d3;
    }
};

/// (|N_1(v)|, |N_2(v)|, |N_3(v)|) for every node, via BFS truncated at 3.
std::vector<Triplet> compute_structural_triplets(const InputGraph& g);

/// Canonical string.  @p deadline may be null for an unlimited budget.
/// @p threads must be >= 1; 1 is the default and the only value that is safe
/// to assume on a cluster (hardware_concurrency() reports the whole node from
/// inside a SLURM cgroup and silently oversubscribes).
std::string canonical_string(const InputGraph& g,
                             const std::chrono::steady_clock::time_point* deadline,
                             int threads);

/// Triplet-pruned canonical string.  Same contract as canonical_string.
std::string pruned_canonical_string(const InputGraph& g,
                                    const std::chrono::steady_clock::time_point* deadline,
                                    int threads);

/// Branch-and-bound toggle.  Present only so the optimisation log can A/B the
/// bound against the faithful port on identical inputs; production wants it on
/// and it is on by default.  Both settings must produce identical strings --
/// the bound prunes only subtrees that provably cannot win.
void set_branch_and_bound(bool on) noexcept;
[[nodiscard]] bool branch_and_bound() noexcept;

}  // namespace isalgraph
