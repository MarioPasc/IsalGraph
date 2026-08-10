#pragma once
// Levenshtein edit distance.
//
// Port of canonical.py::levenshtein -- the standard O(nm) time,
// O(min(n, m)) space single-row dynamic program.  The reference swaps its
// arguments so the shorter string indexes the row; that swap is reproduced
// because it changes nothing observable but keeps the two implementations
// line-comparable.

#include <cstdint>
#include <string>

namespace isalgraph {

[[nodiscard]] int64_t levenshtein(const std::string& s, const std::string& t);

}  // namespace isalgraph
