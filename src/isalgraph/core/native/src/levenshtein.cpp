#include <isalgraph/levenshtein.hpp>

#include <algorithm>
#include <vector>

namespace isalgraph {

int64_t levenshtein(const std::string& s, const std::string& t) {
    // Reference swaps so the shorter string indexes the row.
    const std::string& a = (s.size() < t.size()) ? t : s;
    const std::string& b = (s.size() < t.size()) ? s : t;

    if (b.empty()) return static_cast<int64_t>(a.size());

    const size_t nb = b.size();
    std::vector<int64_t> prev(nb + 1);
    std::vector<int64_t> curr(nb + 1);
    for (size_t j = 0; j <= nb; ++j) prev[j] = static_cast<int64_t>(j);

    for (size_t i = 0; i < a.size(); ++i) {
        curr[0] = static_cast<int64_t>(i) + 1;
        const char ac = a[i];
        for (size_t j = 0; j < nb; ++j) {
            const int64_t insert = prev[j + 1] + 1;
            const int64_t del = curr[j] + 1;
            const int64_t replace = prev[j] + (ac == b[j] ? 0 : 1);
            curr[j + 1] = std::min(insert, std::min(del, replace));
        }
        prev.swap(curr);
    }

    return prev[nb];
}

}  // namespace isalgraph
