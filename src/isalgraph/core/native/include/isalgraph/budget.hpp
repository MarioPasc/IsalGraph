#pragma once
// Wall-clock budget threaded through every canonical-search frame.
//
// A null deadline means unlimited, which is the default and costs one
// predictable branch per frame.  When a deadline is set the clock is only
// read every kCheckMask+1 frames: steady_clock::now() is ~20 ns, which would
// otherwise dominate a frame that does little more than a few array writes.

#include <chrono>

#include <isalgraph/errors.hpp>

namespace isalgraph {

class Budget {
public:
    using Clock = std::chrono::steady_clock;

    Budget() = default;
    explicit Budget(const Clock::time_point* deadline) : deadline_(deadline) {}

    [[nodiscard]] bool bounded() const noexcept { return deadline_ != nullptr; }

    /// Throws CanonicalizationTimeoutError once the deadline has passed.
    void check() {
        if (deadline_ == nullptr) return;
        if ((++tick_ & kCheckMask) != 0u) return;
        if (Clock::now() > *deadline_) {
            throw CanonicalizationTimeoutError(
                "Canonical search exceeded the allotted time budget.");
        }
    }

private:
    static constexpr uint32_t kCheckMask = 0x3FFu;  // check every 1024 frames
    const Clock::time_point* deadline_ = nullptr;
    uint32_t tick_ = 0;
};

}  // namespace isalgraph
