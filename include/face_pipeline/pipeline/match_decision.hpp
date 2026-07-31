#pragma once

#include <optional>
#include <vector>

namespace face_pipeline::pipeline {

/// Margin between the best and second-best candidate similarity.
///
/// When the index returned a single candidate there is nothing to be
/// ambiguous with, so the top-1 similarity itself is used as the margin.
/// That keeps a lone weak hit from being accepted purely for lack of
/// competition.
constexpr float compute_margin(float top1_sim, std::optional<float> top2_sim) noexcept {
    return top2_sim ? top1_sim - *top2_sim : top1_sim;
}

/// Decide whether a probe resolves to the top-1 identity.
///
/// An identity is reported only when the best similarity clears
/// `threshold` AND it leads the runner-up by at least `margin_min`. The
/// margin guards against dense enrollments where several galleries sit
/// just above the threshold and the top-1 is effectively a coin flip.
constexpr bool match_decision(float top1_sim, std::optional<float> top2_sim, float threshold,
                              float margin_min) noexcept {
    return top1_sim >= threshold && compute_margin(top1_sim, top2_sim) >= margin_min;
}

/// Convenience overload for a ranked similarity list, highest first. An
/// empty list is a rejection rather than an error.
inline bool match_decision(const std::vector<float>& ranked_similarities, float threshold,
                           float margin_min) noexcept {
    if (ranked_similarities.empty())
        return false;
    const std::optional<float> second = ranked_similarities.size() > 1
                                            ? std::optional<float>(ranked_similarities[1])
                                            : std::nullopt;
    return match_decision(ranked_similarities[0], second, threshold, margin_min);
}

}  // namespace face_pipeline::pipeline
