#include <gtest/gtest.h>

#include <optional>
#include <vector>

#include "face_pipeline/pipeline/match_decision.hpp"

namespace {

using face_pipeline::pipeline::compute_margin;
using face_pipeline::pipeline::match_decision;

/// The defaults shipped in configs/system_config.yaml.
constexpr float kThreshold = 0.45f;
constexpr float kMarginMin = 0.04f;

TEST(ComputeMargin, IsTheGapBetweenTheTopTwoCandidates) {
    EXPECT_FLOAT_EQ(compute_margin(0.80f, 0.50f), 0.30f);
}

TEST(ComputeMargin, FallsBackToTheTopScoreWhenThereIsNoRunnerUp) {
    // A single candidate has nothing to be confused with, so its own
    // similarity stands in for the margin.
    EXPECT_FLOAT_EQ(compute_margin(0.80f, std::nullopt), 0.80f);
}

TEST(MatchDecision, AcceptsAConfidentUnambiguousTopHit) {
    EXPECT_TRUE(match_decision(0.82f, 0.51f, kThreshold, kMarginMin));
}

TEST(MatchDecision, RejectsWhenTheRunnerUpIsTooClose) {
    // Comfortably over the threshold, but the identity is a coin flip.
    EXPECT_FALSE(match_decision(0.82f, 0.80f, kThreshold, kMarginMin));
}

TEST(MatchDecision, RejectsWhenTheTopHitIsBelowTheThreshold) {
    // Large margin does not rescue a weak top-1.
    EXPECT_FALSE(match_decision(0.30f, 0.01f, kThreshold, kMarginMin));
}

TEST(MatchDecision, RejectsWhenBothCriteriaFail) {
    EXPECT_FALSE(match_decision(0.30f, 0.29f, kThreshold, kMarginMin));
}

TEST(MatchDecision, AcceptsExactlyAtTheThreshold) {
    // The comparison is inclusive, so the boundary value is a match.
    EXPECT_TRUE(match_decision(kThreshold, 0.10f, kThreshold, kMarginMin));
}

TEST(MatchDecision, RejectsJustBelowTheThreshold) {
    EXPECT_FALSE(match_decision(0.4499f, 0.10f, kThreshold, kMarginMin));
}

TEST(MatchDecision, AcceptsExactlyAtTheMinimumMargin) {
    const float top2 = 0.60f;
    const float top1 = top2 + kMarginMin;
    EXPECT_TRUE(match_decision(top1, top2, kThreshold, kMarginMin));
}

TEST(MatchDecision, RejectsJustBelowTheMinimumMargin) {
    const float top2 = 0.60f;
    const float top1 = top2 + kMarginMin - 0.001f;
    EXPECT_FALSE(match_decision(top1, top2, kThreshold, kMarginMin));
}

TEST(MatchDecision, AcceptsALoneCandidateThatClearsTheThreshold) {
    EXPECT_TRUE(match_decision(0.70f, std::nullopt, kThreshold, kMarginMin));
}

TEST(MatchDecision, RejectsALoneCandidateBelowTheThreshold) {
    EXPECT_FALSE(match_decision(0.02f, std::nullopt, kThreshold, kMarginMin));
}

TEST(MatchDecision, AZeroMarginDisablesTheAmbiguityGuard) {
    EXPECT_TRUE(match_decision(0.82f, 0.82f, kThreshold, 0.0f));
}

// ------------------------------------------------------- ranked-list form

TEST(MatchDecisionRanked, RejectsAnEmptyResultList) {
    EXPECT_FALSE(match_decision(std::vector<float>{}, kThreshold, kMarginMin));
}

TEST(MatchDecisionRanked, AcceptsAConfidentTopHit) {
    EXPECT_TRUE(match_decision(std::vector<float>{0.90f, 0.40f, 0.31f}, kThreshold, kMarginMin));
}

TEST(MatchDecisionRanked, RejectsACrowdedTopOfTheList) {
    EXPECT_FALSE(match_decision(std::vector<float>{0.90f, 0.89f, 0.31f}, kThreshold, kMarginMin));
}

TEST(MatchDecisionRanked, HandlesASingleElementList) {
    EXPECT_TRUE(match_decision(std::vector<float>{0.70f}, kThreshold, kMarginMin));
    EXPECT_FALSE(match_decision(std::vector<float>{0.02f}, kThreshold, kMarginMin));
}

TEST(MatchDecisionRanked, IgnoresEverythingBelowTheRunnerUp) {
    // Only the top two entries participate in the decision.
    EXPECT_TRUE(
        match_decision(std::vector<float>{0.90f, 0.40f, 0.39f, 0.38f}, kThreshold, kMarginMin));
}

}  // namespace
