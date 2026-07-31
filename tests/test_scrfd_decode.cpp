#include <gtest/gtest.h>

#include <array>
#include <opencv2/core.hpp>
#include <stdexcept>
#include <vector>

#include "face_pipeline/trt/scrfd_postprocess.hpp"

namespace {

using face_pipeline::trt::anchor_count;
using face_pipeline::trt::distance2bbox;
using face_pipeline::trt::distance2kps;
using face_pipeline::trt::FaceDetection;
using face_pipeline::trt::iou;
using face_pipeline::trt::kAnchorsPerLocation;
using face_pipeline::trt::nms;
using face_pipeline::trt::total_anchor_count;

constexpr int kInput = 640;

FaceDetection make_det(float x, float y, float w, float h, float score) {
    FaceDetection d;
    d.bbox = cv::Rect2f(x, y, w, h);
    d.score = score;
    return d;
}

// ---------------------------------------------------------------- anchors

TEST(AnchorGrid, MatchesTheKnownCountsForA640Input) {
    // 640 / 8 = 80 -> 80 * 80 * 2 anchors, and so on down the pyramid.
    EXPECT_EQ(anchor_count(kInput, kInput, 8), 12800);
    EXPECT_EQ(anchor_count(kInput, kInput, 16), 3200);
    EXPECT_EQ(anchor_count(kInput, kInput, 32), 800);
}

TEST(AnchorGrid, TotalsAcrossEveryStride) {
    EXPECT_EQ(total_anchor_count(kInput, kInput), 12800 + 3200 + 800);
}

TEST(AnchorGrid, ScalesWithTheAnchorsPerLocation) {
    EXPECT_EQ(anchor_count(kInput, kInput, 8, 1), 6400);
    EXPECT_EQ(anchor_count(kInput, kInput, 8, kAnchorsPerLocation), 12800);
}

TEST(AnchorGrid, HandlesNonSquareInputs) {
    EXPECT_EQ(anchor_count(640, 480, 8), 80 * 60 * 2);
}

TEST(AnchorGrid, RejectsANonPositiveStride) {
    EXPECT_THROW(anchor_count(kInput, kInput, 0), std::invalid_argument);
    EXPECT_THROW(anchor_count(kInput, kInput, -8), std::invalid_argument);
}

// ------------------------------------------------------------ distance2*

TEST(Distance2Bbox, ExpandsAnAnchorByTheRegressedDistances) {
    // Distances are in stride units: 2.0 at stride 8 means 16 pixels.
    const std::array<float, 4> distances{2.0f, 2.0f, 2.0f, 2.0f};

    const cv::Rect2f box = distance2bbox({64.0f, 64.0f}, distances.data(), 8.0f,
                                         /*scale=*/1.0f, /*pad_x=*/0, /*pad_y=*/0);

    EXPECT_FLOAT_EQ(box.x, 48.0f);
    EXPECT_FLOAT_EQ(box.y, 48.0f);
    EXPECT_FLOAT_EQ(box.width, 32.0f);
    EXPECT_FLOAT_EQ(box.height, 32.0f);
}

TEST(Distance2Bbox, HandlesAsymmetricDistances) {
    const std::array<float, 4> distances{1.0f, 2.0f, 3.0f, 4.0f};  // l, t, r, b

    const cv::Rect2f box = distance2bbox({80.0f, 80.0f}, distances.data(), 16.0f, 1.0f, 0, 0);

    EXPECT_FLOAT_EQ(box.x, 80.0f - 16.0f);
    EXPECT_FLOAT_EQ(box.y, 80.0f - 32.0f);
    EXPECT_FLOAT_EQ(box.width, 16.0f + 48.0f);
    EXPECT_FLOAT_EQ(box.height, 32.0f + 64.0f);
}

TEST(Distance2Bbox, UndoesTheLetterboxScaleAndPadding) {
    const std::array<float, 4> distances{2.0f, 2.0f, 2.0f, 2.0f};

    // The same anchor, but the frame was halved and shifted before inference.
    const cv::Rect2f box = distance2bbox({64.0f, 64.0f}, distances.data(), 8.0f,
                                         /*scale=*/0.5f, /*pad_x=*/10, /*pad_y=*/20);

    EXPECT_FLOAT_EQ(box.x, (48.0f - 10.0f) / 0.5f);
    EXPECT_FLOAT_EQ(box.y, (48.0f - 20.0f) / 0.5f);
    // Padding cancels out in the extent; only the scale survives.
    EXPECT_FLOAT_EQ(box.width, 32.0f / 0.5f);
    EXPECT_FLOAT_EQ(box.height, 32.0f / 0.5f);
}

TEST(Distance2Kps, OffsetsFiveLandmarksFromTheAnchor) {
    // Five (dx, dy) pairs in stride units.
    const std::array<float, 10> offsets{-1.0f, -1.0f, 1.0f, -1.0f, 0.0f,
                                        0.0f,  -1.0f, 1.0f, 1.0f,  1.0f};

    const auto kps = distance2kps({64.0f, 64.0f}, offsets.data(), 8.0f, 1.0f, 0, 0);

    ASSERT_EQ(kps.size(), 5u);
    EXPECT_FLOAT_EQ(kps[0].x, 56.0f);
    EXPECT_FLOAT_EQ(kps[0].y, 56.0f);
    EXPECT_FLOAT_EQ(kps[1].x, 72.0f);
    EXPECT_FLOAT_EQ(kps[1].y, 56.0f);
    EXPECT_FLOAT_EQ(kps[2].x, 64.0f);  // nose sits on the anchor
    EXPECT_FLOAT_EQ(kps[2].y, 64.0f);
    EXPECT_FLOAT_EQ(kps[4].x, 72.0f);
    EXPECT_FLOAT_EQ(kps[4].y, 72.0f);
}

TEST(Distance2Kps, UndoesTheLetterboxScaleAndPadding) {
    const std::array<float, 10> offsets{0.0f, 0.0f, 0.0f, 0.0f, 0.0f, 0.0f, 0.0f, 0.0f, 0.0f, 0.0f};

    const auto kps = distance2kps({64.0f, 64.0f}, offsets.data(), 8.0f, 0.5f, 10, 20);

    // Every landmark collapses onto the anchor, mapped back to source space.
    EXPECT_FLOAT_EQ(kps[0].x, (64.0f - 10.0f) / 0.5f);
    EXPECT_FLOAT_EQ(kps[0].y, (64.0f - 20.0f) / 0.5f);
}

// -------------------------------------------------------------------- IoU

TEST(Iou, IdenticalBoxesOverlapFully) {
    const cv::Rect2f a(10.0f, 10.0f, 20.0f, 20.0f);
    EXPECT_FLOAT_EQ(iou(a, a), 1.0f);
}

TEST(Iou, DisjointBoxesDoNotOverlap) {
    const cv::Rect2f a(0.0f, 0.0f, 10.0f, 10.0f);
    const cv::Rect2f b(100.0f, 100.0f, 10.0f, 10.0f);
    EXPECT_FLOAT_EQ(iou(a, b), 0.0f);
}

TEST(Iou, TouchingEdgesDoNotOverlap) {
    const cv::Rect2f a(0.0f, 0.0f, 10.0f, 10.0f);
    const cv::Rect2f b(10.0f, 0.0f, 10.0f, 10.0f);
    EXPECT_FLOAT_EQ(iou(a, b), 0.0f);
}

TEST(Iou, MatchesAHandComputedHalfOverlap) {
    // A and B are 10x10 offset by 5 in x: intersection 5*10 = 50,
    // union 100 + 100 - 50 = 150, so IoU = 1/3.
    const cv::Rect2f a(0.0f, 0.0f, 10.0f, 10.0f);
    const cv::Rect2f b(5.0f, 0.0f, 10.0f, 10.0f);
    EXPECT_NEAR(iou(a, b), 1.0f / 3.0f, 1e-6f);
}

TEST(Iou, MatchesAHandComputedContainedBox) {
    // B sits entirely inside A: intersection 25, union 100.
    const cv::Rect2f a(0.0f, 0.0f, 10.0f, 10.0f);
    const cv::Rect2f b(2.0f, 2.0f, 5.0f, 5.0f);
    EXPECT_NEAR(iou(a, b), 0.25f, 1e-6f);
}

TEST(Iou, DegenerateZeroAreaBoxesYieldZero) {
    const cv::Rect2f empty(0.0f, 0.0f, 0.0f, 0.0f);
    EXPECT_FLOAT_EQ(iou(empty, empty), 0.0f);
}

TEST(Iou, IsSymmetric) {
    const cv::Rect2f a(0.0f, 0.0f, 10.0f, 10.0f);
    const cv::Rect2f b(5.0f, 5.0f, 10.0f, 10.0f);
    EXPECT_FLOAT_EQ(iou(a, b), iou(b, a));
}

// -------------------------------------------------------------------- NMS

TEST(Nms, KeepsOnlyTheHighestScoringBoxOfAnOverlappingCluster) {
    std::vector<FaceDetection> dets{
        make_det(0.0f, 0.0f, 10.0f, 10.0f, 0.70f),
        make_det(1.0f, 1.0f, 10.0f, 10.0f, 0.95f),  // best
        make_det(2.0f, 2.0f, 10.0f, 10.0f, 0.60f),
    };

    const auto kept = nms(std::move(dets), 0.4f);

    ASSERT_EQ(kept.size(), 1u);
    EXPECT_FLOAT_EQ(kept.front().score, 0.95f);
    EXPECT_FLOAT_EQ(kept.front().bbox.x, 1.0f);
}

TEST(Nms, KeepsDisjointBoxes) {
    std::vector<FaceDetection> dets{
        make_det(0.0f, 0.0f, 10.0f, 10.0f, 0.9f),
        make_det(100.0f, 100.0f, 10.0f, 10.0f, 0.8f),
        make_det(200.0f, 200.0f, 10.0f, 10.0f, 0.7f),
    };

    const auto kept = nms(std::move(dets), 0.4f);

    EXPECT_EQ(kept.size(), 3u);
}

TEST(Nms, ReturnsResultsSortedByDescendingScore) {
    std::vector<FaceDetection> dets{
        make_det(0.0f, 0.0f, 10.0f, 10.0f, 0.5f),
        make_det(100.0f, 0.0f, 10.0f, 10.0f, 0.9f),
        make_det(200.0f, 0.0f, 10.0f, 10.0f, 0.7f),
    };

    const auto kept = nms(std::move(dets), 0.4f);

    ASSERT_EQ(kept.size(), 3u);
    EXPECT_FLOAT_EQ(kept[0].score, 0.9f);
    EXPECT_FLOAT_EQ(kept[1].score, 0.7f);
    EXPECT_FLOAT_EQ(kept[2].score, 0.5f);
}

TEST(Nms, EmptyInputYieldsEmptyOutput) {
    const auto kept = nms({}, 0.4f);
    EXPECT_TRUE(kept.empty());
}

TEST(Nms, SingleBoxSurvivesUnchanged) {
    std::vector<FaceDetection> dets{make_det(3.0f, 4.0f, 20.0f, 30.0f, 0.42f)};

    const auto kept = nms(std::move(dets), 0.4f);

    ASSERT_EQ(kept.size(), 1u);
    EXPECT_FLOAT_EQ(kept.front().bbox.x, 3.0f);
    EXPECT_FLOAT_EQ(kept.front().bbox.y, 4.0f);
    EXPECT_FLOAT_EQ(kept.front().score, 0.42f);
}

TEST(Nms, AThresholdOfOneSuppressesNothingButExactDuplicates) {
    std::vector<FaceDetection> dets{
        make_det(0.0f, 0.0f, 10.0f, 10.0f, 0.9f),
        make_det(1.0f, 0.0f, 10.0f, 10.0f, 0.8f),  // IoU 0.81 < 1.0
    };

    const auto kept = nms(std::move(dets), 1.0f);

    EXPECT_EQ(kept.size(), 2u);
}

// --------------------------------------------------------- decode_stride

/// Build a 2x2 feature map at stride 8 (i.e. a 16x16 input) with a single
/// above-threshold anchor, and check what comes out of the decoder.
TEST(DecodeStride, EmitsOnlyAnchorsAboveTheScoreThreshold) {
    constexpr int kStride = 8;
    constexpr int kSmallInput = 16;
    const int kAnchors = anchor_count(kSmallInput, kSmallInput, kStride);  // 2*2*2 = 8

    std::vector<float> scores(static_cast<std::size_t>(kAnchors), 0.10f);
    scores[5] = 0.90f;  // feature cell (1, 0), anchor 1
    std::vector<float> bboxes(static_cast<std::size_t>(kAnchors) * 4, 1.0f);
    std::vector<float> kps(static_cast<std::size_t>(kAnchors) * 10, 0.0f);

    std::vector<FaceDetection> out;
    face_pipeline::trt::decode_stride(kStride, kSmallInput, kSmallInput, kAnchorsPerLocation,
                                      scores.data(), bboxes.data(), kps.data(),
                                      /*score_thresh=*/0.5f, /*scale=*/1.0f, 0, 0, out);

    ASSERT_EQ(out.size(), 1u);
    EXPECT_FLOAT_EQ(out.front().score, 0.90f);

    // idx 5 = (y * feat_w + x) * 2 + a  =>  y = 1, x = 0, a = 1.
    // Anchor centre is therefore (0, 8); each distance of 1.0 spans 8 px.
    EXPECT_FLOAT_EQ(out.front().bbox.x, -8.0f);
    EXPECT_FLOAT_EQ(out.front().bbox.y, 0.0f);
    EXPECT_FLOAT_EQ(out.front().bbox.width, 16.0f);
    EXPECT_FLOAT_EQ(out.front().bbox.height, 16.0f);
}

TEST(DecodeStride, EmitsNothingWhenEveryScoreIsBelowThreshold) {
    constexpr int kStride = 8;
    constexpr int kSmallInput = 16;
    const int kAnchors = anchor_count(kSmallInput, kSmallInput, kStride);

    const std::vector<float> scores(static_cast<std::size_t>(kAnchors), 0.01f);
    const std::vector<float> bboxes(static_cast<std::size_t>(kAnchors) * 4, 1.0f);
    const std::vector<float> kps(static_cast<std::size_t>(kAnchors) * 10, 0.0f);

    std::vector<FaceDetection> out;
    face_pipeline::trt::decode_stride(kStride, kSmallInput, kSmallInput, kAnchorsPerLocation,
                                      scores.data(), bboxes.data(), kps.data(), 0.5f, 1.0f, 0, 0,
                                      out);

    EXPECT_TRUE(out.empty());
}

TEST(DecodeStride, AppendsToAnExistingResultVector) {
    constexpr int kStride = 8;
    constexpr int kSmallInput = 16;
    const int kAnchors = anchor_count(kSmallInput, kSmallInput, kStride);

    const std::vector<float> scores(static_cast<std::size_t>(kAnchors), 0.99f);
    const std::vector<float> bboxes(static_cast<std::size_t>(kAnchors) * 4, 1.0f);
    const std::vector<float> kps(static_cast<std::size_t>(kAnchors) * 10, 0.0f);

    std::vector<FaceDetection> out{make_det(0.0f, 0.0f, 1.0f, 1.0f, 0.5f)};
    face_pipeline::trt::decode_stride(kStride, kSmallInput, kSmallInput, kAnchorsPerLocation,
                                      scores.data(), bboxes.data(), kps.data(), 0.5f, 1.0f, 0, 0,
                                      out);

    // One pre-existing entry plus every anchor on the 2x2 map.
    EXPECT_EQ(out.size(), 1u + static_cast<std::size_t>(kAnchors));
}

}  // namespace
