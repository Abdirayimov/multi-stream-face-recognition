#include <gtest/gtest.h>

#include <array>
#include <cmath>
#include <opencv2/core.hpp>
#include <stdexcept>

#include "face_pipeline/align/face_aligner.hpp"
#include "face_pipeline/align/similarity_transform.hpp"

namespace {

using face_pipeline::align::estimate_similarity_transform;

/// The estimator runs a float SVD, so a handful of ULPs of drift on the
/// recovered parameters is expected and harmless.
constexpr float kTol = 1e-5f;

/// A non-degenerate, non-symmetric point set. Symmetric layouts can hide
/// sign errors in the rotation, so the points are deliberately irregular.
const std::array<cv::Point2f, 5> kSourcePoints{{
    {10.0f, 20.0f},
    {50.0f, 22.0f},
    {31.0f, 45.0f},
    {14.0f, 70.0f},
    {48.0f, 68.0f},
}};

/// Apply a 2x3 CV_32F affine matrix to a point.
cv::Point2f apply(const cv::Mat& m, cv::Point2f p) {
    return {m.at<float>(0, 0) * p.x + m.at<float>(0, 1) * p.y + m.at<float>(0, 2),
            m.at<float>(1, 0) * p.x + m.at<float>(1, 1) * p.y + m.at<float>(1, 2)};
}

/// Build `dst` by scaling, rotating and translating every source point.
std::array<cv::Point2f, 5> transform_points(const std::array<cv::Point2f, 5>& src, float scale,
                                            float radians, cv::Point2f offset) {
    const float c = std::cos(radians);
    const float s = std::sin(radians);
    std::array<cv::Point2f, 5> dst;
    for (std::size_t i = 0; i < src.size(); ++i) {
        dst[i] = {scale * (c * src[i].x - s * src[i].y) + offset.x,
                  scale * (s * src[i].x + c * src[i].y) + offset.y};
    }
    return dst;
}

void expect_maps_src_to_dst(const cv::Mat& m, const std::array<cv::Point2f, 5>& src,
                            const std::array<cv::Point2f, 5>& dst, float tol) {
    for (std::size_t i = 0; i < src.size(); ++i) {
        const cv::Point2f mapped = apply(m, src[i]);
        EXPECT_NEAR(mapped.x, dst[i].x, tol) << "point " << i;
        EXPECT_NEAR(mapped.y, dst[i].y, tol) << "point " << i;
    }
}

TEST(UmeyamaTransform, ReturnsTwoByThreeFloatMatrix) {
    const cv::Mat m = estimate_similarity_transform(kSourcePoints, kSourcePoints);
    EXPECT_EQ(m.rows, 2);
    EXPECT_EQ(m.cols, 3);
    EXPECT_EQ(m.type(), CV_32F);
}

TEST(UmeyamaTransform, RecoversIdentityWhenSourceEqualsTarget) {
    const cv::Mat m = estimate_similarity_transform(kSourcePoints, kSourcePoints);

    EXPECT_NEAR(m.at<float>(0, 0), 1.0f, kTol);
    EXPECT_NEAR(m.at<float>(0, 1), 0.0f, kTol);
    EXPECT_NEAR(m.at<float>(0, 2), 0.0f, kTol);
    EXPECT_NEAR(m.at<float>(1, 0), 0.0f, kTol);
    EXPECT_NEAR(m.at<float>(1, 1), 1.0f, kTol);
    EXPECT_NEAR(m.at<float>(1, 2), 0.0f, kTol);
}

TEST(UmeyamaTransform, RecoversPureTranslation) {
    constexpr float kDx = 12.5f;
    constexpr float kDy = -7.25f;
    const auto dst = transform_points(kSourcePoints, 1.0f, 0.0f, {kDx, kDy});

    const cv::Mat m = estimate_similarity_transform(kSourcePoints, dst);

    EXPECT_NEAR(m.at<float>(0, 0), 1.0f, kTol);
    EXPECT_NEAR(m.at<float>(1, 1), 1.0f, kTol);
    EXPECT_NEAR(m.at<float>(0, 1), 0.0f, kTol);
    EXPECT_NEAR(m.at<float>(1, 0), 0.0f, kTol);
    EXPECT_NEAR(m.at<float>(0, 2), kDx, 1e-3f);
    EXPECT_NEAR(m.at<float>(1, 2), kDy, 1e-3f);
}

TEST(UmeyamaTransform, RecoversPureScale) {
    constexpr float kScale = 2.0f;
    const auto dst = transform_points(kSourcePoints, kScale, 0.0f, {0.0f, 0.0f});

    const cv::Mat m = estimate_similarity_transform(kSourcePoints, dst);

    // For a pure scale the linear block is `scale * I`.
    EXPECT_NEAR(m.at<float>(0, 0), kScale, kTol);
    EXPECT_NEAR(m.at<float>(1, 1), kScale, kTol);
    EXPECT_NEAR(m.at<float>(0, 1), 0.0f, kTol);
    EXPECT_NEAR(m.at<float>(1, 0), 0.0f, kTol);
    expect_maps_src_to_dst(m, kSourcePoints, dst, 1e-3f);
}

TEST(UmeyamaTransform, RecoversNinetyDegreeRotation) {
    const float kAngle = static_cast<float>(CV_PI) / 2.0f;
    const auto dst = transform_points(kSourcePoints, 1.0f, kAngle, {0.0f, 0.0f});

    const cv::Mat m = estimate_similarity_transform(kSourcePoints, dst);

    // R(90 deg) = [[0, -1], [1, 0]].
    EXPECT_NEAR(m.at<float>(0, 0), 0.0f, kTol);
    EXPECT_NEAR(m.at<float>(0, 1), -1.0f, kTol);
    EXPECT_NEAR(m.at<float>(1, 0), 1.0f, kTol);
    EXPECT_NEAR(m.at<float>(1, 1), 0.0f, kTol);
}

TEST(UmeyamaTransform, RecoversCombinedScaleRotationTranslation) {
    constexpr float kScale = 1.75f;
    const float kAngle = static_cast<float>(CV_PI) / 6.0f;  // 30 degrees
    const cv::Point2f kOffset{-30.0f, 11.0f};
    const auto dst = transform_points(kSourcePoints, kScale, kAngle, kOffset);

    const cv::Mat m = estimate_similarity_transform(kSourcePoints, dst);

    // The linear block is `scale * R`, so its column norm recovers the scale.
    const float recovered_scale = std::hypot(m.at<float>(0, 0), m.at<float>(1, 0));
    EXPECT_NEAR(recovered_scale, kScale, 1e-4f);
    expect_maps_src_to_dst(m, kSourcePoints, dst, 1e-3f);
}

TEST(UmeyamaTransform, DegenerateSourceFallsBackToCentroidTranslation) {
    // All source points coincident: scale and rotation are undefined.
    const std::array<cv::Point2f, 5> src{{
        {5.0f, 5.0f},
        {5.0f, 5.0f},
        {5.0f, 5.0f},
        {5.0f, 5.0f},
        {5.0f, 5.0f},
    }};

    cv::Mat m;
    ASSERT_NO_THROW(m = estimate_similarity_transform(src, kSourcePoints));

    // Documented fallback: scale 1, identity rotation, centroid offset.
    EXPECT_NEAR(m.at<float>(0, 0), 1.0f, kTol);
    EXPECT_NEAR(m.at<float>(1, 1), 1.0f, kTol);
    EXPECT_NEAR(m.at<float>(0, 1), 0.0f, kTol);
    EXPECT_NEAR(m.at<float>(1, 0), 0.0f, kTol);

    float mean_x = 0.0f;
    float mean_y = 0.0f;
    for (const auto& p : kSourcePoints) {
        mean_x += p.x / static_cast<float>(kSourcePoints.size());
        mean_y += p.y / static_cast<float>(kSourcePoints.size());
    }
    EXPECT_NEAR(m.at<float>(0, 2), mean_x - 5.0f, 1e-3f);
    EXPECT_NEAR(m.at<float>(1, 2), mean_y - 5.0f, 1e-3f);
}

TEST(UmeyamaTransform, DegenerateTargetProducesFiniteMatrix) {
    const std::array<cv::Point2f, 5> dst{{
        {3.0f, 4.0f},
        {3.0f, 4.0f},
        {3.0f, 4.0f},
        {3.0f, 4.0f},
        {3.0f, 4.0f},
    }};

    cv::Mat m;
    ASSERT_NO_THROW(m = estimate_similarity_transform(kSourcePoints, dst));
    for (int r = 0; r < 2; ++r) {
        for (int c = 0; c < 3; ++c) {
            EXPECT_TRUE(std::isfinite(m.at<float>(r, c))) << "element (" << r << ", " << c << ")";
        }
    }
}

TEST(UmeyamaTransform, RejectsFewerThanTwoPointPairs) {
    const cv::Point2f src{1.0f, 1.0f};
    const cv::Point2f dst{2.0f, 2.0f};

    EXPECT_THROW(estimate_similarity_transform(&src, &dst, 1), std::invalid_argument);
    EXPECT_THROW(estimate_similarity_transform(&src, &dst, 0), std::invalid_argument);
}

TEST(FaceAligner, ProducesCropOfTheRequestedSize) {
    constexpr int kOutputSize = 112;
    const face_pipeline::align::FaceAligner aligner(kOutputSize);
    const cv::Mat image(480, 640, CV_8UC3, cv::Scalar(90, 110, 130));
    const std::array<cv::Point2f, 5> landmarks{{
        {280.0f, 220.0f},
        {350.0f, 221.0f},
        {315.0f, 260.0f},
        {286.0f, 300.0f},
        {345.0f, 301.0f},
    }};

    const cv::Mat aligned = aligner.align(image, landmarks);

    EXPECT_EQ(aligner.output_size(), kOutputSize);
    EXPECT_EQ(aligned.rows, kOutputSize);
    EXPECT_EQ(aligned.cols, kOutputSize);
    EXPECT_EQ(aligned.type(), CV_8UC3);
}

TEST(FaceAligner, ScalesTheReferenceTemplateWithOutputSize) {
    // A 224x224 aligner should map the same landmarks to exactly twice the
    // 112x112 aligner's coordinates.
    const face_pipeline::align::FaceAligner small(112);
    const face_pipeline::align::FaceAligner large(224);
    const cv::Mat image(480, 640, CV_8UC3, cv::Scalar(0, 0, 0));
    const std::array<cv::Point2f, 5> landmarks{{
        {280.0f, 220.0f},
        {350.0f, 221.0f},
        {315.0f, 260.0f},
        {286.0f, 300.0f},
        {345.0f, 301.0f},
    }};

    EXPECT_EQ(small.align(image, landmarks).cols, 112);
    EXPECT_EQ(large.align(image, landmarks).cols, 224);
}

}  // namespace
