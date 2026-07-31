#include <gtest/gtest.h>

#include <opencv2/core.hpp>
#include <stdexcept>

#include "face_pipeline/trt/scrfd_postprocess.hpp"

namespace {

using face_pipeline::trt::letterbox;
using face_pipeline::trt::undo_letterbox;

/// The SCRFD engines in this repository take a square 640x640 input.
constexpr int kTargetW = 640;
constexpr int kTargetH = 640;

cv::Mat make_image(int width, int height) {
    return cv::Mat(height, width, CV_8UC3, cv::Scalar(30, 60, 90));
}

TEST(Letterbox, LandscapeSourcePadsTopAndBottom) {
    // 640x480 into 640x640: width already fits, so scale is 1 and the
    // 160 leftover rows split evenly above and below.
    const auto result = letterbox(make_image(640, 480), kTargetW, kTargetH);

    EXPECT_FLOAT_EQ(result.scale, 1.0f);
    EXPECT_EQ(result.pad_x, 0);
    EXPECT_EQ(result.pad_y, 80);
    EXPECT_EQ(result.image.cols, kTargetW);
    EXPECT_EQ(result.image.rows, kTargetH);
}

TEST(Letterbox, PortraitSourcePadsLeftAndRight) {
    const auto result = letterbox(make_image(480, 640), kTargetW, kTargetH);

    EXPECT_FLOAT_EQ(result.scale, 1.0f);
    EXPECT_EQ(result.pad_x, 80);
    EXPECT_EQ(result.pad_y, 0);
}

TEST(Letterbox, SquareSourceNeedsNoPadding) {
    const auto result = letterbox(make_image(320, 320), kTargetW, kTargetH);

    EXPECT_FLOAT_EQ(result.scale, 2.0f);
    EXPECT_EQ(result.pad_x, 0);
    EXPECT_EQ(result.pad_y, 0);
}

TEST(Letterbox, DownscalesSourcesLargerThanTheTarget) {
    // 1920x1080 into 640x640: the width is the binding constraint.
    const auto result = letterbox(make_image(1920, 1080), kTargetW, kTargetH);

    EXPECT_FLOAT_EQ(result.scale, 640.0f / 1920.0f);
    EXPECT_EQ(result.pad_x, 0);
    EXPECT_EQ(result.pad_y, (640 - 360) / 2);
    EXPECT_EQ(result.image.cols, kTargetW);
    EXPECT_EQ(result.image.rows, kTargetH);
}

TEST(Letterbox, OutputAlwaysMatchesTheTargetSize) {
    for (const auto& size : {cv::Size(100, 700), cv::Size(1000, 33), cv::Size(1, 1)}) {
        const auto result = letterbox(make_image(size.width, size.height), kTargetW, kTargetH);
        EXPECT_EQ(result.image.cols, kTargetW) << "source " << size;
        EXPECT_EQ(result.image.rows, kTargetH) << "source " << size;
    }
}

TEST(Letterbox, PreservesTheSourcePixelType) {
    const auto result = letterbox(make_image(640, 480), kTargetW, kTargetH);
    EXPECT_EQ(result.image.type(), CV_8UC3);
}

TEST(Letterbox, RoundTripsCoordinatesBackToTheSourceFrame) {
    const auto result = letterbox(make_image(1920, 1080), kTargetW, kTargetH);

    for (const auto& original :
         {cv::Point2f{0.0f, 0.0f}, cv::Point2f{960.0f, 540.0f}, cv::Point2f{1919.0f, 1079.0f}}) {
        // Forward: source -> letterboxed canvas.
        const float lb_x = original.x * result.scale + static_cast<float>(result.pad_x);
        const float lb_y = original.y * result.scale + static_cast<float>(result.pad_y);

        const cv::Point2f back =
            undo_letterbox(lb_x, lb_y, result.scale, result.pad_x, result.pad_y);

        EXPECT_NEAR(back.x, original.x, 1e-3f);
        EXPECT_NEAR(back.y, original.y, 1e-3f);
    }
}

TEST(Letterbox, RejectsAnEmptySource) {
    EXPECT_THROW(letterbox(cv::Mat(), kTargetW, kTargetH), std::invalid_argument);
}

TEST(Letterbox, RejectsANonPositiveTargetSize) {
    const cv::Mat src = make_image(640, 480);
    EXPECT_THROW(letterbox(src, 0, kTargetH), std::invalid_argument);
    EXPECT_THROW(letterbox(src, kTargetW, -1), std::invalid_argument);
}

TEST(Preprocess, ConvertsBgrHwcToNormalizedRgbChw) {
    using face_pipeline::trt::hwc_bgr_to_chw_rgb_normalized;

    // A single pixel makes the channel reordering unambiguous.
    cv::Mat src(1, 1, CV_8UC3);
    src.at<cv::Vec3b>(0, 0) = cv::Vec3b(0, 128, 255);  // B, G, R

    std::array<float, 3> dst{};
    hwc_bgr_to_chw_rgb_normalized(src, dst.data());

    // R plane first, then G, then B; each mapped by (v - 127.5) / 128.
    EXPECT_NEAR(dst[0], (255.0f - 127.5f) / 128.0f, 1e-6f);
    EXPECT_NEAR(dst[1], (128.0f - 127.5f) / 128.0f, 1e-6f);
    EXPECT_NEAR(dst[2], (0.0f - 127.5f) / 128.0f, 1e-6f);
}

TEST(Preprocess, RejectsANonBgrImage) {
    using face_pipeline::trt::hwc_bgr_to_chw_rgb_normalized;

    const cv::Mat grayscale(4, 4, CV_8UC1, cv::Scalar(0));
    std::array<float, 48> dst{};
    EXPECT_THROW(hwc_bgr_to_chw_rgb_normalized(grayscale, dst.data()), std::invalid_argument);
}

}  // namespace
