#pragma once

#include <array>
#include <cstddef>
#include <opencv2/core.hpp>
#include <vector>

namespace face_pipeline::trt {

/// Feature-map strides emitted by the standard SCRFD export.
inline constexpr std::array<int, 3> kStrides{8, 16, 32};

/// Anchors placed at every feature-map location, for every stride.
inline constexpr int kAnchorsPerLocation = 2;

/// Number of landmarks SCRFD regresses per face.
inline constexpr int kNumLandmarks = 5;

/// One face detection produced by SCRFD.
struct FaceDetection {
    cv::Rect2f bbox;                       ///< In original image coordinates.
    std::array<cv::Point2f, 5> landmarks;  ///< Eye L/R, nose, mouth L/R.
    float score = 0.0f;
};

/// A letterboxed image plus the parameters needed to map coordinates back
/// to the original frame.
struct LetterboxResult {
    cv::Mat image;       ///< `target_h x target_w`, aspect ratio preserved.
    float scale = 1.0f;  ///< Uniform resize factor applied to the source.
    int pad_x = 0;       ///< Left padding, in letterboxed pixels.
    int pad_y = 0;       ///< Top padding, in letterboxed pixels.
};

/// Resize `src` to fit inside `target_w x target_h` without distortion and
/// center it on a zero-filled canvas.
///
/// @throws std::invalid_argument if `src` is empty or the target size is
///         not positive.
LetterboxResult letterbox(const cv::Mat& src, int target_w, int target_h);

/// Map a point from letterboxed coordinates back to the original frame.
inline cv::Point2f undo_letterbox(float x, float y, float scale, int pad_x, int pad_y) noexcept {
    return {(x - static_cast<float>(pad_x)) / scale, (y - static_cast<float>(pad_y)) / scale};
}

/// Convert HWC BGR uint8 -> CHW RGB float32 normalized to [-1, 1] using the
/// SCRFD insightface convention (mean 127.5, scale 1/128). `dst` must hold
/// at least `3 * src.rows * src.cols` floats.
///
/// @throws std::invalid_argument if `src` is not CV_8UC3.
void hwc_bgr_to_chw_rgb_normalized(const cv::Mat& src, float* dst);

/// Number of anchors a single stride level contributes for the given input
/// size, i.e. `(input_h / stride) * (input_w / stride) * anchors_per_loc`.
///
/// @throws std::invalid_argument if `stride` is not positive.
int anchor_count(int input_w, int input_h, int stride,
                 int anchors_per_location = kAnchorsPerLocation);

/// Total anchor count across every stride in `kStrides`.
int total_anchor_count(int input_w, int input_h, int anchors_per_location = kAnchorsPerLocation);

/// Convert an anchor point plus four regressed distances (left, top, right,
/// bottom, in stride units) into a box in original-image coordinates.
cv::Rect2f distance2bbox(cv::Point2f anchor, const float* distances, float stride, float scale,
                         int pad_x, int pad_y) noexcept;

/// Convert an anchor point plus five regressed (dx, dy) offsets (in stride
/// units) into landmarks in original-image coordinates.
std::array<cv::Point2f, 5> distance2kps(cv::Point2f anchor, const float* offsets, float stride,
                                        float scale, int pad_x, int pad_y) noexcept;

/// Intersection over union of two axis-aligned boxes. Returns 0 when the
/// union is degenerate.
float iou(const cv::Rect2f& a, const cv::Rect2f& b) noexcept;

/// Greedy non-maximum suppression. Detections are sorted by descending
/// score; a candidate is dropped when it overlaps an already-kept box by
/// more than `iou_thresh`. An empty input yields an empty output.
std::vector<FaceDetection> nms(std::vector<FaceDetection> dets, float iou_thresh);

/// Decode one stride level from raw SCRFD outputs and append every
/// above-threshold detection to `out`.
///
/// Outputs follow the upstream insightface layout:
///   scores: (A*HW, 1)
///   bboxes: (A*HW, 4)   distances (l, t, r, b) in stride units
///   kps:    (A*HW, 10)  five (x, y) offsets in stride units
void decode_stride(int stride, int input_w, int input_h, int anchors_per_location,
                   const float* scores, const float* bboxes, const float* kps, float score_thresh,
                   float scale, int pad_x, int pad_y, std::vector<FaceDetection>& out);

}  // namespace face_pipeline::trt
