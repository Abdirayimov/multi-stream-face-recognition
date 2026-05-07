#pragma once

#include <opencv2/core.hpp>

#include <array>

namespace face_pipeline::align {

/// 5-point face aligner for the standard insightface 112x112 reference.
///
/// Computes a similarity transform from the detected landmarks (eye L/R,
/// nose, mouth L/R) to the canonical reference points published with the
/// original ArcFace release, then warps the face crop to the target size.
class FaceAligner {
public:
    /// Construct an aligner that produces `output_size x output_size` crops.
    explicit FaceAligner(int output_size = 112);

    /// Align a face. `landmarks` are in the source-image coordinate frame.
    cv::Mat align(const cv::Mat& image,
                  const std::array<cv::Point2f, 5>& landmarks) const;

    int output_size() const noexcept { return output_size_; }

private:
    int output_size_;
    std::array<cv::Point2f, 5> reference_;
};

}  // namespace face_pipeline::align
