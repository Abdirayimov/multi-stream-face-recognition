#include "face_pipeline/align/face_aligner.hpp"

#include <opencv2/imgproc.hpp>

#include "face_pipeline/align/similarity_transform.hpp"

namespace face_pipeline::align {

namespace {

/// Canonical 5-point reference for the 112x112 ArcFace template, as
/// published in the original insightface release. cv::Point2f does not
/// satisfy the literal-type requirement so this is plain const, not
/// constexpr.
const std::array<cv::Point2f, 5> kReference112{{
    {38.2946f, 51.6963f},  // left eye
    {73.5318f, 51.5014f},  // right eye
    {56.0252f, 71.7366f},  // nose
    {41.5493f, 92.3655f},  // left mouth corner
    {70.7299f, 92.2041f},  // right mouth corner
}};

/// Side length the reference points above are expressed in.
constexpr float kReferenceSize = 112.0f;

}  // namespace

FaceAligner::FaceAligner(int output_size) : output_size_(output_size) {
    const float k = static_cast<float>(output_size_) / kReferenceSize;
    for (std::size_t i = 0; i < kReference112.size(); ++i) {
        reference_[i] = cv::Point2f(kReference112[i].x * k, kReference112[i].y * k);
    }
}

cv::Mat FaceAligner::align(const cv::Mat& image,
                           const std::array<cv::Point2f, 5>& landmarks) const {
    const cv::Mat M = estimate_similarity_transform(landmarks, reference_);
    cv::Mat aligned;
    cv::warpAffine(image, aligned, M, cv::Size(output_size_, output_size_), cv::INTER_LINEAR,
                   cv::BORDER_CONSTANT);
    return aligned;
}

}  // namespace face_pipeline::align
