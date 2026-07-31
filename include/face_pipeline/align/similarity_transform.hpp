#pragma once

#include <array>
#include <cstddef>
#include <opencv2/core.hpp>

namespace face_pipeline::align {

/// Estimate the similarity transform (uniform scale + rotation +
/// translation) that maps `src` onto `dst` in the least-squares sense,
/// following Umeyama (1991).
///
/// @param src  Source points.
/// @param dst  Target points, same count as `src`.
/// @param n    Number of point pairs; must be >= 2.
/// @return     A 2x3 CV_32F affine matrix `[sR | t]` suitable for
///             `cv::warpAffine`.
///
/// Degenerate input (all source points coincident) has no defined scale or
/// rotation. Rather than dividing by zero, the estimator falls back to
/// scale 1 and identity rotation, so the result reduces to the translation
/// between the two centroids.
///
/// Reference:
///   Umeyama, "Least-squares estimation of transformation parameters
///   between two point patterns", IEEE TPAMI 13(4), 1991.
///
/// @throws std::invalid_argument if `n < 2`.
cv::Mat estimate_similarity_transform(const cv::Point2f* src, const cv::Point2f* dst,
                                      std::size_t n);

/// Convenience overload for fixed-size landmark sets.
template <std::size_t N>
cv::Mat estimate_similarity_transform(const std::array<cv::Point2f, N>& src,
                                      const std::array<cv::Point2f, N>& dst) {
    static_assert(N >= 2, "a similarity transform needs at least two point pairs");
    return estimate_similarity_transform(src.data(), dst.data(), N);
}

}  // namespace face_pipeline::align
