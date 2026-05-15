#include "face_pipeline/align/face_aligner.hpp"

#include <opencv2/imgproc.hpp>

#include <Eigen/Dense>

namespace face_pipeline::align {

namespace {

/// Canonical 5-point reference for the 112x112 ArcFace template, as
/// published in the original insightface release. cv::Point2f does not
/// satisfy the literal-type requirement so this is plain const, not
/// constexpr.
const std::array<cv::Point2f, 5> kReference112{{
    {38.2946f, 51.6963f},   // left eye
    {73.5318f, 51.5014f},   // right eye
    {56.0252f, 71.7366f},   // nose
    {41.5493f, 92.3655f},   // left mouth corner
    {70.7299f, 92.2041f},   // right mouth corner
}};

/// Estimate a similarity transform (scale + rotation + translation) that
/// maps `src` to `dst` in the least-squares sense, following Umeyama 1991.
cv::Mat estimate_similarity(const std::array<cv::Point2f, 5>& src,
                            const std::array<cv::Point2f, 5>& dst) {
    Eigen::Matrix<float, 5, 2> S;
    Eigen::Matrix<float, 5, 2> D;
    for (int i = 0; i < 5; ++i) {
        S(i, 0) = src[i].x;
        S(i, 1) = src[i].y;
        D(i, 0) = dst[i].x;
        D(i, 1) = dst[i].y;
    }

    const Eigen::RowVector2f s_mean = S.colwise().mean();
    const Eigen::RowVector2f d_mean = D.colwise().mean();
    const Eigen::Matrix<float, 5, 2> S_c = S.rowwise() - s_mean;
    const Eigen::Matrix<float, 5, 2> D_c = D.rowwise() - d_mean;

    const Eigen::Matrix2f cov = (D_c.transpose() * S_c) / static_cast<float>(S.rows());
    Eigen::JacobiSVD<Eigen::Matrix2f> svd(cov, Eigen::ComputeFullU | Eigen::ComputeFullV);

    Eigen::Matrix2f W = Eigen::Matrix2f::Identity();
    if (svd.matrixU().determinant() * svd.matrixV().determinant() < 0.0f) {
        W(1, 1) = -1.0f;
    }

    const float src_var = (S_c.array().square().rowwise().sum()).mean();
    const float scale = (src_var > 0.0f) ? svd.singularValues().dot(W.diagonal()) / src_var : 1.0f;

    const Eigen::Matrix2f R = svd.matrixU() * W * svd.matrixV().transpose();
    const Eigen::Vector2f t = d_mean.transpose() - scale * R * s_mean.transpose();

    cv::Mat M(2, 3, CV_32F);
    M.at<float>(0, 0) = scale * R(0, 0);
    M.at<float>(0, 1) = scale * R(0, 1);
    M.at<float>(0, 2) = t(0);
    M.at<float>(1, 0) = scale * R(1, 0);
    M.at<float>(1, 1) = scale * R(1, 1);
    M.at<float>(1, 2) = t(1);
    return M;
}

}  // namespace

FaceAligner::FaceAligner(int output_size) : output_size_(output_size) {
    const float k = static_cast<float>(output_size_) / 112.0f;
    for (std::size_t i = 0; i < kReference112.size(); ++i) {
        reference_[i] = cv::Point2f(kReference112[i].x * k, kReference112[i].y * k);
    }
}

cv::Mat FaceAligner::align(const cv::Mat& image,
                           const std::array<cv::Point2f, 5>& landmarks) const {
    const cv::Mat M = estimate_similarity(landmarks, reference_);
    cv::Mat aligned;
    cv::warpAffine(image, aligned, M, cv::Size(output_size_, output_size_), cv::INTER_LINEAR,
                   cv::BORDER_CONSTANT);
    return aligned;
}

}  // namespace face_pipeline::align
