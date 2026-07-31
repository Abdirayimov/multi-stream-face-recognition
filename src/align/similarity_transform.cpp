#include "face_pipeline/align/similarity_transform.hpp"

#include <Eigen/Dense>
#include <stdexcept>

namespace face_pipeline::align {

cv::Mat estimate_similarity_transform(const cv::Point2f* src, const cv::Point2f* dst,
                                      std::size_t n) {
    if (n < 2) {
        throw std::invalid_argument("estimate_similarity_transform: need at least two point pairs");
    }

    const auto rows = static_cast<Eigen::Index>(n);
    Eigen::MatrixX2f S(rows, 2);
    Eigen::MatrixX2f D(rows, 2);
    for (Eigen::Index i = 0; i < rows; ++i) {
        const auto k = static_cast<std::size_t>(i);
        S(i, 0) = src[k].x;
        S(i, 1) = src[k].y;
        D(i, 0) = dst[k].x;
        D(i, 1) = dst[k].y;
    }

    const Eigen::RowVector2f s_mean = S.colwise().mean();
    const Eigen::RowVector2f d_mean = D.colwise().mean();
    const Eigen::MatrixX2f S_c = S.rowwise() - s_mean;
    const Eigen::MatrixX2f D_c = D.rowwise() - d_mean;

    const Eigen::Matrix2f cov = (D_c.transpose() * S_c) / static_cast<float>(rows);
    Eigen::JacobiSVD<Eigen::Matrix2f> svd(cov, Eigen::ComputeFullU | Eigen::ComputeFullV);

    // Guard against the reflection case: if U and V have opposite
    // orientation the raw SVD product would mirror rather than rotate.
    Eigen::Matrix2f W = Eigen::Matrix2f::Identity();
    if (svd.matrixU().determinant() * svd.matrixV().determinant() < 0.0f) {
        W(1, 1) = -1.0f;
    }

    const float src_var = (S_c.array().square().rowwise().sum()).mean();
    const float scale = (src_var > 0.0f) ? svd.singularValues().dot(W.diagonal()) / src_var : 1.0f;

    const Eigen::Matrix2f R = (src_var > 0.0f)
                                  ? Eigen::Matrix2f(svd.matrixU() * W * svd.matrixV().transpose())
                                  : Eigen::Matrix2f(Eigen::Matrix2f::Identity());
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

}  // namespace face_pipeline::align
