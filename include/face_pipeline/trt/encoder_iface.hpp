#pragma once

#include <Eigen/Core>
#include <opencv2/core.hpp>
#include <vector>

namespace face_pipeline::trt {

/// An L2-normalized face embedding.
///
/// Every producer in this project returns unit-norm vectors, which is what
/// lets consumers treat a dot product as a cosine similarity.
using Embedding = Eigen::Matrix<float, Eigen::Dynamic, 1>;

/// What the probe chain needs from an embedding backend.
///
/// Narrow on purpose: one method, the batched one. ArcFaceEncoder is the
/// production implementation and needs a TensorRT engine, so depending on
/// the concrete class would put the whole recognition chain behind a GPU
/// and a serialized model. Behind this interface the chain is ordinary
/// C++ and can be tested with a stub.
class IEncoder {
public:
    virtual ~IEncoder() = default;

    /// Encode pre-aligned face crops. The returned vector is expected to
    /// be the same length as `aligned_faces`, in the same order.
    virtual std::vector<Embedding> encode_batch(const std::vector<cv::Mat>& aligned_faces) = 0;
};

}  // namespace face_pipeline::trt
