#pragma once

#include <Eigen/Core>
#include <memory>
#include <opencv2/core.hpp>
#include <vector>

#include "face_pipeline/config/system_config.hpp"
#include "face_pipeline/trt/encoder_iface.hpp"

namespace face_pipeline::trt {

class TrtEngine;

/// ArcFace embedding extractor.
///
/// Accepts pre-aligned 112x112 BGR face crops (use FaceAligner first) and
/// returns L2-normalized embeddings of `embedding_dim` floats. The exact
/// model architecture is whichever ArcFace variant produced the engine
/// (typically ResNet50 trained on a public face corpus).
///
/// Reference:
///   Deng et al., "ArcFace: Additive Angular Margin Loss for Deep Face
///   Recognition", CVPR 2019.
class ArcFaceEncoder final : public IEncoder {
public:
    explicit ArcFaceEncoder(const config::EncodingConfig& cfg);
    ~ArcFaceEncoder() override;

    ArcFaceEncoder(const ArcFaceEncoder&) = delete;
    ArcFaceEncoder& operator=(const ArcFaceEncoder&) = delete;

    /// Encode a single aligned face crop.
    Embedding encode(const cv::Mat& aligned_face);

    /// Encode `faces.size()` aligned face crops in a single TRT call.
    /// Faces beyond `cfg.batch_size` are processed in additional batches.
    std::vector<Embedding> encode_batch(const std::vector<cv::Mat>& aligned_faces) override;

    const config::EncodingConfig& config() const noexcept { return cfg_; }

private:
    config::EncodingConfig cfg_;
    std::unique_ptr<TrtEngine> engine_;
    std::vector<float> input_scratch_;
    std::vector<float> output_scratch_;

    void encode_chunk_(const std::vector<cv::Mat>& chunk, std::vector<Embedding>& out);
};

}  // namespace face_pipeline::trt
