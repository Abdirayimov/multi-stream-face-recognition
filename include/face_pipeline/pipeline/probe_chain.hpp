#pragma once

#include <functional>
#include <memory>
#include <vector>

#include "face_pipeline/config/system_config.hpp"
#include "face_pipeline/indexing/searcher_iface.hpp"
#include "face_pipeline/pipeline/frame_meta.hpp"
#include "face_pipeline/trt/encoder_iface.hpp"

namespace face_pipeline::align {
class FaceAligner;
}

namespace face_pipeline::pipeline {

/// Callback signature for fully-resolved per-frame results.
using ResultCallback = std::function<void(const FrameResult&)>;

/// Builds aligned face crops from detector output, runs them through the
/// encoder in a single batched call, queries the index, and applies the
/// margin rule. Designed to be invoked from a DeepStream src-pad probe but
/// depends on no DeepStream symbol itself.
class ProbeChain {
public:
    /// The encoder and the index arrive as interfaces rather than as
    /// ArcFaceEncoder / FaissSearcher, so the chain can be exercised
    /// without a GPU, a TensorRT engine or a populated FAISS index. The
    /// aligner stays concrete: it is pure OpenCV and Eigen already.
    ProbeChain(align::FaceAligner& aligner, trt::IEncoder& encoder, indexing::ISearcher& searcher,
               const config::RecognitionConfig& recognition_cfg);

    /// Process one frame's detection output. The returned FrameResult is
    /// also forwarded to any callback registered via `set_result_callback`.
    FrameResult process(FrameMeta meta, const cv::Mat& image, std::vector<FaceResult> detections);

    void set_result_callback(ResultCallback cb) { callback_ = std::move(cb); }

private:
    align::FaceAligner& aligner_;
    trt::IEncoder& encoder_;
    indexing::ISearcher& searcher_;
    config::RecognitionConfig recognition_cfg_;
    ResultCallback callback_;
};

}  // namespace face_pipeline::pipeline
