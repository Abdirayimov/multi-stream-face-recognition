#pragma once

#include <functional>
#include <memory>
#include <vector>

#include "face_pipeline/config/system_config.hpp"
#include "face_pipeline/pipeline/frame_meta.hpp"

namespace face_pipeline::align {
class FaceAligner;
}
namespace face_pipeline::trt {
class ArcFaceEncoder;
}
namespace face_pipeline::indexing {
class FaissSearcher;
}

namespace face_pipeline::pipeline {

/// Callback signature for fully-resolved per-frame results.
using ResultCallback = std::function<void(const FrameResult&)>;

/// Builds aligned face crops from detector output, runs them through the
/// encoder in a single batched call, and queries the FAISS index. Designed
/// to be invoked from a DeepStream src-pad probe but does not depend on
/// DeepStream symbols itself, which keeps unit testing simple.
class ProbeChain {
public:
    ProbeChain(align::FaceAligner& aligner,
               trt::ArcFaceEncoder& encoder,
               indexing::FaissSearcher& searcher,
               const config::RecognitionConfig& recognition_cfg);

    /// Process one frame's detection output. The returned FrameResult is
    /// also forwarded to any callback registered via `set_result_callback`.
    FrameResult process(FrameMeta meta,
                        const cv::Mat& image,
                        std::vector<FaceResult> detections);

    void set_result_callback(ResultCallback cb) { callback_ = std::move(cb); }

private:
    align::FaceAligner& aligner_;
    trt::ArcFaceEncoder& encoder_;
    indexing::FaissSearcher& searcher_;
    config::RecognitionConfig recognition_cfg_;
    ResultCallback callback_;
};

}  // namespace face_pipeline::pipeline
