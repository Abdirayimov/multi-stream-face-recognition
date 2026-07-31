#pragma once

#include <array>
#include <map>
#include <memory>
#include <opencv2/core.hpp>
#include <opencv2/core/cuda.hpp>
#include <string>
#include <vector>

#include "face_pipeline/config/system_config.hpp"
#include "face_pipeline/trt/scrfd_postprocess.hpp"

namespace face_pipeline::trt {

class TrtEngine;

// `FaceDetection` and the stride/anchor constants live in
// scrfd_postprocess.hpp so the decoding math stays usable — and unit
// testable — without linking TensorRT.

/// SCRFD (Sample and Computation Redistribution for Face Detection) wrapper.
///
/// Implements the original three-stride decoder (8/16/32) with NMS. Input is
/// a square `input_width x input_height` BGR image; outputs are decoded back
/// into the original (un-resized) coordinate frame using the letterbox scale.
///
/// Reference:
///   Guo et al., "Sample and Computation Redistribution for Efficient Face
///   Detection", arXiv:2105.04714, 2021.
class SCRFDDetector {
public:
    explicit SCRFDDetector(const config::DetectionConfig& cfg);
    ~SCRFDDetector();

    SCRFDDetector(const SCRFDDetector&) = delete;
    SCRFDDetector& operator=(const SCRFDDetector&) = delete;

    /// Run detection on a single image. Image may be CPU or GPU; CPU images
    /// will be uploaded internally.
    std::vector<FaceDetection> detect(const cv::Mat& image);

    /// Batched detection over `images.size()` frames. The TRT engine must be
    /// built with a matching dynamic batch profile.
    std::vector<std::vector<FaceDetection>> detect_batch(const std::vector<cv::Mat>& images);

    const config::DetectionConfig& config() const noexcept { return cfg_; }

private:
    config::DetectionConfig cfg_;
    std::unique_ptr<TrtEngine> engine_;

    // Pre-allocated host scratch space, sized on first use.
    std::vector<float> input_scratch_;

    std::string input_name_;
    /// Output binding names per stride, resolved from binding shapes so
    /// the detector works with either the `score_8`/`bbox_8`/`kps_8`
    /// naming or the numeric output names that insightface's stock
    /// SCRFD exports (e.g. buffalo_l det_10g) produce.
    struct StrideBindings {
        std::string score;  ///< (N, 1)
        std::string bbox;   ///< (N, 4)
        std::string kps;    ///< (N, 10)
    };
    std::map<int, StrideBindings> stride_bindings_;

    void resolve_bindings_();
};

}  // namespace face_pipeline::trt
