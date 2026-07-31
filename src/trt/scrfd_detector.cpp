#include "face_pipeline/trt/scrfd_detector.hpp"

#include <cmath>

#include "face_pipeline/trt/scrfd_postprocess.hpp"
#include "face_pipeline/trt/trt_engine.hpp"
#include "face_pipeline/utils/cuda_helpers.hpp"
#include "face_pipeline/utils/logger.hpp"

namespace face_pipeline::trt {

SCRFDDetector::SCRFDDetector(const config::DetectionConfig& cfg)
    : cfg_(cfg), engine_(std::make_unique<TrtEngine>(cfg.engine_path)) {
    input_scratch_.resize(static_cast<std::size_t>(3 * cfg.input_height * cfg.input_width));
    resolve_bindings_();
}

SCRFDDetector::~SCRFDDetector() = default;

void SCRFDDetector::resolve_bindings_() {
    // Input: first binding. Pin its (dynamic) H/W to the configured
    // size so the device buffer is allocated.
    input_name_ = engine_->bindings().front().name;
    engine_->set_input_shape(input_name_, {1, 3, static_cast<std::int64_t>(cfg_.input_height),
                                           static_cast<std::int64_t>(cfg_.input_width)});

    // Outputs: SCRFD emits three tensors per stride. Identify each by
    // its trailing dimension (1 = score, 4 = bbox, 10 = kps) and map it
    // to a stride via the anchor count N = (input/stride)^2 * anchors.
    // Robust to the output *names*, which differ between a hand-named
    // export and insightface's numeric stock export (buffalo_l).
    for (const auto& b : engine_->bindings()) {
        if (b.is_input || b.shape.empty())
            continue;
        const std::int64_t last = b.shape.back();
        const std::int64_t n = b.shape.front();
        if (n <= 0)
            continue;
        const double cells = static_cast<double>(n) / kAnchorsPerLocation;
        const double feat = std::sqrt(cells);
        if (feat <= 0.0)
            continue;
        const int stride =
            static_cast<int>(std::lround(static_cast<double>(cfg_.input_width) / feat));
        if (stride <= 0)
            continue;
        auto& sb = stride_bindings_[stride];
        if (last == 1)
            sb.score = b.name;
        else if (last == 4)
            sb.bbox = b.name;
        else if (last == 10)
            sb.kps = b.name;
    }
}

std::vector<FaceDetection> SCRFDDetector::detect(const cv::Mat& image) {
    auto results = detect_batch({image});
    return results.empty() ? std::vector<FaceDetection>{} : std::move(results.front());
}

std::vector<std::vector<FaceDetection>> SCRFDDetector::detect_batch(
    const std::vector<cv::Mat>& images) {
    std::vector<std::vector<FaceDetection>> all;
    all.reserve(images.size());

    utils::CudaStream stream;

    for (const auto& src : images) {
        const auto lb =
            letterbox(src, static_cast<int>(cfg_.input_width), static_cast<int>(cfg_.input_height));
        hwc_bgr_to_chw_rgb_normalized(lb.image, input_scratch_.data());

        engine_->copy_input(input_name_, input_scratch_.data(),
                            input_scratch_.size() * sizeof(float), stream.get());
        engine_->infer(stream.get());

        std::vector<FaceDetection> raw;
        for (const auto& [stride, names] : stride_bindings_) {
            if (names.score.empty() || names.bbox.empty() || names.kps.empty())
                continue;

            const auto& sb = engine_->binding(names.score);
            std::vector<float> scores(sb.volume);
            const auto& bb = engine_->binding(names.bbox);
            std::vector<float> bboxes(bb.volume);
            const auto& kb = engine_->binding(names.kps);
            std::vector<float> kps(kb.volume);

            engine_->copy_output(names.score, scores.data(), scores.size() * sizeof(float),
                                 stream.get());
            engine_->copy_output(names.bbox, bboxes.data(), bboxes.size() * sizeof(float),
                                 stream.get());
            engine_->copy_output(names.kps, kps.data(), kps.size() * sizeof(float), stream.get());
            stream.synchronize();

            decode_stride(stride, static_cast<int>(cfg_.input_width),
                          static_cast<int>(cfg_.input_height), kAnchorsPerLocation, scores.data(),
                          bboxes.data(), kps.data(), cfg_.confidence_threshold, lb.scale, lb.pad_x,
                          lb.pad_y, raw);
        }

        all.push_back(nms(std::move(raw), cfg_.nms_iou_threshold));
    }

    return all;
}

}  // namespace face_pipeline::trt
