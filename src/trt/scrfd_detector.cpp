#include "face_pipeline/trt/scrfd_detector.hpp"

#include <opencv2/imgproc.hpp>

#include <algorithm>
#include <cmath>

#include "face_pipeline/trt/trt_engine.hpp"
#include "face_pipeline/utils/cuda_helpers.hpp"
#include "face_pipeline/utils/logger.hpp"

namespace face_pipeline::trt {

namespace {

constexpr std::array<int, 3> kStrides{8, 16, 32};
constexpr int kAnchorsPerLocation = 2;

struct LetterboxResult {
    cv::Mat image;
    float scale = 1.0f;
    int pad_x = 0;
    int pad_y = 0;
};

LetterboxResult letterbox(const cv::Mat& src, int target_w, int target_h) {
    const float r = std::min(static_cast<float>(target_w) / static_cast<float>(src.cols),
                             static_cast<float>(target_h) / static_cast<float>(src.rows));
    const int new_w = static_cast<int>(std::round(static_cast<float>(src.cols) * r));
    const int new_h = static_cast<int>(std::round(static_cast<float>(src.rows) * r));
    const int pad_x = (target_w - new_w) / 2;
    const int pad_y = (target_h - new_h) / 2;

    cv::Mat resized;
    cv::resize(src, resized, cv::Size(new_w, new_h), 0, 0, cv::INTER_LINEAR);

    cv::Mat out(target_h, target_w, src.type(), cv::Scalar(0, 0, 0));
    resized.copyTo(out(cv::Rect(pad_x, pad_y, new_w, new_h)));
    return {out, r, pad_x, pad_y};
}

/// Convert HWC BGR uint8 -> CHW RGB float32 normalized to [-1, 1] using
/// the SCRFD insightface convention (mean 127.5, scale 1/128).
void hwc_bgr_to_chw_rgb_normalized(const cv::Mat& src, float* dst) {
    const int H = src.rows;
    const int W = src.cols;
    const int channel_stride = H * W;
    for (int y = 0; y < H; ++y) {
        const auto* row = src.ptr<cv::Vec3b>(y);
        for (int x = 0; x < W; ++x) {
            const auto& px = row[x];
            const int idx = y * W + x;
            // BGR -> RGB and to CHW
            dst[0 * channel_stride + idx] = (static_cast<float>(px[2]) - 127.5f) / 128.0f;
            dst[1 * channel_stride + idx] = (static_cast<float>(px[1]) - 127.5f) / 128.0f;
            dst[2 * channel_stride + idx] = (static_cast<float>(px[0]) - 127.5f) / 128.0f;
        }
    }
}

float iou(const cv::Rect2f& a, const cv::Rect2f& b) {
    const float xx1 = std::max(a.x, b.x);
    const float yy1 = std::max(a.y, b.y);
    const float xx2 = std::min(a.x + a.width, b.x + b.width);
    const float yy2 = std::min(a.y + a.height, b.y + b.height);
    const float w = std::max(0.0f, xx2 - xx1);
    const float h = std::max(0.0f, yy2 - yy1);
    const float inter = w * h;
    const float union_area = a.area() + b.area() - inter;
    return (union_area > 0.0f) ? inter / union_area : 0.0f;
}

std::vector<FaceDetection> nms(std::vector<FaceDetection> dets, float iou_thresh) {
    std::sort(dets.begin(), dets.end(),
              [](const FaceDetection& a, const FaceDetection& b) { return a.score > b.score; });

    std::vector<FaceDetection> kept;
    std::vector<bool> suppressed(dets.size(), false);
    for (std::size_t i = 0; i < dets.size(); ++i) {
        if (suppressed[i]) continue;
        kept.push_back(dets[i]);
        for (std::size_t j = i + 1; j < dets.size(); ++j) {
            if (suppressed[j]) continue;
            if (iou(dets[i].bbox, dets[j].bbox) > iou_thresh) {
                suppressed[j] = true;
            }
        }
    }
    return kept;
}

/// Decode one stride level from raw SCRFD outputs.
///
/// Outputs follow the upstream insightface naming convention:
///   score_<stride>: (1, A*HW, 1)
///   bbox_<stride>:  (1, A*HW, 4)   distances (l, t, r, b) in stride units
///   kps_<stride>:   (1, A*HW, 10)  five (x, y) landmarks in stride units
void decode_stride(int stride, int input_w, int input_h, int A,
                   const float* scores, const float* bboxes, const float* kps,
                   float score_thresh, float scale, int pad_x, int pad_y,
                   std::vector<FaceDetection>& out) {
    const int feat_h = input_h / stride;
    const int feat_w = input_w / stride;

    for (int y = 0; y < feat_h; ++y) {
        for (int x = 0; x < feat_w; ++x) {
            for (int a = 0; a < A; ++a) {
                const int idx = (y * feat_w + x) * A + a;
                const float score = scores[idx];
                if (score < score_thresh) continue;

                const float cx = static_cast<float>(x * stride);
                const float cy = static_cast<float>(y * stride);

                const float* bb = bboxes + idx * 4;
                const float l = bb[0] * static_cast<float>(stride);
                const float t = bb[1] * static_cast<float>(stride);
                const float r = bb[2] * static_cast<float>(stride);
                const float b = bb[3] * static_cast<float>(stride);

                FaceDetection d;
                const float x1 = ((cx - l) - static_cast<float>(pad_x)) / scale;
                const float y1 = ((cy - t) - static_cast<float>(pad_y)) / scale;
                const float x2 = ((cx + r) - static_cast<float>(pad_x)) / scale;
                const float y2 = ((cy + b) - static_cast<float>(pad_y)) / scale;
                d.bbox = cv::Rect2f(x1, y1, x2 - x1, y2 - y1);
                d.score = score;

                const float* kp = kps + idx * 10;
                for (int k = 0; k < 5; ++k) {
                    const float kx =
                        ((cx + kp[k * 2 + 0] * static_cast<float>(stride)) -
                         static_cast<float>(pad_x)) / scale;
                    const float ky =
                        ((cy + kp[k * 2 + 1] * static_cast<float>(stride)) -
                         static_cast<float>(pad_y)) / scale;
                    d.landmarks[static_cast<std::size_t>(k)] = cv::Point2f(kx, ky);
                }
                out.push_back(d);
            }
        }
    }
}

}  // namespace

SCRFDDetector::SCRFDDetector(const config::DetectionConfig& cfg)
    : cfg_(cfg), engine_(std::make_unique<TrtEngine>(cfg.engine_path)) {
    input_scratch_.resize(static_cast<std::size_t>(3 * cfg.input_height * cfg.input_width));
}

SCRFDDetector::~SCRFDDetector() = default;

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
        const auto lb = letterbox(src, static_cast<int>(cfg_.input_width),
                                  static_cast<int>(cfg_.input_height));
        hwc_bgr_to_chw_rgb_normalized(lb.image, input_scratch_.data());

        // Single input named "input.1" by default in SCRFD ONNX exports.
        const std::string input_name = engine_->bindings().front().name;
        engine_->copy_input(input_name, input_scratch_.data(),
                            input_scratch_.size() * sizeof(float), stream.get());
        engine_->infer(stream.get());

        std::vector<FaceDetection> raw;
        for (int stride : kStrides) {
            const std::string sn = "score_" + std::to_string(stride);
            const std::string bn = "bbox_" + std::to_string(stride);
            const std::string kn = "kps_" + std::to_string(stride);

            const auto& sb = engine_->binding(sn);
            std::vector<float> scores(sb.volume);
            const auto& bb = engine_->binding(bn);
            std::vector<float> bboxes(bb.volume);
            const auto& kb = engine_->binding(kn);
            std::vector<float> kps(kb.volume);

            engine_->copy_output(sn, scores.data(), scores.size() * sizeof(float), stream.get());
            engine_->copy_output(bn, bboxes.data(), bboxes.size() * sizeof(float), stream.get());
            engine_->copy_output(kn, kps.data(), kps.size() * sizeof(float), stream.get());
            stream.synchronize();

            decode_stride(stride, static_cast<int>(cfg_.input_width),
                          static_cast<int>(cfg_.input_height), kAnchorsPerLocation,
                          scores.data(), bboxes.data(), kps.data(), cfg_.confidence_threshold,
                          lb.scale, lb.pad_x, lb.pad_y, raw);
        }

        all.push_back(nms(std::move(raw), cfg_.nms_iou_threshold));
    }

    return all;
}

}  // namespace face_pipeline::trt
