#include "face_pipeline/trt/scrfd_postprocess.hpp"

#include <algorithm>
#include <cmath>
#include <opencv2/imgproc.hpp>
#include <stdexcept>

namespace face_pipeline::trt {

namespace {

/// SCRFD input normalization, matching the insightface preprocessing.
constexpr float kPixelMean = 127.5f;
constexpr float kPixelScale = 128.0f;

}  // namespace

LetterboxResult letterbox(const cv::Mat& src, int target_w, int target_h) {
    if (src.empty()) {
        throw std::invalid_argument("letterbox: source image is empty");
    }
    if (target_w <= 0 || target_h <= 0) {
        throw std::invalid_argument("letterbox: target size must be positive");
    }

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

void hwc_bgr_to_chw_rgb_normalized(const cv::Mat& src, float* dst) {
    if (src.type() != CV_8UC3) {
        throw std::invalid_argument("hwc_bgr_to_chw_rgb_normalized: expected an 8-bit BGR image");
    }

    const int H = src.rows;
    const int W = src.cols;
    const int channel_stride = H * W;
    for (int y = 0; y < H; ++y) {
        const auto* row = src.ptr<cv::Vec3b>(y);
        for (int x = 0; x < W; ++x) {
            const auto& px = row[x];
            const int idx = y * W + x;
            // BGR -> RGB and to CHW
            dst[0 * channel_stride + idx] = (static_cast<float>(px[2]) - kPixelMean) / kPixelScale;
            dst[1 * channel_stride + idx] = (static_cast<float>(px[1]) - kPixelMean) / kPixelScale;
            dst[2 * channel_stride + idx] = (static_cast<float>(px[0]) - kPixelMean) / kPixelScale;
        }
    }
}

int anchor_count(int input_w, int input_h, int stride, int anchors_per_location) {
    if (stride <= 0) {
        throw std::invalid_argument("anchor_count: stride must be positive");
    }
    return (input_h / stride) * (input_w / stride) * anchors_per_location;
}

int total_anchor_count(int input_w, int input_h, int anchors_per_location) {
    int total = 0;
    for (const int stride : kStrides) {
        total += anchor_count(input_w, input_h, stride, anchors_per_location);
    }
    return total;
}

cv::Rect2f distance2bbox(cv::Point2f anchor, const float* distances, float stride, float scale,
                         int pad_x, int pad_y) noexcept {
    const float l = distances[0] * stride;
    const float t = distances[1] * stride;
    const float r = distances[2] * stride;
    const float b = distances[3] * stride;

    const cv::Point2f tl = undo_letterbox(anchor.x - l, anchor.y - t, scale, pad_x, pad_y);
    const cv::Point2f br = undo_letterbox(anchor.x + r, anchor.y + b, scale, pad_x, pad_y);
    return {tl.x, tl.y, br.x - tl.x, br.y - tl.y};
}

std::array<cv::Point2f, 5> distance2kps(cv::Point2f anchor, const float* offsets, float stride,
                                        float scale, int pad_x, int pad_y) noexcept {
    std::array<cv::Point2f, 5> kps;
    for (int k = 0; k < kNumLandmarks; ++k) {
        const float x = anchor.x + offsets[k * 2 + 0] * stride;
        const float y = anchor.y + offsets[k * 2 + 1] * stride;
        kps[static_cast<std::size_t>(k)] = undo_letterbox(x, y, scale, pad_x, pad_y);
    }
    return kps;
}

float iou(const cv::Rect2f& a, const cv::Rect2f& b) noexcept {
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
        if (suppressed[i])
            continue;
        kept.push_back(dets[i]);
        for (std::size_t j = i + 1; j < dets.size(); ++j) {
            if (suppressed[j])
                continue;
            if (iou(dets[i].bbox, dets[j].bbox) > iou_thresh) {
                suppressed[j] = true;
            }
        }
    }
    return kept;
}

void decode_stride(int stride, int input_w, int input_h, int anchors_per_location,
                   const float* scores, const float* bboxes, const float* kps, float score_thresh,
                   float scale, int pad_x, int pad_y, std::vector<FaceDetection>& out) {
    const int feat_h = input_h / stride;
    const int feat_w = input_w / stride;
    const auto stride_f = static_cast<float>(stride);

    for (int y = 0; y < feat_h; ++y) {
        for (int x = 0; x < feat_w; ++x) {
            for (int a = 0; a < anchors_per_location; ++a) {
                const int idx = (y * feat_w + x) * anchors_per_location + a;
                const float score = scores[idx];
                if (score < score_thresh)
                    continue;

                const cv::Point2f anchor(static_cast<float>(x * stride),
                                         static_cast<float>(y * stride));

                FaceDetection d;
                d.bbox = distance2bbox(anchor, bboxes + idx * 4, stride_f, scale, pad_x, pad_y);
                d.landmarks = distance2kps(anchor, kps + idx * 10, stride_f, scale, pad_x, pad_y);
                d.score = score;
                out.push_back(d);
            }
        }
    }
}

}  // namespace face_pipeline::trt
