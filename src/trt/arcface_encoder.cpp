#include "face_pipeline/trt/arcface_encoder.hpp"

#include <algorithm>
#include <cmath>
#include <cstring>
#include <opencv2/imgproc.hpp>
#include <stdexcept>

#include "face_pipeline/trt/trt_engine.hpp"
#include "face_pipeline/utils/cuda_helpers.hpp"

namespace face_pipeline::trt {

namespace {

/// HWC BGR uint8 (112x112) -> CHW RGB float32 in [-1, 1] following the
/// insightface ArcFace preprocessing convention.
void preprocess_face(const cv::Mat& bgr, int input_size, float* dst) {
    cv::Mat resized;
    if (bgr.cols != input_size || bgr.rows != input_size) {
        cv::resize(bgr, resized, cv::Size(input_size, input_size), 0, 0, cv::INTER_LINEAR);
    } else {
        resized = bgr;
    }

    const int channel_stride = input_size * input_size;
    for (int y = 0; y < input_size; ++y) {
        const auto* row = resized.ptr<cv::Vec3b>(y);
        for (int x = 0; x < input_size; ++x) {
            const auto& px = row[x];
            const int idx = y * input_size + x;
            dst[0 * channel_stride + idx] = (static_cast<float>(px[2]) - 127.5f) / 127.5f;
            dst[1 * channel_stride + idx] = (static_cast<float>(px[1]) - 127.5f) / 127.5f;
            dst[2 * channel_stride + idx] = (static_cast<float>(px[0]) - 127.5f) / 127.5f;
        }
    }
}

void l2_normalize(Embedding& v) {
    const float norm = v.norm();
    if (norm > 1e-12f) {
        v /= norm;
    }
}

}  // namespace

ArcFaceEncoder::ArcFaceEncoder(const config::EncodingConfig& cfg)
    : cfg_(cfg), engine_(std::make_unique<TrtEngine>(cfg.engine_path)) {
    input_scratch_.resize(static_cast<std::size_t>(cfg.batch_size) * 3 * cfg.input_size *
                          cfg.input_size);
    output_scratch_.resize(static_cast<std::size_t>(cfg.batch_size) * cfg.embedding_dim);
}

ArcFaceEncoder::~ArcFaceEncoder() = default;

Embedding ArcFaceEncoder::encode(const cv::Mat& aligned_face) {
    auto out = encode_batch({aligned_face});
    if (out.empty()) {
        throw std::runtime_error("ArcFaceEncoder::encode produced no output");
    }
    // `out.front()` is a reference into a local vector, not a local object,
    // so there is no copy elision to defeat here — the move is what keeps
    // this from copying a 512-float embedding.
    // cppcheck-suppress returnStdMoveLocal
    return std::move(out.front());
}

std::vector<Embedding> ArcFaceEncoder::encode_batch(const std::vector<cv::Mat>& aligned_faces) {
    std::vector<Embedding> results;
    results.reserve(aligned_faces.size());

    const std::size_t bsz = cfg_.batch_size;
    for (std::size_t i = 0; i < aligned_faces.size(); i += bsz) {
        const std::size_t end = std::min(i + bsz, aligned_faces.size());
        std::vector<cv::Mat> chunk(aligned_faces.begin() + static_cast<std::ptrdiff_t>(i),
                                   aligned_faces.begin() + static_cast<std::ptrdiff_t>(end));
        encode_chunk_(chunk, results);
    }
    return results;
}

void ArcFaceEncoder::encode_chunk_(const std::vector<cv::Mat>& chunk, std::vector<Embedding>& out) {
    if (chunk.empty())
        return;

    const std::int64_t bsz = static_cast<std::int64_t>(chunk.size());
    const std::int64_t side = static_cast<std::int64_t>(cfg_.input_size);
    const std::int64_t emb = static_cast<std::int64_t>(cfg_.embedding_dim);

    const std::string input_name = engine_->bindings().front().name;
    std::string output_name;
    for (const auto& b : engine_->bindings()) {
        if (!b.is_input) {
            output_name = b.name;
            break;
        }
    }
    if (output_name.empty()) {
        throw std::runtime_error("ArcFace engine has no output binding");
    }

    engine_->set_input_shape(input_name, {bsz, 3, side, side});

    const std::size_t per_face = static_cast<std::size_t>(3 * side * side);
    for (std::size_t i = 0; i < chunk.size(); ++i) {
        preprocess_face(chunk[i], static_cast<int>(side), input_scratch_.data() + i * per_face);
    }

    utils::CudaStream stream;
    engine_->copy_input(input_name, input_scratch_.data(), chunk.size() * per_face * sizeof(float),
                        stream.get());
    engine_->infer(stream.get());
    engine_->copy_output(output_name, output_scratch_.data(),
                         chunk.size() * static_cast<std::size_t>(emb) * sizeof(float),
                         stream.get());
    stream.synchronize();

    for (std::size_t i = 0; i < chunk.size(); ++i) {
        Embedding v(emb);
        std::memcpy(v.data(), output_scratch_.data() + i * static_cast<std::size_t>(emb),
                    static_cast<std::size_t>(emb) * sizeof(float));
        l2_normalize(v);
        out.push_back(std::move(v));
    }
}

}  // namespace face_pipeline::trt
