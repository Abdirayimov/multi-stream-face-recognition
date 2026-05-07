#pragma once

#include <cstdint>
#include <string>

namespace face_pipeline::config {

struct PipelineConfig {
    std::uint32_t max_streams = 8;
    std::uint32_t batch_size = 8;
    std::uint32_t batched_push_timeout_us = 40000;
    std::uint32_t muxer_width = 1280;
    std::uint32_t muxer_height = 720;
};

struct DetectionConfig {
    std::string engine_path;
    std::uint32_t input_width = 640;
    std::uint32_t input_height = 640;
    float confidence_threshold = 0.5f;
    float nms_iou_threshold = 0.4f;
};

struct EncodingConfig {
    std::string engine_path;
    std::uint32_t input_size = 112;
    std::uint32_t embedding_dim = 512;
    std::uint32_t batch_size = 32;
};

enum class FaissIndexType {
    IVFFlat,
    IVFPQ,
};

enum class FaissMetric {
    InnerProduct,
    L2,
};

struct FaissConfig {
    FaissIndexType index_type = FaissIndexType::IVFFlat;
    std::uint32_t ivf_pq_min_size = 100000;
    std::uint32_t nlist = 0;
    std::uint32_t nprobe = 32;
    std::uint32_t pq_m = 64;
    FaissMetric metric = FaissMetric::InnerProduct;
    int gpu_id = 0;
};

struct RecognitionConfig {
    float threshold = 0.45f;
    float margin_min = 0.04f;
    std::uint32_t top_k = 5;
};

struct LoggingConfig {
    std::string level = "info";
    bool json = true;
};

struct SystemConfig {
    PipelineConfig pipeline;
    DetectionConfig detection;
    EncodingConfig encoding;
    FaissConfig faiss;
    RecognitionConfig recognition;
    LoggingConfig logging;

    /// Load a SystemConfig from a YAML file. Throws std::runtime_error on
    /// parse failure or missing required fields.
    static SystemConfig load(const std::string& yaml_path);
};

}  // namespace face_pipeline::config
