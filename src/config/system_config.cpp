#include "face_pipeline/config/system_config.hpp"

#include <yaml-cpp/yaml.h>

#include <filesystem>
#include <stdexcept>
#include <string>

namespace face_pipeline::config {

namespace {

/// Cosine similarity is bounded by [-1, 1], so recognition thresholds are
/// too. A threshold outside that range silently accepts or rejects every
/// probe, which is almost always a typo rather than intent.
constexpr float kMinSimilarity = -1.0f;
constexpr float kMaxSimilarity = 1.0f;

/// TensorRT / DeepStream both expect detector input sizes that survive the
/// stride-32 downsampling path without a ragged final feature map.
constexpr std::uint32_t kInputSizeMultiple = 32;

void require_positive(std::uint32_t value, const char* key) {
    if (value == 0) {
        throw std::runtime_error(std::string("config: ") + key + " must be greater than zero");
    }
}

void require_in_range(float value, float lo, float hi, const char* key) {
    if (!(value >= lo && value <= hi)) {
        throw std::runtime_error(std::string("config: ") + key + " must be in [" +
                                 std::to_string(lo) + ", " + std::to_string(hi) + "], got " +
                                 std::to_string(value));
    }
}

template <typename T>
T require(const YAML::Node& node, const std::string& key) {
    if (!node[key]) {
        throw std::runtime_error("missing required config key: " + key);
    }
    return node[key].as<T>();
}

template <typename T>
T optional(const YAML::Node& node, const std::string& key, T fallback) {
    return node[key] ? node[key].as<T>() : fallback;
}

FaissIndexType parse_index_type(const std::string& s) {
    if (s == "ivf_flat")
        return FaissIndexType::IVFFlat;
    if (s == "ivf_pq")
        return FaissIndexType::IVFPQ;
    throw std::runtime_error("unknown faiss.index_type: " + s);
}

FaissMetric parse_metric(const std::string& s) {
    if (s == "ip" || s == "inner_product")
        return FaissMetric::InnerProduct;
    if (s == "l2")
        return FaissMetric::L2;
    throw std::runtime_error("unknown faiss.metric: " + s);
}

}  // namespace

void SystemConfig::validate() const {
    require_positive(pipeline.max_streams, "pipeline.max_streams");
    require_positive(pipeline.batch_size, "pipeline.batch_size");
    require_positive(pipeline.muxer_width, "pipeline.muxer_width");
    require_positive(pipeline.muxer_height, "pipeline.muxer_height");

    if (detection.engine_path.empty()) {
        throw std::runtime_error("config: detection.engine_path must not be empty");
    }
    require_positive(detection.input_width, "detection.input_width");
    require_positive(detection.input_height, "detection.input_height");
    if (detection.input_width % kInputSizeMultiple != 0 ||
        detection.input_height % kInputSizeMultiple != 0) {
        throw std::runtime_error("config: detection input size must be a multiple of " +
                                 std::to_string(kInputSizeMultiple));
    }
    require_in_range(detection.confidence_threshold, 0.0f, 1.0f, "detection.confidence_threshold");
    require_in_range(detection.nms_iou_threshold, 0.0f, 1.0f, "detection.nms_iou_threshold");

    if (encoding.engine_path.empty()) {
        throw std::runtime_error("config: encoding.engine_path must not be empty");
    }
    require_positive(encoding.input_size, "encoding.input_size");
    require_positive(encoding.embedding_dim, "encoding.embedding_dim");
    require_positive(encoding.batch_size, "encoding.batch_size");

    require_positive(faiss.nprobe, "faiss.nprobe");
    if (faiss.index_type == FaissIndexType::IVFPQ) {
        require_positive(faiss.pq_m, "faiss.pq_m");
        if (encoding.embedding_dim % faiss.pq_m != 0) {
            throw std::runtime_error(
                "config: faiss.pq_m must divide encoding.embedding_dim evenly");
        }
    }

    require_in_range(recognition.threshold, kMinSimilarity, kMaxSimilarity,
                     "recognition.threshold");
    require_in_range(recognition.margin_min, 0.0f, kMaxSimilarity - kMinSimilarity,
                     "recognition.margin_min");
    require_positive(recognition.top_k, "recognition.top_k");
}

SystemConfig SystemConfig::load(const std::string& yaml_path) {
    std::error_code ec;
    if (!std::filesystem::is_regular_file(yaml_path, ec)) {
        throw std::runtime_error("config file not found: " + yaml_path);
    }

    const YAML::Node root = YAML::LoadFile(yaml_path);
    SystemConfig out;

    if (const auto p = root["pipeline"]; p) {
        out.pipeline.max_streams = optional<std::uint32_t>(p, "max_streams", 8);
        out.pipeline.batch_size = optional<std::uint32_t>(p, "batch_size", 8);
        out.pipeline.batched_push_timeout_us =
            optional<std::uint32_t>(p, "batched_push_timeout_us", 40000);
        out.pipeline.muxer_width = optional<std::uint32_t>(p, "muxer_width", 1280);
        out.pipeline.muxer_height = optional<std::uint32_t>(p, "muxer_height", 720);
    }

    if (const auto d = root["detection"]; d) {
        out.detection.engine_path = require<std::string>(d, "engine_path");
        out.detection.input_width = optional<std::uint32_t>(d, "input_width", 640);
        out.detection.input_height = optional<std::uint32_t>(d, "input_height", 640);
        out.detection.confidence_threshold = optional<float>(d, "confidence_threshold", 0.5f);
        out.detection.nms_iou_threshold = optional<float>(d, "nms_iou_threshold", 0.4f);
    } else {
        throw std::runtime_error("missing 'detection' section in config");
    }

    if (const auto e = root["encoding"]; e) {
        out.encoding.engine_path = require<std::string>(e, "engine_path");
        out.encoding.input_size = optional<std::uint32_t>(e, "input_size", 112);
        out.encoding.embedding_dim = optional<std::uint32_t>(e, "embedding_dim", 512);
        out.encoding.batch_size = optional<std::uint32_t>(e, "batch_size", 32);
    } else {
        throw std::runtime_error("missing 'encoding' section in config");
    }

    if (const auto f = root["faiss"]; f) {
        out.faiss.index_type = parse_index_type(optional<std::string>(f, "index_type", "ivf_flat"));
        out.faiss.ivf_pq_min_size = optional<std::uint32_t>(f, "ivf_pq_min_size", 100000);
        out.faiss.nlist = optional<std::uint32_t>(f, "nlist", 0);
        out.faiss.nprobe = optional<std::uint32_t>(f, "nprobe", 32);
        out.faiss.pq_m = optional<std::uint32_t>(f, "pq_m", 64);
        out.faiss.metric = parse_metric(optional<std::string>(f, "metric", "ip"));
        out.faiss.gpu_id = optional<int>(f, "gpu_id", 0);
    }

    if (const auto r = root["recognition"]; r) {
        out.recognition.threshold = optional<float>(r, "threshold", 0.45f);
        out.recognition.margin_min = optional<float>(r, "margin_min", 0.04f);
        out.recognition.top_k = optional<std::uint32_t>(r, "top_k", 5);
    }

    if (const auto l = root["logging"]; l) {
        out.logging.level = optional<std::string>(l, "level", "info");
        out.logging.json = optional<bool>(l, "json", true);
    }

    out.validate();
    return out;
}

}  // namespace face_pipeline::config
