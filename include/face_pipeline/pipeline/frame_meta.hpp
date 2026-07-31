#pragma once

#include <Eigen/Core>
#include <array>
#include <chrono>
#include <cstdint>
#include <opencv2/core.hpp>
#include <optional>
#include <string>
#include <vector>

namespace face_pipeline::pipeline {

using TimePoint = std::chrono::steady_clock::time_point;

/// Per-frame metadata flowing through the pipeline.
struct FrameMeta {
    std::string source_id;       ///< User-defined camera identifier.
    std::uint64_t frame_number;  ///< Monotonic per-source frame counter.
    TimePoint pts;               ///< Wall-clock arrival time.
};

/// Result of running the recognition probe on one detected face.
struct FaceResult {
    cv::Rect2f bbox;
    std::array<cv::Point2f, 5> landmarks;
    float detection_score = 0.0f;

    /// L2-normalized 512-d embedding (filled when encoding succeeded).
    std::optional<Eigen::Matrix<float, Eigen::Dynamic, 1>> embedding;

    /// Resolved identity, if FAISS search returned a confident hit.
    std::optional<std::int64_t> matched_id;
    float top1_similarity = 0.0f;
    float margin_to_top2 = 0.0f;
};

/// Aggregate result for one frame of one source.
struct FrameResult {
    FrameMeta meta;
    std::vector<FaceResult> faces;
};

}  // namespace face_pipeline::pipeline
