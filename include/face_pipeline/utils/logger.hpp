#pragma once

#include <spdlog/spdlog.h>

#include <string>

namespace face_pipeline::utils {

/// Initialize the global spdlog logger.
///
/// @param level   One of "trace", "debug", "info", "warn", "error", "critical".
/// @param json    If true, emit one JSON object per line (suitable for log
///                aggregators). Otherwise plain colored output.
void init_logger(const std::string& level, bool json);

}  // namespace face_pipeline::utils

#define FP_LOG_TRACE(...) SPDLOG_TRACE(__VA_ARGS__)
#define FP_LOG_DEBUG(...) SPDLOG_DEBUG(__VA_ARGS__)
#define FP_LOG_INFO(...) SPDLOG_INFO(__VA_ARGS__)
#define FP_LOG_WARN(...) SPDLOG_WARN(__VA_ARGS__)
#define FP_LOG_ERROR(...) SPDLOG_ERROR(__VA_ARGS__)
#define FP_LOG_CRITICAL(...) SPDLOG_CRITICAL(__VA_ARGS__)
