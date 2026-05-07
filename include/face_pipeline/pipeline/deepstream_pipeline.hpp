#pragma once

#include <atomic>
#include <memory>
#include <mutex>
#include <string>
#include <unordered_map>
#include <vector>

#include "face_pipeline/config/system_config.hpp"

namespace face_pipeline::pipeline {

class ProbeChain;

/// Thin C++ wrapper around a GStreamer / DeepStream pipeline:
///
///     [ N x nvurisrcbin ] -> nvstreammux -> nvinfer (SCRFD) -> appsink
///
/// nvinfer emits tensor metadata (set `output-tensor-meta=1` in
/// `configs/pgie_scrfd.txt`); a src-pad probe parses those tensors back into
/// `FaceResult` objects, then hands them to the supplied `ProbeChain` for
/// alignment, encoding, and FAISS lookup.
///
/// The class is thread-safe with respect to source add/remove operations:
/// the underlying GLib main loop runs on its own thread.
class DeepStreamPipeline {
public:
    DeepStreamPipeline(const config::PipelineConfig& cfg,
                       const std::string& pgie_config_path,
                       ProbeChain& probe_chain);
    ~DeepStreamPipeline();

    DeepStreamPipeline(const DeepStreamPipeline&) = delete;
    DeepStreamPipeline& operator=(const DeepStreamPipeline&) = delete;

    /// Attach a new source to the pipeline. Safe to call before or after
    /// `run()`. Returns false if the source could not be created (e.g.,
    /// invalid URI) or if `max_streams` has been reached.
    bool add_source(const std::string& source_id, const std::string& uri);

    /// Detach an existing source. Idempotent.
    bool remove_source(const std::string& source_id);

    /// Start the GLib main loop in a background thread and return.
    void start();

    /// Block until the pipeline has stopped.
    void wait();

    /// Request graceful shutdown. Safe to call from a signal handler.
    void stop();

    std::size_t source_count() const;

private:
    struct Impl;
    std::unique_ptr<Impl> impl_;
    config::PipelineConfig cfg_;
    ProbeChain& probe_chain_;
    std::atomic<bool> running_{false};
    mutable std::mutex sources_mutex_;
    std::unordered_map<std::string, int> sources_;  ///< source_id -> source_index
};

}  // namespace face_pipeline::pipeline
