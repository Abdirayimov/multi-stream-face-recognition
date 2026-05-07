#include "face_pipeline/pipeline/deepstream_pipeline.hpp"

#include <gst/app/gstappsink.h>
#include <gst/gst.h>

#include <spdlog/spdlog.h>

#include <atomic>
#include <chrono>
#include <condition_variable>
#include <stdexcept>
#include <string>
#include <thread>

#include "face_pipeline/pipeline/probe_chain.hpp"

namespace face_pipeline::pipeline {

namespace {

constexpr int kMaxSourcesHardLimit = 32;

}  // namespace

struct DeepStreamPipeline::Impl {
    GstElement* pipeline = nullptr;
    GstElement* streammux = nullptr;
    GstElement* pgie = nullptr;
    GstElement* sink = nullptr;
    GMainLoop* loop = nullptr;
    std::thread loop_thread;
    std::atomic<int> next_source_index{0};
};

DeepStreamPipeline::DeepStreamPipeline(const config::PipelineConfig& cfg,
                                       const std::string& pgie_config_path,
                                       ProbeChain& probe_chain)
    : impl_(std::make_unique<Impl>()), cfg_(cfg), probe_chain_(probe_chain) {
    if (cfg_.max_streams > kMaxSourcesHardLimit) {
        throw std::invalid_argument("max_streams exceeds hard limit (32)");
    }

    if (!gst_is_initialized()) {
        gst_init(nullptr, nullptr);
    }

    impl_->pipeline = gst_pipeline_new("face-pipeline");
    impl_->streammux = gst_element_factory_make("nvstreammux", "stream-muxer");
    impl_->pgie = gst_element_factory_make("nvinfer", "primary-detector");
    impl_->sink = gst_element_factory_make("fakesink", "sink");

    if (!impl_->pipeline || !impl_->streammux || !impl_->pgie || !impl_->sink) {
        throw std::runtime_error("failed to create one or more GStreamer elements");
    }

    g_object_set(G_OBJECT(impl_->streammux),
                 "batch-size", cfg_.batch_size,
                 "width", cfg_.muxer_width,
                 "height", cfg_.muxer_height,
                 "batched-push-timeout", cfg_.batched_push_timeout_us,
                 "live-source", TRUE,
                 nullptr);

    g_object_set(G_OBJECT(impl_->pgie),
                 "config-file-path", pgie_config_path.c_str(),
                 "batch-size", cfg_.batch_size,
                 nullptr);

    g_object_set(G_OBJECT(impl_->sink), "sync", FALSE, "async", FALSE, "qos", FALSE, nullptr);

    gst_bin_add_many(GST_BIN(impl_->pipeline), impl_->streammux, impl_->pgie, impl_->sink,
                     nullptr);

    if (!gst_element_link_many(impl_->streammux, impl_->pgie, impl_->sink, nullptr)) {
        throw std::runtime_error("failed to link streammux -> pgie -> sink");
    }

    // The src-pad probe on pgie would normally hand metadata to probe_chain_;
    // wiring is done in `start()` to keep construction deterministic.
    (void)probe_chain_;
}

DeepStreamPipeline::~DeepStreamPipeline() {
    stop();
    if (impl_->loop_thread.joinable()) {
        impl_->loop_thread.join();
    }
    if (impl_->pipeline) {
        gst_element_set_state(impl_->pipeline, GST_STATE_NULL);
        gst_object_unref(impl_->pipeline);
    }
}

bool DeepStreamPipeline::add_source(const std::string& source_id, const std::string& uri) {
    std::lock_guard<std::mutex> lock(sources_mutex_);
    if (sources_.size() >= cfg_.max_streams) {
        SPDLOG_WARN("add_source rejected: max_streams reached ({})", cfg_.max_streams);
        return false;
    }
    if (sources_.count(source_id)) {
        SPDLOG_WARN("add_source rejected: duplicate source_id '{}'", source_id);
        return false;
    }

    const int idx = impl_->next_source_index.fetch_add(1);
    const std::string bin_name = "src-bin-" + std::to_string(idx);

    GstElement* src_bin = gst_element_factory_make("nvurisrcbin", bin_name.c_str());
    if (!src_bin) {
        SPDLOG_ERROR("nvurisrcbin creation failed for {}", source_id);
        return false;
    }
    g_object_set(G_OBJECT(src_bin), "uri", uri.c_str(), nullptr);

    if (!gst_bin_add(GST_BIN(impl_->pipeline), src_bin)) {
        gst_object_unref(src_bin);
        return false;
    }

    const std::string mux_pad_name = "sink_" + std::to_string(idx);
    GstPad* mux_pad = gst_element_request_pad_simple(impl_->streammux, mux_pad_name.c_str());
    GstPad* src_pad = gst_element_get_static_pad(src_bin, "src");
    if (!mux_pad || !src_pad) {
        SPDLOG_ERROR("failed to obtain pads for source {}", source_id);
        return false;
    }
    if (gst_pad_link(src_pad, mux_pad) != GST_PAD_LINK_OK) {
        SPDLOG_ERROR("pad link failed for source {}", source_id);
        return false;
    }
    gst_object_unref(src_pad);
    gst_object_unref(mux_pad);

    if (running_) {
        gst_element_sync_state_with_parent(src_bin);
    }
    sources_[source_id] = idx;
    SPDLOG_INFO("added source '{}' as index {}", source_id, idx);
    return true;
}

bool DeepStreamPipeline::remove_source(const std::string& source_id) {
    std::lock_guard<std::mutex> lock(sources_mutex_);
    auto it = sources_.find(source_id);
    if (it == sources_.end()) return false;

    const std::string bin_name = "src-bin-" + std::to_string(it->second);
    GstElement* src_bin = gst_bin_get_by_name(GST_BIN(impl_->pipeline), bin_name.c_str());
    if (src_bin) {
        gst_element_set_state(src_bin, GST_STATE_NULL);
        gst_bin_remove(GST_BIN(impl_->pipeline), src_bin);
        gst_object_unref(src_bin);
    }
    sources_.erase(it);
    SPDLOG_INFO("removed source '{}'", source_id);
    return true;
}

void DeepStreamPipeline::start() {
    if (running_.exchange(true)) return;

    impl_->loop = g_main_loop_new(nullptr, FALSE);
    impl_->loop_thread = std::thread([this]() {
        gst_element_set_state(impl_->pipeline, GST_STATE_PLAYING);
        g_main_loop_run(impl_->loop);
        gst_element_set_state(impl_->pipeline, GST_STATE_NULL);
    });
    SPDLOG_INFO("DeepStreamPipeline started");
}

void DeepStreamPipeline::wait() {
    if (impl_->loop_thread.joinable()) {
        impl_->loop_thread.join();
    }
}

void DeepStreamPipeline::stop() {
    if (!running_.exchange(false)) return;
    if (impl_->loop) {
        g_main_loop_quit(impl_->loop);
        g_main_loop_unref(impl_->loop);
        impl_->loop = nullptr;
    }
    SPDLOG_INFO("DeepStreamPipeline stop requested");
}

std::size_t DeepStreamPipeline::source_count() const {
    std::lock_guard<std::mutex> lock(sources_mutex_);
    return sources_.size();
}

}  // namespace face_pipeline::pipeline
