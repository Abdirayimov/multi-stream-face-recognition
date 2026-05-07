#include <spdlog/spdlog.h>

#include <atomic>
#include <csignal>
#include <cstdlib>
#include <iostream>
#include <memory>
#include <stdexcept>
#include <string>

#include "face_pipeline/align/face_aligner.hpp"
#include "face_pipeline/config/system_config.hpp"
#include "face_pipeline/indexing/faiss_searcher.hpp"
#include "face_pipeline/pipeline/deepstream_pipeline.hpp"
#include "face_pipeline/pipeline/probe_chain.hpp"
#include "face_pipeline/trt/arcface_encoder.hpp"
#include "face_pipeline/utils/logger.hpp"

namespace {

std::atomic<bool> g_shutdown{false};

void signal_handler(int) { g_shutdown = true; }

void print_usage(const char* argv0) {
    std::cerr << "Usage: " << argv0 << " --config CONFIG_YAML [--pgie PGIE_TXT]\n"
              << "\n"
              << "  --config   Path to system_config.yaml.\n"
              << "  --pgie     Path to gst-nvinfer config (default: configs/pgie_scrfd.txt).\n";
}

}  // namespace

int main(int argc, char** argv) {
    std::string config_path;
    std::string pgie_path = "configs/pgie_scrfd.txt";

    for (int i = 1; i < argc; ++i) {
        const std::string arg = argv[i];
        if ((arg == "--config" || arg == "-c") && i + 1 < argc) {
            config_path = argv[++i];
        } else if (arg == "--pgie" && i + 1 < argc) {
            pgie_path = argv[++i];
        } else if (arg == "--help" || arg == "-h") {
            print_usage(argv[0]);
            return EXIT_SUCCESS;
        }
    }

    if (config_path.empty()) {
        print_usage(argv[0]);
        return EXIT_FAILURE;
    }

    try {
        const auto cfg = face_pipeline::config::SystemConfig::load(config_path);
        face_pipeline::utils::init_logger(cfg.logging.level, cfg.logging.json);
        SPDLOG_INFO("face_server starting with config: {}", config_path);

        face_pipeline::align::FaceAligner aligner(static_cast<int>(cfg.encoding.input_size));
        face_pipeline::trt::ArcFaceEncoder encoder(cfg.encoding);
        face_pipeline::indexing::FaissSearcher searcher(cfg.faiss);
        face_pipeline::pipeline::ProbeChain probe(aligner, encoder, searcher, cfg.recognition);

        face_pipeline::pipeline::DeepStreamPipeline pipeline(cfg.pipeline, pgie_path, probe);

        std::signal(SIGINT, signal_handler);
        std::signal(SIGTERM, signal_handler);

        pipeline.start();
        SPDLOG_INFO("face_server is running; awaiting sources via gRPC/REST or static configs");

        while (!g_shutdown.load()) {
            std::this_thread::sleep_for(std::chrono::milliseconds(200));
        }

        SPDLOG_INFO("shutdown signal received, stopping pipeline");
        pipeline.stop();
        pipeline.wait();
        SPDLOG_INFO("face_server exited cleanly");
    } catch (const std::exception& e) {
        SPDLOG_CRITICAL("fatal error: {}", e.what());
        return EXIT_FAILURE;
    }

    return EXIT_SUCCESS;
}
