// benchmark: per-stage latency and throughput numbers for the components
// of the recognition pipeline. Synthetic inputs only.

#include <spdlog/spdlog.h>

#include <Eigen/Core>

#include <algorithm>
#include <chrono>
#include <cstdlib>
#include <iomanip>
#include <iostream>
#include <numeric>
#include <random>
#include <string>
#include <vector>

#include "face_pipeline/config/system_config.hpp"
#include "face_pipeline/indexing/faiss_searcher.hpp"
#include "face_pipeline/utils/logger.hpp"

namespace {

using Clock = std::chrono::steady_clock;

double percentile(std::vector<double> v, double p) {
    if (v.empty()) return 0.0;
    std::sort(v.begin(), v.end());
    const double pos = p * static_cast<double>(v.size() - 1);
    const auto lo = static_cast<std::size_t>(pos);
    const double frac = pos - static_cast<double>(lo);
    if (lo + 1 >= v.size()) return v[lo];
    return v[lo] * (1.0 - frac) + v[lo + 1] * frac;
}

void bench_faiss(const face_pipeline::config::SystemConfig& cfg, std::size_t n_index,
                 std::size_t n_queries, std::size_t iters) {
    face_pipeline::indexing::FaissSearcher searcher(cfg.faiss);

    std::mt19937 rng(42);
    std::normal_distribution<float> nd(0.0f, 1.0f);
    const auto dim = static_cast<Eigen::Index>(cfg.encoding.embedding_dim);

    Eigen::MatrixXf M(dim, static_cast<Eigen::Index>(n_index));
    for (Eigen::Index i = 0; i < M.size(); ++i) {
        M.data()[i] = nd(rng);
    }
    M.colwise().normalize();

    std::vector<std::int64_t> ids(n_index);
    std::iota(ids.begin(), ids.end(), 1);
    searcher.build(M, ids);

    Eigen::MatrixXf Q(dim, static_cast<Eigen::Index>(n_queries));
    for (Eigen::Index i = 0; i < Q.size(); ++i) Q.data()[i] = nd(rng);
    Q.colwise().normalize();

    // Warm-up
    for (std::size_t i = 0; i < 5; ++i) (void)searcher.search_batch(Q, 5);

    std::vector<double> latencies_ms;
    latencies_ms.reserve(iters);
    for (std::size_t i = 0; i < iters; ++i) {
        const auto t0 = Clock::now();
        (void)searcher.search_batch(Q, 5);
        const auto t1 = Clock::now();
        latencies_ms.push_back(std::chrono::duration<double, std::milli>(t1 - t0).count());
    }

    std::cout << "FAISS search (n_index=" << n_index << ", n_queries=" << n_queries
              << ", top_k=5):\n"
              << "  p50: " << std::fixed << std::setprecision(3)
              << percentile(latencies_ms, 0.5) << " ms\n"
              << "  p95: " << percentile(latencies_ms, 0.95) << " ms\n"
              << "  p99: " << percentile(latencies_ms, 0.99) << " ms\n"
              << "  qps: " << std::setprecision(1)
              << static_cast<double>(n_queries) * 1000.0 / percentile(latencies_ms, 0.5) << "\n";
}

void print_usage(const char* argv0) {
    std::cerr << "Usage: " << argv0 << " --config CONFIG_YAML\n";
}

}  // namespace

int main(int argc, char** argv) {
    std::string config_path;
    for (int i = 1; i < argc; ++i) {
        const std::string a = argv[i];
        if ((a == "--config" || a == "-c") && i + 1 < argc) config_path = argv[++i];
        else if (a == "--help" || a == "-h") {
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
        face_pipeline::utils::init_logger("warn", false);

        std::cout << "\n=== FAISS index size sweep ===\n";
        for (std::size_t n : {std::size_t{1000}, std::size_t{10000}, std::size_t{50000}}) {
            bench_faiss(cfg, n, 32, 200);
            std::cout << "\n";
        }
    } catch (const std::exception& e) {
        SPDLOG_CRITICAL("fatal error: {}", e.what());
        return EXIT_FAILURE;
    }
    return EXIT_SUCCESS;
}
