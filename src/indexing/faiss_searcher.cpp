#include "face_pipeline/indexing/faiss_searcher.hpp"

#include <faiss/IndexFlat.h>
#include <faiss/IndexIVFFlat.h>
#include <faiss/IndexIVFPQ.h>
#include <faiss/IndexIDMap.h>
#include <faiss/MetricType.h>
#include <faiss/gpu/GpuCloner.h>
#include <faiss/gpu/StandardGpuResources.h>
#include <faiss/index_io.h>

#include <spdlog/spdlog.h>

#include <algorithm>
#include <cmath>
#include <memory>
#include <stdexcept>

namespace face_pipeline::indexing {

namespace {

faiss::MetricType to_faiss(config::FaissMetric m) {
    return (m == config::FaissMetric::InnerProduct) ? faiss::METRIC_INNER_PRODUCT
                                                    : faiss::METRIC_L2;
}

std::uint32_t default_nlist(std::size_t n) {
    if (n == 0) return 1;
    const auto root = static_cast<std::size_t>(4.0 * std::sqrt(static_cast<double>(n)));
    return static_cast<std::uint32_t>(std::max<std::size_t>(1, std::min(root, n / 39)));
}

}  // namespace

struct FaissSearcher::Impl {
    std::unique_ptr<faiss::Index> index_cpu;
    std::unique_ptr<faiss::Index> index_gpu;
    std::unique_ptr<faiss::gpu::StandardGpuResources> gpu_resources;
    bool on_gpu = false;
};

FaissSearcher::FaissSearcher(const config::FaissConfig& cfg)
    : impl_(std::make_unique<Impl>()), cfg_(cfg) {
    if (cfg_.gpu_id >= 0) {
        impl_->gpu_resources = std::make_unique<faiss::gpu::StandardGpuResources>();
    }
}

FaissSearcher::~FaissSearcher() = default;

void FaissSearcher::build(const Eigen::MatrixXf& embeddings,
                          const std::vector<std::int64_t>& ids) {
    const auto dim = static_cast<std::uint32_t>(embeddings.rows());
    const auto n = static_cast<std::size_t>(embeddings.cols());
    if (n != ids.size()) {
        throw std::invalid_argument("FaissSearcher::build: ids.size() must equal #embeddings");
    }
    dimension_ = dim;

    config::FaissIndexType chosen = cfg_.index_type;
    if (chosen == config::FaissIndexType::IVFPQ && n < cfg_.ivf_pq_min_size) {
        SPDLOG_INFO("FaissSearcher: only {} vectors; falling back from IVF-PQ to IVF-Flat", n);
        chosen = config::FaissIndexType::IVFFlat;
    }

    const std::uint32_t nlist = (cfg_.nlist > 0) ? cfg_.nlist : default_nlist(n);

    auto quantizer = std::make_unique<faiss::IndexFlat>(static_cast<int>(dim), to_faiss(cfg_.metric));
    std::unique_ptr<faiss::Index> ivf;
    if (chosen == config::FaissIndexType::IVFFlat) {
        ivf = std::make_unique<faiss::IndexIVFFlat>(quantizer.release(), dim, nlist,
                                                    to_faiss(cfg_.metric));
    } else {
        ivf = std::make_unique<faiss::IndexIVFPQ>(quantizer.release(), dim, nlist,
                                                  static_cast<int>(cfg_.pq_m), 8,
                                                  to_faiss(cfg_.metric));
    }

    auto wrapped = std::make_unique<faiss::IndexIDMap>(ivf.release());
    wrapped->train(static_cast<faiss::idx_t>(n), embeddings.data());
    wrapped->add_with_ids(static_cast<faiss::idx_t>(n), embeddings.data(), ids.data());

    impl_->index_cpu = std::move(wrapped);

    if (cfg_.gpu_id >= 0) {
        impl_->index_gpu.reset(faiss::gpu::index_cpu_to_gpu(impl_->gpu_resources.get(),
                                                            cfg_.gpu_id, impl_->index_cpu.get()));
        impl_->on_gpu = true;
    }

    SPDLOG_INFO("FaissSearcher: built {} index, n={}, dim={}, nlist={}",
                (chosen == config::FaissIndexType::IVFFlat ? "IVF-Flat" : "IVF-PQ"),
                n, dim, nlist);
}

void FaissSearcher::add(const Eigen::MatrixXf& embeddings,
                        const std::vector<std::int64_t>& ids) {
    if (!impl_->index_cpu) {
        throw std::runtime_error("FaissSearcher::add called before build");
    }
    const auto n = static_cast<std::size_t>(embeddings.cols());
    impl_->index_cpu->add_with_ids(static_cast<faiss::idx_t>(n), embeddings.data(), ids.data());
    if (impl_->on_gpu) {
        impl_->index_gpu.reset(faiss::gpu::index_cpu_to_gpu(impl_->gpu_resources.get(),
                                                            cfg_.gpu_id, impl_->index_cpu.get()));
    }
}

void FaissSearcher::remove(const std::vector<std::int64_t>& ids) {
    if (!impl_->index_cpu) {
        throw std::runtime_error("FaissSearcher::remove called before build");
    }
    faiss::IDSelectorBatch sel(ids.size(), ids.data());
    impl_->index_cpu->remove_ids(sel);
    if (impl_->on_gpu) {
        impl_->index_gpu.reset(faiss::gpu::index_cpu_to_gpu(impl_->gpu_resources.get(),
                                                            cfg_.gpu_id, impl_->index_cpu.get()));
    }
}

std::vector<SearchResult> FaissSearcher::search(const Embedding& query,
                                                std::uint32_t top_k) const {
    Eigen::MatrixXf m(query.size(), 1);
    m.col(0) = query;
    auto batched = search_batch(m, top_k);
    return batched.empty() ? std::vector<SearchResult>{} : std::move(batched.front());
}

std::vector<std::vector<SearchResult>> FaissSearcher::search_batch(
    const Eigen::MatrixXf& queries, std::uint32_t top_k) const {
    if (!impl_->index_cpu) {
        throw std::runtime_error("FaissSearcher::search called before build");
    }
    const auto nq = static_cast<faiss::idx_t>(queries.cols());
    const auto k = static_cast<faiss::idx_t>(top_k);

    std::vector<float> distances(static_cast<std::size_t>(nq * k));
    std::vector<faiss::idx_t> labels(static_cast<std::size_t>(nq * k));

    faiss::Index* idx = impl_->on_gpu ? impl_->index_gpu.get() : impl_->index_cpu.get();
    idx->search(nq, queries.data(), k, distances.data(), labels.data());

    std::vector<std::vector<SearchResult>> out(static_cast<std::size_t>(nq));
    for (std::size_t q = 0; q < static_cast<std::size_t>(nq); ++q) {
        out[q].reserve(top_k);
        for (std::uint32_t r = 0; r < top_k; ++r) {
            const std::size_t pos = q * top_k + r;
            if (labels[pos] < 0) continue;
            out[q].push_back({labels[pos], distances[pos]});
        }
    }
    return out;
}

void FaissSearcher::save(const std::string& path) const {
    if (!impl_->index_cpu) {
        throw std::runtime_error("FaissSearcher::save called before build");
    }
    faiss::write_index(impl_->index_cpu.get(), path.c_str());
}

void FaissSearcher::load(const std::string& path) {
    impl_->index_cpu.reset(faiss::read_index(path.c_str()));
    dimension_ = static_cast<std::uint32_t>(impl_->index_cpu->d);
    if (cfg_.gpu_id >= 0) {
        impl_->index_gpu.reset(faiss::gpu::index_cpu_to_gpu(impl_->gpu_resources.get(),
                                                            cfg_.gpu_id, impl_->index_cpu.get()));
        impl_->on_gpu = true;
    }
}

std::size_t FaissSearcher::size() const noexcept {
    return impl_->index_cpu ? static_cast<std::size_t>(impl_->index_cpu->ntotal) : 0;
}

}  // namespace face_pipeline::indexing
