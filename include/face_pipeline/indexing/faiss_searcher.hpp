#pragma once

#include <Eigen/Core>
#include <cstdint>
#include <memory>
#include <string>
#include <vector>

#include "face_pipeline/config/system_config.hpp"
#include "face_pipeline/indexing/searcher_iface.hpp"

namespace face_pipeline::indexing {

/// FAISS GPU vector index for face embeddings.
///
/// The index type is chosen automatically based on enrollment size:
///   * `n <  ivf_pq_min_size` -> IVF-Flat (exact within an IVF cell)
///   * `n >= ivf_pq_min_size` -> IVF-PQ   (compressed; lower memory)
///
/// `nlist` defaults to the heuristic min(4 * sqrt(n), n / 39) when zero.
/// Cosine similarity is implemented via inner product on L2-normalized
/// embeddings; switch the metric if you store raw vectors.
///
/// Save / load round-trips through CPU (FAISS GPU indexes are not
/// directly serializable). Reload restores the GPU resident copy.
class FaissSearcher final : public ISearcher {
public:
    explicit FaissSearcher(const config::FaissConfig& cfg);
    ~FaissSearcher() override;

    FaissSearcher(const FaissSearcher&) = delete;
    FaissSearcher& operator=(const FaissSearcher&) = delete;

    /// Build the index from scratch over `embeddings` and `ids`. Trains the
    /// IVF quantizer on the provided vectors.
    /// @param embeddings  Each column is one embedding (dim x N).
    /// @param ids         Length-N user-defined identifiers.
    void build(const Eigen::MatrixXf& embeddings, const std::vector<std::int64_t>& ids);

    /// Add additional vectors to an already-built index without retraining.
    void add(const Eigen::MatrixXf& embeddings, const std::vector<std::int64_t>& ids);

    /// Remove enrolled IDs (best-effort; some FAISS index types do not
    /// support deletion and will throw).
    void remove(const std::vector<std::int64_t>& ids);

    /// Top-k search for one query embedding.
    std::vector<SearchResult> search(const Embedding& query, std::uint32_t top_k) const override;

    /// Batched top-k search; outer vector indexed by query.
    std::vector<std::vector<SearchResult>> search_batch(const Eigen::MatrixXf& queries,
                                                        std::uint32_t top_k) const;

    /// Persist to a file (CPU representation).
    void save(const std::string& path) const;

    /// Load and re-upload to GPU (if `gpu_id >= 0`).
    void load(const std::string& path);

    std::size_t size() const noexcept override;
    std::uint32_t dimension() const noexcept { return dimension_; }

private:
    struct Impl;
    std::unique_ptr<Impl> impl_;
    config::FaissConfig cfg_;
    std::uint32_t dimension_ = 0;
};

}  // namespace face_pipeline::indexing
