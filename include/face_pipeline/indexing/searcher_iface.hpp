#pragma once

#include <Eigen/Core>
#include <cstddef>
#include <cstdint>
#include <vector>

namespace face_pipeline::indexing {

/// An L2-normalized face embedding, as stored in the index.
using Embedding = Eigen::Matrix<float, Eigen::Dynamic, 1>;

/// Single (id, similarity) result from a query.
struct SearchResult {
    std::int64_t id = -1;
    float similarity = 0.0f;
};

/// What the probe chain needs from a vector index.
///
/// Narrow on purpose. FaissSearcher is the production implementation and
/// wants a GPU-resident FAISS index; depending on the concrete class
/// would make the recognition chain untestable without one.
class ISearcher {
public:
    virtual ~ISearcher() = default;

    /// Top-k nearest neighbours of `query`, best first. An empty result
    /// is legal and means "nothing enrolled matches".
    virtual std::vector<SearchResult> search(const Embedding& query, std::uint32_t top_k) const = 0;

    /// Number of enrolled vectors. Zero means the chain can skip the
    /// lookup entirely.
    virtual std::size_t size() const noexcept = 0;
};

}  // namespace face_pipeline::indexing
