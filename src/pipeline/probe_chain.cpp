#include "face_pipeline/pipeline/probe_chain.hpp"

#include <spdlog/spdlog.h>

#include <exception>
#include <optional>
#include <utility>
#include <vector>

#include "face_pipeline/align/face_aligner.hpp"
#include "face_pipeline/pipeline/match_decision.hpp"

namespace face_pipeline::pipeline {

ProbeChain::ProbeChain(align::FaceAligner& aligner, trt::IEncoder& encoder,
                       indexing::ISearcher& searcher,
                       const config::RecognitionConfig& recognition_cfg)
    : aligner_(aligner),
      encoder_(encoder),
      searcher_(searcher),
      recognition_cfg_(recognition_cfg) {}

FrameResult ProbeChain::process(FrameMeta meta, const cv::Mat& image,
                                std::vector<FaceResult> detections) {
    FrameResult result;
    result.meta = std::move(meta);
    result.faces = std::move(detections);

    if (result.faces.empty()) {
        if (callback_)
            callback_(result);
        return result;
    }

    // 1. Align all detected faces.
    std::vector<cv::Mat> crops;
    crops.reserve(result.faces.size());
    for (const auto& f : result.faces) {
        crops.push_back(aligner_.align(image, f.landmarks));
    }

    // 2. Encode in a single batched call.
    //
    // A failing engine must cost one frame, not the stream: a live pipeline
    // that terminates because a single inference threw is worse than one
    // that drops a frame and keeps going. The frame is still reported, with
    // no embeddings and therefore no identities.
    std::vector<trt::Embedding> embeddings;
    try {
        embeddings = encoder_.encode_batch(crops);
    } catch (const std::exception& e) {
        SPDLOG_WARN("ProbeChain: encoder failed on frame {} of source '{}', skipping: {}",
                    result.meta.frame_number, result.meta.source_id, e.what());
        if (callback_)
            callback_(result);
        return result;
    }

    // An encoder returning a different count than it was given would leave
    // faces silently paired with another face's embedding, so treat the
    // mismatch as a failed frame rather than indexing past the end.
    if (embeddings.size() != result.faces.size()) {
        SPDLOG_WARN("ProbeChain: encoder returned {} embeddings for {} faces, skipping frame {}",
                    embeddings.size(), result.faces.size(), result.meta.frame_number);
        if (callback_)
            callback_(result);
        return result;
    }

    for (std::size_t i = 0; i < result.faces.size(); ++i) {
        result.faces[i].embedding = embeddings[i];
    }

    // 3. Look up each embedding. Batched search would be more efficient at
    //    high concurrency; we keep it simple here for clarity.
    if (searcher_.size() == 0) {
        SPDLOG_DEBUG("ProbeChain: empty index, skipping search");
        if (callback_)
            callback_(result);
        return result;
    }

    for (auto& f : result.faces) {
        const auto hits = searcher_.search(*f.embedding, recognition_cfg_.top_k);
        if (hits.empty())
            continue;

        const std::optional<float> second =
            (hits.size() > 1) ? std::optional<float>(hits[1].similarity) : std::nullopt;

        f.top1_similarity = hits.front().similarity;
        f.margin_to_top2 = compute_margin(f.top1_similarity, second);

        if (match_decision(f.top1_similarity, second, recognition_cfg_.threshold,
                           recognition_cfg_.margin_min)) {
            f.matched_id = hits.front().id;
        }
    }

    if (callback_)
        callback_(result);
    return result;
}

}  // namespace face_pipeline::pipeline
