#include "face_pipeline/pipeline/probe_chain.hpp"

#include <spdlog/spdlog.h>

#include <utility>
#include <vector>

#include "face_pipeline/align/face_aligner.hpp"
#include "face_pipeline/indexing/faiss_searcher.hpp"
#include "face_pipeline/trt/arcface_encoder.hpp"

namespace face_pipeline::pipeline {

ProbeChain::ProbeChain(align::FaceAligner& aligner,
                       trt::ArcFaceEncoder& encoder,
                       indexing::FaissSearcher& searcher,
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
        if (callback_) callback_(result);
        return result;
    }

    // 1. Align all detected faces.
    std::vector<cv::Mat> crops;
    crops.reserve(result.faces.size());
    for (const auto& f : result.faces) {
        crops.push_back(aligner_.align(image, f.landmarks));
    }

    // 2. Encode in a single batched call.
    const auto embeddings = encoder_.encode_batch(crops);
    for (std::size_t i = 0; i < result.faces.size(); ++i) {
        result.faces[i].embedding = embeddings[i];
    }

    // 3. Look up each embedding. Batched search would be more efficient at
    //    high concurrency; we keep it simple here for clarity.
    if (searcher_.size() == 0) {
        SPDLOG_DEBUG("ProbeChain: empty index, skipping FAISS search");
        if (callback_) callback_(result);
        return result;
    }

    for (auto& f : result.faces) {
        const auto hits = searcher_.search(*f.embedding, recognition_cfg_.top_k);
        if (hits.empty()) continue;

        f.top1_similarity = hits.front().similarity;
        f.margin_to_top2 =
            (hits.size() > 1) ? hits[0].similarity - hits[1].similarity : hits.front().similarity;

        if (f.top1_similarity >= recognition_cfg_.threshold &&
            f.margin_to_top2 >= recognition_cfg_.margin_min) {
            f.matched_id = hits.front().id;
        }
    }

    if (callback_) callback_(result);
    return result;
}

}  // namespace face_pipeline::pipeline
