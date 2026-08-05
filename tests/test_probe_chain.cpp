#include <gtest/gtest.h>

#include <array>
#include <cstdint>
#include <opencv2/core.hpp>
#include <stdexcept>
#include <vector>

#include "face_pipeline/align/face_aligner.hpp"
#include "face_pipeline/pipeline/probe_chain.hpp"

namespace {

using face_pipeline::align::FaceAligner;
using face_pipeline::config::RecognitionConfig;
using face_pipeline::indexing::ISearcher;
using face_pipeline::indexing::SearchResult;
using face_pipeline::pipeline::FaceResult;
using face_pipeline::pipeline::FrameMeta;
using face_pipeline::pipeline::FrameResult;
using face_pipeline::pipeline::ProbeChain;
using face_pipeline::trt::Embedding;
using face_pipeline::trt::IEncoder;

constexpr int kEmbeddingDim = 8;

/// Deterministic stand-in for ArcFace. Derives an embedding from the crop
/// itself so a test can tell one face's vector from another's without a
/// model, and counts calls so the batching claim can be asserted.
class StubEncoder final : public IEncoder {
public:
    std::vector<Embedding> encode_batch(const std::vector<cv::Mat>& aligned) override {
        ++calls;
        last_batch_size = aligned.size();

        std::vector<Embedding> out;
        out.reserve(aligned.size());
        for (std::size_t i = 0; i < aligned.size(); ++i) {
            Embedding e = Embedding::Zero(kEmbeddingDim);
            e[static_cast<Eigen::Index>(i % kEmbeddingDim)] = 1.0f;
            out.push_back(std::move(e));
        }
        if (drop_last && !out.empty())
            out.pop_back();
        return out;
    }

    int calls = 0;
    std::size_t last_batch_size = 0;
    /// Return one fewer embedding than requested, to exercise the guard.
    bool drop_last = false;
};

/// An encoder that always fails, standing in for a dead engine.
class ThrowingEncoder final : public IEncoder {
public:
    std::vector<Embedding> encode_batch(const std::vector<cv::Mat>&) override {
        ++calls;
        throw std::runtime_error("engine exploded");
    }

    int calls = 0;
};

/// Returns a canned hit list, so the margin rule can be driven directly.
class StubSearcher final : public ISearcher {
public:
    std::vector<SearchResult> search(const Embedding&, std::uint32_t) const override {
        ++calls;
        return hits;
    }

    std::size_t size() const noexcept override { return enrolled; }

    std::vector<SearchResult> hits;
    std::size_t enrolled = 1;
    mutable int calls = 0;
};

RecognitionConfig recognition(float threshold = 0.45f, float margin_min = 0.04f) {
    RecognitionConfig cfg;
    cfg.threshold = threshold;
    cfg.margin_min = margin_min;
    cfg.top_k = 5;
    return cfg;
}

cv::Mat frame() {
    return cv::Mat(480, 640, CV_8UC3, cv::Scalar(60, 60, 60));
}

/// A detection with plausible 5-point landmarks around (x, y).
FaceResult face_at(float x, float y) {
    FaceResult f;
    f.bbox = cv::Rect2f(x - 20.0f, y - 25.0f, 40.0f, 50.0f);
    f.landmarks = {{
        cv::Point2f{x - 10.0f, y - 8.0f},
        cv::Point2f{x + 10.0f, y - 8.0f},
        cv::Point2f{x, y},
        cv::Point2f{x - 8.0f, y + 12.0f},
        cv::Point2f{x + 8.0f, y + 12.0f},
    }};
    f.detection_score = 0.9f;
    return f;
}

FrameMeta meta(std::uint64_t frame_number = 1) {
    FrameMeta m;
    m.source_id = "cam-01";
    m.frame_number = frame_number;
    m.pts = std::chrono::steady_clock::now();
    return m;
}

// ------------------------------------------------------------ empty input

TEST(ProbeChain, ReportsAnEmptyFrameWithoutTouchingTheBackends) {
    FaceAligner aligner{112};
    StubEncoder encoder;
    StubSearcher searcher;
    ProbeChain chain{aligner, encoder, searcher, recognition()};

    const FrameResult r = chain.process(meta(), frame(), {});

    EXPECT_TRUE(r.faces.empty());
    EXPECT_EQ(encoder.calls, 0) << "nothing to encode";
    EXPECT_EQ(searcher.calls, 0);
    EXPECT_EQ(r.meta.source_id, "cam-01");
}

TEST(ProbeChain, StillInvokesTheCallbackForAnEmptyFrame) {
    // A downstream consumer counting frames must not silently miss the
    // ones that happened to contain no faces.
    FaceAligner aligner{112};
    StubEncoder encoder;
    StubSearcher searcher;
    ProbeChain chain{aligner, encoder, searcher, recognition()};

    int seen = 0;
    chain.set_result_callback([&seen](const FrameResult&) { ++seen; });
    chain.process(meta(), frame(), {});

    EXPECT_EQ(seen, 1);
}

// --------------------------------------------------------------- batching

TEST(ProbeChain, EncodesEveryFaceOfAFrameInASingleCall) {
    // This is the batching claim the README makes: N faces in a frame
    // become one encoder invocation, not N.
    FaceAligner aligner{112};
    StubEncoder encoder;
    StubSearcher searcher;
    searcher.enrolled = 0;  // keep the test on the encode step
    ProbeChain chain{aligner, encoder, searcher, recognition()};

    chain.process(meta(), frame(), {face_at(100, 100), face_at(300, 200), face_at(500, 150)});

    EXPECT_EQ(encoder.calls, 1);
    EXPECT_EQ(encoder.last_batch_size, 3u);
}

TEST(ProbeChain, AttachesOneEmbeddingPerFaceInOrder) {
    FaceAligner aligner{112};
    StubEncoder encoder;
    StubSearcher searcher;
    searcher.enrolled = 0;
    ProbeChain chain{aligner, encoder, searcher, recognition()};

    const FrameResult r = chain.process(meta(), frame(), {face_at(100, 100), face_at(300, 200)});

    ASSERT_EQ(r.faces.size(), 2u);
    ASSERT_TRUE(r.faces[0].embedding.has_value());
    ASSERT_TRUE(r.faces[1].embedding.has_value());
    // StubEncoder puts a 1 on axis i for face i.
    EXPECT_FLOAT_EQ((*r.faces[0].embedding)[0], 1.0f);
    EXPECT_FLOAT_EQ((*r.faces[1].embedding)[1], 1.0f);
}

TEST(ProbeChain, SkipsTheSearchEntirelyWhenNothingIsEnrolled) {
    FaceAligner aligner{112};
    StubEncoder encoder;
    StubSearcher searcher;
    searcher.enrolled = 0;
    ProbeChain chain{aligner, encoder, searcher, recognition()};

    const FrameResult r = chain.process(meta(), frame(), {face_at(100, 100)});

    EXPECT_EQ(searcher.calls, 0);
    ASSERT_EQ(r.faces.size(), 1u);
    EXPECT_TRUE(r.faces[0].embedding.has_value()) << "the embedding is still produced";
    EXPECT_FALSE(r.faces[0].matched_id.has_value());
}

// ---------------------------------------------------- the margin decision

TEST(ProbeChain, AcceptsAConfidentUnambiguousHit) {
    FaceAligner aligner{112};
    StubEncoder encoder;
    StubSearcher searcher;
    searcher.hits = {{42, 0.82f}, {7, 0.51f}};
    ProbeChain chain{aligner, encoder, searcher, recognition()};

    const FrameResult r = chain.process(meta(), frame(), {face_at(100, 100)});

    ASSERT_EQ(r.faces.size(), 1u);
    ASSERT_TRUE(r.faces[0].matched_id.has_value());
    EXPECT_EQ(*r.faces[0].matched_id, 42);
    EXPECT_FLOAT_EQ(r.faces[0].top1_similarity, 0.82f);
    EXPECT_NEAR(r.faces[0].margin_to_top2, 0.31f, 1e-6f);
}

TEST(ProbeChain, RejectsWhenTheRunnerUpIsTooClose) {
    // Well over the threshold, but the identity is a coin flip — this is
    // the whole point of carrying a margin.
    FaceAligner aligner{112};
    StubEncoder encoder;
    StubSearcher searcher;
    searcher.hits = {{42, 0.82f}, {7, 0.80f}};
    ProbeChain chain{aligner, encoder, searcher, recognition()};

    const FrameResult r = chain.process(meta(), frame(), {face_at(100, 100)});

    ASSERT_EQ(r.faces.size(), 1u);
    EXPECT_FALSE(r.faces[0].matched_id.has_value());
    EXPECT_FLOAT_EQ(r.faces[0].top1_similarity, 0.82f) << "the score is still reported";
}

TEST(ProbeChain, RejectsWhenTheTopHitIsBelowTheThreshold) {
    FaceAligner aligner{112};
    StubEncoder encoder;
    StubSearcher searcher;
    searcher.hits = {{42, 0.30f}, {7, 0.01f}};
    ProbeChain chain{aligner, encoder, searcher, recognition()};

    const FrameResult r = chain.process(meta(), frame(), {face_at(100, 100)});

    ASSERT_EQ(r.faces.size(), 1u);
    EXPECT_FALSE(r.faces[0].matched_id.has_value());
}

TEST(ProbeChain, LeavesAFaceUnmatchedWhenTheIndexReturnsNothing) {
    FaceAligner aligner{112};
    StubEncoder encoder;
    StubSearcher searcher;
    searcher.hits = {};
    searcher.enrolled = 5;  // non-empty index that happens to return no hit
    ProbeChain chain{aligner, encoder, searcher, recognition()};

    const FrameResult r = chain.process(meta(), frame(), {face_at(100, 100)});

    ASSERT_EQ(r.faces.size(), 1u);
    EXPECT_FALSE(r.faces[0].matched_id.has_value());
    EXPECT_FLOAT_EQ(r.faces[0].top1_similarity, 0.0f);
}

TEST(ProbeChain, QueriesTheIndexOncePerFace) {
    FaceAligner aligner{112};
    StubEncoder encoder;
    StubSearcher searcher;
    searcher.hits = {{42, 0.90f}, {7, 0.10f}};
    ProbeChain chain{aligner, encoder, searcher, recognition()};

    chain.process(meta(), frame(), {face_at(100, 100), face_at(300, 200), face_at(500, 150)});

    EXPECT_EQ(searcher.calls, 3);
}

// -------------------------------------------------------- backend failure

TEST(ProbeChain, SurvivesAnEncoderThatThrows) {
    // A dead engine must cost one frame, not the process. The frame is
    // still reported so a consumer's frame accounting stays correct.
    FaceAligner aligner{112};
    ThrowingEncoder encoder;
    StubSearcher searcher;
    ProbeChain chain{aligner, encoder, searcher, recognition()};

    FrameResult r;
    ASSERT_NO_THROW(r = chain.process(meta(9), frame(), {face_at(100, 100)}));

    EXPECT_EQ(encoder.calls, 1);
    EXPECT_EQ(searcher.calls, 0) << "no embeddings means nothing to look up";
    ASSERT_EQ(r.faces.size(), 1u);
    EXPECT_FALSE(r.faces[0].embedding.has_value());
    EXPECT_FALSE(r.faces[0].matched_id.has_value());
    EXPECT_EQ(r.meta.frame_number, 9u);
}

TEST(ProbeChain, StillInvokesTheCallbackWhenTheEncoderThrows) {
    FaceAligner aligner{112};
    ThrowingEncoder encoder;
    StubSearcher searcher;
    ProbeChain chain{aligner, encoder, searcher, recognition()};

    int seen = 0;
    chain.set_result_callback([&seen](const FrameResult&) { ++seen; });
    chain.process(meta(), frame(), {face_at(100, 100)});

    EXPECT_EQ(seen, 1);
}

TEST(ProbeChain, KeepsGoingAfterAFailedFrame) {
    // The chain holds no state across frames, so a failure must not
    // poison the next call.
    FaceAligner aligner{112};
    StubEncoder good;
    StubSearcher searcher;
    searcher.hits = {{42, 0.90f}, {7, 0.10f}};
    ProbeChain chain{aligner, good, searcher, recognition()};

    good.drop_last = true;
    const FrameResult bad = chain.process(meta(1), frame(), {face_at(100, 100)});
    EXPECT_FALSE(bad.faces[0].embedding.has_value());

    good.drop_last = false;
    const FrameResult ok = chain.process(meta(2), frame(), {face_at(100, 100)});
    ASSERT_EQ(ok.faces.size(), 1u);
    EXPECT_TRUE(ok.faces[0].embedding.has_value());
    EXPECT_TRUE(ok.faces[0].matched_id.has_value());
}

TEST(ProbeChain, DropsAFrameWhenTheEncoderReturnsTheWrongCount) {
    // Pairing face i with another face's embedding would attach the wrong
    // identity to a person, so a count mismatch fails the frame instead.
    FaceAligner aligner{112};
    StubEncoder encoder;
    encoder.drop_last = true;
    StubSearcher searcher;
    searcher.hits = {{42, 0.90f}, {7, 0.10f}};
    ProbeChain chain{aligner, encoder, searcher, recognition()};

    const FrameResult r = chain.process(meta(), frame(), {face_at(100, 100), face_at(300, 200)});

    ASSERT_EQ(r.faces.size(), 2u);
    EXPECT_FALSE(r.faces[0].embedding.has_value());
    EXPECT_FALSE(r.faces[1].embedding.has_value());
    EXPECT_EQ(searcher.calls, 0);
}

// ------------------------------------------------------------- end to end

TEST(ProbeChain, RunsTheWholeChainForAMultiFaceFrame) {
    // align -> encode -> search -> margin, in one pass, with every stage
    // observable.
    FaceAligner aligner{112};
    StubEncoder encoder;
    StubSearcher searcher;
    searcher.hits = {{1001, 0.88f}, {1002, 0.40f}};
    ProbeChain chain{aligner, encoder, searcher, recognition()};

    FrameResult delivered;
    chain.set_result_callback([&delivered](const FrameResult& r) { delivered = r; });

    const FrameResult r = chain.process(meta(7), frame(), {face_at(120, 110), face_at(400, 260)});

    EXPECT_EQ(encoder.calls, 1);
    EXPECT_EQ(encoder.last_batch_size, 2u);
    EXPECT_EQ(searcher.calls, 2);

    ASSERT_EQ(r.faces.size(), 2u);
    for (const auto& f : r.faces) {
        ASSERT_TRUE(f.embedding.has_value());
        ASSERT_TRUE(f.matched_id.has_value());
        EXPECT_EQ(*f.matched_id, 1001);
    }

    EXPECT_EQ(delivered.meta.frame_number, 7u);
    EXPECT_EQ(delivered.faces.size(), 2u);
}

}  // namespace
