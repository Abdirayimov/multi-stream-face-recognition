#include <gtest/gtest.h>

#include <stdexcept>
#include <string>

#include "face_pipeline/config/system_config.hpp"

namespace {

using face_pipeline::config::FaissIndexType;
using face_pipeline::config::FaissMetric;
using face_pipeline::config::SystemConfig;

std::string fixture(const std::string& name) {
    return std::string(FACE_PIPELINE_TEST_FIXTURES) + "/" + name;
}

/// A config with every field valid, used as the starting point for the
/// programmatic `validate()` tests.
SystemConfig valid_config() {
    SystemConfig cfg;
    cfg.detection.engine_path = "models/scrfd.engine";
    cfg.encoding.engine_path = "models/arcface.engine";
    return cfg;
}

// ------------------------------------------------------------- happy path

TEST(SystemConfigLoad, ReadsEveryFieldFromAFullyPopulatedFile) {
    const SystemConfig cfg = SystemConfig::load(fixture("valid_full.yaml"));

    EXPECT_EQ(cfg.pipeline.max_streams, 16u);
    EXPECT_EQ(cfg.pipeline.batch_size, 4u);
    EXPECT_EQ(cfg.pipeline.batched_push_timeout_us, 33000u);
    EXPECT_EQ(cfg.pipeline.muxer_width, 1920u);
    EXPECT_EQ(cfg.pipeline.muxer_height, 1080u);

    EXPECT_EQ(cfg.detection.engine_path, "models/scrfd_10g_fp16.engine");
    EXPECT_EQ(cfg.detection.input_width, 320u);
    EXPECT_EQ(cfg.detection.input_height, 320u);
    EXPECT_FLOAT_EQ(cfg.detection.confidence_threshold, 0.6f);
    EXPECT_FLOAT_EQ(cfg.detection.nms_iou_threshold, 0.35f);

    EXPECT_EQ(cfg.encoding.engine_path, "models/arcface_r50_fp16.engine");
    EXPECT_EQ(cfg.encoding.input_size, 112u);
    EXPECT_EQ(cfg.encoding.embedding_dim, 512u);
    EXPECT_EQ(cfg.encoding.batch_size, 64u);

    EXPECT_EQ(cfg.faiss.index_type, FaissIndexType::IVFPQ);
    EXPECT_EQ(cfg.faiss.ivf_pq_min_size, 50000u);
    EXPECT_EQ(cfg.faiss.nlist, 1024u);
    EXPECT_EQ(cfg.faiss.nprobe, 16u);
    EXPECT_EQ(cfg.faiss.pq_m, 64u);
    EXPECT_EQ(cfg.faiss.metric, FaissMetric::L2);
    EXPECT_EQ(cfg.faiss.gpu_id, 1);

    EXPECT_FLOAT_EQ(cfg.recognition.threshold, 0.5f);
    EXPECT_FLOAT_EQ(cfg.recognition.margin_min, 0.08f);
    EXPECT_EQ(cfg.recognition.top_k, 10u);

    EXPECT_EQ(cfg.logging.level, "debug");
    EXPECT_FALSE(cfg.logging.json);
}

TEST(SystemConfigLoad, FillsInDocumentedDefaultsForOmittedFields) {
    const SystemConfig cfg = SystemConfig::load(fixture("minimal.yaml"));

    EXPECT_EQ(cfg.pipeline.max_streams, 8u);
    EXPECT_EQ(cfg.pipeline.batch_size, 8u);
    EXPECT_EQ(cfg.pipeline.batched_push_timeout_us, 40000u);
    EXPECT_EQ(cfg.pipeline.muxer_width, 1280u);
    EXPECT_EQ(cfg.pipeline.muxer_height, 720u);

    EXPECT_EQ(cfg.detection.input_width, 640u);
    EXPECT_EQ(cfg.detection.input_height, 640u);
    EXPECT_FLOAT_EQ(cfg.detection.confidence_threshold, 0.5f);
    EXPECT_FLOAT_EQ(cfg.detection.nms_iou_threshold, 0.4f);

    EXPECT_EQ(cfg.encoding.input_size, 112u);
    EXPECT_EQ(cfg.encoding.embedding_dim, 512u);
    EXPECT_EQ(cfg.encoding.batch_size, 32u);

    // An absent `faiss` section leaves the struct defaults in place.
    EXPECT_EQ(cfg.faiss.index_type, FaissIndexType::IVFFlat);
    EXPECT_EQ(cfg.faiss.metric, FaissMetric::InnerProduct);
    EXPECT_EQ(cfg.faiss.nprobe, 32u);

    EXPECT_FLOAT_EQ(cfg.recognition.threshold, 0.45f);
    EXPECT_FLOAT_EQ(cfg.recognition.margin_min, 0.04f);
    EXPECT_EQ(cfg.recognition.top_k, 5u);

    EXPECT_EQ(cfg.logging.level, "info");
    EXPECT_TRUE(cfg.logging.json);
}

TEST(SystemConfigLoad, AcceptsTheReferenceConfigShippedWithTheRepository) {
    const std::string repo_config =
        std::string(FACE_PIPELINE_TEST_FIXTURES) + "/../../configs/system_config.yaml";
    EXPECT_NO_THROW({
        const SystemConfig cfg = SystemConfig::load(repo_config);
        (void)cfg;
    });
}

// ---------------------------------------------------------------- failures

TEST(SystemConfigLoad, ReportsAMissingFileByName) {
    const std::string path = fixture("this_file_does_not_exist.yaml");

    try {
        SystemConfig::load(path);
        FAIL() << "expected a std::runtime_error";
    } catch (const std::runtime_error& e) {
        EXPECT_NE(std::string(e.what()).find(path), std::string::npos)
            << "error should name the offending path: " << e.what();
    }
}

TEST(SystemConfigLoad, RejectsMalformedYaml) {
    EXPECT_THROW(SystemConfig::load(fixture("malformed.yaml")), std::runtime_error);
}

TEST(SystemConfigLoad, RejectsAMissingDetectionSection) {
    EXPECT_THROW(SystemConfig::load(fixture("missing_detection.yaml")), std::runtime_error);
}

TEST(SystemConfigLoad, RejectsAMissingRequiredKey) {
    try {
        SystemConfig::load(fixture("missing_engine_path.yaml"));
        FAIL() << "expected a std::runtime_error";
    } catch (const std::runtime_error& e) {
        EXPECT_NE(std::string(e.what()).find("engine_path"), std::string::npos)
            << "error should name the missing key: " << e.what();
    }
}

TEST(SystemConfigLoad, RejectsAThresholdAboveOne) {
    try {
        SystemConfig::load(fixture("invalid_threshold.yaml"));
        FAIL() << "expected a std::runtime_error";
    } catch (const std::runtime_error& e) {
        EXPECT_NE(std::string(e.what()).find("recognition.threshold"), std::string::npos)
            << "error should name the offending key: " << e.what();
    }
}

TEST(SystemConfigLoad, RejectsAZeroBatchSize) {
    EXPECT_THROW(SystemConfig::load(fixture("invalid_batch_size.yaml")), std::runtime_error);
}

TEST(SystemConfigLoad, RejectsANegativeBatchSize) {
    EXPECT_THROW(SystemConfig::load(fixture("negative_batch_size.yaml")), std::runtime_error);
}

TEST(SystemConfigLoad, RejectsADetectorInputSizeThatIsNotAMultipleOf32) {
    EXPECT_THROW(SystemConfig::load(fixture("invalid_input_size.yaml")), std::runtime_error);
}

TEST(SystemConfigLoad, RejectsAnUnknownFaissMetric) {
    try {
        SystemConfig::load(fixture("unknown_metric.yaml"));
        FAIL() << "expected a std::runtime_error";
    } catch (const std::runtime_error& e) {
        EXPECT_NE(std::string(e.what()).find("cosine"), std::string::npos)
            << "error should echo the unknown value: " << e.what();
    }
}

// ---------------------------------------------------- programmatic checks

TEST(SystemConfigValidate, AcceptsTheStructDefaultsPlusEnginePaths) {
    EXPECT_NO_THROW(valid_config().validate());
}

TEST(SystemConfigValidate, RejectsAnEmptyEnginePath) {
    SystemConfig cfg = valid_config();
    cfg.detection.engine_path.clear();
    EXPECT_THROW(cfg.validate(), std::runtime_error);
}

TEST(SystemConfigValidate, RejectsANegativeRecognitionThreshold) {
    SystemConfig cfg = valid_config();
    cfg.recognition.threshold = -1.5f;  // below the cosine floor
    EXPECT_THROW(cfg.validate(), std::runtime_error);
}

TEST(SystemConfigValidate, AcceptsTheThresholdBoundaries) {
    SystemConfig cfg = valid_config();
    cfg.recognition.threshold = 1.0f;
    EXPECT_NO_THROW(cfg.validate());
    cfg.recognition.threshold = -1.0f;
    EXPECT_NO_THROW(cfg.validate());
}

TEST(SystemConfigValidate, RejectsAnIouThresholdAboveOne) {
    SystemConfig cfg = valid_config();
    cfg.detection.nms_iou_threshold = 1.2f;
    EXPECT_THROW(cfg.validate(), std::runtime_error);
}

TEST(SystemConfigValidate, RejectsAZeroTopK) {
    SystemConfig cfg = valid_config();
    cfg.recognition.top_k = 0;
    EXPECT_THROW(cfg.validate(), std::runtime_error);
}

TEST(SystemConfigValidate, RejectsAPqSubquantizerCountThatDoesNotDivideTheEmbedding) {
    SystemConfig cfg = valid_config();
    cfg.faiss.index_type = FaissIndexType::IVFPQ;
    cfg.faiss.pq_m = 48;  // 512 % 48 != 0
    EXPECT_THROW(cfg.validate(), std::runtime_error);
}

TEST(SystemConfigValidate, IgnoresPqSettingsForAFlatIndex) {
    SystemConfig cfg = valid_config();
    cfg.faiss.index_type = FaissIndexType::IVFFlat;
    cfg.faiss.pq_m = 48;
    EXPECT_NO_THROW(cfg.validate());
}

}  // namespace
