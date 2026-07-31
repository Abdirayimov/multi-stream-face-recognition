// face_enroll: build a FAISS index from a directory of face images.
//
// Each enrolled identity is taken from the immediate parent directory name
// of every image file, e.g.:
//
//   data/lfw/George_W_Bush/George_W_Bush_0001.jpg  -> id "George_W_Bush"
//   data/lfw/Aaron_Eckhart/Aaron_Eckhart_0001.jpg  -> id "Aaron_Eckhart"
//
// Identities are hashed to a stable 63-bit integer id and the embedding-id
// pairs are persisted to the configured FAISS index path.

#include <spdlog/spdlog.h>

#include <Eigen/Core>
#include <cstdint>
#include <cstdlib>
#include <filesystem>
#include <fstream>
#include <iostream>
#include <opencv2/imgcodecs.hpp>
#include <stdexcept>
#include <string>
#include <unordered_map>
#include <vector>

#include "face_pipeline/align/face_aligner.hpp"
#include "face_pipeline/config/system_config.hpp"
#include "face_pipeline/indexing/faiss_searcher.hpp"
#include "face_pipeline/trt/arcface_encoder.hpp"
#include "face_pipeline/trt/scrfd_detector.hpp"
#include "face_pipeline/utils/logger.hpp"

namespace fs = std::filesystem;

namespace {

constexpr std::array<const char*, 4> kImageExt = {".jpg", ".jpeg", ".png", ".bmp"};

bool is_image(const fs::path& p) {
    const auto ext = p.extension().string();
    for (const char* e : kImageExt) {
        if (ext == e)
            return true;
    }
    return false;
}

/// FNV-1a 64-bit hash with the high bit cleared, so the result fits into a
/// signed faiss::idx_t.
std::int64_t hash_id(const std::string& s) {
    std::uint64_t h = 14695981039346656037ULL;
    for (char c : s) {
        h ^= static_cast<std::uint64_t>(static_cast<unsigned char>(c));
        h *= 1099511628211ULL;
    }
    return static_cast<std::int64_t>(h & 0x7FFFFFFFFFFFFFFFULL);
}

void print_usage(const char* argv0) {
    std::cerr << "Usage: " << argv0
              << " --config CONFIG_YAML --input IMAGE_DIR --output INDEX_PATH "
                 "[--id-map ID_MAP_TSV]\n";
}

}  // namespace

int main(int argc, char** argv) {
    std::string config_path;
    std::string input_dir;
    std::string output_path;
    std::string id_map_path;

    for (int i = 1; i < argc; ++i) {
        const std::string a = argv[i];
        if ((a == "--config" || a == "-c") && i + 1 < argc)
            config_path = argv[++i];
        else if ((a == "--input" || a == "-i") && i + 1 < argc)
            input_dir = argv[++i];
        else if ((a == "--output" || a == "-o") && i + 1 < argc)
            output_path = argv[++i];
        else if (a == "--id-map" && i + 1 < argc)
            id_map_path = argv[++i];
        else if (a == "--help" || a == "-h") {
            print_usage(argv[0]);
            return EXIT_SUCCESS;
        }
    }
    if (config_path.empty() || input_dir.empty() || output_path.empty()) {
        print_usage(argv[0]);
        return EXIT_FAILURE;
    }

    try {
        const auto cfg = face_pipeline::config::SystemConfig::load(config_path);
        face_pipeline::utils::init_logger(cfg.logging.level, cfg.logging.json);

        face_pipeline::trt::SCRFDDetector detector(cfg.detection);
        face_pipeline::trt::ArcFaceEncoder encoder(cfg.encoding);
        face_pipeline::align::FaceAligner aligner(static_cast<int>(cfg.encoding.input_size));
        face_pipeline::indexing::FaissSearcher searcher(cfg.faiss);

        std::vector<std::int64_t> ids;
        std::unordered_map<std::int64_t, std::string> id_to_name;
        std::vector<face_pipeline::indexing::Embedding> embeddings;

        std::size_t scanned = 0;
        std::size_t enrolled = 0;

        for (const auto& entry : fs::recursive_directory_iterator(input_dir)) {
            if (!entry.is_regular_file() || !is_image(entry.path()))
                continue;
            ++scanned;

            const std::string name = entry.path().parent_path().filename().string();
            if (name.empty() || name == ".")
                continue;

            cv::Mat img = cv::imread(entry.path().string());
            if (img.empty()) {
                SPDLOG_WARN("could not read image: {}", entry.path().string());
                continue;
            }

            const auto detections = detector.detect(img);
            if (detections.empty())
                continue;

            // Pick the largest face (by bbox area) when multiple are present.
            const auto* best = &detections.front();
            for (const auto& d : detections) {
                if (d.bbox.area() > best->bbox.area())
                    best = &d;
            }

            const cv::Mat aligned = aligner.align(img, best->landmarks);
            const auto emb = encoder.encode(aligned);

            const std::int64_t id = hash_id(name);
            ids.push_back(id);
            embeddings.push_back(emb);
            id_to_name.emplace(id, name);
            ++enrolled;

            if (enrolled % 100 == 0) {
                SPDLOG_INFO("enrolled {} faces ({} images scanned)", enrolled, scanned);
            }
        }

        if (embeddings.empty()) {
            SPDLOG_ERROR("no faces enrolled from {}", input_dir);
            return EXIT_FAILURE;
        }

        const auto dim = static_cast<Eigen::Index>(embeddings.front().size());
        Eigen::MatrixXf M(dim, static_cast<Eigen::Index>(embeddings.size()));
        for (std::size_t i = 0; i < embeddings.size(); ++i) {
            M.col(static_cast<Eigen::Index>(i)) = embeddings[i];
        }

        searcher.build(M, ids);
        searcher.save(output_path);
        SPDLOG_INFO("wrote FAISS index ({} vectors) to {}", searcher.size(), output_path);

        if (!id_map_path.empty()) {
            std::ofstream out(id_map_path);
            for (const auto& [id, name] : id_to_name) {
                out << id << '\t' << name << '\n';
            }
            SPDLOG_INFO("wrote id->name map ({} entries) to {}", id_to_name.size(), id_map_path);
        }
    } catch (const std::exception& e) {
        SPDLOG_CRITICAL("fatal error: {}", e.what());
        return EXIT_FAILURE;
    }
    return EXIT_SUCCESS;
}
