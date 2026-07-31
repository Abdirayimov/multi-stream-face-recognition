// face_detect: run the SCRFD detector on a single image (or every
// frame of a video) and draw the face boxes + 5-point landmarks.
//
// A small visual smoke-test for the detection front-end that needs
// only the SCRFD engine (no ArcFace, no FAISS, no DeepStream):
//
//   face_detect --config configs/system_config.yaml \
//               --input group.jpg --output group_annotated.jpg

#include <spdlog/spdlog.h>

#include <cstdlib>
#include <iostream>
#include <opencv2/imgcodecs.hpp>
#include <opencv2/imgproc.hpp>
#include <opencv2/videoio.hpp>
#include <string>

#include "face_pipeline/config/system_config.hpp"
#include "face_pipeline/trt/scrfd_detector.hpp"
#include "face_pipeline/utils/logger.hpp"

namespace {

void draw(cv::Mat& img, const std::vector<face_pipeline::trt::FaceDetection>& faces) {
    for (const auto& f : faces) {
        cv::rectangle(img, f.bbox, cv::Scalar(72, 207, 235), 2);
        const std::array<cv::Scalar, 5> lm_colors{cv::Scalar(0, 0, 255), cv::Scalar(0, 255, 0),
                                                  cv::Scalar(255, 0, 0), cv::Scalar(0, 255, 255),
                                                  cv::Scalar(255, 0, 255)};
        for (std::size_t i = 0; i < f.landmarks.size(); ++i) {
            cv::circle(img, f.landmarks[i], 2, lm_colors[i], -1);
        }
        char buf[16];
        std::snprintf(buf, sizeof(buf), "%.2f", static_cast<double>(f.score));
        cv::putText(
            img, buf,
            cv::Point(static_cast<int>(f.bbox.x), std::max(0, static_cast<int>(f.bbox.y) - 5)),
            cv::FONT_HERSHEY_SIMPLEX, 0.45, cv::Scalar(72, 207, 235), 1);
    }
}

bool is_image(const std::string& path) {
    const auto dot = path.find_last_of('.');
    if (dot == std::string::npos)
        return false;
    std::string ext = path.substr(dot);
    for (auto& c : ext)
        c = static_cast<char>(::tolower(c));
    return ext == ".jpg" || ext == ".jpeg" || ext == ".png" || ext == ".bmp";
}

}  // namespace

int main(int argc, char** argv) {
    std::string config_path;
    std::string input_path;
    std::string output_path;

    for (int i = 1; i < argc; ++i) {
        const std::string a = argv[i];
        auto take = [&](const std::string& flag) {
            if (i + 1 >= argc)
                throw std::invalid_argument(flag + " expects a value");
            return std::string(argv[++i]);
        };
        if (a == "--config" || a == "-c")
            config_path = take(a);
        else if (a == "--input" || a == "-i")
            input_path = take(a);
        else if (a == "--output" || a == "-o")
            output_path = take(a);
        else if (a == "--help" || a == "-h") {
            std::cerr << "Usage: " << argv[0]
                      << " --config CFG --input IMG_OR_VIDEO --output OUT\n";
            return EXIT_SUCCESS;
        }
    }
    if (config_path.empty() || input_path.empty() || output_path.empty()) {
        std::cerr << "Usage: " << argv[0] << " --config CFG --input IMG_OR_VIDEO --output OUT\n";
        return EXIT_FAILURE;
    }

    try {
        const auto cfg = face_pipeline::config::SystemConfig::load(config_path);
        face_pipeline::utils::init_logger(cfg.logging.level, cfg.logging.json);
        face_pipeline::trt::SCRFDDetector detector(cfg.detection);

        if (is_image(input_path)) {
            cv::Mat img = cv::imread(input_path);
            if (img.empty())
                throw std::runtime_error("cannot read image: " + input_path);
            const auto faces = detector.detect(img);
            draw(img, faces);
            cv::imwrite(output_path, img);
            SPDLOG_INFO("detected {} face(s); wrote {}", faces.size(), output_path);
        } else {
            cv::VideoCapture cap(input_path);
            if (!cap.isOpened())
                throw std::runtime_error("cannot open video: " + input_path);
            const int W = static_cast<int>(cap.get(cv::CAP_PROP_FRAME_WIDTH));
            const int H = static_cast<int>(cap.get(cv::CAP_PROP_FRAME_HEIGHT));
            const double fps = cap.get(cv::CAP_PROP_FPS);
            cv::VideoWriter writer(output_path, cv::VideoWriter::fourcc('m', 'p', '4', 'v'),
                                   (fps > 0.0 ? fps : 25.0), cv::Size(W, H));
            cv::Mat frame;
            std::uint64_t n = 0;
            std::uint64_t total = 0;
            while (cap.read(frame)) {
                const auto faces = detector.detect(frame);
                total += faces.size();
                draw(frame, faces);
                writer.write(frame);
                ++n;
            }
            SPDLOG_INFO("processed {} frames; {} detections; wrote {}", n, total, output_path);
        }
    } catch (const std::exception& e) {
        SPDLOG_CRITICAL("fatal: {}", e.what());
        return EXIT_FAILURE;
    }
    return EXIT_SUCCESS;
}
