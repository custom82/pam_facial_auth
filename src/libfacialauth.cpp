#include "../include/libfacialauth.h"
#include <opencv2/highgui.hpp>
#include <opencv2/imgproc.hpp>
#include <opencv2/objdetect.hpp>
#include <opencv2/videoio.hpp>
#include <opencv2/face.hpp>
#include <opencv2/dnn.hpp>
#include <filesystem>
#include <iostream>
#include <fstream>
#include <unistd.h>
#include <regex>

namespace fs = std::filesystem;

// Modern C++20 string utility
std::string trim(std::string_view s) {
    auto first = s.find_first_not_of(" \t\r\n");
    if (std::string::npos == first) return "";
    auto last = s.find_last_not_of(" \t\r\n");
    return std::string(s.substr(first, (last - first + 1)));
}

// FIX: Implementation of missing function
bool fa_file_exists(std::string_view path) {
    if (path.empty()) return false;
    return fs::exists(path);
}

class SFacePlugin : public RecognizerPlugin {
    cv::Ptr<cv::FaceRecognizerSF> face_recon;
    cv::Mat registered_embeddings;
public:
    SFacePlugin(const std::string& m_path) {
        if (!m_path.empty() && fs::exists(m_path)) {
            face_recon = cv::FaceRecognizerSF::create(m_path, "");
        }
    }
    bool train(const std::vector<cv::Mat>& faces, const std::vector<int>& labels, const std::string& save_path) override {
        if (!face_recon || faces.empty()) return false;
        cv::Mat all_embeddings;
        for (const auto& face : faces) {
            cv::Mat feature;
            face_recon->feature(face, feature);
            all_embeddings.push_back(feature.clone());
        }
        cv::FileStorage fs_out(save_path, cv::FileStorage::WRITE);
        fs_out << "algorithm" << "sface" << "embeddings" << all_embeddings;
        return true;
    }
    bool load(const std::string& path) override {
        if (!fa_file_exists(path)) return false;
        cv::FileStorage fs_in(path, cv::FileStorage::READ);
        std::string algo; fs_in["algorithm"] >> algo;
        if (algo != "sface") return false;
        fs_in["embeddings"] >> registered_embeddings;
        return !registered_embeddings.empty();
    }
    bool predict(const cv::Mat& face, int& label, double& confidence) override {
        if (!face_recon || registered_embeddings.empty()) return false;
        cv::Mat query; face_recon->feature(face, query);
        double max_s = -1.0;
        for (int i = 0; i < registered_embeddings.rows; i++) {
            double s = face_recon->match(query, registered_embeddings.row(i), cv::FaceRecognizerSF::FR_COSINE);
            if (s > max_s) max_s = s;
        }
        confidence = max_s; label = 0;
        return true;
    }
};

class ClassicPlugin : public RecognizerPlugin {
    cv::Ptr<cv::face::FaceRecognizer> model;
public:
    ClassicPlugin(const std::string& m) {
        if (m == "eigen") model = cv::face::EigenFaceRecognizer::create();
        else if (m == "fisher") model = cv::face::FisherFaceRecognizer::create();
        else model = cv::face::LBPHFaceRecognizer::create();
    }
    bool train(const std::vector<cv::Mat>& f, const std::vector<int>& l, const std::string& p) override {
        if (f.empty()) return false;
        model->train(f, l); model->save(p); return true;
    }
    bool load(const std::string& p) override {
        if (!fa_file_exists(p)) return false;
        model->read(p); return true;
    }
    bool predict(const cv::Mat& f, int& l, double& c) override {
        cv::Mat gray;
        if (f.channels() == 3) cv::cvtColor(f, gray, cv::COLOR_BGR2GRAY); else gray = f;
        model->predict(gray, l, c); return true;
    }
};

std::unique_ptr<RecognizerPlugin> get_plugin(const FacialAuthConfig& cfg) {
    if (cfg.training_method == "sface") return std::make_unique<SFacePlugin>(cfg.rec_model_path);
    return std::make_unique<ClassicPlugin>(cfg.training_method);
}

bool fa_load_config(FacialAuthConfig &cfg, std::string &log, const std::string &path) {
    std::ifstream file(path);
    if (!file.is_open()) { log = "Config not found"; return false; }
    std::string line;
    while (std::getline(file, line)) {
        std::string_view sv = line;
        auto trimmed = trim(sv);
        if (trimmed.empty() || trimmed[0] == '#') continue;
        auto sep = trimmed.find('=');
        if (sep == std::string::npos) continue;

        std::string k = trim(trimmed.substr(0, sep));
        std::string v = trim(trimmed.substr(sep + 1));

        if (k == "basedir") cfg.basedir = v;
        else if (k == "device") cfg.device = v;
        else if (k == "training_method") cfg.training_method = v;
        else if (k == "image_format") cfg.image_format = v;
        else if (k == "detect_yunet") cfg.detect_model_path = v;
        else if (k == "recognize_sface") cfg.rec_model_path = v;
        else if (k == "frames") cfg.frames = std::stoi(v);
        else if (k == "width") cfg.width = std::stoi(v);
        else if (k == "height") cfg.height = std::stoi(v);
        else if (k == "sface_threshold") cfg.sface_threshold = std::stod(v);
        else if (k == "lbph_threshold") cfg.lbph_threshold = std::stod(v);
        else if (k == "debug") cfg.debug = (v == "yes");
        else if (k == "nogui") cfg.nogui = (v == "yes");
    }
    if (cfg.training_method == "auto")
        cfg.training_method = (fa_file_exists(cfg.rec_model_path)) ? "sface" : "lbph";
    return true;
}

bool fa_check_root(std::string_view tool) {
    if (getuid() != 0) { std::cerr << "Error: " << tool << " needs root.\n"; return false; }
    return true;
}

std::string fa_user_model_path(const FacialAuthConfig &cfg, std::string_view u) {
    return (fs::path(cfg.basedir) / u / "model.xml").string();
}

bool fa_capture_user(std::string_view u, const FacialAuthConfig &cfg, std::string_view det, std::string &log) {
    fs::path dir = fs::path(cfg.basedir) / u / "captures";
    if (cfg.force) fs::remove_all(dir);
    fs::create_directories(dir);

    int start_idx = 0;
    std::regex re("img_([0-9]+)\\." + cfg.image_format);
    if (fs::exists(dir)) {
        for (const auto& e : fs::directory_iterator(dir)) {
            std::smatch m; std::string fn = e.path().filename().string();
            if (std::regex_match(fn, m, re)) start_idx = std::max(start_idx, std::stoi(m[1].str()) + 1);
        }
    }

    cv::VideoCapture cap;
    try { cap.open(std::stoi(cfg.device)); } catch(...) { cap.open(cfg.device.c_str()); }
    if (!cap.isOpened()) { log = "Camera error"; return false; }

    cv::Ptr<cv::FaceDetectorYN> yunet;
    if (det == "yunet" && fa_file_exists(cfg.detect_model_path)) {
        yunet = cv::FaceDetectorYN::create(cfg.detect_model_path, "", cv::Size(cfg.width, cfg.height));
    }

    int count = 0;
    while (count < cfg.frames) {
        cv::Mat frame; cap >> frame; if (frame.empty()) break;
        bool ok = true;
        if (yunet) {
            cv::Mat faces; yunet->setInputSize(frame.size()); yunet->detect(frame, faces);
            ok = (faces.rows > 0);
        }
        if (ok) {
            std::string p = (dir / ("img_" + std::to_string(start_idx + count) + "." + cfg.image_format)).string();
            cv::imwrite(p, frame);
            std::cout << "Saved: " << p << std::endl;
            count++;
        }
    }
    return count >= cfg.frames;
}

bool fa_train_user(std::string_view u, const FacialAuthConfig &cfg, std::string &log) {
    fs::path dir = fs::path(cfg.basedir) / u / "captures";
    if (!fs::exists(dir)) return false;
    std::vector<cv::Mat> faces; std::vector<int> labels;
    for (const auto& e : fs::directory_iterator(dir)) {
        cv::Mat img = cv::imread(e.path().string());
        if (!img.empty()) { faces.push_back(img); labels.push_back(0); }
    }
    return get_plugin(cfg)->train(faces, labels, fa_user_model_path(cfg, u));
}
