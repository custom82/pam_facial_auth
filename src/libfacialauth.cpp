#include "../include/libfacialauth.h"
#include <opencv2/highgui.hpp>
#include <opencv2/imgproc.hpp>
#include <opencv2/objdetect.hpp>
#include <opencv2/videoio.hpp>
#include <opencv2/face.hpp>
#include <opencv2/dnn.hpp>
#include <opencv2/core/ocl.hpp>
#include <filesystem>
#include <iostream>
#include <fstream>
#include <thread>
#include <chrono>

namespace fs = std::filesystem;

static std::string trim(std::string_view s) {
    auto first = s.find_first_not_of(" \t\r\n");
    if (std::string::npos == first) return "";
    auto last = s.find_last_not_of(" \t\r\n");
    return std::string(s.substr(first, (last - first + 1)));
}

bool fa_file_exists(std::string_view path) {
    return !path.empty() && fs::exists(path);
}

void get_best_dnn_backend(bool use_accel, int &backend, int &target) {
    backend = cv::dnn::DNN_BACKEND_DEFAULT;
    target = cv::dnn::DNN_TARGET_CPU;
    if (use_accel && cv::ocl::haveOpenCL()) {
        cv::ocl::setUseOpenCL(true);
        backend = cv::dnn::DNN_BACKEND_OPENCV;
        target = cv::dnn::DNN_TARGET_OPENCL;
    }
}

/* --- PLUGINS --- */

class SFacePlugin : public RecognizerPlugin {
    cv::Ptr<cv::FaceRecognizerSF> face_recon;
    cv::Mat registered_embeddings;
public:
    SFacePlugin(const std::string& m_path, bool use_accel) {
        if (fa_file_exists(m_path)) {
            int b, t; get_best_dnn_backend(use_accel, b, t);
            face_recon = cv::FaceRecognizerSF::create(m_path, "", b, t);
        }
    }
    bool load(const std::string& path) override {
        if (!fa_file_exists(path)) return false;
        cv::FileStorage fs_in(path, cv::FileStorage::READ);
        if (!fs_in.isOpened()) return false;
        std::string algo; fs_in["algorithm"] >> algo;
        if (algo != "sface") return false;
        fs_in["embeddings"] >> registered_embeddings;
        return !registered_embeddings.empty();
    }
    bool train(const std::vector<cv::Mat>& faces, const std::vector<int>& labels, const std::string& save_path) override {
        if (!face_recon || faces.empty()) return false;
        cv::Mat all_embeddings;
        for (const auto& face : faces) {
            cv::Mat feature; face_recon->feature(face, feature);
            all_embeddings.push_back(feature.clone());
        }
        cv::FileStorage fs_out(save_path, cv::FileStorage::WRITE);
        fs_out << "algorithm" << "sface" << "embeddings" << all_embeddings;
        return true;
    }
    bool predict(const cv::Mat& face, int& label, double& confidence) override {
        if (!face_recon || registered_embeddings.empty()) return false;
        cv::Mat query; face_recon->feature(face, query);
        double max_s = -1.0;
        for (int i = 0; i < registered_embeddings.rows; i++) {
            double s = face_recon->match(query, registered_embeddings.row(i), cv::FaceRecognizerSF::FR_COSINE);
            if (s > max_s) max_s = s;
        }
        confidence = max_s; label = 0; return true;
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
    bool load(const std::string& p) override {
        if (!fa_file_exists(p)) return false;
        try {
            model->read(p); // FIX: riga 1405 del tuo log
            return true;
        } catch (...) { return false; }
    }
    bool train(const std::vector<cv::Mat>& f, const std::vector<int>& l, const std::string& s) override {
        if (f.empty()) return false;
        model->train(f, l);
        model->save(s);
        return true;
    }
    bool predict(const cv::Mat& f, int& l, double& c) override {
        cv::Mat gray;
        if (f.channels() == 3) cv::cvtColor(f, gray, cv::COLOR_BGR2GRAY);
        else gray = f;
        model->predict(gray, l, c);
        return true;
    }
};

/* --- CORE FUNCTIONS --- */

bool fa_load_config(FacialAuthConfig &cfg, std::string &log, const std::string &path) {
    std::ifstream file(path);
    if (!file.is_open()) { log = "Config not found: " + path; return false; }
    std::string line;
    while (std::getline(file, line)) {
        std::string_view sv = line; auto t = trim(sv);
        if (t.empty() || t[0] == '#') continue;
        auto sep = t.find('='); if (sep == std::string::npos) continue;
        std::string k = trim(t.substr(0, sep)), v = trim(t.substr(sep + 1));
        if (k == "use_accel") cfg.use_accel = (v == "yes");
        else if (k == "basedir") cfg.basedir = v;
        else if (k == "modeldir") cfg.modeldir = v;
        else if (k == "device") cfg.device = v;
        else if (k == "training_method") cfg.training_method = v;
        else if (k == "detect_yunet") cfg.detect_model_path = v;
        else if (k == "recognize_sface") cfg.rec_model_path = v;
        else if (k == "sface_threshold") cfg.sface_threshold = std::stod(v);
        else if (k == "lbph_threshold") cfg.lbph_threshold = std::stod(v);
        else if (k == "nogui") cfg.nogui = (v == "yes");
    }
    return true;
}

// FIX: Qui c'era lo sbilanciamento di graffe (riga 1106)
bool fa_capture_dataset(const FacialAuthConfig &cfg, std::string &log, const std::string &imgdir, int start_index) {
    cv::VideoCapture cap;
    try { cap.open(std::stoi(cfg.device)); } catch(...) { cap.open(cfg.device.c_str()); }
    if (!cap.isOpened()) { log = "Camera error"; return false; }

    cv::Ptr<cv::FaceDetectorYN> detector;
    int b, t; get_best_dnn_backend(cfg.use_accel, b, t);
    detector = cv::FaceDetectorYN::create(cfg.detect_model_path, "", cv::Size(cfg.width, cfg.height), 0.9f, 0.3f, 5000, b, t);

    int saved = 0;
    std::string img_format = "jpg";
    cv::Mat frame;

    for (int i = 0; i < 200 && saved < cfg.frames; i++) {
        cap >> frame;
        if (frame.empty()) continue;

        detector->setInputSize(frame.size());
        cv::Mat faces;
        detector->detect(frame, faces);

        if (faces.rows > 0) {
            int idx = start_index + saved;
            std::string outfile = imgdir + "/" + std::to_string(idx) + "." + img_format;
            if (cv::imwrite(outfile, frame)) {
                saved++;
                if (cfg.debug) std::cout << "[DEBUG] Saved " << saved << std::endl;
            }
        }
        if (cfg.sleep_ms > 0) {
            std::this_thread::sleep_for(std::chrono::milliseconds(cfg.sleep_ms));
        }
    }

    if (saved == 0) {
        log = "No faces captured.";
        return false;
    }
    log = "Captured " + std::to_string(saved) + " images.";
    return true;
}

bool fa_test_user(std::string_view u, const FacialAuthConfig &cfg, const std::string &m_path, double &conf, int &label, std::string &log) {
    std::unique_ptr<RecognizerPlugin> plugin;
    if (cfg.training_method == "sface") plugin = std::make_unique<SFacePlugin>(cfg.rec_model_path, cfg.use_accel);
    else plugin = std::make_unique<ClassicPlugin>(cfg.training_method);

    if (!plugin->load(m_path)) { log = "Model missing"; return false; }

    cv::VideoCapture cap;
    try { cap.open(std::stoi(cfg.device)); } catch(...) { cap.open(cfg.device.c_str()); }
    if (!cap.isOpened()) { log = "Camera error"; return false; }

    cv::Ptr<cv::FaceDetectorYN> detector;
    if (fa_file_exists(cfg.detect_model_path)) {
        int b, t; get_best_dnn_backend(cfg.use_accel, b, t);
        detector = cv::FaceDetectorYN::create(cfg.detect_model_path, "", cv::Size(cfg.width, cfg.height), 0.9f, 0.3f, 5000, b, t);
    }

    cv::Mat frame; bool face_ok = false;
    for (int i = 0; i < 30; i++) {
        cap >> frame; if (frame.empty()) continue;
        if (detector) {
            cv::Mat faces; detector->setInputSize(frame.size()); detector->detect(frame, faces);
            if (faces.rows > 0) { face_ok = true; break; }
        } else { face_ok = true; break; }
    }

    if (!face_ok) { log = "No face"; return false; }
    return plugin->predict(frame, label, conf);
}

std::string fa_user_model_path(const FacialAuthConfig &cfg, std::string_view u) {
    return (fs::path(cfg.modeldir) / (std::string(u) + ".xml")).string();
}
