#include "../include/libfacialauth.h"
#include <opencv2/highgui.hpp>
#include <opencv2/imgproc.hpp>
#include <opencv2/objdetect.hpp>
#include <opencv2/videoio.hpp>
#include <opencv2/face.hpp>
#include <opencv2/dnn.hpp>
#include <fstream>
#include <filesystem>
#include <iostream>
#include <thread>
#include <unistd.h>

namespace fs = std::filesystem;

// --- PLUGIN CLASS: CLASSIC (LBPH, Eigen, Fisher) ---
class ClassicPlugin : public RecognizerPlugin {
    cv::Ptr<cv::face::FaceRecognizer> model;
    std::string method;
public:
    ClassicPlugin(const std::string& m) : method(m) {
        if (method == "eigen") model = cv::face::EigenFaceRecognizer::create();
        else if (method == "fisher") model = cv::face::FisherFaceRecognizer::create();
        else model = cv::face::LBPHFaceRecognizer::create();
    }
    bool load(const std::string& path) override { if(!fs::exists(path)) return false; model->read(path); return true; }
    bool train(const std::vector<cv::Mat>& f, const std::vector<int>& l, const std::string& p) override {
        model->train(f, l); model->save(p); return true;
    }
    bool predict(const cv::Mat& face, int& label, double& conf) override {
        cv::Mat gray; if(face.channels() == 3) cv::cvtColor(face, gray, cv::COLOR_BGR2GRAY); else gray = face;
        model->predict(gray, label, conf); return true;
    }
    std::string get_name() const override { return method; }
};

// --- PLUGIN CLASS: SFACE (Deep Learning) ---
class SFacePlugin : public RecognizerPlugin {
    cv::Ptr<cv::FaceRecognizerSF> model;
    std::string model_path;
public:
    SFacePlugin(const std::string& p) : model_path(p) {
        model = cv::FaceRecognizerSF::create(model_path, "");
    }
    bool load(const std::string& path) override { return fs::exists(path); } // SFace carica i pesi ONNX all'inizio
    bool train(const std::vector<cv::Mat>& f, const std::vector<int>& l, const std::string& p) override { return true; } // Feature extraction logic here
    bool predict(const cv::Mat& face, int& label, double& conf) override { return false; } // Placeholder
    std::string get_name() const override { return "sface"; }
};

std::unique_ptr<RecognizerPlugin> fa_create_plugin(const FacialAuthConfig& cfg) {
    if (cfg.training_method == "sface") return std::make_unique<SFacePlugin>(cfg.recognize_sface);
    return std::make_unique<ClassicPlugin>(cfg.training_method);
}

// --- CORE LOGIC ---

bool fa_capture_user(const std::string &user, const FacialAuthConfig &cfg, const std::string &det_type, std::string &log) {
    std::string user_dir = cfg.basedir + "/" + user + "/captures";
    if (cfg.force) fs::remove_all(user_dir);
    fs::create_directories(user_dir);

    cv::Ptr<cv::FaceDetectorYN> yunet;
    if (det_type == "yunet") yunet = cv::FaceDetectorYN::create(cfg.detect_model_path, "", cv::Size(320, 320));

    cv::VideoCapture cap;
    try { cap.open(std::stoi(cfg.device)); } catch(...) { cap.open(cfg.device); }
    if (!cap.isOpened()) { log = "Camera error"; return false; }

    int count = 0;
    while (count < cfg.frames) {
        cv::Mat frame; cap >> frame;
        if (frame.empty()) break;

        bool detected = true;
        if (det_type == "yunet") {
            cv::Mat faces; yunet->setInputSize(frame.size());
            yunet->detect(frame, faces);
            detected = (faces.rows > 0);
        }

        if (detected) {
            cv::imwrite(user_dir + "/img_" + std::to_string(count++) + "." + cfg.image_format, frame);
        }
        if (!cfg.nogui) { cv::imshow("Capture", frame); if(cv::waitKey(1) == 'q') break; }
    }
    return true;
}

bool fa_train_user(const std::string &user, const FacialAuthConfig &cfg, std::string &log) {
    std::string user_dir = cfg.basedir + "/" + user + "/captures";
    std::vector<cv::Mat> faces; std::vector<int> labels;
    for (const auto& entry : fs::directory_iterator(user_dir)) {
        cv::Mat img = cv::imread(entry.path().string());
        if (!img.empty()) { faces.push_back(img); labels.push_back(0); }
    }
    if (faces.empty()) return false;
    auto plugin = fa_create_plugin(cfg);
    return plugin->train(faces, labels, fa_user_model_path(cfg, user));
}

bool fa_test_user(const std::string &user, const FacialAuthConfig &cfg, const std::string &modelPath, double &best_conf, int &best_label, std::string &log) {
    auto plugin = fa_create_plugin(cfg);
    if (!plugin->load(modelPath)) return false;
    cv::VideoCapture cap;
    try { cap.open(std::stoi(cfg.device)); } catch(...) { cap.open(cfg.device); }
    cv::Mat frame; cap >> frame;
    if (frame.empty()) return false;
    return plugin->predict(frame, best_label, best_conf);
}

// ... helper fa_load_config, fa_check_root, etc ... (come prima)
