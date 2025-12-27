#include "../include/libfacialauth.h"
#include <opencv2/highgui.hpp>
#include <opencv2/imgproc.hpp>
#include <opencv2/objdetect.hpp>
#include <opencv2/videoio.hpp>
#include <opencv2/face.hpp>
#include <opencv2/dnn.hpp>
#include <filesystem>
#include <fstream>
#include <iostream>
#include <unistd.h>

namespace fs = std::filesystem;

// --- IMPLEMENTAZIONE PLUGIN ---
class ClassicPlugin : public RecognizerPlugin {
    cv::Ptr<cv::face::FaceRecognizer> model;
public:
    ClassicPlugin(const std::string& method) {
        if (method == "eigen") model = cv::face::EigenFaceRecognizer::create();
        else if (method == "fisher") model = cv::face::FisherFaceRecognizer::create();
        else model = cv::face::LBPHFaceRecognizer::create();
    }
    bool load(const std::string& path) override { return fs::exists(path) && (model->read(path), true); }
    bool train(const std::vector<cv::Mat>& f, const std::vector<int>& l, const std::string& p) override {
        model->train(f, l); model->save(p); return true;
    }
    bool predict(const cv::Mat& face, int& label, double& conf) override {
        cv::Mat gray;
        if(face.channels() == 3) cv::cvtColor(face, gray, cv::COLOR_BGR2GRAY); else gray = face;
        model->predict(gray, label, conf); return true;
    }
};

// --- API FUNCTIONS ---
bool fa_file_exists(const std::string &path) { return fs::exists(path); }

bool fa_load_config(FacialAuthConfig &cfg, std::string &log, const std::string &path) {
    if (!fs::exists(path)) { log = "Config file missing"; return false; }
    return true;
}

bool fa_check_root(const std::string &tool_name) {
    if (getuid() != 0) { std::cerr << tool_name << " must be run as root\n"; return false; }
    return true;
}

std::string fa_user_model_path(const FacialAuthConfig &cfg, const std::string &user) {
    return cfg.basedir + "/" + user + "/model.xml";
}

bool fa_capture_user(const std::string &user, const FacialAuthConfig &cfg, const std::string &det_type, std::string &log) {
    std::string user_dir = cfg.basedir + "/" + user + "/captures";
    if (cfg.force) fs::remove_all(user_dir);
    fs::create_directories(user_dir);

    cv::Ptr<cv::FaceDetectorYN> detector;
    if (det_type == "yunet") detector = cv::FaceDetectorYN::create(cfg.detect_model_path, "", cv::Size(320, 320));

    cv::VideoCapture cap;
    try { cap.open(std::stoi(cfg.device)); } catch(...) { cap.open(cfg.device); }
    if (!cap.isOpened()) { log = "Camera error"; return false; }

    int count = 0;
    while (count < cfg.frames) {
        cv::Mat frame; cap >> frame;
        if (frame.empty()) break;
        if (detector) {
            cv::Mat faces; detector->setInputSize(frame.size()); detector->detect(frame, faces);
            if (faces.rows == 0) continue;
        }
        cv::imwrite(user_dir + "/img_" + std::to_string(count++) + "." + cfg.image_format, frame);
        if (!cfg.nogui) { cv::imshow("Capture", frame); if(cv::waitKey(1) == 'q') break; }
    }
    return true;
}

bool fa_test_user(const std::string &user, const FacialAuthConfig &cfg, const std::string &model_path, double &conf, int &label, std::string &log) {
    ClassicPlugin plugin(cfg.training_method);
    if (!plugin.load(model_path)) return false;
    cv::VideoCapture cap;
    try { cap.open(std::stoi(cfg.device)); } catch(...) { cap.open(cfg.device); }
    cv::Mat frame; cap >> frame;
    return !frame.empty() && plugin.predict(frame, label, conf);
}

bool fa_train_user(const std::string &user, const FacialAuthConfig &cfg, std::string &log) {
    std::string user_dir = cfg.basedir + "/" + user + "/captures";
    std::vector<cv::Mat> faces; std::vector<int> labels;
    for (const auto& entry : fs::directory_iterator(user_dir)) {
        cv::Mat img = cv::imread(entry.path().string(), cv::IMREAD_GRAYSCALE);
        if (!img.empty()) { faces.push_back(img); labels.push_back(0); }
    }
    ClassicPlugin plugin(cfg.training_method);
    return plugin.train(faces, labels, fa_user_model_path(cfg, user));
}
