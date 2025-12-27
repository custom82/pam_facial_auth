#include "../include/libfacialauth.h"
#include <opencv2/highgui.hpp>
#include <opencv2/imgproc.hpp>
#include <opencv2/objdetect.hpp>
#include <opencv2/face.hpp>
#include <filesystem>
#include <fstream>
#include <iostream>
#include <unistd.h>

namespace fs = std::filesystem;

bool fa_load_config(FacialAuthConfig &cfg, std::string &log, const std::string &path) {
    if (!fs::exists(path)) { log = "Config not found"; return false; }
    return true; // Implementazione parsing .conf qui
}

bool fa_check_root(const std::string &tool_name) {
    if (getuid() != 0) { std::cerr << tool_name << " must be run as root\n"; return false; }
    return true;
}

bool fa_file_exists(const std::string &path) { return fs::exists(path); }

std::string fa_user_model_path(const FacialAuthConfig &cfg, const std::string &user) {
    return cfg.basedir + "/" + user + "/model.xml";
}

bool fa_capture_user(const std::string &user, const FacialAuthConfig &cfg, const std::string &detector_type, std::string &log) {
    std::string user_dir = cfg.basedir + "/" + user + "/captures";
    if (cfg.force) fs::remove_all(user_dir);
    fs::create_directories(user_dir);

    cv::VideoCapture cap;
    try { cap.open(std::stoi(cfg.device)); } catch(...) { cap.open(cfg.device); }
    if (!cap.isOpened()) { log = "Cam error"; return false; }

    int count = 0;
    while (count < cfg.frames) {
        cv::Mat frame; cap >> frame;
        if (frame.empty()) break;
        cv::imwrite(user_dir + "/img_" + std::to_string(count++) + "." + cfg.image_format, frame);
        if (!cfg.nogui) { cv::imshow("Capture", frame); if(cv::waitKey(1) == 'q') break; }
    }
    return true;
}

bool fa_train_user(const std::string &user, const FacialAuthConfig &cfg, std::string &log) {
    std::string user_dir = cfg.basedir + "/" + user + "/captures";
    std::vector<cv::Mat> faces; std::vector<int> labels;
    for (const auto& entry : fs::directory_iterator(user_dir)) {
        cv::Mat img = cv::imread(entry.path().string(), cv::IMREAD_GRAYSCALE);
        if (!img.empty()) { faces.push_back(img); labels.push_back(0); }
    }
    auto model = cv::face::LBPHFaceRecognizer::create();
    model->train(faces, labels);
    model->save(fa_user_model_path(cfg, user));
    return true;
}

bool fa_test_user(const std::string &user, const FacialAuthConfig &cfg, const std::string &model_path, double &confidence, int &label, std::string &log) {
    auto model = cv::face::LBPHFaceRecognizer::create();
    if (!fs::exists(model_path)) return false;
    model->read(model_path);
    cv::VideoCapture cap;
    try { cap.open(std::stoi(cfg.device)); } catch(...) { cap.open(cfg.device); }
    cv::Mat frame, gray; cap >> frame;
    if (frame.empty()) return false;
    cv::cvtColor(frame, gray, cv::COLOR_BGR2GRAY);
    model->predict(gray, label, confidence);
    return true;
}

bool fa_test_user_interactive(const std::string &user, const FacialAuthConfig &cfg, std::string &log) {
    double conf; int label;
    return fa_test_user(user, cfg, fa_user_model_path(cfg, user), conf, label, log);
}
