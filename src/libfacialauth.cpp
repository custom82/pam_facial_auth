/*
 * Project: pam_facial_auth
 * License: GPL-3.0
 */

#include "libfacialauth.h"
#include <iostream>
#include <fstream>
#include <filesystem>
#include <thread>
#include <chrono>
#include <unistd.h>

namespace fs = std::filesystem;

// DICHIARAZIONI ESTERNE (Implementate nei file plugin)
extern "C" std::unique_ptr<RecognizerPlugin> create_classic_plugin(const std::string& method, const FacialAuthConfig& cfg);
extern "C" std::unique_ptr<RecognizerPlugin> create_sface_plugin(const FacialAuthConfig& cfg);

// Factory interna
std::unique_ptr<RecognizerPlugin> create_plugin(const FacialAuthConfig& cfg) {
    if (cfg.method == "sface") return create_sface_plugin(cfg);
    return create_classic_plugin(cfg.method, cfg);
}

bool fa_check_root(const std::string& tool_name) {
    if (getuid() != 0) {
        std::cerr << "ERRORE [" << tool_name << "]: Devi eseguire come root." << std::endl;
        return false;
    }
    return true;
}

// ... (Includi qui le tue funzioni fa_load_config, fa_user_model_path, fa_clean_captures già esistenti) ...

bool fa_capture_user(const std::string& user, const FacialAuthConfig& cfg, const std::string& device_path, std::string& log) {
    cv::VideoCapture cap(device_path);
    if (!cap.isOpened()) { log = "Webcam non trovata"; return false; }

    cv::CascadeClassifier face_cascade;
    face_cascade.load(cfg.cascade_path);

    std::string user_dir = cfg.basedir + "/captures/" + user;
    fs::create_directories(user_dir);

    int count = 0;
    while (count < cfg.frames) {
        cv::Mat frame; cap >> frame;
        if (frame.empty()) break;
        std::vector<cv::Rect> faces;
        face_cascade.detectMultiScale(frame, faces);
        for (const auto& area : faces) {
            cv::Mat face_roi = frame(area);
            cv::resize(face_roi, face_roi, cv::Size(cfg.width, cfg.height));
            cv::imwrite(user_dir + "/" + std::to_string(count) + ".jpg", face_roi);
            count++;
            if (count >= cfg.frames) break;
        }
    }
    return true;
}

bool fa_train_user(const std::string& user, const FacialAuthConfig& cfg, std::string& log) {
    std::string user_dir = cfg.basedir + "/captures/" + user;
    if (!fs::exists(user_dir)) { log = "Nessuna immagine acquisita"; return false; }

    std::vector<cv::Mat> faces;
    std::vector<int> labels;
    for (const auto& entry : fs::directory_iterator(user_dir)) {
        cv::Mat img = cv::imread(entry.path().string());
        if (!img.empty()) { faces.push_back(img); labels.push_back(0); }
    }

    auto plugin = create_plugin(cfg);
    return plugin->train(faces, labels, fa_user_model_path(cfg, user));
}

bool fa_test_user(const std::string& user, const FacialAuthConfig& cfg, const std::string& model_path, double& confidence, int& label, std::string& log) {
    auto plugin = create_plugin(cfg);
    if (!plugin->load(model_path)) { log = "Modello non trovato"; return false; }

    cv::VideoCapture cap(0);
    cv::Mat frame; cap >> frame;
    if (frame.empty()) { log = "Errore webcam"; return false; }

    return plugin->predict(frame, label, confidence);
}
