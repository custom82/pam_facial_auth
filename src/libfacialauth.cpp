#include "../include/libfacialauth.h"
#include <iostream>
#include <fstream>
#include <filesystem>
#include <thread>
#include <chrono>
#include <unistd.h>

namespace fs = std::filesystem;

bool fa_check_root(const std::string& tool_name) {
    if (geteuid() != 0) {
        std::cerr << "[ERROR] " << tool_name << " deve essere eseguito come root.\n";
        return false;
    }
    return true;
}

bool fa_file_exists(const std::string& path) {
    return fs::exists(path);
}

bool fa_load_config(FacialAuthConfig& cfg, std::string& log, const std::string& path) {
    if (!fs::exists(path)) {
        log = "Configurazione non trovata: " + path + " (uso default)";
        return true;
    }
    // Logica di parsing qui...
    return true;
}

std::string fa_user_model_path(const FacialAuthConfig& cfg, const std::string& user) {
    return (fs::path(cfg.basedir) / "models" / (user + ".xml")).string();
}

bool fa_capture_dataset(const FacialAuthConfig& cfg, std::string& log, const std::string& user, int count) {
    cv::VideoCapture cap(0);
    if (!cap.isOpened()) {
        log = "Camera non disponibile.";
        return false;
    }

    fs::path user_dir = fs::path(cfg.basedir) / "data" / user;
    fs::create_directories(user_dir);

    cv::CascadeClassifier face_cascade(cfg.cascade_path);
    int saved = 0;
    cv::Mat frame, gray;

    while (saved < count) {
        cap >> frame;
        if (frame.empty()) break;
        cv::cvtColor(frame, gray, cv::COLOR_BGR2GRAY);

        std::vector<cv::Rect> faces;
        face_cascade.detectMultiScale(gray, faces, 1.1, 4);

        for (const auto& area : faces) {
            cv::Mat face_roi = gray(area);
            cv::resize(face_roi, face_roi, cv::Size(cfg.width, cfg.height));
            std::string filename = (user_dir / (std::to_string(saved) + "." + cfg.image_format)).string();
            cv::imwrite(filename, face_roi);
            saved++;
            if (saved >= count) break;
        }
        if (cfg.sleep_ms > 0) std::this_thread::sleep_for(std::chrono::milliseconds(cfg.sleep_ms));
    }
    return true;
}

bool fa_capture_user(const std::string& user, const FacialAuthConfig& cfg, const std::string& detector, std::string& log) {
    return fa_capture_dataset(cfg, log, user, cfg.frames);
}

bool fa_train_user(const std::string& user, const FacialAuthConfig& cfg, std::string& log) {
    fs::path user_dir = fs::path(cfg.basedir) / "data" / user;
    std::string model_path = fa_user_model_path(cfg, user);

    if (!fs::exists(user_dir)) {
        log = "Dataset non trovato in " + user_dir.string();
        return false;
    }

    std::vector<cv::Mat> images;
    std::vector<int> labels;

    for (const auto& entry : fs::directory_iterator(user_dir)) {
        cv::Mat img = cv::imread(entry.path().string(), cv::IMREAD_GRAYSCALE);
        if (!img.empty()) {
            images.push_back(img);
            labels.push_back(0);
        }
    }

    if (images.empty()) {
        log = "Dataset vuoto.";
        return false;
    }

    cv::Ptr<cv::face::FaceRecognizer> model = cv::face::LBPHFaceRecognizer::create();
    model->train(images, labels);
    fs::create_directories(fs::path(cfg.basedir) / "models");
    model->save(model_path);
    return true;
}

bool fa_test_user(const std::string& user, const FacialAuthConfig& cfg, const std::string& model_path, double& confidence, int& label, std::string& log) {
    if (!fa_file_exists(model_path)) {
        log = "Modello non trovato.";
        return false;
    }

    cv::Ptr<cv::face::FaceRecognizer> model = cv::face::LBPHFaceRecognizer::create();
    model->read(model_path);

    cv::VideoCapture cap(0);
    cv::Mat frame, gray;
    cap >> frame;
    if (frame.empty()) return false;

    cv::cvtColor(frame, gray, cv::COLOR_BGR2GRAY);
    model->predict(gray, label, confidence);
    return true;
}
