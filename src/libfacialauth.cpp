#include "../include/libfacialauth.h"
#include <iostream>
#include <fstream>
#include <filesystem>
#include <thread>
#include <chrono>
#include <unistd.h>

namespace fs = std::filesystem;

// Verify root privileges for system tools
bool fa_check_root(const std::string& tool_name) {
    if (geteuid() != 0) {
        std::cerr << "[ERROR] " << tool_name << " must be run as root.\n";
        return false;
    }
    return true;
}

// Helper to check file existence
bool fa_file_exists(const std::string& path) {
    if (path.empty()) return false;
    return fs::exists(path);
}

// Load configuration from file or apply dynamic defaults
bool fa_load_config(FacialAuthConfig& cfg, std::string& log, const std::string& path) {
    // Dynamic fallback values if config is missing or incomplete
    if (cfg.basedir.empty()) cfg.basedir = "/var/lib/pam_facial_auth";
    if (cfg.cascade_path.empty()) cfg.cascade_path = "/usr/share/opencv4/haarcascades/haarcascade_frontalface_default.xml";

    if (!fs::exists(path)) {
        log = "Info: " + path + " not found. Using system defaults.";
        return true;
    }

    // Config parser logic will be implemented here to override cfg members
    log = "Configuration loaded from " + path;
    return true;
}

// Construct user model path based on loaded basedir
std::string fa_user_model_path(const FacialAuthConfig& cfg, const std::string& user) {
    return (fs::path(cfg.basedir) / "models" / (user + ".xml")).string();
}

// Capture face dataset for a specific user
bool fa_capture_user(const std::string& user, const FacialAuthConfig& cfg, const std::string& detector, std::string& log) {
    cv::VideoCapture cap(0);
    if (!cap.isOpened()) {
        log = "Error: Camera not accessible.";
        return false;
    }

    // Dynamic data path derived from basedir
    fs::path user_dir = fs::path(cfg.basedir) / "data" / user;
    try {
        fs::create_directories(user_dir);
    } catch (const fs::filesystem_error& e) {
        log = "Filesystem error: " + std::string(e.what());
        return false;
    }

    cv::CascadeClassifier face_cascade(cfg.cascade_path);
    if (face_cascade.empty()) {
        log = "Error: Haar classifier not found at " + cfg.cascade_path;
        return false;
    }

    int saved = 0;
    cv::Mat frame, gray;
    while (saved < cfg.frames) {
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
        }
        if (cfg.sleep_ms > 0) std::this_thread::sleep_for(std::chrono::milliseconds(cfg.sleep_ms));
    }
    return true;
}

// Train the LBPH model for a user
bool fa_train_user(const std::string& user, const FacialAuthConfig& cfg, std::string& log) {
    fs::path user_dir = fs::path(cfg.basedir) / "data" / user;
    fs::path model_dir = fs::path(cfg.basedir) / "models";
    std::string model_full_path = fa_user_model_path(cfg, user);

    if (!fs::exists(user_dir)) {
        log = "Error: Data directory not found: " + user_dir.string();
        return false;
    }

    std::vector<cv::Mat> images;
    std::vector<int> labels;
    for (const auto& entry : fs::directory_iterator(user_dir)) {
        cv::Mat img = cv::imread(entry.path().string(), cv::IMREAD_GRAYSCALE);
        if (!img.empty()) {
            images.push_back(img);
            labels.push_back(0); // All images belong to the same user
        }
    }

    if (images.empty()) {
        log = "Error: No data available for training.";
        return false;
    }

    cv::Ptr<cv::face::FaceRecognizer> model = cv::face::LBPHFaceRecognizer::create();
    model->train(images, labels);

    fs::create_directories(model_dir);
    model->save(model_full_path);
    log = "Model saved to " + model_full_path;
    return true;
}

// Test face recognition against a trained model
bool fa_test_user(const std::string& user, const FacialAuthConfig& cfg, const std::string& model_path, double& confidence, int& label, std::string& log) {
    if (!fa_file_exists(model_path)) {
        log = "Error: Model does not exist at " + model_path;
        return false;
    }

    cv::Ptr<cv::face::FaceRecognizer> model = cv::face::LBPHFaceRecognizer::create();
    model->read(model_path);

    cv::VideoCapture cap(0);
    cv::Mat frame, gray;
    cap >> frame;
    if (frame.empty()) {
        log = "Error: Camera failure during test.";
        return false;
    }

    cv::cvtColor(frame, gray, cv::COLOR_BGR2GRAY);
    model->predict(gray, label, confidence);
    return true;
}
