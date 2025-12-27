#include "../include/libfacialauth.h"
#include <opencv2/highgui.hpp>
#include <opencv2/imgproc.hpp>
#include <opencv2/objdetect.hpp>
#include <opencv2/videoio.hpp>
#include <opencv2/face.hpp>
#include <fstream>
#include <filesystem>
#include <iostream>
#include <thread>

namespace fs = std::filesystem;

// Helper interno per il rilevamento
bool internal_detect(cv::Mat& frame, const std::string& method, cv::Ptr<cv::CascadeClassifier>& haar, cv::Ptr<cv::FaceDetectorYN>& yunet) {
    if (method == "haar" && !haar->empty()) {
        std::vector<cv::Rect> faces;
        cv::Mat gray; cv::cvtColor(frame, gray, cv::COLOR_BGR2GRAY);
        haar->detectMultiScale(gray, faces, 1.1, 3, 0, cv::Size(30, 30));
        return !faces.empty();
    } else if (method == "yunet" && yunet) {
        cv::Mat faces; yunet->setInputSize(frame.size());
        yunet->detect(frame, faces);
        return faces.rows > 0;
    }
    return true;
}

bool fa_load_config(FacialAuthConfig &cfg, std::string &log, const std::string &path) {
    std::ifstream file(path);
    if (!file.is_open()) return false;
    std::string line;
    while (std::getline(file, line)) {
        if (line.empty() || line[0] == '#') continue;
        std::istringstream is_line(line);
        std::string key, value;
        if (std::getline(is_line, key, '=') && std::getline(is_line, value)) {
            key.erase(key.find_last_not_of(" \t\r\n") + 1);
            value.erase(0, value.find_first_not_of(" \t\r\n"));
            if (key == "basedir") cfg.basedir = value;
            else if (key == "device") cfg.device = value;
            else if (key == "detect_model_path") cfg.detect_model_path = value;
            else if (key == "frames") cfg.frames = std::stoi(value);
        }
    }
    return true;
}

bool fa_capture_user(const std::string &user, const FacialAuthConfig &cfg, const std::string &detector_type, std::string &log) {
    std::string user_dir = cfg.basedir + "/" + user + "/captures";
    if (cfg.force) fs::remove_all(user_dir);
    fs::create_directories(user_dir);

    cv::Ptr<cv::CascadeClassifier> haar;
    cv::Ptr<cv::FaceDetectorYN> yunet;
    if (detector_type == "haar") haar = cv::makePtr<cv::CascadeClassifier>("/usr/share/opencv4/haarcascades/haarcascade_frontalface_default.xml");
    else if (detector_type == "yunet") yunet = cv::FaceDetectorYN::create(cfg.detect_model_path, "", cv::Size(320, 320));

    cv::VideoCapture cap;
    try { cap.open(std::stoi(cfg.device)); } catch(...) { cap.open(cfg.device); }
    if (!cap.isOpened()) { log = "Impossibile aprire camera"; return false; }

    int count = 0;
    while (count < cfg.frames) {
        cv::Mat frame; cap >> frame;
        if (frame.empty()) break;
        if (internal_detect(frame, detector_type, haar, yunet)) {
            cv::imwrite(user_dir + "/img_" + std::to_string(count++) + "." + cfg.image_format, frame);
        }
        if (!cfg.nogui) {
            cv::imshow("Capture", frame);
            if (cv::waitKey(1) == 'q') break;
        }
        if (cfg.sleep_ms > 0) std::this_thread::sleep_for(std::chrono::milliseconds(cfg.sleep_ms));
    }
    return true;
}

bool fa_test_user_interactive(const std::string &user, const FacialAuthConfig &cfg, std::string &log) {
    double conf = 0; int label = -1;
    bool res = fa_test_user(user, cfg, fa_user_model_path(cfg, user), conf, label, log);
    std::cout << "User: " << user << " | Success: " << (res ? "YES" : "NO") << " | Conf: " << conf << "\n";
    return res;
}

bool fa_check_root(const std::string &t) {
    if (getuid() != 0) { std::cerr << t << " richiede permessi root.\n"; return false; }
    return true;
}

std::string fa_user_model_path(const FacialAuthConfig &cfg, const std::string &user) {
    return cfg.basedir + "/" + user + "/model.xml";
}

bool fa_file_exists(const std::string &path) { return fs::exists(path); }
