#include "../include/libfacialauth.h"
#include <opencv2/highgui.hpp>
#include <opencv2/imgproc.hpp>
#include <opencv2/objdetect.hpp>
#include <opencv2/videoio.hpp>
#include <opencv2/face.hpp>
#include <opencv2/imgcodecs.hpp>
#include <fstream>
#include <filesystem>
#include <iostream>
#include <thread>
#include <unistd.h>

namespace fs = std::filesystem;

bool fa_load_config(FacialAuthConfig &cfg, std::string &log, const std::string &path) {
    std::ifstream file(path);
    if (!file.is_open()) { log = "Config not found"; return false; }
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
    cv::VideoCapture cap;
    try { cap.open(std::stoi(cfg.device)); } catch(...) { cap.open(cfg.device); }
    if (!cap.isOpened()) { log = "Camera error"; return false; }
    int count = 0;
    while (count < cfg.frames) {
        cv::Mat frame; cap >> frame;
        if (frame.empty()) break;
        cv::imwrite(user_dir + "/img_" + std::to_string(count++) + "." + cfg.image_format, frame);
        if (!cfg.nogui) {
            cv::imshow("Capture", frame);
            if (cv::waitKey(1) == 'q') break;
        }
    }
    return true;
}

// IMPLEMENTAZIONE MANCANTE CHE CAUSAVA L'ERRORE
bool fa_train_user(const std::string &user, const FacialAuthConfig &cfg, std::string &log) {
    std::string user_dir = cfg.basedir + "/" + user + "/captures";
    std::vector<cv::Mat> faces;
    std::vector<int> labels;

    for (const auto& entry : fs::directory_iterator(user_dir)) {
        cv::Mat img = cv::imread(entry.path().string(), cv::IMREAD_GRAYSCALE);
        if (!img.empty()) {
            faces.push_back(img);
            labels.push_back(0); // Label fissa per singolo utente
        }
    }

    if (faces.empty()) { log = "No images found in " + user_dir; return false; }

    cv::Ptr<cv::face::LBPHFaceRecognizer> model = cv::face::LBPHFaceRecognizer::create();
    model->train(faces, labels);
    model->save(fa_user_model_path(cfg, user));
    return true;
}

bool fa_test_user(const std::string &user, const FacialAuthConfig &cfg, const std::string &modelPath,
                  double &best_conf, int &best_label, std::string &log) {
    if (!fs::exists(modelPath)) { log = "Model missing"; return false; }
    cv::Ptr<cv::face::LBPHFaceRecognizer> model = cv::face::LBPHFaceRecognizer::create();
    model->read(modelPath);
    cv::VideoCapture cap;
    try { cap.open(std::stoi(cfg.device)); } catch(...) { cap.open(cfg.device); }
    if (!cap.isOpened()) return false;
    cv::Mat frame, gray; cap >> frame;
    if (frame.empty()) return false;
    cv::cvtColor(frame, gray, cv::COLOR_BGR2GRAY);
    model->predict(gray, best_label, best_conf);
    return (best_conf < cfg.lbph_threshold);
                  }

                  bool fa_test_user_interactive(const std::string &user, const FacialAuthConfig &cfg, std::string &log) {
                      double conf = 0; int label = -1;
                      bool res = fa_test_user(user, cfg, fa_user_model_path(cfg, user), conf, label, log);
                      std::cout << "Test for " << user << ": " << (res ? "OK" : "FAILED") << " (Conf: " << conf << ")\n";
                      return res;
                  }

                  std::string fa_user_model_path(const FacialAuthConfig &cfg, const std::string &user) {
                      return cfg.basedir + "/" + user + "/model.xml";
                  }

                  bool fa_check_root(const std::string &t) {
                      if (getuid() != 0) { std::cerr << t << " requires root.\n"; return false; }
                      return true;
                  }

                  bool fa_file_exists(const std::string &path) { return fs::exists(path); }
