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
#include <unistd.h> // PER GETUID()

namespace fs = std::filesystem;

// Helper per il caricamento configurazione
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

// Logica di cattura (centralizzata)
bool fa_capture_user(const std::string &user, const FacialAuthConfig &cfg, const std::string &detector_type, std::string &log) {
    std::string user_dir = cfg.basedir + "/" + user + "/captures";
    if (cfg.force) fs::remove_all(user_dir);
    fs::create_directories(user_dir);

    cv::VideoCapture cap;
    try { cap.open(std::stoi(cfg.device)); } catch(...) { cap.open(cfg.device); }
    if (!cap.isOpened()) { log = "Camera non accessibile"; return false; }

    int count = 0;
    while (count < cfg.frames) {
        cv::Mat frame; cap >> frame;
        if (frame.empty()) break;
        // Salva sempre o aggiungi logica detector qui
        cv::imwrite(user_dir + "/img_" + std::to_string(count++) + "." + cfg.image_format, frame);
        if (!cfg.nogui) {
            cv::imshow("Cattura", frame);
            if (cv::waitKey(1) == 'q') break;
        }
    }
    return true;
}

// Logica di test (silenziosa per PAM)
bool fa_test_user(const std::string &user, const FacialAuthConfig &cfg, const std::string &modelPath,
                  double &best_conf, int &best_label, std::string &log) {
    if (!fs::exists(modelPath)) { log = "Modello non trovato"; return false; }

    cv::Ptr<cv::face::LBPHFaceRecognizer> model = cv::face::LBPHFaceRecognizer::create();
    model->read(modelPath);

    cv::VideoCapture cap;
    try { cap.open(std::stoi(cfg.device)); } catch(...) { cap.open(cfg.device); }
    if (!cap.isOpened()) { log = "Camera offline"; return false; }

    cv::Mat frame; cap >> frame;
    if (frame.empty()) return false;

    cv::Mat gray; cv::cvtColor(frame, gray, cv::COLOR_BGR2GRAY);
    model->predict(gray, best_label, best_conf);

    return (best_conf < cfg.lbph_threshold);
                  }

                  // Logica di test (interattiva per CLI)
                  bool fa_test_user_interactive(const std::string &user, const FacialAuthConfig &cfg, std::string &log) {
                      double conf = 0; int label = -1;
                      bool res = fa_test_user(user, cfg, fa_user_model_path(cfg, user), conf, label, log);
                      std::cout << "Utente: " << user << " | Riconosciuto: " << (res ? "SI" : "NO") << " | Confidenza: " << conf << "\n";
                      return res;
                  }

                  // Implementazione fa_train_user omessa per brevità ma necessaria nel file...

                  std::string fa_user_model_path(const FacialAuthConfig &cfg, const std::string &user) {
                      return cfg.basedir + "/" + user + "/model.xml";
                  }

                  bool fa_check_root(const std::string &t) {
                      if (getuid() != 0) { std::cerr << t << " richiede root.\n"; return false; }
                      return true;
                  }

                  bool fa_file_exists(const std::string &path) { return fs::exists(path); }
