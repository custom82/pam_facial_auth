/*
 * Project: pam_facial_auth
 * License: GPL-3.0
 */

#include "libfacialauth.h"
#include <iostream>
#include <fstream>
#include <filesystem>
#include <algorithm>
#include <unistd.h>

namespace fs = std::filesystem;

bool fa_check_root(const std::string& tool_name) {
    if (getuid() != 0) {
        std::cerr << "ERRORE [" << tool_name << "]: Devi eseguire questo comando come root (sudo)." << std::endl;
        return false;
    }
    return true;
}

static std::string trim(const std::string& s) {
    auto start = s.begin();
    while (start != s.end() && std::isspace(*start)) start++;
    auto end = s.end();
    if (start == s.end()) return "";
    do { end--; } while (std::distance(start, end) > 0 && std::isspace(*end));
    return std::string(start, end + 1);
}

bool fa_file_exists(const std::string& path) {
    return fs::exists(path);
}

bool fa_load_config(FacialAuthConfig& cfg, std::string& log, const std::string& path) {
    std::ifstream file(path);
    if (!file.is_open()) {
        log = "Configurazione non trovata, uso i default.";
        return true;
    }
    std::string line;
    while (std::getline(file, line)) {
        line = trim(line);
        if (line.empty() || line[0] == '#') continue;
        auto sep = line.find('=');
        if (sep == std::string::npos) continue;
        std::string key = trim(line.substr(0, sep));
        std::string val = trim(line.substr(sep + 1));
        if (key == "basedir") cfg.basedir = val;
        else if (key == "cascade_path") cfg.cascade_path = val;
        else if (key == "threshold") cfg.threshold = std::stod(val);
        else if (key == "frames") cfg.frames = std::stoi(val);
        else if (key == "width") cfg.width = std::stoi(val);
        else if (key == "height") cfg.height = std::stoi(val);
    }
    return true;
}

std::string fa_user_model_path(const FacialAuthConfig& cfg, const std::string& user) {
    return cfg.basedir + "/" + user + ".xml";
}

bool fa_clean_captures(const std::string& user, const FacialAuthConfig& cfg, std::string& log) {
    std::string user_dir = cfg.basedir + "/captures/" + user;
    try {
        if (fs::exists(user_dir)) {
            fs::remove_all(user_dir);
            log = "Pulizia completata per l'utente: " + user;
        } else {
            log = "Nessun dato da pulire per l'utente: " + user;
        }
        return true;
    } catch (const fs::filesystem_error& e) {
        log = "Errore filesystem: " + std::string(e.what());
        return false;
    }
}

bool fa_capture_user(const std::string& user, const FacialAuthConfig& cfg, const std::string& detector, std::string& log) {
    #ifdef HAVE_OPENCV
    cv::VideoCapture cap(0);
    cv::CascadeClassifier face_cascade(cfg.cascade_path);
    if (!cap.isOpened() || face_cascade.empty()) {
        log = "Errore: Webcam o file cascade non accessibili.";
        return false;
    }
    std::string user_dir = cfg.basedir + "/captures/" + user;
    fs::create_directories(user_dir);
    int count = 0;
    while (count < cfg.frames) {
        cv::Mat frame, gray;
        cap >> frame;
        if (frame.empty()) break;
        cv::cvtColor(frame, gray, cv::COLOR_BGR2GRAY);
        std::vector<cv::Rect> faces;
        face_cascade.detectMultiScale(gray, faces, 1.1, 4);
        for (const auto& area : faces) {
            cv::Mat face_roi = gray(area);
            cv::resize(face_roi, face_roi, cv::Size(cfg.width, cfg.height));
            cv::imwrite(user_dir + "/" + std::to_string(count) + ".jpg", face_roi);
            count++;
            std::cout << "\rCattura frame: " << count << "/" << cfg.frames << std::flush;
            if (count >= cfg.frames) break;
        }
    }
    std::cout << std::endl;
    return true;
    #else
    log = "Errore: OpenCV non abilitato."; return false;
    #endif
}

bool fa_train_user(const std::string& user, const FacialAuthConfig& cfg, std::string& log) {
    #ifdef HAVE_OPENCV
    std::string user_dir = cfg.basedir + "/captures/" + user;
    if (!fs::exists(user_dir)) {
        log = "Nessuna cattura trovata. Esegui prima la cattura.";
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
        log = "Immagini non valide."; return false;
    }
    auto model = cv::face::LBPHFaceRecognizer::create();
    model->train(images, labels);
    model->write(fa_user_model_path(cfg, user));
    log = "Training completato per " + user;
    return true;
    #else
    log = "Errore: OpenCV non abilitato."; return false;
    #endif
}

bool fa_test_user(const std::string& user, const FacialAuthConfig& cfg, const std::string& model_path, double& confidence, int& label, std::string& log) {
    #ifdef HAVE_OPENCV
    if (!fs::exists(model_path)) {
        log = "Modello mancante: " + model_path; return false;
    }
    auto model = cv::face::LBPHFaceRecognizer::create();
    model->read(model_path);
    cv::VideoCapture cap(0);
    if (!cap.isOpened()) {
        log = "Webcam non trovata."; return false;
    }
    cv::Mat frame, gray;
    cap >> frame;
    if (frame.empty()) {
        log = "Impossibile leggere dalla webcam."; return false;
    }
    cv::cvtColor(frame, gray, cv::COLOR_BGR2GRAY);
    model->predict(gray, label, confidence);
    return true;
    #else
    log = "Errore: OpenCV non abilitato."; return false;
    #endif
}
