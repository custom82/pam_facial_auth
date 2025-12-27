/*
 * Project: pam_facial_auth
 * License: GPL-3.0
 */

#include "libfacialauth.h"
#include <iostream>
#include <fstream>
#include <sstream>
#include <filesystem>
#include <algorithm>

namespace fs = std::filesystem;

/**
 * Helper per rimuovere spazi bianchi all'inizio e alla fine di una stringa
 */
static std::string trim(const std::string& s) {
    auto start = s.begin();
    while (start != s.end() && std::isspace(*start)) start++;
    auto end = s.end();
    do { end--; } while (std::distance(start, end) > 0 && std::isspace(*end));
    return std::string(start, end + 1);
}

bool fa_file_exists(const std::string& path) {
    return fs::exists(path);
}

bool fa_load_config(FacialAuthConfig& cfg, std::string& log, const std::string& path) {
    std::ifstream file(path);
    if (!file.is_open()) {
        log = "Could not open config file: " + path;
        return false;
    }

    std::string line;
    while (std::getline(file, line)) {
        line = trim(line);
        if (line.empty() || line[0] == '#') continue;

        std::size_t sep = line.find('=');
        if (sep == std::string::npos) continue;

        std::string key = trim(line.substr(0, sep));
        std::string val = trim(line.substr(sep + 1));

        if (key == "basedir") cfg.basedir = val;
        else if (key == "cascade_path") cfg.cascade_path = val;
        else if (key == "threshold") cfg.threshold = std::stod(val);
        else if (key == "frames") cfg.frames = std::stoi(val);
        else if (key == "width") cfg.width = std::stoi(val);
        else if (key == "height") cfg.height = std::stoi(val);
        else if (key == "debug") cfg.debug = (val == "true" || val == "1");
        else if (key == "verbose") cfg.verbose = (val == "true" || val == "1");
    }
    return true;
}

std::string fa_user_model_path(const FacialAuthConfig& cfg, const std::string& user) {
    return cfg.basedir + "/" + user + ".xml";
}

bool fa_capture_user(const std::string& user, const FacialAuthConfig& cfg, const std::string& detector, std::string& log) {
    #ifdef HAVE_OPENCV
    cv::VideoCapture cap(0);
    if (!cap.isOpened()) {
        log = "Errore: Impossibile aprire la fotocamera.";
        return false;
    }

    cv::CascadeClassifier face_cascade(cfg.cascade_path);
    if (face_cascade.empty()) {
        log = "Errore: Impossibile caricare il file cascade: " + cfg.cascade_path;
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
            std::string filename = user_dir + "/" + std::to_string(count) + ".jpg";
            cv::imwrite(filename, face_roi);
            count++;
            if (count >= cfg.frames) break;
        }
    }
    return true;
    #else
    log = "OpenCV support not enabled at compile time.";
    return false;
    #endif
}

bool fa_train_user(const std::string& user, const FacialAuthConfig& cfg, std::string& log) {
    #ifdef HAVE_OPENCV
    std::string user_dir = cfg.basedir + "/captures/" + user;
    std::vector<cv::Mat> images;
    std::vector<int> labels;

    for (const auto& entry : fs::directory_iterator(user_dir)) {
        cv::Mat img = cv::imread(entry.path().string(), cv::IMREAD_GRAYSCALE);
        if (!img.empty()) {
            images.push_back(img);
            labels.push_back(0); // Singolo utente, label fissa
        }
    }

    if (images.empty()) {
        log = "Nessuna immagine trovata per il training in: " + user_dir;
        return false;
    }

    auto model = cv::face::LBPHFaceRecognizer::create();
    model->train(images, labels);
    model->write(fa_user_model_path(cfg, user));
    return true;
    #else
    log = "OpenCV support not enabled.";
    return false;
    #endif
}

bool fa_test_user(const std::string& user, const FacialAuthConfig& cfg, const std::string& model_path, double& confidence, int& label, std::string& log) {
    #ifdef HAVE_OPENCV
    if (!fa_file_exists(model_path)) {
        log = "Modello non trovato per l'utente: " + user;
        return false;
    }

    auto model = cv::face::LBPHFaceRecognizer::create();
    model->read(model_path);

    cv::VideoCapture cap(0);
    cv::Mat frame, gray;
    cap >> frame;
    if (frame.empty()) {
        log = "Errore cattura fotocamera.";
        return false;
    }

    cv::cvtColor(frame, gray, cv::COLOR_BGR2GRAY);
    // Nota: Qui potresti voler aggiungere la detezione del volto prima del predict
    model->predict(gray, label, confidence);

    return true;
    #else
    log = "OpenCV support not enabled.";
    return false;
    #endif
}
