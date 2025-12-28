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

// Dichiarazioni esterne per i plugin (saranno linkati da Ninja/CMake)
// Questo risolve l'errore di file non trovato senza usare #include ".cpp"
std::unique_ptr<RecognizerPlugin> create_classic_plugin(const std::string& method, const FacialAuthConfig& cfg);
std::unique_ptr<RecognizerPlugin> create_sface_plugin(const FacialAuthConfig& cfg);

// --- Utility Functions ---

bool fa_check_root(const std::string& tool_name) {
    if (getuid() != 0) {
        std::cerr << "ERRORE [" << tool_name << "]: Devi eseguire come root." << std::endl;
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

bool fa_load_config(FacialAuthConfig& cfg, std::string& log, const std::string& path) {
    std::ifstream file(path);
    if (!file.is_open()) {
        log = "Configurazione non trovata (" + path + "), uso i default.";
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
        else if (key == "method") cfg.method = val;
        else if (key == "threshold") cfg.threshold = std::stod(val);
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
    if (fs::exists(user_dir)) {
        fs::remove_all(user_dir);
        log = "Catture rimosse per l'utente: " + user;
    }
    return true;
}

// --- Factory for Plugins ---

std::unique_ptr<RecognizerPlugin> create_plugin(const FacialAuthConfig& cfg) {
    if (cfg.method == "sface") {
        return create_sface_plugin(cfg);
    } else {
        return create_classic_plugin(cfg.method, cfg);
    }
}

// --- Core Logic ---

bool fa_capture_user(const std::string& user, const FacialAuthConfig& cfg, const std::string& device_path, std::string& log) {
    cv::VideoCapture cap(device_path);
    if (!cap.isOpened()) {
        log = "Impossibile aprire il device: " + device_path;
        return false;
    }

    cv::CascadeClassifier face_cascade;
    if (!face_cascade.load(cfg.cascade_path)) {
        log = "Errore caricamento cascade: " + cfg.cascade_path;
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
            cv::Mat face_roi = frame(area);
            cv::resize(face_roi, face_roi, cv::Size(cfg.width, cfg.height));
            cv::imwrite(user_dir + "/" + std::to_string(count) + ".jpg", face_roi);
            count++;
            if (cfg.verbose) std::cout << "\r[*] Frame acquisiti: " << count << "/" << cfg.frames << std::flush;
            if (!cfg.nogui) {
                cv::rectangle(frame, area, cv::Scalar(0, 255, 0), 2);
                cv::imshow("Capture", frame);
                if (cv::waitKey(1) == 27) return false;
            }
            if (count >= cfg.frames) break;
        }
    }
    return true;
}

bool fa_train_user(const std::string& user, const FacialAuthConfig& cfg, std::string& log) {
    std::string user_dir = cfg.basedir + "/captures/" + user;
    std::vector<cv::Mat> faces;
    std::vector<int> labels;
    for (const auto& entry : fs::directory_iterator(user_dir)) {
        cv::Mat img = cv::imread(entry.path().string());
        if (!img.empty()) { faces.push_back(img); labels.push_back(0); }
    }
    if (faces.empty()) { log = "Nessuna immagine trovata."; return false; }

    auto plugin = create_plugin(cfg);
    return plugin->train(faces, labels, fa_user_model_path(cfg, user));
}

bool fa_test_user(const std::string& user, const FacialAuthConfig& cfg, const std::string& model_path, double& confidence, int& label, std::string& log) {
    auto plugin = create_plugin(cfg);
    if (!plugin->load(model_path)) return false;

    cv::VideoCapture cap(0);
    cv::Mat frame;
    cap >> frame;
    if (frame.empty()) return false;
    return plugin->predict(frame, label, confidence);
}
