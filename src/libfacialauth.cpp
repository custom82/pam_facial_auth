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
#include <sstream>
#include <unistd.h>
#include <iomanip>
#include <regex>

namespace fs = std::filesystem;

extern "C" std::unique_ptr<RecognizerPlugin> create_classic_plugin(const std::string& method, const FacialAuthConfig& cfg);
extern "C" std::unique_ptr<RecognizerPlugin> create_sface_plugin(const FacialAuthConfig& cfg);

std::unique_ptr<RecognizerPlugin> create_plugin(const FacialAuthConfig& cfg) {
    if (cfg.method == "sface" || cfg.method == "auto") return create_sface_plugin(cfg);
    return create_classic_plugin(cfg.method, cfg);
}

int get_last_index(const std::string& dir) {
    int max_idx = -1;
    if (!fs::exists(dir)) return max_idx;
    std::regex re("frame_(\\d+)\\.\\w+");
    for (const auto& entry : fs::directory_iterator(dir)) {
        std::string filename = entry.path().filename().string();
        std::smatch match;
        if (std::regex_match(filename, match, re)) {
            try {
                int idx = std::stoi(match[1].str());
                if (idx > max_idx) max_idx = idx;
            } catch (...) {}
        }
    }
    return max_idx;
}

extern "C" {

    FA_EXPORT bool fa_check_root(const std::string& tool_name) {
        if (getuid() != 0) {
            std::cerr << "ERRORE [" << tool_name << "]: Devi essere root." << std::endl;
            return false;
        }
        return true;
    }

    FA_EXPORT bool fa_load_config(FacialAuthConfig& cfg, std::string& log, const std::string& path) {
        std::ifstream file(path);
        if (!file.is_open()) { log = "File non trovato: " + path; return false; }
        std::string line;
        while (std::getline(file, line)) {
            if (line.empty() || line[0] == '#') continue;
            std::istringstream is_line(line);
            std::string key, value;
            if (std::getline(is_line, key, '=') && std::getline(is_line, value)) {
                key.erase(key.find_last_not_of(" \t") + 1);
                value.erase(0, value.find_first_not_of(" \t"));
                if (key == "basedir") cfg.basedir = value;
                else if (key == "device") cfg.device = value;
                else if (key == "detect_yunet") cfg.detect_yunet = value;
                else if (key == "training_method") cfg.method = value;
                else if (key == "detector") cfg.detector = value;
                else if (key == "image_format") cfg.image_format = value;
                else if (key == "frames") cfg.frames = std::stoi(value);
                else if (key == "width") cfg.width = std::stoi(value);
                else if (key == "height") cfg.height = std::stoi(value);
                else if (key == "sleep_ms") { cfg.sleep_ms = std::stoi(value); cfg.capture_delay = (double)cfg.sleep_ms / 1000.0; }
                else if (key == "debug") cfg.debug = (value == "yes");
                else if (key == "verbose") cfg.verbose = (value == "yes");
            }
        }
        return true;
    }

    FA_EXPORT bool fa_clean_captures(const std::string& user, const FacialAuthConfig& cfg, std::string& log) {
        std::string user_dir = cfg.basedir + "/captures/" + user;
        if (fs::exists(user_dir)) fs::remove_all(user_dir);
        log = "Cartella pulita.";
        return true;
    }

    FA_EXPORT bool fa_capture_user(const std::string& user, const FacialAuthConfig& cfg, const std::string& device_path, std::string& log) {
        cv::VideoCapture cap(device_path);
        if (!cap.isOpened()) { log = "Webcam non disponibile: " + device_path; return false; }

        cv::Ptr<cv::FaceDetectorYN> detector;
        if (cfg.detector == "yunet") {
            if (cfg.verbose) std::cout << "[INFO] Caricamento modello YuNet: " << cfg.detect_yunet << std::endl;
            detector = cv::FaceDetectorYN::create(cfg.detect_yunet, "", cv::Size(320, 320));
            if (detector.empty()) { log = "Errore init YuNet!"; return false; }
        }

        std::string user_dir = cfg.basedir + "/captures/" + user;
        fs::create_directories(user_dir);
        int start_idx = get_last_index(user_dir) + 1;

        std::cout << "[START] Acquisizione per '" << user << "' (Richiesti: " << cfg.frames << " frame)" << std::endl;

        int count = 0;
        int dropped = 0;
        while (count < cfg.frames) {
            cv::Mat frame; cap >> frame;
            if (frame.empty()) continue;

            bool face_found = true;
            if (cfg.detector == "yunet") {
                detector->setInputSize(frame.size());
                cv::Mat faces;
                detector->detect(frame, faces);
                face_found = (faces.rows > 0);
            }

            if (face_found) {
                cv::Mat resized;
                cv::resize(frame, resized, cv::Size(cfg.width, cfg.height));
                std::string path = user_dir + "/frame_" + std::to_string(start_idx + count) + "." + cfg.image_format;
                if (cv::imwrite(path, resized)) {
                    count++;
                }
            } else {
                dropped++;
            }

            // OUTPUT VERBOSO IN TEMPO REALE
            if (cfg.verbose || cfg.debug) {
                std::cout << "\r[CATTURA] Salvati: " << count << "/" << cfg.frames
                << " | Scartati (No Volto): " << dropped << std::flush;
            }

            if (cfg.sleep_ms > 0) std::this_thread::sleep_for(std::chrono::milliseconds(cfg.sleep_ms));
        }
        std::cout << "\n[FINISH] Salvate " << count << " immagini in " << user_dir << std::endl;
        return true;
    }

    FA_EXPORT std::string fa_user_model_path(const FacialAuthConfig& cfg, const std::string& user) {
        std::string ext = (cfg.method == "sface" || cfg.method == "auto") ? ".yml" : ".xml";
        return cfg.basedir + "/models/" + user + ext;
    }

    FA_EXPORT bool fa_train_user(const std::string& user, const FacialAuthConfig& cfg, std::string& log) {
        std::string user_dir = cfg.basedir + "/captures/" + user;
        std::vector<cv::Mat> faces; std::vector<int> labels;
        if (!fs::exists(user_dir)) return false;
        for (const auto& entry : fs::directory_iterator(user_dir)) {
            cv::Mat img = cv::imread(entry.path().string(), cv::IMREAD_GRAYSCALE);
            if (!img.empty()) { faces.push_back(img); labels.push_back(0); }
        }
        fs::create_directories(cfg.basedir + "/models");
        auto plugin = create_plugin(cfg);
        return plugin->train(faces, labels, fa_user_model_path(cfg, user));
    }

    FA_EXPORT bool fa_test_user(const std::string& user, const FacialAuthConfig& cfg, const std::string& model_path, double& confidence, int& label, std::string& log) {
        auto plugin = create_plugin(cfg);
        if (!plugin->load(model_path)) return false;
        cv::VideoCapture cap(cfg.device);
        cv::Mat frame; cap >> frame;
        return plugin->predict(frame, label, confidence);
    }

} // extern "C"
