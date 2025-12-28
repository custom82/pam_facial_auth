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
        std::smatch match;
        std::string fname = entry.path().filename().string();
        if (std::regex_match(fname, match, re)) {
            max_idx = std::max(max_idx, std::stoi(match[1].str()));
        }
    }
    return max_idx;
}

extern "C" {

    FA_EXPORT bool fa_check_root(const std::string& tool_name) {
        if (getuid() != 0) {
            std::cerr << "ERRORE [" << tool_name << "]: Eseguire come root." << std::endl;
            return false;
        }
        return true;
    }

    FA_EXPORT bool fa_load_config(FacialAuthConfig& cfg, std::string& log, const std::string& path) {
        std::ifstream file(path);
        if (!file.is_open()) { log = "Config non trovata: " + path; return false; }
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
                else if (key == "detector") cfg.detector = value;
                else if (key == "image_format") cfg.image_format = value;
                else if (key == "frames") cfg.frames = std::stoi(value);
                else if (key == "width") cfg.width = std::stoi(value);
                else if (key == "height") cfg.height = std::stoi(value);
                else if (key == "sleep_ms") cfg.sleep_ms = std::stoi(value);
            }
        }
        return true;
    }

    FA_EXPORT bool fa_capture_user(const std::string& user, const FacialAuthConfig& cfg, const std::string& device_path, std::string& log) {
        cv::VideoCapture cap(device_path);
        if (!cap.isOpened()) { log = "Webcam non accessibile"; return false; }

        cv::Ptr<cv::FaceDetectorYN> detector;
        if (cfg.detector == "yunet") {
            detector = cv::FaceDetectorYN::create(cfg.detect_yunet, "", cv::Size(320, 320));
            if (detector.empty()) { log = "Errore: Modello YuNet non trovato in " + cfg.detect_yunet; return false; }
        }

        std::string user_dir = cfg.basedir + "/captures/" + user;
        fs::create_directories(user_dir);
        int start_idx = get_last_index(user_dir) + 1;

        std::cout << "[INFO] Inizio acquisizione per: " << user << std::endl;
        if (cfg.detector == "yunet") std::cout << "[INFO] Validazione YuNet ATTIVA. Mostra il volto alla camera." << std::endl;

        int saved = 0;
        int dropped = 0;

        while (saved < cfg.frames) {
            cv::Mat frame;
            cap >> frame;
            if (frame.empty()) continue;

            bool face_valid = false; // DI DEFAULT NON SALVARE

            if (detector) {
                detector->setInputSize(frame.size());
                cv::Mat faces;
                detector->detect(frame, faces);
                face_valid = (faces.rows > 0);
            } else {
                // Se non c'è detector, salva tutto (comportamento legacy)
                face_valid = true;
            }

            if (face_valid) {
                cv::Mat resized;
                cv::resize(frame, resized, cv::Size(cfg.width, cfg.height));
                std::string path = user_dir + "/frame_" + std::to_string(start_idx + saved) + "." + cfg.image_format;
                if (cv::imwrite(path, resized)) {
                    saved++;
                }
            } else {
                dropped++;
            }

            // Forza l'output real-time nel terminale
            std::cout << "\r[STATO] Acquisiti: " << saved << "/" << cfg.frames
            << " | Scartati: " << dropped
            << (face_valid ? " [VOLTO TROVATO]  " : " [NESSUN VOLTO]   ") << std::flush;

            if (cfg.sleep_ms > 0) std::this_thread::sleep_for(std::chrono::milliseconds(cfg.sleep_ms));
        }

        std::cout << "\n[SUCCESS] Sessione completata." << std::endl;
        return true;
    }

    FA_EXPORT bool fa_clean_captures(const std::string& user, const FacialAuthConfig& cfg, std::string& log) {
        std::string user_dir = cfg.basedir + "/captures/" + user;
        if (fs::exists(user_dir)) fs::remove_all(user_dir);
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
