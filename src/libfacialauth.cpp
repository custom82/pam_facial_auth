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

namespace fs = std::filesystem;

// DICHIARAZIONI ESTERNE (Implementate nei file plugin)
extern "C" std::unique_ptr<RecognizerPlugin> create_classic_plugin(const std::string& method, const FacialAuthConfig& cfg);
extern "C" std::unique_ptr<RecognizerPlugin> create_sface_plugin(const FacialAuthConfig& cfg);

// Factory interna (Linkage C++ perché usa unique_ptr)
std::unique_ptr<RecognizerPlugin> create_plugin(const FacialAuthConfig& cfg) {
    if (cfg.method == "sface" || cfg.method == "auto") return create_sface_plugin(cfg);
    return create_classic_plugin(cfg.method, cfg);
}

// Inizio blocco esportazione simboli C per il linker
extern "C" {

    FA_EXPORT bool fa_check_root(const std::string& tool_name) {
        if (getuid() != 0) {
            std::cerr << "ERRORE [" << tool_name << "]: Devi eseguire come root (sudo)." << std::endl;
            return false;
        }
        return true;
    }

    FA_EXPORT bool fa_load_config(FacialAuthConfig& cfg, std::string& log, const std::string& path) {
        std::ifstream file(path);
        if (!file.is_open()) {
            log = "Impossibile aprire il file di configurazione: " + path;
            return false;
        }

        std::string line;
        while (std::getline(file, line)) {
            if (line.empty() || line[0] == '#') continue;

            std::istringstream is_line(line);
            std::string key;
            if (std::getline(is_line, key, '=')) {
                std::string value;
                if (std::getline(is_line, value)) {
                    // Pulizia spazi
                    key.erase(key.find_last_not_of(" \t") + 1);
                    value.erase(0, value.find_first_not_of(" \t"));

                    if (key == "basedir") cfg.basedir = value;
                    else if (key == "device") cfg.device = value;
                    else if (key == "detect_yunet") cfg.detect_yunet = value;
                    else if (key == "recognize_sface") cfg.recognize_sface = value;
                    else if (key == "training_method") cfg.method = value;
                    else if (key == "sface_threshold") cfg.sface_threshold = std::stod(value);
                    else if (key == "lbph_threshold") cfg.lbph_threshold = std::stod(value);
                    else if (key == "frames") cfg.frames = std::stoi(value);
                    else if (key == "width") cfg.width = std::stoi(value);
                    else if (key == "height") cfg.height = std::stoi(value);
                    else if (key == "sleep_ms") cfg.sleep_ms = std::stoi(value);
                    else if (key == "debug") cfg.debug = (value == "yes");
                    else if (key == "verbose") cfg.verbose = (value == "yes");
                    else if (key == "nogui") cfg.nogui = (value == "yes");
                }
            }
        }
        return true;
    }

    FA_EXPORT std::string fa_user_model_path(const FacialAuthConfig& cfg, const std::string& user) {
        // Estensione .yml per SFace, .xml per i metodi classici (LBPH)
        std::string ext = (cfg.method == "sface" || cfg.method == "auto") ? ".yml" : ".xml";
        return cfg.basedir + "/models/" + user + ext;
    }

    FA_EXPORT bool fa_clean_captures(const std::string& user, const FacialAuthConfig& cfg, std::string& log) {
        std::string user_dir = cfg.basedir + "/captures/" + user;
        try {
            if (fs::exists(user_dir)) {
                fs::remove_all(user_dir);
            }
            return true;
        } catch (const fs::filesystem_error& e) {
            log = "Errore pulizia: " + std::string(e.what());
            return false;
        }
    }

    FA_EXPORT bool fa_capture_user(const std::string& user, const FacialAuthConfig& cfg, const std::string& device_path, std::string& log) {
        cv::VideoCapture cap;

        // Supporta sia indici (0) che percorsi (/dev/video0)
        if (device_path.find_first_not_of("0123456789") == std::string::npos)
            cap.open(std::stoi(device_path));
        else
            cap.open(device_path);

        if (!cap.isOpened()) { log = "Webcam non accessibile: " + device_path; return false; }

        std::string user_dir = cfg.basedir + "/captures/" + user;
        fs::create_directories(user_dir);

        int count = 0;
        while (count < cfg.frames) {
            cv::Mat frame;
            cap >> frame;
            if (frame.empty()) break;

            cv::Mat face_roi;
            cv::resize(frame, face_roi, cv::Size(cfg.width, cfg.height));

            std::string img_path = user_dir + "/" + std::to_string(count) + ".jpg";
            cv::imwrite(img_path, face_roi);

            count++;
            if (cfg.verbose) std::cout << "Catturato frame " << count << "/" << cfg.frames << std::endl;

            if (cfg.sleep_ms > 0)
                std::this_thread::sleep_for(std::chrono::milliseconds(cfg.sleep_ms));
        }
        return true;
    }

    FA_EXPORT bool fa_train_user(const std::string& user, const FacialAuthConfig& cfg, std::string& log) {
        std::string user_dir = cfg.basedir + "/captures/" + user;
        if (!fs::exists(user_dir)) { log = "Directory catture non trovata"; return false; }

        std::vector<cv::Mat> faces;
        std::vector<int> labels;

        for (const auto& entry : fs::directory_iterator(user_dir)) {
            cv::Mat img = cv::imread(entry.path().string(), cv::IMREAD_GRAYSCALE);
            if (!img.empty()) {
                faces.push_back(img);
                labels.push_back(0);
            }
        }

        if (faces.empty()) { log = "Nessuna immagine valida per il training"; return false; }

        fs::create_directories(cfg.basedir + "/models");
        auto plugin = create_plugin(cfg);
        return plugin->train(faces, labels, fa_user_model_path(cfg, user));
    }

    FA_EXPORT bool fa_test_user(const std::string& user, const FacialAuthConfig& cfg, const std::string& model_path, double& confidence, int& label, std::string& log) {
        auto plugin = create_plugin(cfg);
        if (!plugin->load(model_path)) { log = "Caricamento modello fallito"; return false; }

        cv::VideoCapture cap(cfg.device);
        if (!cap.isOpened()) { log = "Webcam non accessibile"; return false; }

        cv::Mat frame;
        cap >> frame; // Warm-up
        cap >> frame;

        if (frame.empty()) { log = "Frame vuoto"; return false; }

        return plugin->predict(frame, label, confidence);
    }

} // Fine extern "C"
