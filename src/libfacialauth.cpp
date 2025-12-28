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

// Factory interna (Non esportata con linkage C perché usa unique_ptr)
std::unique_ptr<RecognizerPlugin> create_plugin(const FacialAuthConfig& cfg) {
    if (cfg.method == "sface") return create_sface_plugin(cfg);
    return create_classic_plugin(cfg.method, cfg);
}

// Inizio blocco esportazione simboli per il linker
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
            // Salta commenti e righe vuote
            if (line.empty() || line[0] == '#') continue;

            std::istringstream is_line(line);
            std::string key;
            if (std::getline(is_line, key, '=')) {
                std::string value;
                if (std::getline(is_line, value)) {
                    // Rimuove eventuali spazi bianchi
                    key.erase(key.find_last_not_of(" \n\r\t") + 1);
                    value.erase(0, value.find_first_not_of(" \n\r\t"));

                    if (key == "basedir") cfg.basedir = value;
                    else if (key == "cascade_path") cfg.cascade_path = value;
                    else if (key == "sface_model") cfg.recognize_sface = value;
                    else if (key == "detector") cfg.detector = value;
                    else if (key == "method") cfg.method = value;
                    else if (key == "threshold") cfg.threshold = std::stod(value);
                    else if (key == "sface_threshold") cfg.sface_threshold = std::stod(value);
                    else if (key == "lbph_threshold") cfg.lbph_threshold = std::stod(value);
                    else if (key == "frames") cfg.frames = std::stoi(value);
                    else if (key == "width") cfg.width = std::stoi(value);
                    else if (key == "height") cfg.height = std::stoi(value);
                    else if (key == "capture_delay") cfg.capture_delay = std::stod(value);
                    else if (key == "debug") cfg.debug = (value == "true" || value == "1");
                }
            }
        }
        return true;
    }

    FA_EXPORT std::string fa_user_model_path(const FacialAuthConfig& cfg, const std::string& user) {
        // Percorso dinamico: <basedir>/models/<user>.xml
        return cfg.basedir + "/models/" + user + ".xml";
    }

    FA_EXPORT bool fa_clean_captures(const std::string& user, const FacialAuthConfig& cfg, std::string& log) {
        std::string user_dir = cfg.basedir + "/captures/" + user;
        try {
            if (fs::exists(user_dir)) {
                fs::remove_all(user_dir);
            }
            return true;
        } catch (const fs::filesystem_error& e) {
            log = "Errore pulizia directory: " + std::string(e.what());
            return false;
        }
    }

    FA_EXPORT bool fa_capture_user(const std::string& user, const FacialAuthConfig& cfg, const std::string& device_path, std::string& log) {
        cv::VideoCapture cap;

        // Gestisce sia indici numerici (0) che percorsi (/dev/video0)
        if (device_path.find_first_not_of("0123456789") == std::string::npos)
            cap.open(std::stoi(device_path));
        else
            cap.open(device_path);

        if (!cap.isOpened()) { log = "Webcam non trovata o occupata"; return false; }

        cv::CascadeClassifier face_cascade;
        if (!face_cascade.load(cfg.cascade_path)) {
            log = "Impossibile caricare il classificatore: " + cfg.cascade_path;
            return false;
        }

        std::string user_dir = cfg.basedir + "/captures/" + user;
        fs::create_directories(user_dir);

        int count = 0;
        while (count < cfg.frames) {
            cv::Mat frame; cap >> frame;
            if (frame.empty()) break;

            std::vector<cv::Rect> faces;
            face_cascade.detectMultiScale(frame, faces, 1.1, 3, 0, cv::Size(30, 30));

            for (const auto& area : faces) {
                cv::Mat face_roi = frame(area);
                cv::resize(face_roi, face_roi, cv::Size(cfg.width, cfg.height));
                cv::imwrite(user_dir + "/" + std::to_string(count) + ".jpg", face_roi);
                count++;

                if (cfg.debug) {
                    std::cout << "\rCattura frame " << count << "/" << cfg.frames << std::flush;
                }

                if (count >= cfg.frames) break;
                if (cfg.capture_delay > 0)
                    std::this_thread::sleep_for(std::chrono::milliseconds(static_cast<int>(cfg.capture_delay * 1000)));
            }
        }
        std::cout << std::endl;
        return true;
    }

    FA_EXPORT bool fa_train_user(const std::string& user, const FacialAuthConfig& cfg, std::string& log) {
        std::string user_dir = cfg.basedir + "/captures/" + user;
        if (!fs::exists(user_dir)) { log = "Nessuna immagine trovata in " + user_dir; return false; }

        std::vector<cv::Mat> faces;
        std::vector<int> labels;

        for (const auto& entry : fs::directory_iterator(user_dir)) {
            if (entry.path().extension() == ".jpg") {
                cv::Mat img = cv::imread(entry.path().string(), cv::IMREAD_GRAYSCALE);
                if (!img.empty()) {
                    faces.push_back(img);
                    labels.push_back(0); // Etichetta singola per utente
                }
            }
        }

        if (faces.empty()) { log = "Directory catture vuota"; return false; }

        std::string model_dir = cfg.basedir + "/models";
        fs::create_directories(model_dir);

        auto plugin = create_plugin(cfg);
        return plugin->train(faces, labels, fa_user_model_path(cfg, user));
    }

    FA_EXPORT bool fa_test_user(const std::string& user, const FacialAuthConfig& cfg, const std::string& model_path, double& confidence, int& label, std::string& log) {
        auto plugin = create_plugin(cfg);
        if (!plugin->load(model_path)) { log = "Modello utente non trovato: " + model_path; return false; }

        cv::VideoCapture cap(0);
        if (!cap.isOpened()) { log = "Errore apertura webcam"; return false; }

        cv::Mat frame;
        // Warm-up della camera
        for(int i=0; i<5; i++) cap >> frame;

        if (frame.empty()) { log = "Impossibile leggere frame dalla webcam"; return false; }

        return plugin->predict(frame, label, confidence);
    }

} // Fine extern "C"
