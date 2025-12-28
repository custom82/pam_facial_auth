/*
 * Project: pam_facial_auth
 * License: GPL-3.0
 */

#include "libfacialauth.h"
#include <iostream>
#include <fstream>
#include <filesystem>
#include <thread>
#include <regex>

namespace fs = std::filesystem;

// Trova l'ultimo indice frame_N per continuare la sequenza
int get_last_index(const std::string& dir) {
    int max_idx = -1;
    if (!fs::exists(dir)) return max_idx;
    std::regex re("frame_(\\d+)\\.\\w+");
    for (const auto& entry : fs::directory_iterator(dir)) {
        std::smatch match;
        std::string fname = entry.path().filename().string();
        if (std::regex_match(fname, match, re)) {
            try {
                max_idx = std::max(max_idx, std::stoi(match[1].str()));
            } catch (...) {}
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
            size_t sep = line.find('=');
            if (sep == std::string::npos) continue;
            std::string key = line.substr(0, sep);
            std::string val = line.substr(sep + 1);
            // Trim
            key.erase(key.find_last_not_of(" \t") + 1);
            val.erase(0, val.find_first_not_of(" \t"));

            if (key == "basedir") cfg.basedir = val;
            else if (key == "device") cfg.device = val;
            else if (key == "detect_yunet") cfg.detect_yunet = val;
            else if (key == "detector") cfg.detector = val;
            else if (key == "image_format") cfg.image_format = val;
            else if (key == "frames") cfg.frames = std::stoi(val);
            else if (key == "width") cfg.width = std::stoi(val);
            else if (key == "height") cfg.height = std::stoi(val);
            else if (key == "sleep_ms") cfg.sleep_ms = std::stoi(val);
            else if (key == "debug") cfg.debug = (val == "yes");
        }
        return true;
    }

    FA_EXPORT bool fa_capture_user(const std::string& user, const FacialAuthConfig& cfg, const std::string& device_path, std::string& log) {
        cv::VideoCapture cap(device_path);
        if (!cap.isOpened()) { log = "Webcam off: " + device_path; return false; }

        cv::Ptr<cv::FaceDetectorYN> detector;
        if (cfg.detector == "yunet") {
            if (!fs::exists(cfg.detect_yunet)) {
                log = "ERRORE: Modello YuNet non trovato in " + cfg.detect_yunet;
                return false;
            }
            detector = cv::FaceDetectorYN::create(cfg.detect_yunet, "", cv::Size(320, 320));
        }

        std::string user_dir = cfg.basedir + "/captures/" + user;
        fs::create_directories(user_dir);
        int start_idx = get_last_index(user_dir) + 1;

        std::cout << "[INFO] Detector: " << cfg.detector << " | Sequenza da: " << start_idx << std::endl;

        int saved = 0;
        int dropped = 0;

        while (saved < cfg.frames) {
            cv::Mat frame;
            cap >> frame;
            if (frame.empty()) continue;

            bool face_found = false;

            if (detector) {
                detector->setInputSize(frame.size());
                cv::Mat faces;
                detector->detect(frame, faces);
                if (faces.rows > 0) face_found = true;
            } else {
                // Se nessun detector è attivo, salva tutto (comportamento legacy)
                face_found = true;
            }

            if (face_found) {
                cv::Mat res;
                cv::resize(frame, res, cv::Size(cfg.width, cfg.height));
                std::string path = user_dir + "/frame_" + std::to_string(start_idx + saved) + "." + cfg.image_format;
                if (cv::imwrite(path, res)) {
                    saved++;
                }
            } else {
                dropped++;
            }

            // Feedback verboso obbligatorio per debug
            std::cout << "\r[STATO] Salvati: " << saved << "/" << cfg.frames
            << " | Scartati: " << dropped
            << (face_found ? " [VOLTO TROVATO] " : " [NESSUN VOLTO]  ") << std::flush;

            if (cfg.sleep_ms > 0) std::this_thread::sleep_for(std::chrono::milliseconds(cfg.sleep_ms));
        }
        std::cout << "\n[SUCCESS] Acquisizione completata." << std::endl;
        return true;
    }

    FA_EXPORT bool fa_clean_captures(const std::string& user, const FacialAuthConfig& cfg, std::string& log) {
        std::string user_dir = cfg.basedir + "/captures/" + user;
        if (fs::exists(user_dir)) fs::remove_all(user_dir);
        log = "Cartella svuotata.";
        return true;
    }

    FA_EXPORT std::string fa_user_model_path(const FacialAuthConfig& cfg, const std::string& user) {
        return cfg.basedir + "/models/" + user + ".yml";
    }

}
