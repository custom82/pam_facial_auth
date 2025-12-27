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
        else if (key == "method") cfg.method = val;
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
            log = "Pulizia completata per: " + user;
        } else {
            log = "Nessun dato trovato per: " + user;
        }
        return true;
    } catch (const fs::filesystem_error& e) {
        log = "Errore filesystem: " + std::string(e.what());
        return false;
    }
}

bool fa_capture_user(const std::string& user, const FacialAuthConfig& cfg, const std::string& device_path, std::string& log) {
    #ifdef HAVE_OPENCV
    int device_id = 0;
    if (device_path.find("/dev/video") != std::string::npos) {
        try { device_id = std::stoi(device_path.substr(10)); } catch (...) { device_id = 0; }
    }

    cv::VideoCapture cap(device_id);
    cv::CascadeClassifier face_cascade;
    if (!face_cascade.load(cfg.cascade_path)) {
        log = "Errore caricamento detector: " + cfg.cascade_path;
        return false;
    }
    if (!cap.isOpened()) {
        log = "Impossibile aprire device: " + device_path;
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

            if (cfg.verbose) std::cout << "\r[*] Frame: " << count << "/" << cfg.frames << std::flush;

            if (!cfg.nogui) {
                cv::rectangle(frame, area, cv::Scalar(0, 255, 0), 2);
                cv::imshow("Facial Capture", frame);
                if (cv::waitKey(1) == 27) return false;
            }

            if (cfg.capture_delay > 0)
                std::this_thread::sleep_for(std::chrono::milliseconds(static_cast<int>(cfg.capture_delay * 1000)));

            if (count >= cfg.frames) break;
        }
    }
    if (!cfg.nogui) cv::destroyAllWindows();
    std::cout << std::endl;
    return true;
    #else
    log = "OpenCV non disponibile."; return false;
    #endif
}

// Nota: fa_train_user e fa_test_user rimangono simili a quelli caricati,
// ma useranno la logica dei plugin una volta integrati nel main factory.
