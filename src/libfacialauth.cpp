/*
 * Project: pam_facial_auth
 * License: GPL-3.0
 */

#include "libfacialauth.h"
#include <iostream>
#include <fstream>
#include <filesystem>
#include <thread>
#include <unistd.h>
#include <opencv2/objdetect.hpp>
#include <opencv2/imgproc.hpp>

namespace fs = std::filesystem;

extern "C" {

    FA_EXPORT bool fa_check_root(const std::string& tool_name) {
        if (getuid() != 0) {
            std::cerr << "Errore: " << tool_name << " richiede privilegi di root." << std::endl;
            return false;
        }
        return true;
    }

    FA_EXPORT std::string fa_user_model_path(const FacialAuthConfig& cfg, const std::string& user) {
        return cfg.basedir + "/models/" + user + ".yml";
    }

    FA_EXPORT bool fa_load_config(FacialAuthConfig& cfg, std::string& log, const std::string& path) {
        std::ifstream file(path);
        if (!file.is_open()) { log = "Config mancante: " + path; return false; }
        std::string line;
        while (std::getline(file, line)) {
            if (line.empty() || line[0] == '#') continue;
            size_t sep = line.find('=');
            if (sep == std::string::npos) continue;
            std::string key = line.substr(0, sep);
            std::string val = line.substr(sep + 1);
            key.erase(key.find_last_not_of(" \t") + 1);
            val.erase(0, val.find_first_not_of(" \t"));

            if (key == "basedir") cfg.basedir = val;
            else if (key == "device") cfg.device = val;
            else if (key == "detect_yunet") cfg.detect_yunet = val;
            else if (key == "cascade_path") cfg.cascade_path = val;
            else if (key == "detector") cfg.detector = val;
            else if (key == "frames") cfg.frames = std::stoi(val);
            else if (key == "width") cfg.width = std::stoi(val);
            else if (key == "height") cfg.height = std::stoi(val);
            else if (key == "sleep_ms") { cfg.sleep_ms = std::stoi(val); cfg.capture_delay = (double)cfg.sleep_ms / 1000.0; }
            else if (key == "debug") cfg.debug = (val == "yes");
            else if (key == "nogui") cfg.nogui = (val == "yes");
        }
        return true;
    }

    FA_EXPORT bool fa_capture_user(const std::string& user, const FacialAuthConfig& cfg, const std::string& device_path, std::string& log) {
        cv::VideoCapture cap(device_path);
        if (!cap.isOpened()) { log = "Webcam off: " + device_path; return false; }

        cv::Ptr<cv::FaceDetectorYN> detector_yn;
        cv::CascadeClassifier detector_cascade;

        if (cfg.detector == "yunet") {
            detector_yn = cv::FaceDetectorYN::create(cfg.detect_yunet, "", cv::Size(320, 320));
        } else if (cfg.detector == "cascade") {
            if (!detector_cascade.load(cfg.cascade_path)) { log = "Errore caricamento XML Cascade"; return false; }
        }

        std::string user_dir = cfg.basedir + "/captures/" + user;
        fs::create_directories(user_dir);

        int saved = 0;
        while (saved < cfg.frames) {
            cv::Mat frame; cap >> frame;
            if (frame.empty()) continue;

            bool face_found = false;
            if (cfg.detector == "yunet" && detector_yn) {
                detector_yn->setInputSize(frame.size());
                cv::Mat faces; detector_yn->detect(frame, faces);
                face_found = (faces.rows > 0);
            } else if (cfg.detector == "cascade") {
                cv::Mat gray; cv::cvtColor(frame, gray, cv::COLOR_BGR2GRAY);
                std::vector<cv::Rect> faces;
                detector_cascade.detectMultiScale(gray, faces, 1.1, 3);
                face_found = !faces.empty();
            } else { face_found = true; }

            if (face_found) {
                cv::Mat res; cv::resize(frame, res, cv::Size(cfg.width, cfg.height));
                cv::imwrite(user_dir + "/frame_" + std::to_string(saved) + "." + cfg.image_format, res);
                saved++;
            }

            std::cout << "\r[*] Cattura: " << saved << "/" << cfg.frames << std::flush;
            std::this_thread::sleep_for(std::chrono::milliseconds((int)(cfg.capture_delay * 1000)));
        }
        std::cout << std::endl;
        return true;
    }

    // Stub per evitare errori di link, queste andranno implementate nei plugin/training
    FA_EXPORT bool fa_train_user(const std::string& user, const FacialAuthConfig& cfg, std::string& log) { return false; }
    FA_EXPORT bool fa_test_user(const std::string& user, const FacialAuthConfig& cfg, const std::string& model_path, double& confidence, int& label, std::string& log) { return false; }
    FA_EXPORT bool fa_clean_captures(const std::string& user, const FacialAuthConfig& cfg, std::string& log) {
        fs::remove_all(cfg.basedir + "/captures/" + user);
        return true;
    }

}
