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

namespace fs = std::filesystem;

extern "C" {

    FA_EXPORT bool fa_check_root(const std::string& tool_name) {
        return (getuid() == 0);
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
            else if (key == "verbose") cfg.verbose = (val == "yes");
            else if (key == "nogui") cfg.nogui = (val == "yes");
        }
        return true;
    }

    FA_EXPORT bool fa_capture_user(const std::string& user, const FacialAuthConfig& cfg, const std::string& device_path, std::string& log) {
        cv::VideoCapture cap(device_path);
        if (!cap.isOpened()) { log = "Webcam non accessibile: " + device_path; return false; }

        cv::Ptr<cv::FaceDetectorYN> detector_yn;
        cv::CascadeClassifier detector_cascade;

        if (cfg.detector == "yunet") {
            if (cfg.detect_yunet.empty() || !fs::exists(cfg.detect_yunet)) { log = "YuNet model non trovato"; return false; }
            detector_yn = cv::FaceDetectorYN::create(cfg.detect_yunet, "", cv::Size(320, 320));
        } else if (cfg.detector == "cascade") {
            if (cfg.cascade_path.empty() || !detector_cascade.load(cfg.cascade_path)) { log = "Haar Cascade XML non trovato o invalido"; return false; }
        }

        std::string user_dir = cfg.basedir + "/captures/" + user;
        fs::create_directories(user_dir);

        int saved = 0, dropped = 0;
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
                detector_cascade.detectMultiScale(gray, faces, 1.1, 3, 0, cv::Size(30, 30));
                face_found = !faces.empty();
            } else { face_found = (cfg.detector == "none"); }

            if (face_found) {
                cv::Mat res; cv::resize(frame, res, cv::Size(cfg.width, cfg.height));
                std::string path = user_dir + "/frame_" + std::to_string(saved) + "." + cfg.image_format;
                if (cv::imwrite(path, res)) {
                    saved++;
                    if (cfg.debug) std::cout << "[DEBUG] Salvato: " << path << std::endl;
                }
            } else { dropped++; }

            if (!cfg.debug) {
                std::cout << "\r[*] Acquisizione: " << saved << "/" << cfg.frames
                << " | Saltati: " << dropped << (face_found ? " [OK]" : " [NO VOLTO]") << std::flush;
            }

            if (!cfg.nogui) {
                cv::imshow("Facial Capture", frame);
                if (cv::waitKey(1) == 27) break;
            }

            std::this_thread::sleep_for(std::chrono::milliseconds((int)(cfg.capture_delay * 1000)));
        }
        cv::destroyAllWindows();
        std::cout << std::endl;
        return true;
    }

    FA_EXPORT bool fa_clean_captures(const std::string& user, const FacialAuthConfig& cfg, std::string& log) {
        std::string user_dir = cfg.basedir + "/captures/" + user;
        if (fs::exists(user_dir)) fs::remove_all(user_dir);
        return true;
    }

}
