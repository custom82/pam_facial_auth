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

// Dichiarazioni esterne per i plugin linkati nella stessa lib
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
            try { max_idx = std::max(max_idx, std::stoi(match[1].str())); } catch (...) {}
        }
    }
    return max_idx;
}

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
            else if (key == "detector") cfg.detector = val;
            else if (key == "frames") cfg.frames = std::stoi(val);
            else if (key == "width") cfg.width = std::stoi(val);
            else if (key == "height") cfg.height = std::stoi(val);
            else if (key == "sleep_ms") { cfg.sleep_ms = std::stoi(val); cfg.capture_delay = (double)cfg.sleep_ms / 1000.0; }
            else if (key == "sface_threshold") { cfg.sface_threshold = std::stod(val); cfg.threshold = cfg.sface_threshold; }
            else if (key == "lbph_threshold") { cfg.lbph_threshold = std::stod(val); if(cfg.method=="lbph") cfg.threshold=val; }
        }
        return true;
    }

    FA_EXPORT bool fa_capture_user(const std::string& user, const FacialAuthConfig& cfg, const std::string& device_path, std::string& log) {
        cv::VideoCapture cap(device_path);
        if (!cap.isOpened()) { log = "Webcam non accessibile"; return false; }

        cv::Ptr<cv::FaceDetectorYN> detector;
        if (cfg.detector == "yunet") {
            if (!fs::exists(cfg.detect_yunet)) { log = "File YuNet ONNX non trovato"; return false; }
            detector = cv::FaceDetectorYN::create(cfg.detect_yunet, "", cv::Size(320, 320));
        }

        std::string user_dir = cfg.basedir + "/captures/" + user;
        fs::create_directories(user_dir);
        int start_idx = get_last_index(user_dir) + 1;

        int saved = 0, dropped = 0;
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
                face_found = (cfg.detector == "none");
            }

            if (face_found) {
                cv::Mat res;
                cv::resize(frame, res, cv::Size(cfg.width, cfg.height));
                std::string path = user_dir + "/frame_" + std::to_string(start_idx + saved) + "." + cfg.image_format;
                if (cv::imwrite(path, res)) saved++;
            } else {
                dropped++;
            }

            if (cfg.debug || cfg.verbose) {
                std::cout << "\r[DEBUG] Detector: " << cfg.detector << " | Salvati: " << saved << "/" << cfg.frames
                << " | Scartati: " << dropped << (face_found ? " [VOLTO OK]" : " [COPERTA]") << "    " << std::flush;
            }

            int wait = (cfg.capture_delay > 0) ? (int)(cfg.capture_delay * 1000) : cfg.sleep_ms;
            std::this_thread::sleep_for(std::chrono::milliseconds(wait));
        }
        std::cout << std::endl;
        return true;
    }

    FA_EXPORT bool fa_train_user(const std::string& user, const FacialAuthConfig& cfg, std::string& log) {
        std::string user_dir = cfg.basedir + "/captures/" + user;
        if (!fs::exists(user_dir)) return false;
        std::vector<cv::Mat> faces; std::vector<int> labels;
        for (const auto& entry : fs::directory_iterator(user_dir)) {
            cv::Mat img = cv::imread(entry.path().string(), cv::IMREAD_GRAYSCALE);
            if (!img.empty()) { faces.push_back(img); labels.push_back(0); }
        }
        auto plugin = create_plugin(cfg);
        return plugin->train(faces, labels, fa_user_model_path(cfg, user));
    }

    FA_EXPORT bool fa_test_user(const std::string& user, const FacialAuthConfig& cfg, const std::string& model_path, double& conf, int& lbl, std::string& log) {
        auto plugin = create_plugin(cfg);
        if (!plugin->load(model_path)) return false;
        cv::VideoCapture cap(cfg.device);
        cv::Mat frame; cap >> frame;
        if (frame.empty()) return false;
        return plugin->predict(frame, lbl, conf);
    }

    FA_EXPORT std::string fa_user_model_path(const FacialAuthConfig& cfg, const std::string& user) {
        std::string ext = (cfg.method == "sface" || cfg.method == "auto") ? ".yml" : ".xml";
        return cfg.basedir + "/models/" + user + ext;
    }

    FA_EXPORT bool fa_clean_captures(const std::string& user, const FacialAuthConfig& cfg, std::string& log) {
        std::string user_dir = cfg.basedir + "/captures/" + user;
        if (fs::exists(user_dir)) fs::remove_all(user_dir);
        log = "Dati eliminati.";
        return true;
    }

}
