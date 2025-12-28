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
#include <unistd.h>
#include <ctime>
#include <opencv2/face.hpp>

namespace fs = std::filesystem;

// Funzione interna (non esportata)
int get_last_index(const std::string& dir) {
    int max_idx = -1;
    if (!fs::exists(dir)) return max_idx;
    std::regex re("frame_(\\d+)\\.\\w+");
    for (const auto& entry : fs::directory_iterator(dir)) {
        std::string filename = entry.path().filename().string();
        std::smatch match;
        if (std::regex_match(filename, match, re)) {
            try { max_idx = std::max(max_idx, std::stoi(match[1].str())); } catch (...) {}
        }
    }
    return max_idx;
}

extern "C" {

    bool fa_check_root(const std::string& tool_name) {
        if (getuid() != 0) {
            std::cerr << "Errore: " << tool_name << " richiede privilegi di root." << std::endl;
            return false;
        }
        return true;
    }

    std::string fa_user_model_path(const FacialAuthConfig& cfg, const std::string& user) {
        return cfg.modeldir + "/" + user + ".xml";
    }

    bool fa_load_config(FacialAuthConfig& cfg, std::string& log, const std::string& path) {
        std::ifstream file(path);
        if (!file.is_open()) { log = "Config mancante: " + path; return false; }
        std::string line;
        while (std::getline(file, line)) {
            if (line.empty() || line[0] == '#') continue;
            size_t sep = line.find('=');
            if (sep == std::string::npos) continue;
            std::string key = line.substr(0, sep), val = line.substr(sep + 1);
            if (key == "basedir") cfg.basedir = val;
            else if (key == "modeldir") cfg.modeldir = val;
            else if (key == "threshold") cfg.threshold = std::stod(val);
            else if (key == "detector") cfg.detector = val;
            else if (key == "nogui") cfg.nogui = (val == "yes");
        }
        return true;
    }

    bool fa_capture_user(const std::string& user, const FacialAuthConfig& cfg, const std::string& device_path, std::string& log) {
        cv::VideoCapture cap(device_path);
        if (!cap.isOpened()) { log = "Webcam non accessibile."; return false; }

        std::string user_dir = cfg.basedir + "/captures/" + user;
        fs::create_directories(user_dir);

        int start_idx = get_last_index(user_dir) + 1;
        int current_saved = 0;

        while (current_saved < cfg.frames) {
            cv::Mat frame; cap >> frame;
            if (frame.empty()) continue;

            cv::Mat res; cv::resize(frame, res, cv::Size(cfg.width, cfg.height));
            std::string img_path = user_dir + "/frame_" + std::to_string(start_idx + current_saved) + "." + cfg.image_format;
            cv::imwrite(img_path, res);

            current_saved++;
            std::cout << "\r[*] Cattura: " << current_saved << "/" << cfg.frames << std::flush;

            if (!cfg.nogui) {
                cv::imshow("Cattura", frame);
                if (cv::waitKey(1) == 27) break;
            }
            std::this_thread::sleep_for(std::chrono::milliseconds(100));
        }
        if (!cfg.nogui) cv::destroyAllWindows();
        std::cout << std::endl;
        return true;
    }

    bool fa_train_user(const std::string& user, const FacialAuthConfig& cfg, std::string& log) {
        std::string user_dir = cfg.basedir + "/captures/" + user;
        if (!fs::exists(user_dir)) { log = "Directory non trovata: " + user_dir; return false; }

        std::vector<cv::Mat> images;
        std::vector<int> labels;
        for (const auto& entry : fs::directory_iterator(user_dir)) {
            cv::Mat img = cv::imread(entry.path().string(), cv::IMREAD_GRAYSCALE);
            if (!img.empty()) {
                images.push_back(img);
                labels.push_back(1);
            }
        }

        if (images.empty()) { log = "Nessuna immagine per il training."; return false; }

        cv::Ptr<cv::face::LBPHFaceRecognizer> model = cv::face::LBPHFaceRecognizer::create();
        model->train(images, labels);

        std::string model_path = fa_user_model_path(cfg, user);
        fs::create_directories(cfg.modeldir);
        model->write(model_path);

        log = "Modello XML salvato in " + model_path;
        return true;
    }

    bool fa_clean_captures(const std::string& user, const FacialAuthConfig& cfg, std::string& log) {
        fs::remove_all(cfg.basedir + "/captures/" + user);
        return true;
    }

    bool fa_test_user(const std::string& user, const FacialAuthConfig& cfg, const std::string& model_path, double& confidence, int& label, std::string& log) {
        log = "Funzione test non implementata";
        return false;
    }

}
