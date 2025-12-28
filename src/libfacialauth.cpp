/*
 * Project: pam_facial_auth
 * License: GPL-3.0
 */

#include "libfacialauth.h"
#include <iostream>
#include <fstream>
#include <filesystem>
#include <thread>
#include <opencv2/face.hpp>

namespace fs = std::filesystem;

extern "C" {

    bool fa_check_root(const std::string& tool_name) {
        if (getuid() != 0) {
            std::cerr << "Errore: " << tool_name << " richiede root." << std::endl;
            return false;
        }
        return true;
    }

    bool fa_load_config(FacialAuthConfig& cfg, std::string& log, const std::string& path) {
        std::string final_path = path.empty() ? "/etc/security/pam_facial_auth.conf" : path;
        std::ifstream file(final_path);
        if (!file.is_open()) {
            log = "Config non trovato: " + final_path;
            return false;
        }
        // Logica di parsing semplificata
        cfg.basedir = "/var/lib/pam_facial_auth";
        return true;
    }

    std::string fa_user_model_path(const FacialAuthConfig& cfg, const std::string& user) {
        return cfg.modeldir + "/" + user + ".xml";
    }

    bool fa_capture_user(const std::string& user, const FacialAuthConfig& cfg, const std::string& device_path, std::string& log) {
        cv::VideoCapture cap(device_path);
        if (!cap.isOpened()) { log = "Webcam non disponibile"; return false; }

        std::string path = cfg.basedir + "/captures/" + user;
        fs::create_directories(path);

        for (int i = 0; i < cfg.frames; ++i) {
            cv::Mat frame; cap >> frame;
            if (frame.empty()) continue;
            cv::imwrite(path + "/f_" + std::to_string(i) + ".jpg", frame);
            std::this_thread::sleep_for(std::chrono::milliseconds(100));
        }
        return true;
    }

    bool fa_train_user(const std::string& user, const FacialAuthConfig& cfg, std::string& log) {
        std::string path = cfg.basedir + "/captures/" + user;
        if (!fs::exists(path)) { log = "No catture per " + user; return false; }

        std::vector<cv::Mat> faces;
        std::vector<int> labels;
        for (const auto& entry : fs::directory_iterator(path)) {
            cv::Mat img = cv::imread(entry.path().string(), cv::IMREAD_GRAYSCALE);
            if (!img.empty()) { faces.push_back(img); labels.push_back(1); }
        }

        auto model = cv::face::LBPHFaceRecognizer::create();
        model->train(faces, labels);
        model->write(fa_user_model_path(cfg, user));
        log = "Training completato";
        return true;
    }

    bool fa_clean_captures(const std::string& user, const FacialAuthConfig& cfg, std::string& log) {
        fs::remove_all(cfg.basedir + "/captures/" + user);
        return true;
    }

    bool fa_test_user(const std::string& user, const FacialAuthConfig& cfg, const std::string& model_path, double& confidence, int& label, std::string& log) {
        log = "Test non implementato in questa versione";
        return false;
    }

}
