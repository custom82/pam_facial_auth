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
#include <unistd.h>

namespace fs = std::filesystem;

extern "C" {

    bool fa_check_root(const std::string& tool_name) {
        if (getuid() != 0) {
            std::cerr << "Errore: " << tool_name << " richiede privilegi di root." << std::endl;
            return false;
        }
        return true;
    }

    bool fa_load_config(FacialAuthConfig& cfg, std::string& log, const std::string& path) {
        std::string real_path = path.empty() ? "/etc/security/pam_facial_auth.conf" : path;
        std::ifstream file(real_path);
        if (!file.is_open()) {
            log = "Configurazione non trovata in " + real_path + ". Uso i default.";
            return true; // Non falliamo se il config non c'è, usiamo i default della struct
        }

        std::string line;
        while (std::getline(file, line)) {
            if (line.empty() || line[0] == '#') continue;
            size_t sep = line.find('=');
            if (sep == std::string::npos) continue;
            std::string key = line.substr(0, sep);
            std::string val = line.substr(sep + 1);

            if (key == "detector") cfg.detector = val;
            else if (key == "method") cfg.method = val;
            else if (key == "cascade_path") cfg.cascade_path = val;
            else if (key == "modeldir") cfg.modeldir = val;
            else if (key == "basedir") cfg.basedir = val;
        }
        return true;
    }

    std::string fa_user_model_path(const FacialAuthConfig& cfg, const std::string& user) {
        return cfg.modeldir + "/" + user + ".xml";
    }

    bool fa_capture_user(const std::string& user, const FacialAuthConfig& cfg, const std::string& device_path, std::string& log) {
        cv::VideoCapture cap(device_path);
        if (!cap.isOpened()) { log = "Impossibile aprire la webcam: " + device_path; return false; }

        std::string user_dir = cfg.basedir + "/captures/" + user;
        fs::create_directories(user_dir);

        int count = 0;
        while (count < cfg.frames) {
            cv::Mat frame;
            cap >> frame;
            if (frame.empty()) continue;

            std::string img_name = user_dir + "/f_" + std::to_string(count) + "." + cfg.image_format;
            cv::imwrite(img_name, frame);
            count++;

            if (!cfg.nogui) {
                cv::imshow("Cattura", frame);
                if (cv::waitKey(1) == 27) break;
            }
            std::this_thread::sleep_for(std::chrono::milliseconds(static_cast<int>(cfg.capture_delay * 1000)));
        }
        return true;
    }

    bool fa_train_user(const std::string& user, const FacialAuthConfig& cfg, std::string& log) {
        std::string user_dir = cfg.basedir + "/captures/" + user;
        if (!fs::exists(user_dir)) { log = "Directory catture mancante"; return false; }

        std::string method = cfg.method;
        if (method == "auto") method = "lbph"; // oppure "sface" se vuoi default moderno

        std::unique_ptr<RecognizerPlugin> plugin;
        if (method == "sface") plugin = std::make_unique<SFacePlugin>(cfg);
        else                   plugin = std::make_unique<ClassicPlugin>(method, cfg);

        std::vector<cv::Mat> images;
        std::vector<int> labels;

        for (const auto& entry : fs::directory_iterator(user_dir)) {
            // per sface meglio caricare a colori (BGR) e poi resize 112x112 nel plugin
            int mode = (method == "sface") ? cv::IMREAD_COLOR : cv::IMREAD_GRAYSCALE;
            cv::Mat img = cv::imread(entry.path().string(), mode);
            if (!img.empty()) {
                images.push_back(img);
                labels.push_back(1);
            }
        }

        if (images.empty()) { log = "Nessuna immagine valida trovata"; return false; }

        fs::create_directories(cfg.modeldir);

        std::string out = cfg.modeldir + "/" + user + ".xml";
        if (!plugin->train(images, labels, out)) {
            log = "Training fallito";
            return false;
        }

        log = "Modello XML salvato: " + out + " (algorithm=" + plugin->get_name() + ")";
        return true;
    }

