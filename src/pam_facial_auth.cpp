#include <string>
#include <iostream>
#include <fstream>
#include <filesystem>
#include <chrono>
#include <thread>
#include <syslog.h>
#include <opencv2/opencv.hpp>

#include <security/pam_appl.h>
#include <security/pam_modules.h>
#include <security/pam_ext.h>

#include "FaceRecWrapper.h"

namespace fs = std::filesystem;

struct Config {
    std::string device = "/dev/video0";
    int width = 640;
    int height = 480;
    int timeout = 10;
    float threshold = 80.0f;
    bool nogui = false;
    bool debug = false;
};

Config load_config(const std::string& config_path) {
    Config cfg;
    std::ifstream conf(config_path);
    if (!conf.is_open()) {
        throw std::runtime_error("Impossibile aprire il file di configurazione");
    }
    conf >> cfg.device >> cfg.width >> cfg.height >> cfg.timeout >> cfg.threshold;
    return cfg;
}

void apply_pam_args(Config& cfg, pam_handle_t* pamh, int argc, const char** argv) {
    for (int i = 0; i < argc; ++i) {
        if (strcmp(argv[i], "nogui") == 0)
            cfg.nogui = true;
        else if (strcmp(argv[i], "debug") == 0)
            cfg.debug = true;
        else if (strncmp(argv[i], "threshold=", 10) == 0)
            cfg.threshold = std::stof(argv[i] + 10);
        else if (strncmp(argv[i], "timeout=", 8) == 0)
            cfg.timeout = std::stoi(argv[i] + 8);
    }

    if (cfg.debug)
        pam_syslog(pamh, LOG_DEBUG,
                   "pam_facial_auth: threshold=%.2f timeout=%d nogui=%d",
                   cfg.threshold, cfg.timeout, cfg.nogui);
}

extern "C" int pam_sm_authenticate(pam_handle_t* pamh, int, int argc, const char** argv) {
    const char* user = nullptr;
    if (pam_get_user(pamh, &user, nullptr) != PAM_SUCCESS || !user) {
        pam_syslog(pamh, LOG_ERR, "pam_facial_auth: impossibile ottenere l'utente");
        return PAM_AUTH_ERR;
    }

    Config cfg;
    try {
        cfg = load_config("/etc/pam_facial_auth/pam_facial.conf");
    } catch (...) {
        pam_syslog(pamh, LOG_ERR, "pam_facial_auth: impossibile leggere il file di configurazione");
        return PAM_AUTH_ERR;
    }

    apply_pam_args(cfg, pamh, argc, argv);

    std::string user_dir = std::string("/etc/pam_facial_auth/") + user;
    std::string model_file = user_dir + "/models/" + user + ".xml";

    if (!fs::exists(model_file)) {
        pam_syslog(pamh, LOG_ERR, "pam_facial_auth: modello non trovato per %s (%s)",
                   user, model_file.c_str());
        return PAM_AUTH_ERR;
    }

    // Istanzia FaceRecWrapper
    FaceRecWrapper fr(model_file, user);

    cv::VideoCapture cap(cfg.device);
    if (!cap.isOpened()) {
        pam_syslog(pamh, LOG_ERR, "pam_facial_auth: impossibile aprire webcam %s", cfg.device.c_str());
        return PAM_AUTH_ERR;
    }

    cap.set(cv::CAP_PROP_FRAME_WIDTH, cfg.width);
    cap.set(cv::CAP_PROP_FRAME_HEIGHT, cfg.height);

    cv::CascadeClassifier face_cascade;
    face_cascade.load("/usr/share/opencv4/haarcascades/haarcascade_frontalface_default.xml");

    auto start = std::chrono::steady_clock::now();
    pam_syslog(pamh, LOG_INFO, "pam_facial_auth: in attesa del volto di %s", user);

    while (true) {
        cv::Mat frame, gray;
        cap >> frame;
        if (frame.empty())
            continue;

        cv::cvtColor(frame, gray, cv::COLOR_BGR2GRAY);
        std::vector<cv::Rect> faces;
        face_cascade.detectMultiScale(gray, faces, 1.1, 4, 0, cv::Size(100, 100));

        if (!faces.empty()) {
            int predicted = -1;
            double confidence = 0.0;
            cv::Mat roi = gray(faces[0]);
            fr.Predict(roi, predicted, confidence);

            if (cfg.debug)
                pam_syslog(pamh, LOG_DEBUG,
                           "pam_facial_auth: pred=%d conf=%.2f thr=%.2f",
                           predicted, confidence, cfg.threshold);

                if (confidence <= cfg.threshold) {
                    pam_syslog(pamh, LOG_INFO,
                               "pam_facial_auth: riconoscimento riuscito per %s (%.2f)",
                               user, confidence);
                    return PAM_SUCCESS;
                }
        }

        auto elapsed = std::chrono::duration_cast<std::chrono::seconds>(
            std::chrono::steady_clock::now() - start)
        .count();

        if (elapsed > cfg.timeout) {
            pam_syslog(pamh, LOG_ERR,
                       "pam_facial_auth: timeout (%d s) per %s",
                       cfg.timeout, user);
            break;
        }

        std::this_thread::sleep_for(std::chrono::milliseconds(100));
    }

    pam_syslog(pamh, LOG_WARNING, "pam_facial_auth: autenticazione fallita per %s", user);
    return PAM_AUTH_ERR;
}

extern "C" int pam_sm_setcred(pam_handle_t*, int, int, const char**) {
    return PAM_SUCCESS;
}
