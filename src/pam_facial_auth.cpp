// pam_facial_auth.cpp
#include <security/pam_modules.h>
#include <security/pam_ext.h>
#include <opencv2/opencv.hpp>
#include <filesystem>
#include <string>
#include <iostream>
#include <chrono>
#include <thread>
#include "FaceRecWrapper.h"
#include <syslog.h>

namespace fs = std::filesystem;

// Configurazione del modulo PAM
struct FacialAuthConfig {
    bool debug = false;
    bool nogui = false;
    double threshold = 80.0;
    int timeout = 10;
    std::string model_path = "/etc/pam_facial_auth";
    std::string device = "/dev/video0";
};

// Analisi parametri passati dallo stack PAM
static FacialAuthConfig parse_args(int argc, const char **argv) {
    FacialAuthConfig cfg;
    for (int i = 0; i < argc; ++i) {
        std::string arg(argv[i]);
        if (arg == "debug") cfg.debug = true;
        else if (arg == "nogui") cfg.nogui = true;
        else if (arg.starts_with("threshold=")) cfg.threshold = std::stod(arg.substr(10));
        else if (arg.starts_with("timeout=")) cfg.timeout = std::stoi(arg.substr(8));
        else if (arg.starts_with("model_path=")) cfg.model_path = arg.substr(11);
        else if (arg.starts_with("device=")) cfg.device = arg.substr(7);
    }
    return cfg;
}

// Implementazione del modulo PAM
extern "C" {

    PAM_EXTERN int pam_sm_authenticate(pam_handle_t *pamh, int flags, int argc, const char **argv) {
        FacialAuthConfig cfg = parse_args(argc, argv);
        const char *user = nullptr;

        // Ottiene l'utente corrente
        if (pam_get_user(pamh, &user, NULL) != PAM_SUCCESS || !user) {
            pam_syslog(pamh, LOG_ERR, "Unable to get user");
            return PAM_USER_UNKNOWN;
        }

        std::string user_dir = fs::path(cfg.model_path) / user / "models";
        std::string model_file = (fs::path(user_dir) / (std::string(user) + ".xml")).string();

        if (!fs::exists(model_file)) {
            if (cfg.debug)
                pam_syslog(pamh, LOG_DEBUG, "Model not found for user %s at %s", user, model_file.c_str());
            return PAM_IGNORE; // Non bloccare, lascia passare ad altri moduli
        }

        // Se è stato già autenticato (es. pam_unix ha successo), non rifare autenticazione
        int retval = 0;
        retval = pam_get_item(pamh, PAM_AUTHTOK, nullptr);
        if (retval == PAM_SUCCESS && !(flags & PAM_SILENT)) {
            if (cfg.debug)
                pam_syslog(pamh, LOG_DEBUG, "Authentication token already set, skipping facial auth");
            return PAM_IGNORE;
        }

        if (cfg.debug)
            pam_syslog(pamh, LOG_INFO, "Starting facial authentication for user %s", user);

        cv::VideoCapture cap(cfg.device);
        if (!cap.isOpened()) {
            pam_syslog(pamh, LOG_ERR, "Cannot open webcam device %s", cfg.device.c_str());
            return PAM_AUTH_ERR;
        }

        FaceRecWrapper fr(model_file, user);

        auto start_time = std::chrono::steady_clock::now();
        bool recognized = false;
        double confidence = 0.0;
        int label = -1;

        if (cfg.debug)
            pam_syslog(pamh, LOG_INFO, "Webcam opened, beginning recognition loop");

        while (true) {
            cv::Mat frame;
            cap >> frame;
            if (frame.empty()) {
                pam_syslog(pamh, LOG_ERR, "Empty frame captured");
                continue;
            }

            int pred = -1;
            double conf = 0.0;
            fr.Predict(frame, pred, conf);

            if (cfg.debug)
                pam_syslog(pamh, LOG_INFO, "Prediction=%d Confidence=%.2f", pred, conf);

            if (pred == 1 && conf >= cfg.threshold) {
                recognized = true;
                confidence = conf;
                break;
            }

            if (std::chrono::duration_cast<std::chrono::seconds>(
                std::chrono::steady_clock::now() - start_time).count() > cfg.timeout) {
                break;
                }

                std::this_thread::sleep_for(std::chrono::milliseconds(200));
        }

        cap.release();

        if (recognized) {
            pam_syslog(pamh, LOG_INFO, "Facial authentication successful (conf=%.2f)", confidence);
            return PAM_SUCCESS;
        } else {
            pam_syslog(pamh, LOG_WARNING, "Facial authentication failed for user %s", user);
            return PAM_AUTH_ERR;
        }
    }

    PAM_EXTERN int pam_sm_setcred(pam_handle_t *pamh, int flags, int argc, const char **argv) {
        // Gestione credenziali: non serve modificare le credenziali di sistema
        return PAM_SUCCESS;
    }

} // extern "C"
