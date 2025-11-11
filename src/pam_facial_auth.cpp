#include <security/pam_modules.h>
#include <security/pam_ext.h>
#include <opencv2/opencv.hpp>
#include <filesystem>
#include <string>
#include <iostream>
#include <chrono>
#include <thread>
#include <fstream>
#include <sstream>
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
    std::string model = "lbph"; // Default model
};

// Funzione per caricare il file di configurazione
FacialAuthConfig load_config(const std::string &config_file) {
    FacialAuthConfig cfg;
    std::ifstream file(config_file);
    std::string line;

    if (!file.is_open()) {
        std::cerr << "Unable to open configuration file: " << config_file << std::endl;
        return cfg;
    }

    while (std::getline(file, line)) {
        std::istringstream iss(line);
        std::string key;
        if (std::getline(iss, key, ' ')) {
            std::string value;
            if (std::getline(iss, value)) {
                // Assegna i parametri in base alla chiave
                if (key == "device") cfg.device = value;
                else if (key == "threshold") cfg.threshold = std::stod(value);
                else if (key == "timeout") cfg.timeout = std::stoi(value);
                else if (key == "model_path") cfg.model_path = value;
                else if (key == "model") cfg.model = value;  // Imposta il modello
                else if (key == "debug") cfg.debug = (value == "true");
                else if (key == "nogui") cfg.nogui = (value == "true");
            }
        }
    }
    return cfg;
}

extern "C" {

    PAM_EXTERN int pam_sm_authenticate(pam_handle_t *pamh, int flags, int argc, const char **argv) {
        // Carica la configurazione dal file
        FacialAuthConfig cfg = load_config("/etc/pam_facial_auth/pam_facial.conf");

        const char *user = nullptr;

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

        // Se è stato già autenticato, non rifare autenticazione
        int retval = pam_get_item(pamh, PAM_AUTHTOK, nullptr);
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

        // Passa anche il modello dal file di configurazione
        FaceRecWrapper fr(model_file, user, cfg.model);

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
