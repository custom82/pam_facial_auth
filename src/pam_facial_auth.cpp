#include <security/pam_modules.h>
#include <security/pam_ext.h>
#include <security/pam_appl.h>
#include <syslog.h>
#include <fstream>

#include <opencv2/opencv.hpp>
#include <opencv2/face.hpp>

#include <fstream>
#include <thread>
#include <chrono>
#include <string>
#include <iostream>
#include <filesystem>

#include "FaceRecWrapper.h"

namespace fs = std::filesystem;

extern "C" {

    // ============================================================================
    // Funzione richiesta da PAM (necessaria per evitare errori di simbolo mancanti)
    // ============================================================================
    int pam_sm_setcred(pam_handle_t *pamh, int flags, int argc, const char **argv) {
        return PAM_SUCCESS;
    }

    // ============================================================================
    // Funzione per scrivere il log nel file di debug
    // ============================================================================
    void write_debug_log(const std::string &message) {
        std::ofstream log_file("/var/log/pam_facial_auth.log", std::ios_base::app);
        if (log_file.is_open()) {
            log_file << message << std::endl;
            log_file.close();
        } else {
            syslog(LOG_ERR, "Unable to open debug log file");
        }
    }

    // ============================================================================
    // Funzione principale di autenticazione PAM
    // ============================================================================
    int pam_sm_authenticate(pam_handle_t *pamh, int flags, int argc, const char **argv) {
        const char *user = nullptr;
        int pam_result = pam_get_user(pamh, &user, nullptr);

        if (pam_result != PAM_SUCCESS || user == nullptr) {
            pam_syslog(pamh, LOG_ERR, "Unable to get PAM user");
            return PAM_AUTH_ERR;
        }

        // Parametri di configurazione di default
        std::string model_root = "/etc/pam_facial_auth";
        bool nogui = false;
        bool debug = false;
        double threshold = 80.0;
        int timeout = 5;
        std::string device = "/dev/video0";

        // Parsing dei parametri dallo stack PAM
        for (int i = 0; i < argc; ++i) {
            std::string arg(argv[i]);
            if (arg == "nogui") nogui = true;
            else if (arg == "debug") debug = true;
            else if (arg.rfind("threshold=", 0) == 0)
                threshold = std::stod(arg.substr(10));
            else if (arg.rfind("timeout=", 0) == 0)
                timeout = std::stoi(arg.substr(8));
            else if (arg.rfind("device=", 0) == 0)
                device = arg.substr(7);
            else if (arg.rfind("model_path=", 0) == 0)
                model_root = arg.substr(11);
        }

        // Percorso modello: /etc/pam_facial_auth/<utente>/models/<utente>.xml
        fs::path user_model_dir = fs::path(model_root) / user / "models";
        fs::path model_file = user_model_dir / (std::string(user) + ".xml");

        if (!fs::exists(model_file)) {
            pam_syslog(pamh, LOG_ERR, "Model not found: %s", model_file.c_str());
            return PAM_AUTH_ERR;
        }

        // Inizializzazione riconoscitore facciale
        FaceRecWrapper faceRec(model_root, user);
        try {
            faceRec.Load(model_file);
        } catch (const std::exception &e) {
            pam_syslog(pamh, LOG_ERR, "Failed to load model: %s", e.what());
            return PAM_AUTH_ERR;
        }

        // Apertura webcam
        cv::VideoCapture cap(device);
        if (!cap.isOpened()) {
            pam_syslog(pamh, LOG_ERR, "Unable to open webcam device: %s", device.c_str());
            return PAM_AUTH_ERR;
        }

        if (debug)
            pam_syslog(pamh, LOG_INFO, "Webcam %s opened successfully", device.c_str());

        pam_syslog(pamh, LOG_INFO, "Capturing face for user %s", user);

        if (debug) write_debug_log("Capturing face for user: " + std::string(user));

        auto start_time = std::chrono::steady_clock::now();
        bool recognized = false;

        while (true) {
            cv::Mat frame;
            if (!cap.read(frame) || frame.empty()) {
                pam_syslog(pamh, LOG_ERR, "Failed to capture frame from webcam");
                break;
            }

            // Conversione in grayscale
            cv::Mat gray;
            cv::cvtColor(frame, gray, cv::COLOR_BGR2GRAY);

            int predicted_label = -1;
            double confidence = 0.0;

            int ret = faceRec.Predict(gray, predicted_label, confidence);

            if (ret == 0 && confidence <= threshold) {
                recognized = true;
                if (debug) {
                    pam_syslog(pamh, LOG_INFO, "Face match for %s (confidence=%.2f)", user, confidence);
                    write_debug_log("Face match for user: " + std::string(user) + " (confidence=" + std::to_string(confidence) + ")");
                }
                break;
            }

            // Controllo timeout
            auto elapsed = std::chrono::duration_cast<std::chrono::seconds>(
                std::chrono::steady_clock::now() - start_time
            ).count();

            if (elapsed >= timeout) {
                pam_syslog(pamh, LOG_ERR, "Timeout reached for user %s", user);
                if (debug) write_debug_log("Timeout reached for user: " + std::string(user));
                break;
            }

            std::this_thread::sleep_for(std::chrono::milliseconds(300));
        }

        cap.release();

        if (!recognized) {
            pam_syslog(pamh, LOG_ERR, "Authentication failed for %s", user);
            if (debug) write_debug_log("Authentication failed for user: " + std::string(user));
            return PAM_AUTH_ERR;
        }

        pam_syslog(pamh, LOG_INFO, "Facial authentication succeeded for %s", user);
        if (debug) write_debug_log("Facial authentication succeeded for user: " + std::string(user));

        return PAM_SUCCESS;
    }

} // extern "C"
