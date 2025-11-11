#include <security/pam_modules.h>  // Include PAM per l'autenticazione
#include <security/pam_appl.h>     // Include PAM per il logging
#include <security/pam_ext.h>      // Include PAM per il syslog
#include <syslog.h>                // Per i macro LOG_* (LOG_ERR, LOG_INFO, etc.)
#include <string>
#include <opencv2/opencv.hpp>
#include <opencv2/face.hpp>
#include <iostream>
#include <filesystem>
#include <fstream>                // Per std::ifstream
#include <thread>                 // Per std::this_thread
#include <chrono>                 // Per gestione del timeout

namespace fs = std::filesystem;

// Struttura per configurazione
struct Config {
    std::string device;
    int timeout;
    double threshold;
    bool nogui;
    bool debug;
};

Config load_config(const std::string& config_path) {
    Config cfg;

    // Legge il file di configurazione
    std::ifstream conf(config_path);
    if (!conf.is_open()) {
        std::cerr << "Error: Cannot open config file: " << config_path << std::endl;
        return cfg;
    }

    std::string line;
    while (std::getline(conf, line)) {
        if (line.find("device") != std::string::npos) {
            cfg.device = line.substr(line.find('=') + 1);
        } else if (line.find("timeout") != std::string::npos) {
            cfg.timeout = std::stoi(line.substr(line.find('=') + 1));
        } else if (line.find("threshold") != std::string::npos) {
            cfg.threshold = std::stod(line.substr(line.find('=') + 1));
        } else if (line.find("nogui") != std::string::npos) {
            cfg.nogui = line.substr(line.find('=') + 1) == "true";
        } else if (line.find("debug") != std::string::npos) {
            cfg.debug = line.substr(line.find('=') + 1) == "true";
        }
    }

    return cfg;
}

void apply_pam_args(Config& cfg, pam_handle_t* pamh, int argc, const char** argv) {
    // Aggiungi eventuali logiche per applicare i parametri PAM
    for (int i = 0; i < argc; ++i) {
        if (strncmp(argv[i], "device=", 7) == 0) {
            cfg.device = argv[i] + 7;
        } else if (strncmp(argv[i], "timeout=", 8) == 0) {
            cfg.timeout = std::stoi(argv[i] + 8);
        } else if (strncmp(argv[i], "threshold=", 10) == 0) {
            cfg.threshold = std::stod(argv[i] + 10);
        } else if (strncmp(argv[i], "nogui=", 6) == 0) {
            cfg.nogui = std::string(argv[i] + 6) == "true";
        } else if (strncmp(argv[i], "debug=", 6) == 0) {
            cfg.debug = std::string(argv[i] + 6) == "true";
        }
    }
}

extern "C" {
    int pam_sm_authenticate(pam_handle_t *pamh, int flags, int argc, const char **argv) {
        const char *user = NULL;
        const char *model_path = NULL;

        // Ottieni il nome dell'utente dal PAM
        if (pam_get_user(pamh, &user, NULL) != PAM_SUCCESS) {
            std::cerr << "Unable to get user" << std::endl;
            return PAM_AUTH_ERR;
        }

        // Carica la configurazione
        Config cfg = load_config("/etc/pam_facial_auth/pam_facial.conf");

        // Applica i parametri passati nel modulo PAM
        apply_pam_args(cfg, pamh, argc, argv);

        if (cfg.debug) {
            std::cerr << "Configuration loaded. Device: " << cfg.device << ", Timeout: " << cfg.timeout << ", Threshold: " << cfg.threshold << std::endl;
        }

        // Ottieni il percorso del modello
        if (argc > 1) {
            model_path = argv[1];  // Percorso fornito come argomento
        } else {
            model_path = (std::string("/etc/pam_facial_auth/") + user + "/models/" + user + ".xml").c_str();  // Percorso di default
        }

        // Verifica se il modello esiste
        if (!fs::exists(model_path)) {
            std::cerr << "Model not found at " << model_path << std::endl;
            return PAM_AUTH_ERR;
        }

        // Carica il modello
        cv::Ptr<cv::face::FaceRecognizer> model = cv::face::LBPHFaceRecognizer::create();
        try {
            model->read(model_path);
        } catch (const cv::Exception& e) {
            std::cerr << "Error loading model: " << e.what() << std::endl;
            return PAM_AUTH_ERR;
        }

        // Apre la webcam
        cv::VideoCapture cap(cfg.device);  // Apri il dispositivo webcam

        if (!cap.isOpened()) {
            pam_syslog(pamh, LOG_ERR, "Error: Unable to open webcam device %s", cfg.device.c_str());
            return PAM_AUTH_ERR;
        }

        pam_syslog(pamh, LOG_INFO, "Webcam %s opened successfully", cfg.device.c_str());

        cv::Mat frame;
        int prediction = -1;
        double confidence = 0.0;
        while (true) {
            cap >> frame;  // Leggi un frame dalla webcam

            if (frame.empty()) {
                pam_syslog(pamh, LOG_ERR, "Error: Failed to capture frame from webcam");
                return PAM_AUTH_ERR;
            }

            // Previsione del volto
            model->predict(frame, prediction, confidence);

            if (cfg.debug) {
                pam_syslog(pamh, LOG_DEBUG, "Prediction: %d, Confidence: %.2f", prediction, confidence);
            }

            if (confidence < cfg.threshold) {
                pam_syslog(pamh, LOG_INFO, "Authentication successful for user %s", user);
                return PAM_SUCCESS;  // Autenticazione riuscita
            }

            // Timeout dopo il periodo definito
            if (cfg.timeout > 0) {
                std::this_thread::sleep_for(std::chrono::seconds(cfg.timeout));
            }

            // Se il ciclo è stato interrotto
            if (prediction != -1) {
                break;
            }
        }

        return PAM_AUTH_ERR;  // Se non corrisponde
    }
}
