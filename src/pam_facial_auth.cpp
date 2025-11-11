#include <string>
#include <iostream>
#include <fstream>
#include <filesystem>
#include <syslog.h>  // Per LOG_ERR, LOG_INFO, ecc.
#include <opencv2/opencv.hpp>
#include <opencv2/face.hpp>
#include <chrono>
#include <thread>

#include <security/pam_appl.h>     // pam_handle_t
#include <security/pam_modules.h>  // PAM interface
#include <security/pam_ext.h>      // pam_syslog()

namespace fs = std::filesystem;

// =============================
// Struttura di configurazione
// =============================
struct Config {
    std::string device = "/dev/video0";
    int width = 640;
    int height = 480;
    int timeout = 10;
    float threshold = 80.0f;
    bool nogui = false;
    bool debug = false;
};

// =============================
// Caricamento configurazione
// =============================
Config load_config(const std::string& config_path) {
    Config cfg;
    std::ifstream conf(config_path);

    if (!conf.is_open()) {
        std::cerr << "Errore nell'aprire il file di configurazione: " << config_path << std::endl;
        throw std::runtime_error("Errore nel caricare il file di configurazione");
    }

    conf >> cfg.device >> cfg.width >> cfg.height >> cfg.timeout >> cfg.threshold;
    return cfg;
}

// =============================
// Parametri PAM
// =============================
void apply_pam_args(Config& cfg, pam_handle_t* pamh, int argc, const char** argv) {
    for (int i = 0; i < argc; ++i) {
        if (strcmp(argv[i], "nogui") == 0) {
            cfg.nogui = true;
        } else if (strcmp(argv[i], "debug") == 0) {
            cfg.debug = true;
        } else if (strncmp(argv[i], "threshold=", 10) == 0) {
            cfg.threshold = std::stof(argv[i] + 10);
        } else if (strncmp(argv[i], "timeout=", 8) == 0) {
            cfg.timeout = std::stoi(argv[i] + 8);
        }
    }

    if (cfg.debug)
        pam_syslog(pamh, LOG_DEBUG, "pam_facial_auth: threshold=%.2f timeout=%d nogui=%d",
                   cfg.threshold, cfg.timeout, cfg.nogui);
}

// =============================
// Funzione principale PAM
// =============================
extern "C" int pam_sm_authenticate(pam_handle_t* pamh, int flags, int argc, const char** argv) {
    const char* user = nullptr;
    const char* model_path = nullptr;

    // Ottieni l’utente PAM
    if (pam_get_user(pamh, &user, nullptr) != PAM_SUCCESS || !user) {
        pam_syslog(pamh, LOG_ERR, "pam_facial_auth: unable to retrieve username");
        return PAM_AUTH_ERR;
    }

    // Carica configurazione
    Config cfg;
    try {
        cfg = load_config("/etc/pam_facial_auth/pam_facial.conf");
    } catch (...) {
        pam_syslog(pamh, LOG_ERR, "pam_facial_auth: unable to read config file");
        return PAM_AUTH_ERR;
    }

    apply_pam_args(cfg, pamh, argc, argv);

    // Percorso modello
    std::string user_model_dir = "/etc/pam_facial_auth/" + std::string(user) + "/models/";
    std::string default_model = user_model_dir + std::string(user) + ".xml";
    model_path = default_model.c_str();

    if (!fs::exists(model_path)) {
        pam_syslog(pamh, LOG_ERR, "pam_facial_auth: model not found for user %s at %s", user, model_path);
        return PAM_AUTH_ERR;
    }

    // TODO: Inserire qui la logica di riconoscimento facciale
    // Per ora simuliamo un successo di autenticazione
    pam_syslog(pamh, LOG_INFO, "pam_facial_auth: authentication succeeded for %s", user);
    return PAM_SUCCESS;
}

// =============================
// Sessione PAM (stub vuoti)
// =============================
extern "C" int pam_sm_setcred(pam_handle_t*, int, int, const char**) {
    return PAM_SUCCESS;
}
