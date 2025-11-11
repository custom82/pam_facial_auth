#include <string>
#include <iostream>
#include <fstream>
#include <filesystem>
#include <syslog.h>  // Per LOG_ERR, LOG_INFO, ecc.
#include <opencv2/opencv.hpp>
#include <opencv2/face.hpp>
#include <chrono>
#include <thread>
#include <security/pam_appl.h>  // Necessario per pam_handle_t e PAM
#include <security/pam_modules.h>  // Per definire la funzione pam_sm_authenticate

namespace fs = std::filesystem;

// Definizione della struttura Config
struct Config {
    std::string device = "/dev/video0";  // Dispositivo predefinito
    int width = 640;
    int height = 480;
    int timeout = 10;
    float threshold = 80.0f;
    bool nogui = false;
    bool debug = false;
};

// Funzione per caricare la configurazione
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

// Funzione per applicare i parametri PAM
void apply_pam_args(Config& cfg, pam_handle_t* pamh, int argc, const char** argv) {
    for (int i = 0; i < argc; ++i) {
        if (strcmp(argv[i], "nogui") == 0) {
            cfg.nogui = true;
        }
        else if (strcmp(argv[i], "debug") == 0) {
            cfg.debug = true;
        }
        else if (strncmp(argv[i], "threshold=", 10) == 0) {
            cfg.threshold = std::stof(argv[i] + 10);
        }
        else if (strncmp(argv[i], "timeout=", 8) == 0) {
            cfg.timeout = std::stoi(argv[i] + 8);
        }
    }
}

// Funzione di autenticazione PAM
int pam_sm_authenticate(pam_handle_t* pamh, int flags, int argc, const char** argv) {
    const char* user = NULL;
    const char* model_path = NULL;

    // Ottieni il nome dell'utente dal PAM
    if (pam_get_user(pamh, &user, NULL) != PAM_SUCCESS) {
        std::cerr << "Unable to get user" << std::endl;
        return PAM_AUTH_ERR;
    }

    // Carica la configurazione da file o dai parametri
    Config cfg = load_config("/etc/pam_facial_auth/pam_facial.conf");

    // Leggi i parametri dalla riga di comando
    apply_pam_args(cfg, pamh, argc, argv);

    // Se il percorso del modello è specificato, usalo, altrimenti cerca nella directory dell'utente
    if (argc > 1) {
        model_path = argv[1];
    } else {
        model_path = (std::string("/etc/pam_facial_auth/") + user + "/models/" + user + ".xml").c_str();
    }

    // Verifica se il modello esiste
    if (!fs::exists(model_path)) {
        pam_syslog(pamh, LOG_ERR, "Model not found for user %s at %s", user, model_path);
        return PAM_AUTH_ERR;
    }

    // Aggiungi qui la logica di autenticazione facciale, ad esempio caricando il modello e facendo una previsione
    // Esegui il riconoscimento facciale (aggiungi la logica necessaria)

    // Log dell'esito dell'autenticazione
    pam_syslog(pamh, LOG_INFO, "Authentication succeeded for user %s", user);

    return PAM_SUCCESS;
}
