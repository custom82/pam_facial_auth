// pam_facial_auth.cpp
#include <string>
#include <opencv2/opencv.hpp>
#include <iostream>
#include <filesystem>

namespace fs = std::filesystem;

extern "C" {
    #include <security/pam_modules.h>  // Include PAM

    int pam_sm_authenticate(pam_handle_t *pamh, int flags, int argc, const char **argv) {
        const char *user = NULL;
        const char *model_path = NULL;

        // Ottieni il nome dell'utente dal PAM
        if (pam_get_user(pamh, &user, NULL) != PAM_SUCCESS) {
            std::cerr << "Unable to get user" << std::endl;
            return PAM_AUTH_ERR;
        }

        // Verifica se il percorso del modello è passato, altrimenti usa il percorso predefinito
        if (argc > 1) {
            model_path = argv[1];  // Percorso fornito come argomento
        } else {
            model_path = (std::string("/etc/pam_facial_auth/") + user).c_str();  // Percorso di default
        }

        // Verifica se il modello esiste
        if (!fs::exists(model_path)) {
            std::cerr << "Model not found at " << model_path << std::endl;
            return PAM_AUTH_ERR;
        }

        // Aggiungi qui la logica di autenticazione facciale, ad esempio caricando il modello

        return PAM_SUCCESS;
    }
}
