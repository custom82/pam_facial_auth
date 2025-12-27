#include "../include/libfacialauth.h"
#include <iostream>

int main(int argc, char** argv) {
    // Controllo permessi root (necessari per scrivere in /var/lib/...)
    if (!fa_check_root("facial_training")) return 1;

    if (argc < 2) {
        std::cerr << "Uso: facial_training <username>\n";
        return 1;
    }

    std::string user = argv[1];
    FacialAuthConfig cfg;
    std::string log;

    // Carica la configurazione globale
    fa_load_config(cfg, log, FACIALAUTH_DEFAULT_CONFIG);

    std::cout << "[INFO] Inizio addestramento per l'utente: " << user << "\n";

    if (fa_train_user(user, cfg, log)) {
        std::cout << "[SUCCESS] Modello generato correttamente.\n";
    } else {
        std::cerr << "[ERROR] Addestramento fallito: " << log << "\n";
        return 1;
    }

    return 0;
}
