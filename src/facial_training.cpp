#include "../include/libfacialauth.h"
#include <iostream>

int main(int argc, char** argv) {
    if (!fa_check_root("facial_training")) return 1;
    if (argc < 2) {
        std::cerr << "Uso: " << argv[0] << " <username>\n";
        return 1;
    }

    std::string user = argv[1];
    FacialAuthConfig cfg;
    std::string log;

    if (!fa_load_config(cfg, log, FACIALAUTH_DEFAULT_CONFIG)) {
        std::cout << "[INFO] Usando parametri di default (config non trovata).\n";
    }

    std::cout << "Inizio addestramento per: " << user << " (Metodo: " << cfg.training_method << ")\n";

    if (fa_train_user(user, cfg, log)) {
        std::cout << "Modello salvato con successo in: " << fa_user_model_path(cfg, user) << "\n";
    } else {
        std::cerr << "Errore durante l'addestramento: " << log << "\n";
        return 1;
    }

    return 0;
}
