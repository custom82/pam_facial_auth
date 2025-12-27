#include "../include/libfacialauth.h"
#include <iostream>
#include <iomanip>

int main(int argc, char** argv) {
    if (argc < 2) {
        std::cerr << "Uso: " << argv[0] << " <username>\n";
        return 1;
    }

    std::string user = argv[1];
    FacialAuthConfig cfg;
    std::string log;
    fa_load_config(cfg, log, FACIALAUTH_DEFAULT_CONFIG);

    std::string modelPath = fa_user_model_path(cfg, user);
    if (!fa_file_exists(modelPath)) {
        std::cerr << "Errore: Modello non trovato per " << user << ". Esegui prima facial_training.\n";
        return 1;
    }

    double confidence = 0.0;
    int label = -1;

    std::cout << "Test di riconoscimento in corso... Guarda la camera.\n";

    if (fa_test_user(user, cfg, modelPath, confidence, label, log)) {
        std::cout << "******************************************\n";
        std::cout << " RISULTATO: UTENTE RICONOSCIUTO!\n";
        std::cout << " Confidenza: " << std::fixed << std::setprecision(4) << confidence << "\n";
        std::cout << "******************************************\n";
    } else {
        std::cout << "------------------------------------------\n";
        std::cout << " RISULTATO: ACCESSO NEGATO\n";
        std::cout << " Confidenza misurata: " << confidence << "\n";
        std::cout << " Errore/Log: " << log << "\n";
        std::cout << "------------------------------------------\n";
    }

    return 0;
}
