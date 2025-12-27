#include "../include/libfacialauth.h"
#include <iostream>

int main(int argc, char** argv) {
    if (argc < 2) {
        std::cerr << "Uso: facial_test <username>\n";
        return 1;
    }

    std::string user = argv[1];
    FacialAuthConfig cfg;
    std::string log;

    fa_load_config(cfg, log, FACIALAUTH_DEFAULT_CONFIG);

    std::cout << "[INFO] Avvio test interattivo per: " << user << "\n";
    std::cout << "[INFO] Assicurati di essere davanti alla camera.\n";

    if (!fa_test_user_interactive(user, cfg, log)) {
        std::cerr << "[FAIL] Riconoscimento fallito o errore: " << log << "\n";
        return 1;
    }

    return 0;
}
