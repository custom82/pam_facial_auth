/*
 * Project: pam_facial_auth
 * License: GPL-3.0
 */

#include "libfacialauth.h"
#include <iostream>
#include <vector>
#include <string>

void usage() {
    std::cout << "Utilizzo: facial_capture -u <utente> [--clean]" << std::endl;
}

int main(int argc, char** argv) {
    if (!fa_check_root("facial_capture")) return 1;

    std::string user;
    bool clean_requested = false;
    std::vector<std::string> args(argv + 1, argv + argc);

    for (size_t i = 0; i < args.size(); ++i) {
        if (args[i] == "-u" && i + 1 < args.size()) {
            user = args[++i];
        } else if (args[i] == "--clean") {
            clean_requested = true;
        }
    }

    if (user.empty()) {
        usage();
        return 1;
    }

    FacialAuthConfig cfg;
    std::string log;
    fa_load_config(cfg, log);

    if (clean_requested) {
        if (fa_clean_captures(user, cfg, log)) {
            std::cout << log << std::endl;
            return 0;
        } else {
            std::cerr << "Errore: " << log << std::endl;
            return 1;
        }
    }

    std::cout << "Avvio cattura per l'utente: " << user << std::endl;
    if (!fa_capture_user(user, cfg, "default", log)) {
        std::cerr << "Errore: " << log << std::endl;
        return 1;
    }

    std::cout << "\nOperazione riuscita. Puoi ora allenare il modello con: facial_training -u " << user << std::endl;
    return 0;
}
