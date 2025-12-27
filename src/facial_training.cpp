/*
 * Project: pam_facial_auth
 * License: GPL-3.0
 */

#include "libfacialauth.h"
#include <iostream>
#include <string>
#include <vector>

int main(int argc, char** argv) {
    if (!fa_check_root("facial_training")) return 1;

    std::string user;
    std::vector<std::string> args(argv + 1, argv + argc);

    for (size_t i = 0; i < args.size(); ++i) {
        if (args[i] == "-u" && i + 1 < args.size()) {
            user = args[++i];
        }
    }

    if (user.empty()) {
        std::cout << "Uso: facial_training -u <utente>" << std::endl;
        return 1;
    }

    FacialAuthConfig cfg;
    std::string log;
    fa_load_config(cfg, log);

    std::cout << "[*] Avvio training per: " << user << std::endl;
    if (!fa_train_user(user, cfg, log)) {
        std::cerr << "ERRORE: " << log << std::endl;
        return 1;
    }

    std::cout << "[SUCCESS] " << log << std::endl;
    return 0;
}
