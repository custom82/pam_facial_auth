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
    std::string config_path = "/etc/security/pam_facial_auth.conf";
    FacialAuthConfig cfg;
    std::string log;

    std::vector<std::string> args(argv + 1, argv + argc);
    for (size_t i = 0; i < args.size(); ++i) {
        if ((args[i] == "-u" || args[i] == "--user") && i + 1 < args.size()) user = args[++i];
        if ((args[i] == "-c" || args[i] == "--config") && i + 1 < args.size()) config_path = args[++i];
    }

    if (user.empty()) {
        std::cout << "Uso: facial_training -u <utente> [-c <config>]" << std::endl;
        return 1;
    }

    fa_load_config(cfg, log, config_path);
    std::cout << "[*] Avvio training per: " << user << " (Metodo: " << cfg.method << ")" << std::endl;

    if (!fa_train_user(user, cfg, log)) {
        std::cerr << "ERRORE: " << log << std::endl;
        return 1;
    }

    std::cout << "[SUCCESS] " << log << std::endl;
    return 0;
}
