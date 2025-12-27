/*
 * Project: pam_facial_auth
 * License: GPL-3.0
 */

#include "libfacialauth.h"
#include <iostream>
#include <vector>
#include <string>

int main(int argc, char** argv) {
    if (!fa_check_root("facial_test")) return 1;

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
        std::cout << "Uso: facial_test -u <utente> [-c <config>]" << std::endl;
        return 1;
    }

    fa_load_config(cfg, log, config_path);
    double confidence = 0.0;
    int label = -1;
    std::string model = fa_user_model_path(cfg, user);

    std::cout << "[*] Test riconoscimento per utente: " << user << std::endl;
    if (!fa_test_user(user, cfg, model, confidence, label, log)) {
        std::cerr << "ERRORE: " << log << std::endl;
        return 1;
    }

    bool ok = (confidence <= cfg.threshold);
    std::cout << "-----------------------------------" << std::endl;
    std::cout << "Risultato:   " << (ok ? "AUTENTICATO" : "FALLITO") << std::endl;
    std::cout << "Confidence:  " << confidence << " (Soglia: " << cfg.threshold << ")" << std::endl;
    std::cout << "Label:       " << label << std::endl;
    std::cout << "-----------------------------------" << std::endl;

    return ok ? 0 : 1;
}
