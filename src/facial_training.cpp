/*
 * Project: pam_facial_auth
 * License: GPL-3.0
 */

#include "libfacialauth.h"
#include <iostream>
#include <string>
#include <vector>
#include <filesystem>

namespace fs = std::filesystem;

void usage() {
    std::cout << "Usage: facial_training -u <user> [options]\n\n"
    << "Options:\n"
    << "  -u, --user <name>           Utente per cui generare il modello\n"
    << "  -c, --config <file>         Percorso config (default: /etc/security/pam_facial_auth.conf)\n"
    << "  -m, --method <type>         Metodo: lbph, eigen, fisher, sface\n"
    << "  -o, --output <file>         Percorso salvataggio modello (XML/YML)\n"
    << "  -f, --force                 Sovrascrivi se esistente\n"
    << "  -v, --verbose               Log dettagliati\n"
    << "  -h, --help                  Mostra questo aiuto\n";
}

int main(int argc, char** argv) {
    if (!fa_check_root("facial_training")) return 1;

    std::string user;
    // CORRETTO: Percorso allineato con il sistema
    std::string config_path = "/etc/security/pam_facial_auth.conf";
    std::string output_path;
    bool force = false;

    FacialAuthConfig cfg;
    std::string log;

    std::vector<std::string> args(argv + 1, argv + argc);
    if (args.empty()) { usage(); return 1; }

    // Parsing preliminare per il config path
    for (size_t i = 0; i < args.size(); ++i) {
        if ((args[i] == "-c" || args[i] == "--config") && i + 1 < args.size()) config_path = args[++i];
    }

    if (!fa_load_config(cfg, log, config_path)) {
        std::cerr << "ERRORE CRITICO: " << log << std::endl;
        return 1;
    }

    for (size_t i = 0; i < args.size(); ++i) {
        if (args[i] == "-h" || args[i] == "--help") { usage(); return 0; }
        else if ((args[i] == "-u" || args[i] == "--user") && i + 1 < args.size()) user = args[++i];
        else if ((args[i] == "-m" || args[i] == "--method") && i + 1 < args.size()) cfg.method = args[++i];
        else if ((args[i] == "-o" || args[i] == "--output") && i + 1 < args.size()) output_path = args[++i];
        else if (args[i] == "-f" || args[i] == "--force") force = true;
        else if (args[i] == "-v" || args[i] == "--verbose") cfg.verbose = true;
    }

    if (user.empty()) {
        std::cerr << "Errore: L'utente (-u) è obbligatorio.\n";
        return 1;
    }

    if (output_path.empty()) output_path = fa_user_model_path(cfg, user);

    if (fs::exists(output_path) && !force) {
        std::cerr << "Errore: Il modello esiste già: " << output_path << "\nUsare -f per sovrascrivere.\n";
        return 1;
    }

    if (!fa_train_user(user, cfg, log)) {
        std::cerr << "ERRORE TRAINING: " << log << std::endl;
        return 1;
    }

    std::cout << "[SUCCESSO] " << log << std::endl;
    return 0;
}
