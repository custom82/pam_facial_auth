/*
 * Project: pam_facial_auth
 * License: GPL-3.0
 */

#include "libfacialauth.h"
#include <iostream>
#include <vector>
#include <filesystem>

namespace fs = std::filesystem;

void usage() {
    std::cout << "Usage: facial_training -u <user> [options]\n\n"
    << "Options:\n"
    << "  -u, --user <name>       Utente target\n"
    << "  -c, --config <file>     Config (default: /etc/security/pam_facial_auth.conf)\n"
    << "  -m, --method <type>     Metodo: lbph, eigen, fisher, sface\n"
    << "  -o, --output <file>     Percorso manuale del modello\n"
    << "  -f, --force             Sovrascrivi modello esistente\n"
    << "  -v, --verbose           Log dettagliati\n"
    << "  -h, --help              Mostra questo aiuto\n";
}

int main(int argc, char** argv) {
    if (!fa_check_root("facial_training")) return 1;

    std::string user, config_path = "/etc/security/pam_facial_auth.conf", out_path, log;
    FacialAuthConfig cfg;
    bool force = false;

    std::vector<std::string> args(argv + 1, argv + argc);
    for (size_t i = 0; i < args.size(); ++i) {
        if ((args[i] == "-c" || args[i] == "--config") && i + 1 < args.size()) config_path = args[++i];
    }
    if (!fa_load_config(cfg, log, config_path)) { std::cerr << "Errore Config: " << log << std::endl; return 1; }

    for (size_t i = 0; i < args.size(); ++i) {
        if (args[i] == "-u" || args[i] == "--user") user = args[++i];
        else if (args[i] == "-m" || args[i] == "--method") cfg.method = args[++i];
        else if (args[i] == "-o" || args[i] == "--output") out_path = args[++i];
        else if (args[i] == "-f" || args[i] == "--force") force = true;
        else if (args[i] == "-v" || args[i] == "--verbose") cfg.verbose = true;
        else if (args[i] == "-h" || args[i] == "--help") { usage(); return 0; }
    }

    if (user.empty()) { usage(); return 1; }
    std::string final_path = out_path.empty() ? fa_user_model_path(cfg, user) : out_path;

    if (fs::exists(final_path) && !force) {
        std::cerr << "Il modello esiste gia'. Usa -f per sovrascrivere." << std::endl;
        return 1;
    }

    if (!fa_train_user(user, cfg, log)) { std::cerr << "ERRORE TRAINING: " << log << std::endl; return 1; }
    std::cout << "[OK] " << log << std::endl;
    return 0;
}
