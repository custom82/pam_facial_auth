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
    << "  -u, --user <name>           Specify the username to train the model for\n"
    << "  -m, --method <type>         Specify the training method (lbph, eigen, fisher, sface)\n"
    << "  -o, --output <file>         Path to save the trained model (XML)\n"
    << "  -f, --force                 Force overwrite of existing model file\n"
    << "  -v, --verbose               Enable detailed output\n"
    << "  -h, --help                  Show this help message\n";
}

int main(int argc, char** argv) {
    if (!fa_check_root("facial_training")) return 1;

    std::string user;
    std::string config_path = "/etc/pam_facial_auth/pam_facial.conf";
    std::string output_path;
    bool force = false;

    FacialAuthConfig cfg;
    std::string log;

    std::vector<std::string> args(argv + 1, argv + argc);
    if (args.empty()) { usage(); return 1; }

    // Caricamento configurazione base
    fa_load_config(cfg, log, config_path);

    // Parsing parametri CLI
    for (size_t i = 0; i < args.size(); ++i) {
        if (args[i] == "-h" || args[i] == "--help") { usage(); return 0; }
        else if ((args[i] == "-u" || args[i] == "--user") && i + 1 < args.size()) user = args[++i];
        else if ((args[i] == "-m" || args[i] == "--method") && i + 1 < args.size()) cfg.method = args[++i];
        else if ((args[i] == "-o" || args[i] == "--output") && i + 1 < args.size()) output_path = args[++i];
        else if (args[i] == "-f" || args[i] == "--force") force = true;
        else if (args[i] == "-v" || args[i] == "--verbose") cfg.verbose = true;
    }

    if (user.empty()) {
        std::cerr << "Error: User (-u) is mandatory.\n";
        return 1;
    }

    // Se l'output non è specificato, usa il path di default
    if (output_path.empty()) {
        output_path = fa_user_model_path(cfg, user);
    }

    // Controllo esistenza file se non forzato
    if (fs::exists(output_path) && !force) {
        std::cerr << "Error: Model file already exists: " << output_path << "\n";
        std::cerr << "Use -f or --force to overwrite.\n";
        return 1;
    }

    if (cfg.verbose) {
        std::cout << "[*] Starting training for: " << user << "\n";
        std::cout << "[*] Method: " << cfg.method << "\n";
        std::cout << "[*] Output: " << output_path << "\n";
    }

    // Esecuzione del training tramite la libreria
    if (!fa_train_user(user, cfg, log)) {
        std::cerr << "TRAINING ERROR: " << log << std::endl;
        return 1;
    }

    std::cout << "[SUCCESS] " << log << std::endl;
    return 0;
}
