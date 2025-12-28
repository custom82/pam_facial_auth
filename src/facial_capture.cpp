/*
 * Project: pam_facial_auth
 * License: GPL-3.0
 */

#include "libfacialauth.h"
#include <iostream>
#include <vector>
#include <string>

// ... usage() rimane uguale ...

int main(int argc, char** argv) {
    if (!fa_check_root("facial_capture")) return 1;

    std::string user;
    std::string device = "/dev/video0";
    std::string config_path = "/etc/security/pam_facial_auth.conf";
    FacialAuthConfig cfg;
    std::string log;

    std::vector<std::string> args(argv + 1, argv + argc);
    if (args.empty()) { usage(); return 1; }

    // 1. Carica config file
    fa_load_config(cfg, log, config_path);

    // 2. Parsa argomenti CLI (sovrascrivono config)
    for (size_t i = 0; i < args.size(); ++i) {
        if (args[i] == "--help" || args[i] == "-H") { usage(); return 0; }
        else if ((args[i] == "-u" || args[i] == "--user") && i + 1 < args.size()) user = args[++i];
        else if ((args[i] == "-d" || args[i] == "--device") && i + 1 < args.size()) device = args[++i];
        else if (args[i] == "--detector" && i + 1 < args.size()) cfg.detector = args[++i];
        else if (args[i] == "-n" || args[i] == "--num_images") cfg.frames = std::stoi(args[++i]);
        else if (args[i] == "-s" || args[i] == "--sleep") cfg.capture_delay = std::stod(args[++i]);
        else if (args[i] == "-v" || args[i] == "--verbose") cfg.verbose = true;
        else if (args[i] == "--debug") cfg.debug = true;
    }

    if (user.empty()) { std::cerr << "Errore: -u obbligatorio\n"; return 1; }

    // Forza il caricamento del detector se passato da CLI
    if (!fa_capture_user(user, cfg, device, log)) {
        std::cerr << "ERRORE: " << log << std::endl;
        return 1;
    }

    return 0;
}
