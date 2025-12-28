/*
 * Project: pam_facial_auth
 * License: GPL-3.0
 */

#include "libfacialauth.h"
#include <iostream>
#include <vector>
#include <string>

void usage() {
    std::cout << "Usage: facial_capture -u <user> [options]\n\n"
    << "Options:\n"
    << "  -u, --user <name>       Nome utente\n"
    << "  -d, --device <path>     Dispositivo (default: /dev/video0)\n"
    << "  -c, --config <file>     Config (default: /etc/security/pam_facial_auth.conf)\n"
    << "  --detector <name>       Detector: yunet, none\n"
    << "  -n, --num_images <num>  Immagini da acquisire\n"
    << "  -s, --sleep <sec>       Pausa tra catture (es: 0.1)\n"
    << "  -v, --verbose           Output dettagliato\n"
    << "  --debug                 Output di debug\n";
}

int main(int argc, char** argv) {
    if (!fa_check_root("facial_capture")) return 1;

    std::string user;
    std::string config_path = "/etc/security/pam_facial_auth.conf";
    FacialAuthConfig cfg;
    std::string log;

    std::vector<std::string> args(argv + 1, argv + argc);
    if (args.empty()) { usage(); return 1; }

    fa_load_config(cfg, log, config_path);

    for (size_t i = 0; i < args.size(); ++i) {
        if (args[i] == "--help" || args[i] == "-h") { usage(); return 0; }
        else if ((args[i] == "-u" || args[i] == "--user") && i + 1 < args.size()) user = args[++i];
        else if ((args[i] == "-d" || args[i] == "--device") && i + 1 < args.size()) cfg.device = args[++i];
        else if (args[i] == "--detector" && i + 1 < args.size()) cfg.detector = args[++i];
        else if ((args[i] == "-n" || args[i] == "--num_images") && i + 1 < args.size()) cfg.frames = std::stoi(args[++i]);
        else if ((args[i] == "-s" || args[i] == "--sleep") && i + 1 < args.size()) cfg.capture_delay = std::stod(args[++i]);
        else if (args[i] == "-v" || args[i] == "--verbose") cfg.verbose = true;
        else if (args[i] == "--debug") cfg.debug = true;
    }

    if (user.empty()) { std::cerr << "Errore: specificare l'utente con -u\n"; return 1; }

    if (!fa_capture_user(user, cfg, cfg.device, log)) {
        std::cerr << "ERRORE: " << log << std::endl;
        return 1;
    }

    return 0;
}
