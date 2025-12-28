/*
 * Project: pam_facial_auth
 * License: GPL-3.0
 */

#include "libfacialauth.h"
#include <iostream>
#include <vector>

void usage() {
    std::cout << "Usage: facial_capture -u <user> [options]\n\n"
    << "Options:\n"
    << "  -u, --user <name>       Utente\n"
    << "  -c, --config <file>     Config (default: /etc/security/pam_facial_auth.conf)\n"
    << "  -n, --num_images <n>    Numero immagini\n"
    << "  --detector <name>       yunet, cascade, none\n"
    << "  --format <ext>          jpg, png, bmp\n"
    << "  -f, --force             Pulisce prima di iniziare\n"
    << "  --clean                 Elimina catture utente ed esce\n"
    << "  --nogui                 Niente finestra video (evita crash display)\n"
    << "  --debug                 Log dettagliati\n";
}

int main(int argc, char** argv) {
    if (!fa_check_root("facial_capture")) return 1;

    std::string user, config_path = "/etc/security/pam_facial_auth.conf", log;
    FacialAuthConfig cfg;
    bool force = false, clean_only = false;

    std::vector<std::string> args(argv + 1, argv + argc);
    for (size_t i = 0; i < args.size(); ++i) {
        if ((args[i] == "-c" || args[i] == "--config") && i + 1 < args.size()) config_path = args[++i];
    }
    fa_load_config(cfg, log, config_path);

    for (size_t i = 0; i < args.size(); ++i) {
        if (args[i] == "-u" || args[i] == "--user") user = args[++i];
        else if (args[i] == "-n" || args[i] == "--num_images") cfg.frames = std::stoi(args[++i]);
        else if (args[i] == "--detector") cfg.detector = args[++i];
        else if (args[i] == "--format") cfg.image_format = args[++i];
        else if (args[i] == "-f" || args[i] == "--force") force = true;
        else if (args[i] == "--clean") clean_only = true;
        else if (args[i] == "--nogui") cfg.nogui = true;
        else if (args[i] == "--debug") cfg.debug = true;
    }

    if (user.empty()) { usage(); return 1; }

    if (clean_only || force) fa_clean_captures(user, cfg, log);
    if (clean_only) return 0;

    if (!fa_capture_user(user, cfg, cfg.device, log)) {
        std::cerr << "[ERRORE] " << log << std::endl;
        return 1;
    }
    return 0;
}
