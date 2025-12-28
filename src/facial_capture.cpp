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
    << "  -u, --user <name>       Nome utente per cui salvare le immagini\n"
    << "  -c, --config <file>     File di configurazione (default: /etc/pam_facial_auth/pam_facial.conf)\n"
    << "  -d, --device <path>     Device della webcam (es: /dev/video0)\n"
    << "  -w, --width <px>        Larghezza frame\n"
    << "  -h, --height <px>       Altezza frame\n"
    << "  -f, --force             Sovrascrive immagini esistenti e riparte da 1\n"
    << "  --flush, --clean        Elimina tutte le immagini per l'utente specificato\n"
    << "  -n, --num_images <num>  Numero di immagini da acquisire\n"
    << "  -s, --sleep <sec>       Pausa tra una cattura e l'altra (in secondi)\n"
    << "  -v, --verbose           Output dettagliato\n"
    << "  --debug                 Abilita output di debug\n"
    << "  --nogui                 Disabilita GUI, cattura solo da console\n"
    << "  --help, -H              Mostra questo messaggio\n"
    << "  --detector              yunet or haar\n";
}

int main(int argc, char** argv) {
    if (!fa_check_root("facial_capture")) return 1;

    std::string user, config_path = "/etc/pam_facial_auth/pam_facial.conf", log;
    FacialAuthConfig cfg;
    bool force = false, clean_only = false;

    std::vector<std::string> args(argv + 1, argv + argc);
    for (size_t i = 0; i < args.size(); ++i) {
        if ((args[i] == "-c" || args[i] == "--config") && i + 1 < args.size()) config_path = args[++i];
    }
    fa_load_config(cfg, log, config_path);

    for (size_t i = 0; i < args.size(); ++i) {
        if (args[i] == "--help" || args[i] == "-H") {
            usage();
            return 0;
        } else if ((args[i] == "-u" || args[i] == "--user") && i + 1 < args.size()) {
            user = args[++i];
        } else if ((args[i] == "-d" || args[i] == "--device") && i + 1 < args.size()) {
            cfg.device = args[++i];
        } else if ((args[i] == "-w" || args[i] == "--width") && i + 1 < args.size()) {
            cfg.width = std::stoi(args[++i]);
        } else if ((args[i] == "-h" || args[i] == "--height") && i + 1 < args.size()) {
            cfg.height = std::stoi(args[++i]);
        } else if ((args[i] == "-n" || args[i] == "--num_images") && i + 1 < args.size()) {
            cfg.frames = std::stoi(args[++i]);
        } else if ((args[i] == "-s" || args[i] == "--sleep") && i + 1 < args.size()) {
            cfg.sleep_ms = std::stoi(args[++i]) * 1000;
        } else if (args[i] == "--detector" && i + 1 < args.size()) {
            cfg.detector = args[++i];
            if (cfg.detector == "haar") {
                cfg.detector = "cascade";
            }
        } else if (args[i] == "--format" && i + 1 < args.size()) {
            cfg.image_format = args[++i];
        } else if (args[i] == "-f" || args[i] == "--force") {
            force = true;
        } else if (args[i] == "--clean" || args[i] == "--flush") {
            clean_only = true;
        } else if (args[i] == "--nogui") {
            cfg.nogui = true;
        } else if (args[i] == "-v" || args[i] == "--verbose") {
            cfg.verbose = true;
        } else if (args[i] == "--debug") {
            cfg.debug = true;
        }
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
