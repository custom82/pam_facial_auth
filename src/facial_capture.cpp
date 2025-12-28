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
    << "  -u, --user <name>       Utente obbligatorio\n"
    << "  -c, --config <file>     Percorso config\n"
    << "  -d, --device <path>     Video device\n"
    << "  -w, --width <px>        Larghezza frame\n"
    << "  -h, --height <px>       Altezza frame\n"
    << "  -n, --num_images <n>    Immagini da scattare\n"
    << "  -s, --sleep <sec>       Pausa tra scatti\n"
    << "  --detector <name>       yunet, cascade, none\n"
    << "  -f, --force             Pulisce la cartella prima della cattura\n"
    << "  --clean, --flush        Elimina le catture dell'utente ed esce\n"
    << "  --debug                 Modalità debug\n"
    << "  --nogui                 Niente anteprima video\n";
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
        else if (args[i] == "-d" || args[i] == "--device") cfg.device = args[++i];
        else if (args[i] == "-w" || args[i] == "--width") cfg.width = std::stoi(args[++i]);
        else if (args[i] == "-h" || args[i] == "--height") cfg.height = std::stoi(args[++i]);
        else if (args[i] == "-n" || args[i] == "--num_images") cfg.frames = std::stoi(args[++i]);
        else if (args[i] == "-s" || args[i] == "--sleep") cfg.capture_delay = std::stod(args[++i]);
        else if (args[i] == "--detector") cfg.detector = args[++i];
        else if (args[i] == "-f" || args[i] == "--force") force = true;
        else if (args[i] == "--clean" || args[i] == "--flush") clean_only = true;
        else if (args[i] == "--debug") cfg.debug = true;
        else if (args[i] == "--nogui") cfg.nogui = true;
        else if (args[i] == "--help") { usage(); return 0; }
    }

    if (user.empty()) { usage(); return 1; }

    if (clean_only || force) {
        fa_clean_captures(user, cfg, log);
        if (clean_only) { std::cout << "[*] Dati utente " << user << " rimossi." << std::endl; return 0; }
    }

    if (!fa_capture_user(user, cfg, cfg.device, log)) {
        std::cerr << "[ERRORE] " << log << std::endl;
        return 1;
    }

    return 0;
}
