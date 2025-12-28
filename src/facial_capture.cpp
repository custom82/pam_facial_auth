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
    << "  -u, --user <name>       Nome utente per cui salvare le immagini\n"
    << "  -c, --config <file>     File di configurazione (default: /etc/security/pam_facial_auth.conf)\n"
    << "  -d, --device <path>     Device della webcam (es: /dev/video0)\n"
    << "  -w, --width <px>        Larghezza frame\n"
    << "  -h, --height <px>       Altezza frame\n"
    << "  -f, --force             Sovrascrive immagini esistenti e riparte da 1\n"
    << "  --flush, --clean        Elimina tutte le immagini per l'utente specificato\n"
    << "  -n, --num_images <num>  Numero di immagini da acquisire\n"
    << "  -s, --sleep <sec>       Pausa tra una cattura e l'altra (in secondi)\n"
    << "  --detector <name>       Detector da usare: yunet, none\n"
    << "  -v, --verbose           Output dettagliato\n"
    << "  --debug                 Abilita output di debug\n"
    << "  --nogui                 Disabilita GUI, cattura solo da console\n"
    << "  --help, -H              Mostra questo messaggio\n";
}

int main(int argc, char** argv) {
    if (!fa_check_root("facial_capture")) return 1;

    std::string user;
    std::string config_path = "/etc/security/pam_facial_auth.conf";
    FacialAuthConfig cfg;
    std::string log;
    bool force_restart = false;
    bool clean_only = false;

    std::vector<std::string> args(argv + 1, argv + argc);
    if (args.empty()) { usage(); return 1; }

    // Primo passaggio per individuare il file di configurazione
    for (size_t i = 0; i < args.size(); ++i) {
        if ((args[i] == "-c" || args[i] == "--config") && i + 1 < args.size()) {
            config_path = args[++i];
        }
    }

    // Carica configurazione base
    fa_load_config(cfg, log, config_path);

    // Secondo passaggio per i parametri CLI
    for (size_t i = 0; i < args.size(); ++i) {
        if (args[i] == "--help" || args[i] == "-H") {
            usage();
            return 0;
        }
        else if ((args[i] == "-u" || args[i] == "--user") && i + 1 < args.size()) {
            user = args[++i];
        }
        else if ((args[i] == "-d" || args[i] == "--device") && i + 1 < args.size()) {
            cfg.device = args[++i];
        }
        else if ((args[i] == "-w" || args[i] == "--width") && i + 1 < args.size()) {
            cfg.width = std::stoi(args[++i]);
        }
        else if ((args[i] == "-h" || args[i] == "--height") && i + 1 < args.size()) {
            cfg.height = std::stoi(args[++i]);
        }
        else if (args[i] == "-n" || args[i] == "--num_images") {
            cfg.frames = std::stoi(args[++i]);
        }
        else if (args[i] == "-s" || args[i] == "--sleep") {
            cfg.capture_delay = std::stod(args[++i]);
        }
        else if (args[i] == "--detector" && i + 1 < args.size()) {
            cfg.detector = args[++i];
        }
        else if (args[i] == "-f" || args[i] == "--force") {
            force_restart = true;
        }
        else if (args[i] == "--flush" || args[i] == "--clean") {
            clean_only = true;
        }
        else if (args[i] == "-v" || args[i] == "--verbose") {
            cfg.verbose = true;
        }
        else if (args[i] == "--debug") {
            cfg.debug = true;
        }
        else if (args[i] == "--nogui") {
            cfg.nogui = true;
        }
    }

    if (user.empty()) {
        std::cerr << "[ERRORE] Utente non specificato. Usa -u <nome>" << std::endl;
        return 1;
    }

    // Gestione --flush o --clean
    if (clean_only || force_restart) {
        if (cfg.verbose || cfg.debug) std::cout << "[INFO] Pulizia immagini per: " << user << std::endl;
        fa_clean_captures(user, cfg, log);
        if (clean_only) return 0;
    }

    // Avvio cattura
    if (cfg.verbose || cfg.debug) {
        std::cout << "[INFO] Inizio acquisizione per: " << user << std::endl;
    }

    if (!fa_capture_user(user, cfg, cfg.device, log)) {
        std::cerr << "[ERRORE] " << log << std::endl;
        return 1;
    }

    std::cout << "[SUCCESS] Sessione completata." << std::endl;
    return 0;
}
