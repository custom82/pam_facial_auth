#include "../include/libfacialauth.h"
#include <iostream>
#include <getopt.h>

void print_usage(const char* prog) {
    std::cout << "Uso: " << prog << " -u <utente> [OPZIONI]\n"
    << "  -u, --user <nome>        Utente per la cattura\n"
    << "  -D, --detector <tipo>    Detector: 'yunet', 'haar', 'none' (default)\n"
    << "  -f, --force              Svuota samples esistenti\n"
    << "  -h, --help               Mostra questo aiuto\n";
}

int main(int argc, char** argv) {
    FacialAuthConfig cfg;
    std::string user, detector = "none", log;

    static struct option long_opts[] = {
        {"user", 1, 0, 'u'}, {"detector", 1, 0, 'D'}, {"force", 0, 0, 'f'}, {"help", 0, 0, 'h'}, {0,0,0,0}
    };

    int opt;
    while ((opt = getopt_long(argc, argv, "u:D:fh", long_opts, nullptr)) != -1) {
        switch (opt) {
            case 'u': user = optarg; break;
            case 'D': detector = optarg; break;
            case 'f': cfg.force = true; break;
            case 'h': print_usage(argv[0]); return 0;
            default: return 1;
        }
    }

    if (user.empty()) { print_usage(argv[0]); return 1; }

    fa_load_config(cfg, log, FACIALAUTH_DEFAULT_CONFIG);

    std::cout << "[INFO] Avvio cattura per: " << user << " (Detector: " << detector << ")\n";
    if (!fa_capture_user(user, cfg, detector, log)) {
        std::cerr << "[ERRORE] " << log << "\n";
        return 1;
    }

    std::cout << "[OK] Completato.\n";
    return 0;
}
