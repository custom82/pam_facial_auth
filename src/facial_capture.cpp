#include "../include/libfacialauth.h"
#include <iostream>
#include <getopt.h>

void print_usage(const char* p) {
    std::cout << "Utilizzo: " << p << " [OPZIONI]\n"
    << "  -u, --user <nome>      Utente target\n"
    << "  -D, --detector <tipo>  Detector: yunet, haar, none (default: none)\n"
    << "  -n, --number <num>     Numero di frame (default: 50)\n"
    << "  -w, --width <px>       Larghezza (default: 640)\n"
    << "  -H, --height <px>      Altezza (default: 480)\n"
    << "  -c, --clean            Cancella dati utente\n"
    << "  -f, --force            Svuota captures prima\n"
    << "  -g, --nogui            Nasconde finestra video\n"
    << "  -d, --debug            Messaggi di log estesi\n"
    << "  -h, --help             Aiuto\n";
}

int main(int argc, char** argv) {
    FacialAuthConfig cfg;
    std::string user, det = "none", log;
    bool clean_only = false;

    static struct option long_opts[] = {
        {"user", 1, 0, 'u'}, {"detector", 1, 0, 'D'}, {"number", 1, 0, 'n'},
        {"width", 1, 0, 'w'}, {"height", 1, 0, 'H'}, {"clean", 0, 0, 'c'},
        {"force", 0, 0, 'f'}, {"nogui", 0, 0, 'g'}, {"debug", 0, 0, 'd'},
        {"help", 0, 0, 'h'}, {0,0,0,0}
    };

    int opt;
    while ((opt = getopt_long(argc, argv, "u:D:n:w:H:cfgdh", long_opts, NULL)) != -1) {
        switch (opt) {
            case 'u': user = optarg; break;
            case 'D': det = optarg; break;
            case 'n': cfg.frames = std::stoi(optarg); break;
            case 'w': cfg.width = std::stoi(optarg); break;
            case 'H': cfg.height = std::stoi(optarg); break;
            case 'c': clean_only = true; break;
            case 'f': cfg.force = true; break;
            case 'g': cfg.nogui = true; break;
            case 'd': cfg.debug = true; break;
            case 'h': print_usage(argv[0]); return 0;
            default: return 1;
        }
    }

    if (user.empty()) { print_usage(argv[0]); return 1; }
    if (!fa_check_root(argv[0])) return 1;

    if (clean_only) {
        std::cout << "Pulizia utente: " << user << "\n";
        return fa_delete_user_data(user, cfg) ? 0 : 1;
    }

    fa_load_config(cfg, log, FACIALAUTH_DEFAULT_CONFIG);
    if (!fa_capture_user(user, cfg, det, log)) {
        std::cerr << "Errore: " << log << "\n";
        return 1;
    }

    std::cout << "Completato.\n";
    return 0;
}
