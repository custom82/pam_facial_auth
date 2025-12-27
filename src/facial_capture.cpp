#include "../include/libfacialauth.h"
#include <iostream>
#include <getopt.h>

void print_usage(const char* prog_name) {
    std::cout << "Utilizzo: " << prog_name << " [OPZIONI]\n\n"
    << "Opzioni obbligatorie:\n"
    << "  -u, --user <nome>        Specifica l'utente per la cattura\n\n"
    << "Opzioni facoltative:\n"
    << "  -D, --detector <tipo>    Tipo di detector: 'none' (default), 'haar', 'yunet'\n"
    << "  -d, --device <id/path>   ID camera (0, 1...) o path stream/file\n"
    << "  -f, --force              Svuota la cartella dei sample prima di iniziare\n"
    << "  -G, --nogui              Esegue la cattura senza mostrare la finestra video\n"
    << "  -c, --config <path>      Path alternativo per pam_facial.conf\n"
    << "  -h, --help               Mostra questo messaggio di aiuto\n\n"
    << "Esempi:\n"
    << "  " << prog_name << " -u mario --detector yunet\n"
    << "  " << prog_name << " -u luigi -d /dev/video2 --force\n";
}

int main(int argc, char** argv) {
    FacialAuthConfig cfg;
    std::string user, detector = "none", config_path = FACIALAUTH_DEFAULT_CONFIG;
    std::string log;

    static struct option long_opts[] = {
        {"user",     required_argument, 0, 'u'},
        {"detector", required_argument, 0, 'D'},
        {"device",   required_argument, 0, 'd'},
        {"config",   required_argument, 0, 'c'},
        {"force",    no_argument,       0, 'f'},
        {"nogui",    no_argument,       0, 'G'},
        {"help",     no_argument,       0, 'h'},
        {0, 0, 0, 0}
    };

    int opt;
    while ((opt = getopt_long(argc, argv, "u:D:d:c:fGh", long_opts, nullptr)) != -1) {
        switch (opt) {
            case 'u': user = optarg; break;
            case 'D': detector = optarg; break;
            case 'd': cfg.device = optarg; break;
            case 'c': config_path = optarg; break;
            case 'f': cfg.force = true; break;
            case 'G': cfg.nogui = true; break;
            case 'h': print_usage(argv[0]); return 0;
            default:  print_usage(argv[0]); return 1;
        }
    }

    if (user.empty()) {
        std::cerr << "Errore: l'opzione --user è obbligatoria.\n";
        print_usage(argv[0]);
        return 1;
    }

    // Carichiamo la config (che sovrascriverà i default, ma non quelli passati da CLI)
    fa_load_config(cfg, log, config_path);

    // Riapplichiamo l'id device se passato da CLI (getopt ha priorità su config file)
    // Nota: in un'architettura a wrapper ideale, fa_load_config non dovrebbe
    // sovrascrivere valori già impostati manualmente.

    std::cout << "[INFO] Avvio cattura per l'utente '" << user << "' utilizzando detector: " << detector << "\n";

    if (!fa_capture_user(user, cfg, detector, log)) {
        std::cerr << "[ERRORE] " << log << "\n";
        return 1;
    }

    std::cout << "[OK] Cattura completata con successo.\n";
    return 0;
}
