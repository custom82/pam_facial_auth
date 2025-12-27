#include "../include/libfacialauth.h"
#include <iostream>
#include <getopt.h>

int main(int argc, char** argv) {
    FacialAuthConfig cfg;
    std::string user, detector = "none", log;

    static struct option long_opts[] = {
        {"user", 1, 0, 'u'}, {"detector", 1, 0, 'D'}, {"force", 0, 0, 'f'}, {"nogui", 0, 0, 'G'}, {0,0,0,0}
    };

    int opt;
    while ((opt = getopt_long(argc, argv, "u:D:fG", long_opts, nullptr)) != -1) {
        switch (opt) {
            case 'u': user = optarg; break;
            case 'D': detector = optarg; break;
            case 'f': cfg.force = true; break;
            case 'G': cfg.nogui = true; break;
        }
    }

    if (user.empty()) { std::cerr << "Uso: facial_capture -u <utente> [--detector <haar|yunet>]\n"; return 1; }

    fa_load_config(cfg, log, FACIALAUTH_DEFAULT_CONFIG);
    if (!fa_capture_user(user, cfg, detector, log)) {
        std::cerr << "Errore: " << log << "\n";
        return 1;
    }
    return 0;
}
