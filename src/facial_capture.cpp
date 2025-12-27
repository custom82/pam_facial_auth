#include "../include/libfacialauth.h"
#include <iostream>
#include <getopt.h>

void help(const char* p) {
    std::cout << "Uso: " << p << " [OPZIONI]\n"
    << "  -u, --user <nome>     Utente (obbligatorio)\n"
    << "  -D, --detector <tipo> yunet, haar, none\n"
    << "  -f, --force           Pulisce i vecchi sample\n"
    << "  -h, --help            Mostra questo aiuto\n";
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
            case 'h': help(argv[0]); return 0;
        }
    }
    if (user.empty()) { help(argv[0]); return 1; }
    fa_load_config(cfg, log, FACIALAUTH_DEFAULT_CONFIG);
    if (!fa_capture_user(user, cfg, detector, log)) return 1;
    return 0;
}
