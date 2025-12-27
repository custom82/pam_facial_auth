#include "../include/libfacialauth.h"
#include <iostream>
#include <getopt.h>

int main(int argc, char** argv) {
    FacialAuthConfig cfg;
    std::string user, log;
    static struct option long_opts[] = {
        {"user", 1, 0, 'u'}, {"help", 0, 0, 'h'}, {0,0,0,0}
    };

    int opt;
    while ((opt = getopt_long(argc, argv, "u:h", long_opts, NULL)) != -1) {
        if (opt == 'u') user = optarg;
        else { std::cout << "Usage: " << argv[0] << " -u <user>\n"; return 0; }
    }

    if (user.empty()) return 1;
    fa_load_config(cfg, log, FACIALAUTH_DEFAULT_CONFIG);
    std::cout << "Training utente: " << user << "...\n";
    return fa_train_user(user, cfg, log) ? 0 : 1;
}
