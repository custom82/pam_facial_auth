#include "../include/libfacialauth.h"
#include <iostream>
#include <getopt.h>

int main(int argc, char** argv) {
    FacialAuthConfig cfg;
    std::string user, detector = "none", log;
    int opt;
    while ((opt = getopt(argc, argv, "u:D:f")) != -1) {
        switch (opt) {
            case 'u': user = optarg; break;
            case 'D': detector = optarg; break;
            case 'f': cfg.force = true; break;
        }
    }
    if (user.empty()) return 1;
    fa_load_config(cfg, log, FACIALAUTH_DEFAULT_CONFIG);
    return fa_capture_user(user, cfg, detector, log) ? 0 : 1;
}
