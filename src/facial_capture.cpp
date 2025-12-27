#include "../include/libfacialauth.h"
#include <iostream>
#include <getopt.h>

int main(int argc, char** argv) {
    FacialAuthConfig cfg;
    std::string user, log;

    static struct option long_opts[] = {
        {"user", 1, 0, 'u'}, {"number", 1, 0, 'n'}, {"width", 1, 0, 'w'},
        {"height", 1, 0, 'H'}, {"format", 1, 0, 'F'}, {"force", 0, 0, 'f'},
        {"nogui", 0, 0, 'g'}, {"help", 0, 0, 'h'}, {0,0,0,0}
    };

    int opt;
    while ((opt = getopt_long(argc, argv, "u:n:w:H:F:fgh", long_opts, NULL)) != -1) {
        switch (opt) {
            case 'u': user = optarg; break;
            case 'n': cfg.frames = std::stoi(optarg); break;
            case 'w': cfg.width = std::stoi(optarg); break;
            case 'H': cfg.height = std::stoi(optarg); break;
            case 'F': cfg.image_format = optarg; break;
            case 'f': cfg.force = true; break;
            case 'g': cfg.nogui = true; break;
            case 'h': std::cout << "Usage: facial_capture -u <user> [options]\n"; return 0;
        }
    }

    if (user.empty() || !fa_check_root("facial_capture")) return 1;
    fa_load_config(cfg, log, FACIALAUTH_DEFAULT_CONFIG);

    std::string det = (cfg.training_method == "sface" || cfg.training_method == "auto") ? "yunet" : "none";
    if (fa_capture_user(user, cfg, det, log)) {
        std::cout << "[SUCCESS] Capture completed.\n";
    } else return 1;

    return 0;
}
