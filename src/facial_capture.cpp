#include "../include/libfacialauth.h"
#include <getopt.h>
#include <iostream>

int main(int argc, char **argv) {
    std::string user;
    std::string device;
    int width = 640, height = 480, frames = 5;
    bool force = false, nogui = false, verbose = false;
    int sleep = 500;
    std::string config = "/etc/security/pam_facial.conf";

    FacialAuthConfig cfg;
    static struct option long_opts[] = {
        {"user", required_argument, 0, 'u'},
        {"device", required_argument, 0, 'd'},
        {"width", required_argument, 0, 'w'},
        {"height", required_argument, 0, 'h'},
        {"frames", required_argument, 0, 'n'},
        {"sleep", required_argument, 0, 's'},
        {"force", no_argument, 0, 'f'},
        {"nogui", no_argument, 0, 'g'},
        {"verbose", no_argument, 0, 'v'},
        {"config", required_argument, 0, 'c'},
        {0,0,0,0}
    };
    int opt, idx;
    while ((opt = getopt_long(argc, argv, "u:d:w:h:n:s:fgvc:", long_opts, &idx)) != -1) {
        switch (opt) {
            case 'u': user = optarg; break;
            case 'd': device = optarg; break;
            case 'w': width = atoi(optarg); break;
            case 'h': height = atoi(optarg); break;
            case 'n': frames = atoi(optarg); break;
            case 's': sleep = atoi(optarg); break;
            case 'f': force = true; break;
            case 'g': nogui = true; break;
            case 'v': verbose = true; break;
            case 'c': config = optarg; break;
        }
    }

    read_kv_config(config, cfg);
    if (!device.empty()) cfg.device = device;
    cfg.width = width;
    cfg.height = height;
    cfg.frames = frames;
    cfg.sleep_ms = sleep;
    cfg.nogui = nogui;
    if (verbose) cfg.debug = true;

    std::string log;
    bool ok = fa_capture_images(user, cfg, force, log);
    std::cerr << log;
    return ok ? 0 : 1;
}
