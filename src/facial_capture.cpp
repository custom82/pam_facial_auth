#include "../include/libfacialauth.h"

#include <iostream>
#include <getopt.h>

static const char *DEFAULT_CONFIG_PATH = "/etc/security/pam_facial.conf";

int facial_capture_main(int argc, char *argv[])
{
    std::string user;
    std::string cfg_path = DEFAULT_CONFIG_PATH;
    std::string fmt;
    std::string detector_cli;
    std::string device_cli;
    int width_cli  = -1;
    int height_cli = -1;
    int frames_cli = -1;
    bool debug_cli = false;

    static struct option long_opts[] = {
        {"user",     required_argument, nullptr, 'u'},
        {"config",   required_argument, nullptr, 'c'},
        {"format",   required_argument, nullptr, 'f'},
        {"detector", required_argument, nullptr, 'D'},
        {"device",   required_argument, nullptr, 'd'},
        {"width",    required_argument, nullptr, 'w'},
        {"height",   required_argument, nullptr, 'h'},
        {"frames",   required_argument, nullptr, 'n'},
        {"debug",    no_argument,       nullptr, 'g'},
        {nullptr,    0,                 nullptr,  0 }
    };

    int opt, idx;
    while ((opt = getopt_long(argc, argv, "u:c:f:D:d:w:h:n:g", long_opts, &idx)) != -1) {
        switch (opt) {
            case 'u': user = optarg; break;
            case 'c': cfg_path = optarg; break;
            case 'f': fmt = optarg; break;
            case 'D': detector_cli = optarg; break;
            case 'd': device_cli = optarg; break;
            case 'w': width_cli = std::atoi(optarg); break;
            case 'h': height_cli = std::atoi(optarg); break;
            case 'n': frames_cli = std::atoi(optarg); break;
            case 'g': debug_cli = true; break;
            default:
                std::cerr << "Usage: facial_capture -u USER [--config FILE] [--detector PROFILE] "
                << "[-d DEVICE] [-w WIDTH] [-h HEIGHT] [-n FRAMES] [--format fmt] [--debug]\n";
                return 1;
        }
    }

    if (user.empty()) {
        std::cerr << "facial_capture: --user is required\n";
        return 1;
    }

    FacialAuthConfig cfg;
    std::string log;
    if (!fa_load_config(cfg, log, cfg_path)) {
        std::cerr << log;
        return 1;
    }

    if (!device_cli.empty())
        cfg.device = device_cli;
    if (width_cli > 0)
        cfg.width = width_cli;
    if (height_cli > 0)
        cfg.height = height_cli;
    if (frames_cli > 0)
        cfg.frames = frames_cli;
    if (!detector_cli.empty())
        cfg.detector_profile = detector_cli;
    if (debug_cli)
        cfg.debug = true;

    bool ok = fa_capture_images(user, cfg, fmt, log);

    std::cout << log;
    return ok ? 0 : 1;
}
