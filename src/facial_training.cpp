#include "../include/libfacialauth.h"

#include <iostream>
#include <getopt.h>

static const char *DEFAULT_CONFIG_PATH = "/etc/security/pam_facial.conf";

int facial_training_cli_main(int argc, char *argv[])
{
    std::string user;
    std::string cfg_path = DEFAULT_CONFIG_PATH;
    std::string training_method_cli;
    std::string recognizer_cli;
    bool force = false;
    bool debug_cli = false;

    static struct option long_opts[] = {
        {"user",       required_argument, nullptr, 'u'},
        {"config",     required_argument, nullptr, 'c'},
        {"method",     required_argument, nullptr, 'm'},
        {"recognizer", required_argument, nullptr, 'r'},
        {"force",      no_argument,       nullptr, 'f'},
        {"debug",      no_argument,       nullptr, 'g'},
        {nullptr,      0,                 nullptr,  0}
    };

    int opt, idx;
    while ((opt = getopt_long(argc, argv, "u:c:m:r:fg", long_opts, &idx)) != -1) {
        switch (opt) {
            case 'u': user = optarg; break;
            case 'c': cfg_path = optarg; break;
            case 'm': training_method_cli = optarg; break;
            case 'r': recognizer_cli = optarg; break;
            case 'f': force = true; break;
            case 'g': debug_cli = true; break;
            default:
                std::cerr << "Usage: facial_training -u USER [--config FILE] "
                << "[--method auto|lbph|eigen|fisher|sface] "
                << "[--recognizer PROFILE] [--force] [--debug]\n";
                return 1;
        }
    }

    if (user.empty()) {
        std::cerr << "facial_training: --user is required\n";
        return 1;
    }

    FacialAuthConfig cfg;
    std::string log;
    if (!fa_load_config(cfg, log, cfg_path)) {
        std::cerr << log;
        return 1;
    }

    if (!training_method_cli.empty())
        cfg.training_method = training_method_cli;
    if (!recognizer_cli.empty())
        cfg.recognizer_profile = recognizer_cli;
    if (force)
        cfg.force_overwrite = true;
    if (debug_cli)
        cfg.debug = true;

    bool ok = fa_train_user(user, cfg, log);

    std::cout << log;
    return ok ? 0 : 1;
}
