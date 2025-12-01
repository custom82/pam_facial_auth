#include "../include/libfacialauth.h"

#include <iostream>
#include <getopt.h>

static const char *DEFAULT_CONFIG_PATH = "/etc/security/pam_facial.conf";

int facial_test_cli_main(int argc, char *argv[])
{
    std::string user;
    std::string cfg_path = DEFAULT_CONFIG_PATH;
    std::string recognizer_cli;
    std::string method_cli;
    double threshold_override = -1.0;
    bool debug_cli = false;

    static struct option long_opts[] = {
        {"user",       required_argument, nullptr, 'u'},
        {"config",     required_argument, nullptr, 'c'},
        {"recognizer", required_argument, nullptr, 'r'},
        {"method",     required_argument, nullptr, 'm'},
        {"threshold",  required_argument, nullptr, 't'},
        {"debug",      no_argument,       nullptr, 'g'},
        {nullptr,      0,                 nullptr,  0}
    };

    int opt, idx;
    while ((opt = getopt_long(argc, argv, "u:c:r:m:t:g", long_opts, &idx)) != -1) {
        switch (opt) {
            case 'u': user = optarg; break;
            case 'c': cfg_path = optarg; break;
            case 'r': recognizer_cli = optarg; break;
            case 'm': method_cli = optarg; break;
            case 't': threshold_override = std::atof(optarg); break;
            case 'g': debug_cli = true; break;
            default:
                std::cerr << "Usage: facial_test -u USER [--config FILE] "
                << "[--recognizer PROFILE] [--method auto|lbph|eigen|fisher|sface] "
                << "[--threshold X] [--debug]\n";
                return 1;
        }
    }

    if (user.empty()) {
        std::cerr << "facial_test: --user is required\n";
        return 1;
    }

    FacialAuthConfig cfg;
    std::string log;
    if (!fa_load_config(cfg, log, cfg_path)) {
        std::cerr << log;
        return 1;
    }

    if (!recognizer_cli.empty())
        cfg.recognizer_profile = recognizer_cli;
    if (!method_cli.empty())
        cfg.training_method = method_cli;
    if (debug_cli)
        cfg.debug = true;

    std::string model_path = fa_user_model_path(cfg, user);
    double best_conf = 0.0;
    int best_label = -1;
    bool ok = fa_test_user(user, cfg, model_path, best_conf, best_label, log, threshold_override);

    std::cout << log;
    if (ok) {
        std::cout << "Authentication OK, label=" << best_label
        << " score=" << best_conf << "\n";
        return 0;
    } else {
        std::cout << "Authentication FAILED\n";
        return 1;
    }
}
