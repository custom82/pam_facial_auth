#include "../include/libfacialauth.h"

#include <getopt.h>
#include <iostream>

static void usage(const char *prog)
{
    std::cerr <<
    "Usage: " << prog << " [options]\n"
    "  -u, --user USER             User name\n"
    "  -c, --config FILE           Config file (default " FACIALAUTH_CONFIG_DEFAULT ")\n"
    "  -f, --force                 Overwrite / append images without asking\n"
    "      --format EXT            Image format (png,jpg,...) [default: png]\n"
    "      --frames N              Number of frames to capture\n"
    "      --device DEV            Camera device (/dev/video0, etc.)\n"
    "      --debug                 Enable debug logging\n";
}

int main(int argc, char *argv[])
{
    std::string user;
    std::string config_path = FACIALAUTH_CONFIG_DEFAULT;
    std::string img_format  = "png";
    bool force = false;

    static struct option long_opts[] = {
        {"user",   required_argument, nullptr, 'u'},
        {"config", required_argument, nullptr, 'c'},
        {"force",  no_argument,       nullptr, 'f'},
        {"format", required_argument, nullptr,  1 },
        {"frames", required_argument, nullptr,  2 },
        {"device", required_argument, nullptr,  3 },
        {"debug",  no_argument,       nullptr,  4 },
        {nullptr,  0,                 nullptr,  0 }
    };

    FacialAuthConfig cfg;
    std::string log;

    int opt, idx;
    while ((opt = getopt_long(argc, argv, "u:c:f", long_opts, &idx)) != -1) {
        switch (opt) {
            case 'u':
                user = optarg;
                break;
            case 'c':
                config_path = optarg;
                break;
            case 'f':
                force = true;
                cfg.force_overwrite = true;
                break;
            case 1:
                img_format = optarg;
                break;
            case 2:
                cfg.frames = std::stoi(optarg);
                break;
            case 3:
                cfg.device = optarg;
                break;
            case 4:
                cfg.debug = true;
                break;
            default:
                usage(argv[0]);
                return 1;
        }
    }

    if (user.empty()) {
        usage(argv[0]);
        return 1;
    }

    fa_load_config(config_path, cfg, log);

    std::cout << "[INFO] Starting capture for user: " << user << "\n";

    if (!fa_capture_images(user, cfg, force, log, img_format)) {
        std::cerr << "[ERROR] Capture failed\n";
        return 1;
    }

    std::cout << "[INFO] Capture completed\n";
    return 0;
}
