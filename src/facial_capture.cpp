#include "../include/libfacialauth.h"

#include <getopt.h>
#include <iostream>
#include <filesystem>

namespace fs = std::filesystem;

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
    "      --debug                 Enable debug logging\n"
    "  -h, --help                  Show this help and exit\n";
}

int main(int argc, char *argv[])
{
    FacialAuthConfig cfg;

    std::string user;
    std::string config_path = FACIALAUTH_CONFIG_DEFAULT;
    std::string img_format = "png";
    std::string log;

    bool force = false;

    static struct option long_opts[] = {
        {"user",     required_argument, nullptr, 'u'},
        {"config",   required_argument, nullptr, 'c'},
        {"force",    no_argument,       nullptr, 'f'},
        {"format",   required_argument, nullptr,  1 },
        {"frames",   required_argument, nullptr,  2 },
        {"device",   required_argument, nullptr,  3 },
        {"debug",    no_argument,       nullptr,  4 },
        {"help",     no_argument,       nullptr, 'h'},
        {nullptr,    0,                 nullptr,  0 }
    };

    int opt, idx;
    while ((opt = getopt_long(argc, argv, "u:c:fh", long_opts, &idx)) != -1) {
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

            case 'h':
                usage(argv[0]);
                return 0; // terminate cleanly

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

    // Load configuration
    fa_load_config(config_path, cfg, log);

    // Capture images
    if (!fa_capture_images(user, cfg, force, log, img_format)) {
        std::cerr << "[ERROR] facial_capture failed for user " << user << "\n";
        return 1;
    }

    std::cout << "[INFO] Image capture complete for user " << user << "\n";
    return 0;
}
