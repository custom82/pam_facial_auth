#include "../include/libfacialauth.h"

#include <getopt.h>
#include <iostream>
#include <filesystem>

namespace fs = std::filesystem;

static void usage(const char *prog)
{
    std::cerr <<
    "\nUsage: " << prog << " [options]\n"
    "  -u, --user USER              User name (REQUIRED)\n"
    "  -c, --config FILE            Config file (default /etc/security/pam_facial.conf)\n"
    "  -f, --force                  Force overwrite existing images\n"
    "      --format EXT             Image format (png,jpg,...) [default: png]\n"
    "      --frames N               Number of frames to capture\n"
    "      --device DEV             Video device (/dev/video0, etc.)\n"
    "      --width W                Capture width override\n"
    "      --height H               Capture height override\n"
    "      --clean                  Delete all user images\n"
    "      --reset                  Delete images AND user model\n"
    "      --debug                  Enable debug logging\n"
    "      --help                   Show this help\n\n";
}

int main(int argc, char *argv[])
{
    FacialAuthConfig cfg;
    std::string log;

    std::string user;
    std::string config_path = FACIALAUTH_CONFIG_DEFAULT;
    std::string format = "png";

    bool force = false;
    bool clean = false;
    bool reset = false;
    int override_frames = -1;
    int override_width  = -1;
    int override_height = -1;

    static struct option long_opts[] = {
        {"user",     required_argument, nullptr, 'u'},
        {"config",   required_argument, nullptr, 'c'},
        {"force",    no_argument,       nullptr, 'f'},
        {"format",   required_argument, nullptr,  1 },
        {"frames",   required_argument, nullptr,  2 },
        {"device",   required_argument, nullptr,  3 },
        {"width",    required_argument, nullptr,  4 },
        {"height",   required_argument, nullptr,  5 },
        {"clean",    no_argument,       nullptr,  6 },
        {"reset",    no_argument,       nullptr,  7 },
        {"debug",    no_argument,       nullptr,  8 },
        {"help",     no_argument,       nullptr,  9 },
        {nullptr,    0,                 nullptr,  0 }
    };

    int opt, idx;
    while ((opt = getopt_long(argc, argv, "u:c:f", long_opts, &idx)) != -1) {
        switch (opt) {
            case 'u': user = optarg; break;
            case 'c': config_path = optarg; break;
            case 'f': force = true; cfg.force_overwrite = true; break;

            case 1: format = optarg; break;
            case 2: override_frames = std::stoi(optarg); break;
            case 3: cfg.device = optarg; break;
            case 4: override_width = std::stoi(optarg); break;
            case 5: override_height = std::stoi(optarg); break;

            case 6: clean = true; break;
            case 7: reset = true; break;
            case 8: cfg.debug = true; break;
            case 9: usage(argv[0]); return 0;

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

    // Applica override runtime
    if (override_frames > 0) cfg.frames = override_frames;
    if (override_width  > 0) cfg.width  = override_width;
    if (override_height > 0) cfg.height = override_height;

    std::string img_dir = fa_user_image_dir(cfg, user);

    // ================================
    // --clean
    // ================================
    if (clean && !reset) {
        std::cout << "[INFO] Cleaning images for user " << user << "\n";
        fa_clean_images(cfg, user);
        return 0;
    }

    // ================================
    // --reset
    // ================================
    if (reset) {
        std::cout << "[INFO] Resetting user " << user << ": images + model\n";
        fa_clean_images(cfg, user);
        fa_clean_model(cfg, user);
        return 0;
    }

    // ================================
    // Capture
    // ================================
    std::cout << "[INFO] Capturing images for user " << user << "\n";

    if (!fa_capture_images(user, cfg, force, log, format)) {
        std::cerr << "[ERROR] Capture failed.\n";
        return 1;
    }

    std::cout << "[INFO] Capture completed.\n";
    return 0;
}
