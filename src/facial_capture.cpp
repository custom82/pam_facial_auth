#include "../include/libfacialauth.h"
#include <getopt.h>
#include <iostream>
#include <cstring>
#include <filesystem>

namespace fs = std::filesystem;

static void usage(const char *prog)
{
    std::cout <<
    "Usage: " << prog << " [OPTIONS]\n"
    "Options:\n"
    "  -u, --user <name>            User name (REQUIRED)\n"
    "  -d, --device <dev>           Video device (/dev/video0)\n"
    "  -w, --width <px>             Frame width\n"
    "  -h, --height <px>            Frame height\n"
    "  -n, --frames <num>           Number of frames\n"
    "  -c, --config <file>          Config file path\n"
    "  -f, --force                  Overwrite images (reset index to 1)\n"
    "      --clean                  Delete ALL images for user\n"
    "      --reset                  Delete images + model\n"
    "      --list                   List user images\n"
    "      --format <ext>           Image format (png,jpg) [default: jpg]\n"
    "      --debug                  Enable verbose debug\n"
    "      --help                   Show this help\n";
}

int main(int argc, char *argv[])
{
    FacialAuthConfig cfg;

    std::string config_path = "/etc/pam_facial_auth/pam_facial.conf";
    std::string user;
    std::string img_format = "jpg";
    bool force = false;
    bool clean = false;
    bool reset = false;
    bool list_images = false;

    bool debug_cli = false;

    int width = 0, height = 0, frames = 0;

    static struct option long_opts[] = {
        {"user",        required_argument, nullptr, 'u'},
        {"device",      required_argument, nullptr, 'd'},
        {"width",       required_argument, nullptr, 'w'},
        {"height",      required_argument, nullptr, 'h'},
        {"frames",      required_argument, nullptr, 'n'},
        {"config",      required_argument, nullptr, 'c'},
        {"force",       no_argument,       nullptr, 'f'},
        {"format",      required_argument, nullptr,  1 },
        {"debug",       no_argument,       nullptr,  2 },
        {"clean",       no_argument,       nullptr,  3 },
        {"reset",       no_argument,       nullptr,  4 },
        {"list",        no_argument,       nullptr,  5 },
        {"help",        no_argument,       nullptr,  6 },
        {nullptr,       0,                 nullptr,  0 }
    };

    int opt, idx;
    while ((opt = getopt_long(argc, argv, "u:d:w:h:n:c:f", long_opts, &idx)) != -1)
    {
        switch (opt)
        {
            case 'u': user = optarg; break;
            case 'd': cfg.device = optarg; break;
            case 'w': width = std::stoi(optarg); break;
            case 'h': height = std::stoi(optarg); break;
            case 'n': frames = std::stoi(optarg); break;
            case 'c': config_path = optarg; break;
            case 'f': force = true; cfg.force_overwrite = true; break;

            case 1: img_format = optarg; break;

            case 2:
                debug_cli = true;
                break;

            case 3: clean = true; break;
            case 4: reset = true; break;
            case 5: list_images = true; break;
            case 6: usage(argv[0]); return 0;

            default:
                usage(argv[0]);
                return 1;
        }
    }

    if (user.empty()) {
        std::cerr << "[ERROR] Missing --user\n";
        usage(argv[0]);
        return 1;
    }

    std::string log;
    fa_load_config(config_path, cfg, log);

    // ======================================================
    //  CLI DEBUG OVERRIDE (sempre dopo il load del config)
    // ======================================================
    if (debug_cli) {
        cfg.debug = true;
        std::cout << "[DEBUG] Debug mode FORZATO da CLI (--debug)\n";
    }

    if (width > 0)  cfg.width = width;
    if (height > 0) cfg.height = height;
    if (frames > 0) cfg.frames = frames;

    // =============== LIST IMAGES ==========================
    if (list_images) {
        fa_list_images(cfg, user);
        return 0;
    }

    // =============== CLEAN ================================
    if (clean) {
        std::cout << "[INFO] Removing all images for user " << user << "\n";
        fa_clean_images(cfg, user);
        return 0;
    }

    // =============== RESET ================================
    if (reset) {
        std::cout << "[INFO] Reset user data (images + model)\n";
        fa_clean_images(cfg, user);
        fa_clean_model(cfg, user);
        return 0;
    }

    // =============== FORCE FIX =============================
    if (force) {
        std::cout << "[INFO] FORCE enabled: cleaning images before capture\n";
        fa_clean_images(cfg, user);
    }

    // =======================================================
    // CAPTURE
    // =======================================================
    std::cout << "[INFO] Starting capture for user: " << user << "\n";

    if (!fa_capture_images(user, cfg, force, log, img_format)) {
        std::cerr << "[ERROR] Capture failed\n";
        return 1;
    }

    std::cout << "[INFO] Capture completed\n";
    return 0;
}
