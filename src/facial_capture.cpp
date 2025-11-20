#include "../include/libfacialauth.h"
#include <getopt.h>
#include <iostream>
#include <cstring>
#include <filesystem>

namespace fs = std::filesystem;

// ==========================================================
// Remove all face images for a given user
// ==========================================================

static bool fa_clean_images(const FacialAuthConfig &cfg, const std::string &user) {
    std::string imgdir = fa_user_image_dir(cfg, user);
    if (!fs::exists(imgdir)) return true;

    try {
        for (auto &entry : fs::directory_iterator(imgdir)) {
            if (entry.is_regular_file()) {
                fs::remove(entry.path());
            }
        }
        return true;
    } catch (...) {
        return false;
    }
}

// ==========================================================
// Remove model file
// ==========================================================

static bool fa_clean_model(const FacialAuthConfig &cfg, const std::string &user) {
    std::string model = fa_user_model_path(cfg, user);
    if (!fs::exists(model)) return true;

    try {
        fs::remove(model);
        return true;
    } catch (...) {
        return false;
    }
}

// ==========================================================
// List all face images
// ==========================================================

static void fa_list_images(const FacialAuthConfig &cfg, const std::string &user) {
    std::string imgdir = fa_user_image_dir(cfg, user);

    if (!fs::exists(imgdir)) {
        std::cout << "[INFO] No images for user: " << user << "\n";
        return;
    }

    std::cout << "[INFO] Images for user " << user << ":\n";

    for (auto &entry : fs::directory_iterator(imgdir)) {
        if (entry.is_regular_file()) {
            std::cout << "  " << entry.path().filename().string() << "\n";
        }
    }
}

// ==========================================================
// MAIN
// ==========================================================

int main(int argc, char *argv[]) {
    FacialAuthConfig cfg;

    std::string config_path = "/etc/pam_facial_auth/pam_facial.conf";
    std::string user;
    bool force = false;
    std::string img_format = "jpg";

    bool clean_only = false;
    bool clean_model = false;
    bool reset_all = false;
    bool list_images = false;

    int width = -1, height = -1;

    auto print_help = []() {
        std::cout <<
        "Usage: facial_capture [OPTIONS]\n"
        "Options:\n"
        "  -u, --user <name>            User name (REQUIRED)\n"
        "  -d, --device <dev>           Video device (/dev/video0)\n"
        "  -w, --width <px>             Width\n"
        "  -h, --height <px>            Height\n"
        "  -n, --frames <num>           Number of frames\n"
        "  -c, --config <file>          Config file path\n"
        "  -f, --force                  Overwrite existing images\n"
        "  -v, --debug                  Debug logs\n"
        "  -g, --nogui                  Disable preview window\n"
        "      --clean                  Delete images\n"
        "      --clean-model            Delete model\n"
        "      --reset                  Delete images + model\n"
        "      --list                   List images\n"
        "      --format <png|jpg>       Image format\n"
        "      --help                   Show help\n";
    };

    int frames = -1;

    static struct option long_opts[] = {
        {"user",        required_argument, 0, 'u'},
        {"device",      required_argument, 0, 'd'},
        {"width",       required_argument, 0, 'w'},
        {"height",      required_argument, 0, 'h'},
        {"frames",      required_argument, 0, 'n'},
        {"config",      required_argument, 0, 'c'},
        {"force",       no_argument,       0, 'f'},
        {"debug",       no_argument,       0, 'v'},
        {"nogui",       no_argument,       0, 'g'},
        {"clean",       no_argument,       0, 1000},
        {"clean-model", no_argument,       0, 1001},
        {"reset",       no_argument,       0, 1002},
        {"list",        no_argument,       0, 1003},
        {"format",      required_argument, 0, 1005},
        {"help",        no_argument,       0, 2000},
        {0, 0, 0, 0}
    };

    int opt, idx;
    while ((opt = getopt_long(argc, argv, "u:d:w:h:n:c:fvg", long_opts, &idx)) != -1) {
        switch (opt) {
            case 'u':
                user = optarg;
                break;
            case 'd':
                cfg.device = optarg;
                break;
            case 'w':
                width = std::stoi(optarg);
                break;
            case 'h':
                height = std::stoi(optarg);
                break;
            case 'n':
                frames = std::stoi(optarg);
                break;
            case 'c':
                config_path = optarg;
                break;
            case 'f':
                force = true;
                break;
            case 'v':
                cfg.debug = true;
                break;
            case 'g':
                cfg.nogui = true;
                break;

            case 1000:
                clean_only = true;
                break;
            case 1001:
                clean_model = true;
                break;
            case 1002:
                reset_all = true;
                break;
            case 1003:
                list_images = true;
                break;

            default:
                std::cerr << "Unknown option\n";
                return 1;
        }
    }

    if (user.empty()) {
        std::cerr << "Error: --user is required\n";
        return 1;
    }

    std::string log;
    read_kv_config(config_path, cfg, &log);

    if (width > 0)  cfg.width = width;
    if (height > 0) cfg.height = height;
    if (frames > 0) cfg.frames = frames;

    // ==========================================================
    // --list
    // ==========================================================

    if (list_images) {
        fa_list_images(cfg, user);
        return 0;
    }

    // ==========================================================
    // --clean-model
    // ==========================================================

    if (clean_model) {
        if (fa_clean_model(cfg, user)) {
            std::cout << "[INFO] Model deleted for user: " << user << "\n";
            return 0;
        } else {
            std::cerr << "[ERROR] Failed to delete model\n";
            return 1;
        }
    }

    // ==========================================================
    // --reset = clean + clean-model
    // ==========================================================

    if (reset_all) {
        bool ok1 = fa_clean_images(cfg, user);
        bool ok2 = fa_clean_model(cfg, user);

        if (ok1 && ok2) {
            std::cout << "[INFO] Reset completed for user: " << user << "\n";
            return 0;
        } else {
            std::cerr << "[ERROR] Reset failed\n";
            return 1;
        }
    }

    // ==========================================================
    // --clean
    // ==========================================================

    if (clean_only) {
        if (fa_clean_images(cfg, user)) {
            std::cout << "[INFO] All images cleaned for user: " << user << "\n";
            return 0;
        } else {
            std::cerr << "[ERROR] Failed to clean images\n";
            return 1;
        }
    }

    // ==========================================================
    // Capture mode
    // ==========================================================

    std::cout << "[INFO] Starting capture for user: " << user << "\n";

    if (!fa_capture_images(user, cfg, force, log, img_format)) {
        std::cerr << "[ERROR] Capture failed\n";
        return 1;
    }

    std::cout << "[INFO] Capture completed\n";
    return 0;
}
