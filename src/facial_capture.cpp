/*
 * Project: pam_facial_auth
 * License: GPL-3.0
 */

#include "libfacialauth.h"
#include <iostream>
#include <vector>

void usage() {
    std::cout << "Usage: facial_capture -u <user> [options]\n\n"
    << "Options:\n"
    << "  -u, --user <name>       User name to save images for\n"
    << "  -c, --config <file>     Configuration file (default: /etc/security/pam_facial_auth.conf)\n"
    << "  -d, --device <path>     Webcam device (e.g., /dev/video0)\n"
    << "  -w, --width <px>        Frame width\n"
    << "  -h, --height <px>       Frame height\n"
    << "  -f, --force             Overwrite existing images and restart from 1\n"
    << "  --flush, --clean        Delete all images for the specified user\n"
    << "  -n, --num_images <num>  Number of images to capture\n"
    << "  -s, --sleep <sec>       Pause between captures (seconds)\n"
    << "  -v, --verbose           Verbose output\n"
    << "  --debug                 Enable debug output\n"
    << "  --nogui                 Disable GUI, capture from console only\n"
    << "  --help, -H              Show this message\n"
    << "  --detector              yunet or haar\n";
}

int main(int argc, char** argv) {
    if (!fa_check_root("facial_capture")) return 1;

    std::string user, config_path = "/etc/security/pam_facial_auth.conf", log;
    FacialAuthConfig cfg;
    bool force = false, clean_only = false;

    std::vector<std::string> args(argv + 1, argv + argc);
    for (size_t i = 0; i < args.size(); ++i) {
        if ((args[i] == "-c" || args[i] == "--config") && i + 1 < args.size()) config_path = args[++i];
    }
    fa_load_config(cfg, log, config_path);

    for (size_t i = 0; i < args.size(); ++i) {
        if (args[i] == "--help" || args[i] == "-H") {
            usage();
            return 0;
        } else if ((args[i] == "-u" || args[i] == "--user") && i + 1 < args.size()) {
            user = args[++i];
        } else if ((args[i] == "-d" || args[i] == "--device") && i + 1 < args.size()) {
            cfg.device = args[++i];
        } else if ((args[i] == "-w" || args[i] == "--width") && i + 1 < args.size()) {
            cfg.width = std::stoi(args[++i]);
        } else if ((args[i] == "-h" || args[i] == "--height") && i + 1 < args.size()) {
            cfg.height = std::stoi(args[++i]);
        } else if ((args[i] == "-n" || args[i] == "--num_images") && i + 1 < args.size()) {
            cfg.frames = std::stoi(args[++i]);
        } else if ((args[i] == "-s" || args[i] == "--sleep") && i + 1 < args.size()) {
            cfg.sleep_ms = std::stoi(args[++i]) * 1000;
        } else if (args[i] == "--detector" && i + 1 < args.size()) {
            cfg.detector = args[++i];
            if (cfg.detector == "haar") {
                cfg.detector = "cascade";
            }
        } else if (args[i] == "--format" && i + 1 < args.size()) {
            cfg.image_format = args[++i];
        } else if (args[i] == "-f" || args[i] == "--force") {
            force = true;
        } else if (args[i] == "--clean" || args[i] == "--flush") {
            clean_only = true;
        } else if (args[i] == "--nogui") {
            cfg.nogui = true;
        } else if (args[i] == "-v" || args[i] == "--verbose") {
            cfg.verbose = true;
        } else if (args[i] == "--debug") {
            cfg.debug = true;
        }
    }

    if (user.empty()) { usage(); return 1; }

    if (clean_only || force) {
        fa_clean_captures(user, cfg, log);
        if (cfg.verbose || cfg.debug) {
            std::cout << "[INFO] " << log << "\n";
        }
    }
    if (clean_only) return 0;

    if (!fa_capture_user(user, cfg, cfg.device, log)) {
        std::cerr << "[ERROR] " << log << std::endl;
        return 1;
    }
    if (cfg.verbose || cfg.debug) {
        std::cout << "[INFO] " << log << "\n";
    }
    return 0;
}
