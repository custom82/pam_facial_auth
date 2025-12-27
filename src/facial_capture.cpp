#include "../include/libfacialauth.h"
#include <iostream>
#include <getopt.h>

void print_usage(const char* p) {
    std::cout << "Usage: " << p << " [OPTIONS]\n"
    << "  -u, --user <name>      Target username (required)\n"
    << "  -n, --number <num>     Number of frames to capture\n"
    << "  -w, --width <px>       Capture width\n"
    << "  -H, --height <px>      Capture height\n"
    << "  -F, --format <ext>     Image format (jpg, png)\n"
    << "  -f, --force            Clear directory and restart from img_0\n"
    << "  -g, --nogui            Disable preview window\n"
    << "  -d, --debug            Enable debug output\n"
    << "  -h, --help             Show this help\n";
}

int main(int argc, char** argv) {
    FacialAuthConfig cfg;
    std::string user, log;

    // 1. Carica i default dal file /etc/security/pam_facial.conf
    fa_load_config(cfg, log, FACIALAUTH_DEFAULT_CONFIG);

    // 2. Override tramite riga di comando
    static struct option long_opts[] = {
        {"user",   required_argument, 0, 'u'},
        {"number", required_argument, 0, 'n'},
        {"width",  required_argument, 0, 'w'},
        {"height", required_argument, 0, 'H'},
        {"format", required_argument, 0, 'F'},
        {"force",  no_argument,       0, 'f'},
        {"nogui",  no_argument,       0, 'g'},
        {"debug",  no_argument,       0, 'd'},
        {"help",   no_argument,       0, 'h'},
        {0, 0, 0, 0}
    };

    int opt;
    while ((opt = getopt_long(argc, argv, "u:n:w:H:F:fgdh", long_opts, NULL)) != -1) {
        switch (opt) {
            case 'u': user = optarg; break;
            case 'n': cfg.frames = std::stoi(optarg); break;
            case 'w': cfg.width = std::stoi(optarg); break;
            case 'H': cfg.height = std::stoi(optarg); break;
            case 'F': cfg.image_format = optarg; break;
            case 'f': cfg.force = true; break;
            case 'g': cfg.nogui = true; break;
            case 'd': cfg.debug = true; break;
            case 'h': print_usage(argv[0]); return 0;
            default: return 1;
        }
    }

    if (user.empty()) {
        std::cerr << "Error: User (-u) is required.\n";
        return 1;
    }

    if (!fa_check_root("facial_capture")) return 1;

    // Logica di selezione detector basata sul metodo di training
    std::string detector = (cfg.training_method == "sface" || cfg.training_method == "auto") ? "yunet" : "none";

    std::cout << "[INFO] Session for: " << user << "\n"
    << "[INFO] Resolution: " << cfg.width << "x" << cfg.height << "\n"
    << "[INFO] Format:     " << cfg.image_format << "\n"
    << "[INFO] Target:     " << cfg.frames << " frames\n"
    << "[INFO] Detector:   " << detector << "\n";

    if (fa_capture_user(user, cfg, detector, log)) {
        std::cout << "[SUCCESS] Capture session completed for " << user << std::endl;
    } else {
        std::cerr << "[ERROR] " << log << std::endl;
        return 1;
    }

    return 0;
}
