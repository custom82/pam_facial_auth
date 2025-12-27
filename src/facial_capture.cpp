#include "../include/libfacialauth.h"
#include <iostream>
#include <getopt.h>

void print_usage(const char* p) {
    std::cout << "Usage: " << p << " [OPTIONS]\n"
    << "  -u, --user <name>      Target username (required)\n"
    << "  -D, --detector <type>  Face detector: yunet, haar, none (default: none)\n"
    << "  -F, --format <ext>     Image format: jpg, png, bmp (default: jpg)\n"
    << "  -n, --number <num>     Number of frames (default: 50)\n"
    << "  -w, --width <px>       Capture width (default: 640)\n"
    << "  -H, --height <px>      Capture height (default: 480)\n"
    << "  -c, --clean            Delete user data and exit\n"
    << "  -f, --force            Clear capture directory first\n"
    << "  -g, --nogui            Headless mode (disable preview window)\n"
    << "  -d, --debug            Verbose logging\n"
    << "  -h, --help             Show help\n";
}

int main(int argc, char** argv) {
    FacialAuthConfig cfg;
    std::string user, det = "none", log;
    bool clean_only = false;

    static struct option long_opts[] = {
        {"user", 1, 0, 'u'}, {"detector", 1, 0, 'D'}, {"format", 1, 0, 'F'},
        {"number", 1, 0, 'n'}, {"width", 1, 0, 'w'}, {"height", 1, 0, 'H'},
        {"clean", 0, 0, 'c'}, {"force", 0, 0, 'f'}, {"nogui", 0, 0, 'g'},
        {"debug", 0, 0, 'd'}, {"help", 0, 0, 'h'}, {0,0,0,0}
    };

    int opt;
    while ((opt = getopt_long(argc, argv, "u:D:F:n:w:H:cfgdh", long_opts, NULL)) != -1) {
        switch (opt) {
            case 'u': user = optarg; break;
            case 'D': det = optarg; break;
            case 'F': cfg.image_format = optarg; break;
            case 'n': cfg.frames = std::stoi(optarg); break;
            case 'w': cfg.width = std::stoi(optarg); break;
            case 'H': cfg.height = std::stoi(optarg); break;
            case 'c': clean_only = true; break;
            case 'f': cfg.force = true; break;
            case 'g': cfg.nogui = true; break;
            case 'd': cfg.debug = true; break;
            case 'h': print_usage(argv[0]); return 0;
            default: return 1;
        }
    }

    if (user.empty()) {
        std::cerr << "Error: --user is required.\n";
        return 1;
    }

    if (!fa_check_root(argv[0])) return 1;

    if (clean_only) {
        std::cout << "Cleaning data for user: " << user << "\n";
        return fa_delete_user_data(user, cfg) ? 0 : 1;
    }

    fa_load_config(cfg, log, FACIALAUTH_DEFAULT_CONFIG);

    if (cfg.debug) std::cout << "[INFO] Session start for user: " << user << std::endl;

    if (!fa_capture_user(user, cfg, det, log)) {
        std::cerr << "[ERROR] " << log << std::endl;
        return 1;
    }

    std::cout << "[SUCCESS] Capture completed for " << user << std::endl;
    return 0;
}
