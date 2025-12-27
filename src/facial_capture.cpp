#include "../include/libfacialauth.h"
#include <iostream>
#include <getopt.h>

/**
 * Display help message in English with updated parameter descriptions
 */
void print_usage(const char* p) {
    std::cout << "Usage: " << p << " [OPTIONS]\n"
    << "  -u, --user <name>      Target username (required)\n"
    << "  -D, --detector <type>  Face detector: yunet, haar, none (default: none)\n"
    << "  -F, --format <ext>     Image format: jpg, png, bmp (default: jpg)\n"
    << "  -n, --number <num>     Number of frames to capture in this session (default: 50)\n"
    << "  -w, --width <px>       Capture width (default: 640)\n"
    << "  -H, --height <px>      Capture height (default: 480)\n"
    << "  -c, --clean            Delete all user data and exit\n"
    << "  -f, --force            Clear capture directory (restart from img_0)\n"
    << "  -g, --nogui            Headless mode (disable preview window)\n"
    << "  -d, --debug            Enable verbose debug logging (e.g., detection misses)\n"
    << "  -h, --help             Show this help message\n\n"
    << "Note: Without --force, the tool automatically resumes naming from the last saved image index.\n";
}

int main(int argc, char** argv) {
    FacialAuthConfig cfg;
    std::string user, det = "none", log;
    bool clean_only = false;

    // Define long options for getopt_long
    static struct option long_opts[] = {
        {"user",     required_argument, 0, 'u'},
        {"detector", required_argument, 0, 'D'},
        {"format",   required_argument, 0, 'F'},
        {"number",   required_argument, 0, 'n'},
        {"width",    required_argument, 0, 'w'},
        {"height",   required_argument, 0, 'H'},
        {"clean",    no_argument,       0, 'c'},
        {"force",    no_argument,       0, 'f'},
        {"nogui",    no_argument,       0, 'g'},
        {"debug",    no_argument,       0, 'd'},
        {"help",     no_argument,       0, 'h'},
        {0, 0, 0, 0}
    };

    int opt;
    while ((opt = getopt_long(argc, argv, "u:D:F:n:w:H:cfgdh", long_opts, NULL)) != -1) {
        switch (opt) {
            case 'u': user = optarg; break;
            case 'D': det = optarg; break;
            case 'F': cfg.image_format = optarg; break;
            case 'n': try { cfg.frames = std::stoi(optarg); } catch(...) {} break;
            case 'w': try { cfg.width = std::stoi(optarg); } catch(...) {} break;
            case 'H': try { cfg.height = std::stoi(optarg); } catch(...) {} break;
            case 'c': clean_only = true; break;
            case 'f': cfg.force = true; break;
            case 'g': cfg.nogui = true; break;
            case 'd': cfg.debug = true; break;
            case 'h': print_usage(argv[0]); return 0;
            default: return 1;
        }
    }

    // Validation: Username is mandatory
    if (user.empty()) {
        std::cerr << "Error: --user is required.\n";
        print_usage(argv[0]);
        return 1;
    }

    // Root check for system directories (/var/lib/...)
    if (!fa_check_root(argv[0])) return 1;

    // Handle data cleanup if requested
    if (clean_only) {
        std::cout << "Cleaning data for user: " << user << "\n";
        if (fa_delete_user_data(user, cfg)) {
            std::cout << "User data deleted successfully.\n";
            return 0;
        } else {
            std::cerr << "Error: Could not delete user data (maybe it doesn't exist).\n";
            return 1;
        }
    }

    // Load configuration (defaults or from file)
    fa_load_config(cfg, log, FACIALAUTH_DEFAULT_CONFIG);

    // Minimalist session start info
    std::cout << "[INFO] Starting capture session for user: " << user << "\n";
    if (cfg.force) std::cout << "[INFO] Force mode enabled: Directory will be cleared.\n";
    else std::cout << "[INFO] Auto-increment enabled: Resuming from last index.\n";

    // Execute capture logic from libfacialauth
    if (!fa_capture_user(user, cfg, det, log)) {
        std::cerr << "[ERROR] " << log << std::endl;
        return 1;
    }

    std::cout << "[SUCCESS] Capture completed for " << user << std::endl;
    return 0;
}
