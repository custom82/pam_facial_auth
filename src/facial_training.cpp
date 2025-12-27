#include "../include/libfacialauth.h"
#include <iostream>
#include <getopt.h>
#include <filesystem>

namespace fs = std::filesystem;

/**
 * Display help message for the training tool
 */
void print_usage(const char* p) {
    std::cout << "Usage: " << p << " [OPTIONS]\n"
    << "  -u, --user <name>      Target username to train (required)\n"
    << "  -m, --method <type>    Training method: lbph, eigen, fisher (default: lbph)\n"
    << "  -d, --debug            Enable verbose debug logging\n"
    << "  -h, --help             Show this help message\n\n"
    << "Example:\n"
    << "  " << p << " --user phoenix --method lbph\n";
}

int main(int argc, char** argv) {
    FacialAuthConfig cfg;
    std::string user, log;

    static struct option long_opts[] = {
        {"user",   required_argument, 0, 'u'},
        {"method", required_argument, 0, 'm'},
        {"debug",  no_argument,       0, 'd'},
        {"help",   no_argument,       0, 'h'},
        {0, 0, 0, 0}
    };

    int opt;
    while ((opt = getopt_long(argc, argv, "u:m:dh", long_opts, NULL)) != -1) {
        switch (opt) {
            case 'u': user = optarg; break;
            case 'm': cfg.training_method = optarg; break;
            case 'd': cfg.debug = true; break;
            case 'h': print_usage(argv[0]); return 0;
            default: return 1;
        }
    }

    if (user.empty()) {
        std::cerr << "Error: --user is required.\n";
        print_usage(argv[0]);
        return 1;
    }

    // Ensure tool is run with necessary privileges
    if (!fa_check_root(argv[0])) return 1;

    // Load configuration for paths and defaults
    fa_load_config(cfg, log, FACIALAUTH_DEFAULT_CONFIG);

    std::string captures_path = cfg.basedir + "/" + user + "/captures";

    if (!fs::exists(captures_path) || fs::is_empty(captures_path)) {
        std::cerr << "[ERROR] No images found in " << captures_path << "\n"
        << "Please run facial_capture first.\n";
        return 1;
    }

    std::cout << "[INFO] Starting training for user: " << user << "\n";
    std::cout << "[INFO] Using method: " << cfg.training_method << "\n";

    // fa_train_user handles the internal OpenCV model creation and saving
    if (!fa_train_user(user, cfg, log)) {
        std::cerr << "[ERROR] Training failed: " << log << "\n";
        return 1;
    }

    std::cout << "[SUCCESS] Model trained and saved at: " << fa_user_model_path(cfg, user) << "\n";

    return 0;
}
