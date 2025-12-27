#include "../include/libfacialauth.h"
#include <iostream>
#include <getopt.h>
#include <filesystem>
#include <vector>
#include <string>

namespace fs = std::filesystem;

/**
 * Display an improved, professional help message
 */
void print_usage(const char* p) {
    std::cout << "Facial Authentication Training Tool\n"
    << "Usage: " << p << " [OPTIONS]\n\n"
    << "Options:\n"
    << "  -u, --user <name>      Target username (must have captured images)\n"
    << "  -m, --method <type>    Recognition algorithm: lbph, eigen, fisher\n"
    << "                         (default: lbph - recommended for lighting stability)\n"
    << "  -d, --debug            Show detailed processing information\n"
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
        std::cerr << "[ERROR] Target user is mandatory. Use -u <username>.\n";
        return 1;
    }

    if (!fa_check_root(argv[0])) return 1;

    fa_load_config(cfg, log, FACIALAUTH_DEFAULT_CONFIG);

    // 1. Validate training method
    std::string m = cfg.training_method;
    if (m != "lbph" && m != "eigen" && m != "fisher") {
        std::cerr << "[ERROR] Invalid method: '" << m << "'. Supported: lbph, eigen, fisher.\n";
        return 1;
    }

    // 2. Dataset Pre-check
    fs::path captures_path = fs::path(cfg.basedir) / user / "captures";
    if (!fs::exists(captures_path)) {
        std::cerr << "[ERROR] Captures directory not found: " << captures_path << "\n";
        return 1;
    }

    std::vector<std::string> image_files;
    for (const auto& entry : fs::directory_iterator(captures_path)) {
        if (entry.is_regular_file()) image_files.push_back(entry.path().string());
    }

    if (image_files.empty()) {
        std::cerr << "[ERROR] No images found for user '" << user << "'. Capture some faces first.\n";
        return 1;
    }

    // 3. Informative Output
    std::cout << "--------------------------------------------------\n";
    std::cout << " Training Profile for: " << user << "\n";
    std::cout << "--------------------------------------------------\n";
    std::cout << " > Method:      " << cfg.training_method << "\n";
    std::cout << " > Dataset:    " << image_files.size() << " images found\n";
    std::cout << " > Source:     " << captures_path << "\n";
    std::cout << "--------------------------------------------------\n";

    std::cout << "[WAIT] Training the mathematical model... " << std::flush;

    if (!fa_train_user(user, cfg, log)) {
        std::cout << "FAILED\n";
        std::cerr << "[ERROR] " << log << "\n";
        return 1;
    }

    std::cout << "DONE\n";
    std::cout << "[SUCCESS] Model generated: " << fa_user_model_path(cfg, user) << "\n";

    return 0;
}
