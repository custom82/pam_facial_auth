#include "../include/libfacialauth.h"
#include <iostream>
#include <getopt.h>

/**
 * Display help message for the training tool
 */
void print_usage(const char* p) {
    std::cout << "Usage: " << p << " [OPTIONS]\n"
    << "  -u, --user <name>      Target username to train (required)\n"
    << "  -d, --debug            Enable verbose debug logging\n"
    << "  -h, --help             Show this help message\n\n"
    << "Example:\n"
    << "  " << p << " --user phoenix\n";
}

int main(int argc, char** argv) {
    FacialAuthConfig cfg;
    std::string user, log;

    static struct option long_opts[] = {
        {"user",  required_argument, 0, 'u'},
        {"debug", no_argument,       0, 'd'},
        {"help",  no_argument,       0, 'h'},
        {0, 0, 0, 0}
    };

    int opt;
    while ((opt = getopt_long(argc, argv, "u:dh", long_opts, NULL)) != -1) {
        switch (opt) {
            case 'u': user = optarg; break;
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

    // Require root privileges to write to /var/lib/pam_facial_auth
    if (!fa_check_root(argv[0])) return 1;

    // Load configuration for base directory and training method
    fa_load_config(cfg, log, FACIALAUTH_DEFAULT_CONFIG);

    std::cout << "Training model for user: " << user << "...\n";

    if (!fa_train_user(user, cfg, log)) {
        std::cerr << "Training error: " << log << "\n";
        return 1;
    }

    std::cout << "Training completed. Model saved at: " << fa_user_model_path(cfg, user) << "\n";
    return 0;
}
