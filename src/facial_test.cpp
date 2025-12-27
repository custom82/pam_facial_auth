#include "../include/libfacialauth.h"
#include <iostream>
#include <getopt.h>

/**
 * Display help message for the testing tool
 */
void print_usage(const char* p) {
    std::cout << "Usage: " << p << " [OPTIONS]\n"
    << "  -u, --user <name>      Target username to test against (required)\n"
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

    // Load configuration for threshold settings
    fa_load_config(cfg, log, FACIALAUTH_DEFAULT_CONFIG);

    std::string model_path = fa_user_model_path(cfg, user);
    if (!fa_file_exists(model_path)) {
        std::cerr << "Error: Model file not found for user " << user << ".\n"
        << "Run facial_training first.\n";
        return 1;
    }

    double confidence = 0;
    int label = -1;

    std::cout << "Starting recognition test for user: " << user << "\n";
    std::cout << "Look at the camera...\n";

    if (fa_test_user(user, cfg, model_path, confidence, label, log)) {
        // Label 0 is the current user. Higher confidence values mean lower accuracy in LBPH.
        bool is_match = (label == 0 && confidence <= cfg.lbph_threshold);

        std::cout << "-----------------------------\n";
        std::cout << "Result: " << (is_match ? "SUCCESS (MATCH)" : "FAILED (NO MATCH)") << "\n";
        std::cout << "Label: " << label << "\n";
        std::cout << "Confidence: " << confidence << " (Threshold: " << cfg.lbph_threshold << ")\n";
        std::cout << "-----------------------------\n";

        return is_match ? 0 : 1;
    } else {
        std::cerr << "Test error: " << log << "\n";
        return 1;
    }
}
