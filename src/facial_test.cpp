/*
 * Project: pam_facial_auth
 * License: GPL-3.0
 */

#include "libfacialauth.h"
#include <iostream>
#include <vector>
#include <string>

void usage() {
    std::cout << "Usage: facial_test -u <user> [-m <path>] [options]\n\n"
    << "Options:\n"
    << "  -u, --user <user>        User to verify (required)\n"
    << "  -m, --model <path>       XML model file (default: /etc/security/pam_facial_auth/<user>.xml)\n"
    << "  -c, --config <file>      Configuration file (default: /etc/security/pam_facial_auth.conf)\n"
    << "  -d, --device <device>    Webcam device (e.g., /dev/video0)\n"
    << "  --threshold <value>      Match confidence threshold (default: 80.0)\n"
    << "  -v, --verbose            Verbose mode\n"
    << "  --nogui                  Disable GUI (console only)\n"
    << "  -h, --help               Show this message\n";
}

int main(int argc, char** argv) {
    if (!fa_check_root("facial_test")) return 1;

    std::string user;
    std::string model_path;
    std::string device = "/dev/video0";
    std::string config_path = "/etc/security/pam_facial_auth.conf";

    FacialAuthConfig cfg;
    std::string log;
    bool model_provided = false;

    std::vector<std::string> args(argv + 1, argv + argc);
    if (args.empty()) { usage(); return 1; }

    // Load base configuration
    fa_load_config(cfg, log, config_path);

    // Parse CLI parameters (override config)
    for (size_t i = 0; i < args.size(); ++i) {
        if (args[i] == "-h" || args[i] == "--help") { usage(); return 0; }
        else if ((args[i] == "-u" || args[i] == "--user") && i + 1 < args.size()) user = args[++i];
        else if ((args[i] == "-m" || args[i] == "--model") && i + 1 < args.size()) { model_path = args[++i]; model_provided = true; }
        else if ((args[i] == "-c" || args[i] == "--config") && i + 1 < args.size()) config_path = args[++i];
        else if ((args[i] == "-d" || args[i] == "--device") && i + 1 < args.size()) device = args[++i];
        else if (args[i] == "--threshold" && i + 1 < args.size()) cfg.threshold = std::stod(args[++i]);
        else if (args[i] == "-v" || args[i] == "--verbose") cfg.verbose = true;
        else if (args[i] == "--nogui") cfg.nogui = true;
    }

    if (user.empty()) {
        std::cerr << "Error: -u (user) is required.\n";
        return 1;
    }

    if (!model_provided) {
        model_path = fa_user_model_path(cfg, user);
    }

    double confidence = 0.0;
    int label = -1;

    if (cfg.verbose) {
        std::cout << "[*] Starting recognition test...\n";
        std::cout << "[*] User:        " << user << "\n";
        std::cout << "[*] Model:       " << model_path << "\n";
        std::cout << "[*] Threshold:   " << cfg.threshold << "\n";
        std::cout << "[*] Device:      " << device << "\n";
    }

    cfg.device = device;

    // Run the test (fa_test_user handles device opening and prediction)
    if (!fa_test_user(user, cfg, model_path, confidence, label, log)) {
        std::cerr << "ERROR: " << log << std::endl;
        return 1;
    }

    // Threshold-based validation (lower is better for LBPH/Eigen, higher for SFace).
    // Assume standard logic: confidence <= threshold is a match (for classic plugins).
    bool is_authenticated = (confidence <= cfg.threshold);

    std::cout << "\n-----------------------------------" << std::endl;
    std::cout << " RESULT:      " << (is_authenticated ? "AUTHENTICATED" : "FAILED") << std::endl;
    std::cout << " Confidence:  " << confidence << " (Threshold: " << cfg.threshold << ")" << std::endl;
    std::cout << " Label:       " << label << std::endl;
    std::cout << "-----------------------------------" << std::endl;

    return is_authenticated ? 0 : 1;
}
