#include "../include/libfacialauth.h"
#include <iostream>
#include <getopt.h>
#include <iomanip>

void print_usage(const char* prog) {
    std::cout << "Facial Auth Diagnostic Tool\n"
    << "Usage: " << prog << " -u <username> [OPTIONS]\n\n"
    << "Options:\n"
    << "  -u, --user <name>       User to test against (required)\n"
    << "  -c, --config <path>     Path to config file\n"
    << "  -g, --gui               Enable live camera preview\n"
    << "  -d, --debug             Enable verbose debug logs\n"
    << "  -h, --help              Show this help menu\n\n"
    << "This tool captures a frame and compares it with the stored model,\n"
    << "reporting the confidence score and the final decision.\n";
}

int main(int argc, char** argv) {
    FacialAuthConfig cfg;
    std::string user, log, config_path = FACIALAUTH_DEFAULT_CONFIG;

    // Default: caricamento config
    fa_load_config(cfg, log, config_path);

    static struct option long_opts[] = {
        {"user",   required_argument, 0, 'u'},
        {"config", required_argument, 0, 'c'},
        {"gui",    no_argument,       0, 'g'},
        {"debug",  no_argument,       0, 'd'},
        {"help",   no_argument,       0, 'h'},
        {0, 0, 0, 0}
    };

    int opt;
    while ((opt = getopt_long(argc, argv, "u:c:gdh", long_opts, NULL)) != -1) {
        switch (opt) {
            case 'u': user = optarg; break;
            case 'c': config_path = optarg; break;
            case 'g': cfg.nogui = false; break;
            case 'd': cfg.debug = true; break;
            case 'h': print_usage(argv[0]); return 0;
            default: return 1;
        }
    }

    if (user.empty()) {
        std::cerr << "Error: User (-u) is required.\n";
        return 1;
    }

    std::string model_path = fa_user_model_path(cfg, user);
    if (!fa_file_exists(model_path)) {
        std::cerr << "[ERROR] Model not found: " << model_path << "\n"
        << "Run 'facial_training -u " << user << "' first.\n";
        return 1;
    }

    std::cout << "[*] Testing recognition for user: " << user << "\n"
    << "[*] Using model: " << model_path << "\n"
    << "[*] Method: " << cfg.training_method << "\n";

    double confidence = 0.0;
    int label = -1;

    // Esecuzione del test (chiama la funzione nel .cpp che abbiamo aggiornato prima)
    if (fa_test_user(user, cfg, model_path, confidence, label, log)) {
        bool authenticated = false;

        // Logica di soglia (Threshold)
        if (cfg.training_method == "sface") {
            authenticated = (confidence >= cfg.sface_threshold);
        } else {
            authenticated = (confidence <= cfg.lbph_threshold);
        }

        std::cout << "\n--- Results ---\n"
        << "Score:      " << std::fixed << std::setprecision(4) << confidence << "\n"
        << "Threshold:  " << (cfg.training_method == "sface" ? cfg.sface_threshold : cfg.lbph_threshold) << "\n"
        << "Status:     " << (authenticated ? "\033[1;32mMATCHED\033[0m" : "\033[1;31mREJECTED\033[0m") << "\n"
        << "---------------\n";
    } else {
        std::cerr << "[FATAL] Test failed: " << log << "\n";
        return 1;
    }

    return 0;
}
