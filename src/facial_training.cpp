#include "../include/libfacialauth.h"
#include <iostream>
#include <getopt.h>
#include <filesystem>

namespace fs = std::filesystem;

void print_usage(const char* prog) {
    std::cout << "Facial Auth Training Tool - System-wide Recognition Generator\n"
    << "Usage: " << prog << " -u <username> [OPTIONS]\n\n"
    << "Core Options:\n"
    << "  -u, --user <name>       Target user for model generation (required)\n"
    << "  -m, --method <method>   Training method: sface, lbph, eigen, fisher (default: auto)\n"
    << "  -c, --config <path>     Path to custom config file\n\n"
    << "Maintenance Options:\n"
    << "  -f, --force             Overwrite existing model without asking\n"
    << "  -d, --debug             Enable verbose output during training\n"
    << "  -p, --purge             Delete all captured images after successful training\n"
    << "  -h, --help              Show this help menu\n\n"
    << "Example:\n"
    << "  " << prog << " -u phoenix -m sface -d\n";
}

int main(int argc, char** argv) {
    FacialAuthConfig cfg;
    std::string user, log, config_path = FACIALAUTH_DEFAULT_CONFIG;
    bool purge = false;

    // 1. Caricamento configurazione base
    if (!fa_load_config(cfg, log, config_path)) {
        std::cerr << "[WARN] Could not load config, using internal defaults.\n";
    }

    static struct option long_opts[] = {
        {"user",   required_argument, 0, 'u'},
        {"method", required_argument, 0, 'm'},
        {"config", required_argument, 0, 'c'},
        {"force",  no_argument,       0, 'f'},
        {"debug",  no_argument,       0, 'd'},
        {"purge",  no_argument,       0, 'p'},
        {"help",   no_argument,       0, 'h'},
        {0, 0, 0, 0}
    };

    int opt;
    while ((opt = getopt_long(argc, argv, "u:m:c:fdph", long_opts, NULL)) != -1) {
        switch (opt) {
            case 'u': user = optarg; break;
            case 'm': cfg.training_method = optarg; break;
            case 'c': config_path = optarg; break;
            case 'f': cfg.force = true; break;
            case 'd': cfg.debug = true; break;
            case 'p': purge = true; break;
            case 'h': print_usage(argv[0]); return 0;
            default: return 1;
        }
    }

    // Validazione input
    if (user.empty()) {
        std::cerr << "Error: User (-u) is mandatory.\n";
        print_usage(argv[0]);
        return 1;
    }

    if (!fa_check_root("facial_training")) return 1;

    // Controllo esistenza cartella catture
    fs::path capture_path = fs::path(cfg.basedir) / user / "captures";
    if (!fs::exists(capture_path) || fs::is_empty(capture_path)) {
        std::cerr << "[ERROR] No captures found for user '" << user << "' in: " << capture_path << "\n"
        << "Please run 'facial_capture -u " << user << "' first.\n";
        return 1;
    }

    std::cout << "[*] Starting training for user: " << user << "\n"
    << "[*] Method: " << (cfg.training_method == "auto" ? "Detecting..." : cfg.training_method) << "\n";

    // Esecuzione training
    if (fa_train_user(user, cfg, log)) {
        std::string final_path = fa_user_model_path(cfg, user);
        std::cout << "[SUCCESS] Model generated successfully: " << final_path << "\n";

        if (purge) {
            std::cout << "[*] Purging capture directory as requested...\n";
            fs::remove_all(capture_path);
            fs::create_directories(capture_path);
        }
    } else {
        std::cerr << "[FATAL] Training failed: " << log << "\n";
        return 1;
    }

    return 0;
}
