#include "../include/libfacialauth.h"

#include <iostream>
#include <string>
#include <filesystem>
#include <cstdlib>
#include <vector>
#include <algorithm>

namespace fs = std::filesystem;

// ------------------------------------------------------------
// HELP DINAMICO (legge i profili da config)
// ------------------------------------------------------------
static void print_help_dynamic(const FacialAuthConfig *cfg = nullptr)
{
    std::cout <<
    "Usage: facial_training -u <user> -m <method> [options]\n"
    "\nOptions:\n"
    "  -u, --user <name>           Username (required)\n"
    "  -m, --method <type>         Recognizer profile (required)\n"
    "                              (see \"Available recognizers\" below)\n"
    "  -i, --input <dir>           Directory containing training images\n"
    "  -o, --output <file>         Where to save the trained model (XML)\n"
    "  -f, --force                 Overwrite existing model file\n"
    "  -v, --verbose               Enable verbose output\n"
    "      --debug                 Enable debug (CLI override)\n"
    "      --no-debug              Disable debug (CLI override)\n"
    "      --detector <name>       Detector profile to use (for classic\n"
    "                              recognizers only, see \"Available detectors\")\n"
    "  -c, --config <file>         Alternative config file\n"
    "  -h, --help                  Show this help\n"
    "\nIf -i or -o are not specified, defaults from configuration are used.\n";

    if (!cfg)
        return;

    // -------------------------------------------
    // Detector profiles (detect_*)
    // -------------------------------------------
    if (!cfg->detector_models.empty()) {
        std::cout << "\nAvailable detectors (from config detect_*):\n";
        // keys già normalizzati (es: haar, yunet_fp32, yunet_int8)
        std::vector<std::string> det_names;
        det_names.reserve(cfg->detector_models.size());
        for (const auto &kv : cfg->detector_models)
            det_names.push_back(kv.first);
        std::sort(det_names.begin(), det_names.end());
        for (const auto &name : det_names)
            std::cout << "  " << name << "  -> " << cfg->detector_models.at(name) << "\n";
    } else {
        std::cout << "\nAvailable detectors: (none found in config)\n";
    }

    // -------------------------------------------
    // Recognizer profiles (recognize_*)
    // -------------------------------------------
    if (!cfg->recognizer_models.empty()) {
        std::cout << "\nAvailable recognizers (from config recognize_*):\n";
        std::vector<std::string> rec_names;
        rec_names.reserve(cfg->recognizer_models.size());
        for (const auto &kv : cfg->recognizer_models)
            rec_names.push_back(kv.first);
        std::sort(rec_names.begin(), rec_names.end());
        for (const auto &name : rec_names)
            std::cout << "  " << name << "  -> " << cfg->recognizer_models.at(name) << "\n";
    } else {
        std::cout << "\nAvailable recognizers:\n"
        "  sface_fp32   (DNN SFace, FP32)\n"
        "  sface_int8   (DNN SFace, INT8)\n"
        "  lbph         (classical)\n"
        "  eigen        (classical)\n"
        "  fisher       (classical)\n";
    }

    std::cout << std::endl;
}

// ------------------------------------------------------------
// MAIN
// ------------------------------------------------------------
int facial_training_main(int argc, char *argv[])
{
    std::string user;
    std::string method;        // recognizer_profile
    std::string input_dir;
    std::string output_model;
    std::string cfg_path;

    bool opt_force   = false;
    bool opt_verbose = false;
    bool opt_debug   = false;
    bool opt_no_debug = false;
    std::string opt_detector;  // detector_profile override

    // -------------------------------------
    // Parse arguments
    // -------------------------------------
    for (int i = 1; i < argc; ++i) {
        std::string a = argv[i];

        auto need_value = [&](const char *optname) -> bool {
            if (i + 1 >= argc) {
                std::cerr << "Option " << optname << " requires a value.\n";
                return false;
            }
            return true;
        };

        if ((a == "-u" || a == "--user") && i + 1 < argc) {
            user = argv[++i];
        } else if ((a == "-m" || a == "--method") && i + 1 < argc) {
            method = argv[++i];
        } else if ((a == "-i" || a == "--input") && i + 1 < argc) {
            input_dir = argv[++i];
        } else if ((a == "-o" || a == "--output") && i + 1 < argc) {
            output_model = argv[++i];
        } else if (a == "-f" || a == "--force") {
            opt_force = true;
        } else if (a == "-v" || a == "--verbose") {
            opt_verbose = true;
        } else if (a == "--debug") {
            opt_debug = true;
        } else if (a == "--no-debug") {
            opt_no_debug = true;
        } else if (a == "--detector") {
            if (!need_value(a.c_str())) return 1;
            opt_detector = argv[++i];
        } else if ((a == "-c" || a == "--config") && i + 1 < argc) {
            cfg_path = argv[++i];
        } else if (a == "-h" || a == "--help") {
            // stampa help generico (senza parsing config)
            print_help_dynamic(nullptr);
            return 0;
        } else {
            std::cerr << "Unknown option: " << a << "\n";
            print_help_dynamic(nullptr);
            return 1;
        }
    }

    // -------------------------------------
    // Required arguments
    // -------------------------------------
    if (user.empty() || method.empty()) {
        std::cerr << "[ERROR] -u and -m are required.\n";
        print_help_dynamic(nullptr);
        return 1;
    }

    // -------------------------------------
    // Load config
    // -------------------------------------
    FacialAuthConfig cfg;
    std::string logbuf;

    if (!fa_load_config(cfg, logbuf,
        cfg_path.empty() ? FACIALAUTH_CONFIG_DEFAULT : cfg_path)) {
        std::cerr << "[ERROR] Cannot load configuration from '"
        << (cfg_path.empty() ? FACIALAUTH_CONFIG_DEFAULT : cfg_path)
        << "'\n";
    if (!logbuf.empty())
        std::cerr << logbuf;
        return 1;
        }

        if (!logbuf.empty() && cfg.debug)
            std::cerr << "[DEBUG] Config load log:\n" << logbuf << "\n";
    logbuf.clear();

    // Override debug from CLI
    if (opt_debug)
        cfg.debug = true;
    if (opt_no_debug)
        cfg.debug = false;

    // Recognizer profile from CLI (-m) ha la precedenza
    cfg.recognizer_profile = method;

    // Detector CLI override (solo per classic recognizer; per SFace non serve)
    if (!opt_detector.empty())
        cfg.detector_profile = opt_detector;

    // -------------------------------------
    // Determine paths (input / output)
    // -------------------------------------
    if (input_dir.empty())
        input_dir = fa_user_image_dir(cfg, user);

    if (output_model.empty())
        output_model = fa_user_model_path(cfg, user);

    // Informa l'utente se verbose
    if (opt_verbose) {
        std::cout << "[INFO] Training model\n"
        << "  User:          " << user << "\n"
        << "  Method:        " << cfg.recognizer_profile << "\n"
        << "  Input dir:     " << input_dir << "\n"
        << "  Output model:  " << output_model << "\n";
        if (!cfg.detector_profile.empty())
            std::cout << "  Detector:      " << cfg.detector_profile << "\n";
    }

    // -------------------------------------
    // Check input directory
    // -------------------------------------
    if (!fs::exists(input_dir) || !fs::is_directory(input_dir)) {
        std::cerr << "[ERROR] Image directory missing: " << input_dir << "\n";
        return 1;
    }

    // -------------------------------------
    // Handle overwrite
    // -------------------------------------
    if (fs::exists(output_model) && !opt_force) {
        std::cerr << "[ERROR] Model already exists: " << output_model << "\n";
        std::cerr << "Use --force to overwrite.\n";
        return 1;
    }

    if (opt_force && fs::exists(output_model)) {
        fs::remove(output_model);
        if (opt_verbose)
            std::cout << "[INFO] Existing model removed (force).\n";
    }

    // -------------------------------------
    // Invoke library training
    // -------------------------------------
    bool ok = fa_train_user(user, cfg, logbuf);

    if (!logbuf.empty())
        std::cerr << logbuf;

    if (!ok) {
        std::cerr << "[ERROR] Training failed.\n";
        return 1;
    }

    if (opt_verbose)
        std::cout << "[INFO] Training completed successfully.\n";

    return 0;
}

int main(int argc, char *argv[])
{
    return facial_training_main(argc, argv);
}
