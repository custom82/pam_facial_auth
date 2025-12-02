#include "libfacialauth.h"

#include <opencv2/core.hpp>

#include <iostream>
#include <string>
#include <algorithm>

static void print_test_help()
{
    std::cout <<
    "Usage: facial_test -u <user> [options]\n\n"
    "Test facial recognition accuracy for a specific user.\n"
    "\n"
    "Options:\n"
    "  -u, --user <name>\n"
    "      Username to evaluate (required).\n"
    "\n"
    "  -c, --config <file>\n"
    "      Path to configuration file.\n"
    "      Default: " FACIALAUTH_DEFAULT_CONFIG "\n"
    "\n"
    "      --threshold <value>\n"
    "      Override matching confidence threshold.\n"
    "\n"
    "  -v, --verbose\n"
    "      Enable informational messages.\n"
    "\n"
    "      --debug\n"
    "      Enable detailed debugging output.\n"
    "\n"
    "  -H, --help\n"
    "      Show this help message and exit.\n";
}

int facial_test_cli_main(int argc, char **argv)
{
    std::string user;
    std::string config_path = FACIALAUTH_DEFAULT_CONFIG;
    double threshold_override = -1.0;
    bool verbose = false;
    bool debug = false;

    for (int i = 1; i < argc; ++i) {
        std::string arg = argv[i];

        auto take_value = [&](const std::string &opt) -> std::string {
            if (i + 1 >= argc) {
                std::cerr << "Manca il valore per l'opzione " << opt << "\n";
                exit(1);
            }
            return argv[++i];
        };

        if (arg == "-u" || arg == "--user") {
            user = take_value(arg);
        } else if (arg == "-c" || arg == "--config") {
            config_path = take_value(arg);
        } else if (arg == "--threshold") {
            threshold_override = std::stod(take_value(arg));
        } else if (arg == "-v" || arg == "--verbose") {
            verbose = true;
        } else if (arg == "--debug") {
            debug = true;
        } else if (arg == "-H" || arg == "--help") {
            print_test_help();
            return 0;
        } else {
            std::cerr << "Opzione sconosciuta: " << arg << "\n";
            print_test_help();
            return 1;
        }
    }

    if (user.empty()) {
        std::cerr << "[ERRORE] Devi specificare --user <name>.\n";
        return 1;
    }

    FacialAuthConfig cfg;
    std::string log;
    if (!fa_load_config(cfg, log, config_path)) {
        std::cerr << log;
        // continuiamo con i defaults
    }

    if (verbose || debug)
        cfg.debug = true;

    if (!fa_check_root("facial_test")) {
        std::cerr << "[ERRORE] Questo strumento deve essere eseguito come root.\n";
        return 1;
    }

    std::string model_path = fa_user_model_path(cfg, user);
    double best_conf = 0.0;
    int best_label   = -1;
    std::string test_log;

    if (!fa_test_user(user, cfg, model_path, best_conf, best_label, test_log, threshold_override)) {
        std::cerr << test_log;
        return 1;
    }

    if (!test_log.empty())
        std::cout << test_log;

    std::cout << "[RISULTATO] best_label=" << best_label
    << " best_conf=" << best_conf << "\n";

    return 0;
}

int main(int argc, char **argv)
{
    try {
        return facial_test_cli_main(argc, argv);
    }
    catch (const cv::Exception &e) {
        std::cerr << "[OpenCV ERROR] " << e.what() << std::endl;
        return 1;
    }
    catch (const std::exception &e) {
        std::cerr << "[ERROR] " << e.what() << std::endl;
        return 1;
    }
}
