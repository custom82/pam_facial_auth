#include "libfacialauth.h"

#include <opencv2/core.hpp>

#include <iostream>
#include <string>
#include <algorithm>

static void print_training_help()
{
    std::cout <<
    "Usage: facial_training -u <user> [options]\n\n"
    "Options:\n"
    "  -u, --user <name>      Nome utente da addestrare\n"
    "  -c, --config <file>    File di configurazione\n"
    "                         (default: /etc/pam_facial_auth/pam_facial.conf)\n"
    "      --threshold <val>  Soglia opzionale per il training (override)\n"
    "  -v, --verbose          Output dettagliato\n"
    "      --debug            Abilita debug\n"
    "  -H, --help             Mostra questo messaggio\n";
}

int facial_training_cli_main(int argc, char **argv)
{
    std::string user;
    std::string config_path = FACIALAUTH_DEFAULT_CONFIG;
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
        } else if (arg == "-v" || arg == "--verbose") {
            verbose = true;
        } else if (arg == "--debug") {
            debug = true;
        } else if (arg == "-H" || arg == "--help") {
            print_training_help();
            return 0;
        } else {
            std::cerr << "Opzione sconosciuta: " << arg << "\n";
            print_training_help();
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
        // usiamo comunque i default
    }

    if (verbose || debug)
        cfg.debug = true;

    if (!fa_check_root("facial_training")) {
        std::cerr << "[ERRORE] Questo strumento deve essere eseguito come root.\n";
        return 1;
    }

    std::string train_log;
    if (!fa_train_user(user, cfg, train_log)) {
        std::cerr << train_log;
        return 1;
    }

    if (!train_log.empty())
        std::cout << train_log;

    return 0;
}

int main(int argc, char **argv)
{
    try {
        return facial_training_cli_main(argc, argv);
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
