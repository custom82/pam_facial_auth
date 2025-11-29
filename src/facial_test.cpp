#include "../include/libfacialauth.h"

#include <iostream>
#include <string>
#include <vector>
#include <cstring>
#include <cstdlib>
#include <unistd.h>

// ------------------------------------------------------------
// HELP
// ------------------------------------------------------------
static void print_help()
{
    std::cout <<
    "Usage: facial_test -u <user> [options]\n\n"
    "Options:\n"
    "  -u, --user <user>        Utente da verificare (obbligatorio)\n"
    "  -c, --config <file>      File di configurazione (default: /etc/security/pam_facial.conf)\n"
    "  -d, --device <device>    Dispositivo webcam (es. /dev/video0)\n"
    "  --threshold <value>      Soglia di confidenza (override)\n"
    "  --detector <name>        Forza detector (auto|haar|yunet|yunet_int8)\n"
    "  -v, --verbose            Modalità verbosa\n"
    "  --nogui                  Disabilita GUI\n"
    "  -h, --help               Mostra questo messaggio\n";
}

// ------------------------------------------------------------
// MAIN
// ------------------------------------------------------------
int facial_test_cli_main(int argc, char *argv[])
{
    const char *prog = "facial_test";

    if (!fa_check_root(prog))
        return 1;

    std::string user;
    std::string cfg_path;

    std::string opt_device;
    std::string opt_detector;
    bool opt_debug = false;
    bool opt_nogui = false;

    int opt_width  = -1;
    int opt_height = -1;
    int opt_frames = -1;
    int opt_sleep  = -1;

    double opt_threshold = -1.0;

    // ---------------------------------------------------------
    // Parse args
    // ---------------------------------------------------------
    for (int i = 1; i < argc; ++i) {
        std::string a = argv[i];

        if ((a == "-u" || a == "--user") && i + 1 < argc)
            user = argv[++i];
        else if ((a == "-c" || a == "--config") && i + 1 < argc)
            cfg_path = argv[++i];
        else if ((a == "-d" || a == "--device") && i + 1 < argc)
            opt_device = argv[++i];
        else if ((a == "-w" || a == "--width") && i + 1 < argc)
            opt_width = atoi(argv[++i]);
        else if ((a == "-h" || a == "--height") && i + 1 < argc)
            opt_height = atoi(argv[++i]);
        else if ((a == "-n" || a == "--frames") && i + 1 < argc)
            opt_frames = atoi(argv[++i]);
        else if ((a == "-s" || a == "--sleep") && i + 1 < argc)
            opt_sleep = atoi(argv[++i]);
        else if (a == "--detector" && i + 1 < argc)
            opt_detector = argv[++i];
        else if ((a == "-t" || a == "--threshold") && i + 1 < argc)
            opt_threshold = atof(argv[++i]);
        else if (a == "-v" || a == "--debug")
            opt_debug = true;
        else if (a == "--nogui")
            opt_nogui = true;
        else if (a == "--help") {
            std::cout << "Usage: facial_test -u <user> [options]\n";
            return 0;
        }
    }

    if (user.empty()) {
        std::cerr << "[ERROR] Missing required parameter -u\n";
        std::cout << "Usage: facial_test -u <user> [options]\n";
        return 1;
    }

    // ---------------------------------------------------------
    // Load config
    // ---------------------------------------------------------
    FacialAuthConfig cfg;
    std::string logbuf;

    fa_load_config(cfg, logbuf,
                   cfg_path.empty() ? FACIALAUTH_CONFIG_DEFAULT : cfg_path);

    if (!logbuf.empty())
        std::cerr << logbuf;
    logbuf.clear();

    if (!opt_device.empty())   cfg.device           = opt_device;
    if (!opt_detector.empty()) cfg.detector_profile = opt_detector;
    if (opt_debug)             cfg.debug            = true;
    if (opt_nogui)             cfg.nogui            = true;
    if (opt_width  > 0)        cfg.width            = opt_width;
    if (opt_height > 0)        cfg.height           = opt_height;
    if (opt_frames > 0)        cfg.frames           = opt_frames;
    if (opt_sleep >= 0)        cfg.sleep_ms         = opt_sleep;

    double best_conf  = -1.0;
    int    best_label = -1;

    // Determina percorso modello automaticamente
    std::string model_path =
    cfg.model_path.empty()
    ? fa_user_model_path(cfg, user)
    : cfg.model_path;

    // ---------------------------------------------------------
    // Esegui test
    // ---------------------------------------------------------
    bool ok = fa_test_user(
        user,
        cfg,
        model_path,
        best_conf,
        best_label,
        logbuf,
        opt_threshold
    );

    if (!logbuf.empty())
        std::cerr << logbuf;

    // ---------------------------------------------------------
    // RISULTATI FINALI
    // ---------------------------------------------------------
    if (ok) {
        std::cout << "\n✔ AUTH SUCCESS for user '" << user << "'\n";
        if (cfg.debug)
            std::cout << "  Best similarity = " << best_conf << "\n";
        return 0;
    } else {
        std::cout << "\n❌ AUTH FAILED for user '" << user << "'\n";
        if (cfg.debug)
            std::cout << "  Best similarity = " << best_conf << "\n";
        return 1;
    }
}

int main(int argc, char *argv[])
{
    return facial_test_cli_main(argc, argv);
}
