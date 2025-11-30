// =============================================================
// facial_test.cpp - CLI per test riconoscimento facciale
// =============================================================

#include "../include/libfacialauth.h"
#include <iostream>
#include <string>
#include <cstring>
#include <cstdlib>

static void print_usage()
{
    std::cout <<
    "Usage: facial_test -u <user> [options]\n"
    "\n"
    "Options:\n"
    "  -u, --user <user>        Utente da testare (obbligatorio)\n"
    "  -c, --config <file>      File di configurazione\n"
    "  -d, --device <path>      Dispositivo camera (es. /dev/video0)\n"
    "  --detector <profile>     Forza profilo detector (haar, yunet, yunet_int8, auto)\n"
    "  --backend <type>         Backend DNN (cpu, cuda, cuda_fp16, opencl, auto)\n"
    "  --target <type>          Target DNN (cpu, cuda, cuda_fp16, opencl, auto)\n"
    "  -t, --threshold <value>  Soglia SFace (default da config)\n"
    "  -w, --width <px>         Larghezza webcam\n"
    "  -h, --height <px>        Altezza webcam\n"
    "  -n, --frames <num>       Numero frame da analizzare\n"
    "  -s, --sleep <ms>         Ritardo tra frame\n"
    "  -v, --debug              Debug verboso\n"
    "  -g, --nogui              Disabilita GUI\n"
    "  --help                   Mostra questo messaggio\n\n";
}

// =============================================================
// ENTRY POINT
// =============================================================

int main(int argc, char *argv[])
{
    const char *prog = "facial_test";

    if (!fa_check_root(prog))
        return 1;

    std::string user;
    std::string cfg_path;

    std::string opt_device;
    std::string opt_detector;
    std::string opt_backend;
    std::string opt_target;

    bool opt_debug = false;
    bool opt_nogui = false;

    int opt_width  = -1;
    int opt_height = -1;
    int opt_frames = -1;
    int opt_sleep  = -1;

    double opt_threshold = -1.0;

    // =============================================================
    // PARSING ARGOMENTI CLI
    // =============================================================

    for (int i = 1; i < argc; ++i) {
        std::string a = argv[i];

        if ((a == "-u" || a == "--user") && i + 1 < argc)
            user = argv[++i];

        else if ((a == "-c" || a == "--config") && i + 1 < argc)
            cfg_path = argv[++i];

        else if ((a == "-d" || a == "--device") && i + 1 < argc)
            opt_device = argv[++i];

        else if (a == "--detector" && i + 1 < argc)
            opt_detector = argv[++i];

        else if (a == "--backend" && i + 1 < argc)
            opt_backend = argv[++i];

        else if (a == "--target" && i + 1 < argc)
            opt_target = argv[++i];

        else if ((a == "-t" || a == "--threshold") && i + 1 < argc)
            opt_threshold = std::atof(argv[++i]);

        else if ((a == "-w" || a == "--width") && i + 1 < argc)
            opt_width = std::atoi(argv[++i]);

        else if ((a == "-h" || a == "--height") && i + 1 < argc)
            opt_height = std::atoi(argv[++i]);

        else if ((a == "-n" || a == "--frames") && i + 1 < argc)
            opt_frames = std::atoi(argv[++i]);

        else if ((a == "-s" || a == "--sleep") && i + 1 < argc)
            opt_sleep = std::atoi(argv[++i]);

        else if (a == "-v" || a == "--debug")
            opt_debug = true;

        else if (a == "-g" || a == "--nogui")
            opt_nogui = true;

        else if (a == "--help") {
            print_usage();
            return 0;
        }
    }

    if (user.empty()) {
        print_usage();
        return 1;
    }

    // =============================================================
    // CARICAMENTO CONFIG
    // =============================================================

    FacialAuthConfig cfg;
    std::string logbuf;

    fa_load_config(cfg, logbuf,
                   cfg_path.empty() ? FACIALAUTH_CONFIG_DEFAULT : cfg_path);

    if (!logbuf.empty())
        std::cerr << logbuf;
    logbuf.clear();

    // =============================================================
    // OVERRIDE DELLA CLI (PRIORITARIO SU CONFIG FILE)
    // =============================================================

    if (!opt_device.empty())    cfg.device           = opt_device;
    if (!opt_detector.empty())  cfg.detector_profile = opt_detector;
    if (!opt_backend.empty())   cfg.dnn_backend      = opt_backend;
    if (!opt_target.empty())    cfg.dnn_target       = opt_target;

    if (opt_debug)              cfg.debug            = true;
    if (opt_nogui)              cfg.nogui            = true;

    if (opt_width  > 0)         cfg.width            = opt_width;
    if (opt_height > 0)         cfg.height           = opt_height;
    if (opt_frames > 0)         cfg.frames           = opt_frames;
    if (opt_sleep >= 0)         cfg.sleep_ms         = opt_sleep;

    if (opt_threshold >= 0.0)   cfg.sface_threshold  = opt_threshold;

    // =============================================================
    // ESECUZIONE TEST
    // =============================================================

    double best_conf = -1.0;
    int    best_label = -1;

    bool ok = fa_test_user(
        user,
        cfg,
        cfg.model_path,   // path modello (se vuoto usa quello di default)
    best_conf,
    best_label,
    logbuf,
    (opt_threshold >= 0.0 ? opt_threshold : -1.0)
    );

    if (!logbuf.empty())
        std::cerr << logbuf;

    return ok ? 0 : 1;
}
