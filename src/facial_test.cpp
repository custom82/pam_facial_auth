#include "../include/libfacialauth.h"

#include <iostream>
#include <string>
#include <vector>
#include <filesystem>

using namespace std;
namespace fs = std::filesystem;

static void print_facial_test_help(const char *prog)
{
    std::cout <<
    "Usage: " << prog << " -u USER [options]\n"
    "\n"
    "Options:\n"
    "  -u, --user USER          Utente da verificare (obbligatorio)\n"
    "  -c, --config FILE        Percorso file di configurazione\n"
    "  -d, --device DEV         Dispositivo video (es. /dev/video0)\n"
    "  -w, --width N            Larghezza frame\n"
    "  -h, --height N           Altezza frame\n"
    "  -n, --frames N           Numero di frame da acquisire\n"
    "  -s, --sleep MS           Delay tra frame (ms)\n"
    "  --detector X             auto | haar | yunet | yunet_int8\n"
    "  -t, --threshold VAL      Override soglia (SFace o LBPH/Eigen/Fisher)\n"
    "  -v, --debug              Abilita debug\n"
    "  -g, --nogui              Disabilita GUI\n"
    "  --help                   Mostra questo messaggio\n"
    "\n"
    "Riconoscitori supportati:\n"
    "  recognizer_profile = sface | sface_int8 | lbph | eigen | fisher\n"
    "\n"
    "Backend DNN (da file di configurazione):\n"
    "  dnn_backend = cpu | cuda | cuda_fp16 | opencl | auto\n"
    "  dnn_target  = cpu | cuda | cuda_fp16 | opencl | auto\n"
    "\n";
}

int facial_test_cli_main(int argc, char *argv[])
{
    const char *prog = "facial_test";

    // PAM e gli strumenti devono essere root
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
    // PARSING PARAMETRI CLI
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

        else if (a == "-g" || a == "--nogui")
            opt_nogui = true;

        else if (a == "--help" || a == "-h") {
            print_facial_test_help(prog);
            return 0;
        }
    }

    // ---------------------------------------------------------
    // PARAMETRO OBBLIGATORIO: USER
    // ---------------------------------------------------------
    if (user.empty()) {
        print_facial_test_help(prog);
        return 1;
    }

    // ---------------------------------------------------------
    // CARICA CONFIG
    // ---------------------------------------------------------
    FacialAuthConfig cfg;
    std::string logbuf;

    fa_load_config(
        cfg,
        logbuf,
        cfg_path.empty() ? FACIALAUTH_CONFIG_DEFAULT : cfg_path
    );

    if (!logbuf.empty())
        std::cerr << logbuf;
    logbuf.clear();

    // ---------------------------------------------------------
    // OVERRIDE PARAMETRI DA CLI
    // ---------------------------------------------------------
    if (!opt_device.empty())   cfg.device           = opt_device;
    if (!opt_detector.empty()) cfg.detector_profile = opt_detector;
    if (opt_debug)             cfg.debug            = true;
    if (opt_nogui)             cfg.nogui            = true;
    if (opt_width  > 0)        cfg.width            = opt_width;
    if (opt_height > 0)        cfg.height           = opt_height;
    if (opt_frames > 0)        cfg.frames           = opt_frames;
    if (opt_sleep >= 0)        cfg.sleep_ms         = opt_sleep;

    // ---------------------------------------------------------
    // ESECUZIONE TEST DI AUTENTICAZIONE
    // ---------------------------------------------------------
    double best_conf  = 0.0;
    int    best_label = -1;

    bool ok = fa_test_user(
        user,
        cfg,
        cfg.model_path,
        best_conf,
        best_label,
        logbuf,
        opt_threshold
    );

    if (!logbuf.empty())
        std::cerr << logbuf;

    return ok ? 0 : 1;
}
