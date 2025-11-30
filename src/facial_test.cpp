#include <iostream>
#include <string>
#include <vector>
#include <cstdlib>

#include "libfacialauth.h"

using std::cout;
using std::cerr;
using std::endl;
using std::string;

struct TestOptions {
    string user;
    string device = "/dev/video0";
    string config_path = "/etc/security/pam_facial.conf";
    string backend;            // cpu / cuda (vuoto = usa config)
    string target;             // cpu / cuda (vuoto = usa config)
    string detector;           // auto / haar / yunet (vuoto = usa config)
    double threshold = -1.0;   // <0 = usa soglia da config
    bool debug = false;
    bool debug_override = false;  // nuovo: indica che la CLI ha forzato debug
};

static void print_usage(const char *progname) {
    cout << "Uso: " << progname << " -u USER [opzioni]\n"
    << "\n"
    << "Opzioni:\n"
    << "  -u, --user USER         Utente da testare (obbligatorio)\n"
    << "  -d, --device DEV        Dispositivo video (default: /dev/video0)\n"
    << "      --config FILE       File di configurazione (default: /etc/security/pam_facial.conf)\n"
    << "      --backend B         Backend DNN: cpu | cuda (CLI > config)\n"
    << "      --target T          Target DNN: cpu | cuda (CLI > config)\n"
    << "      --detector P        Profilo detector: auto | haar | yunet (CLI > config)\n"
    << "      --threshold X       Soglia di similarità per SFace (CLI > config)\n"
    << "      --debug             Abilita debug verboso (CLI > config)\n"
    << "      --no-debug          Disabilita debug (CLI > config)\n"
    << "  -h, --help              Mostra questo aiuto\n";
}

static bool parse_args(int argc, char *argv[], TestOptions &opt) {
    if (argc <= 1) {
        print_usage(argv[0]);
        return false;
    }

    for (int i = 1; i < argc; ++i) {
        string arg = argv[i];

        auto need_value = [&](const char *optname) -> bool {
            if (i + 1 >= argc) {
                cerr << "Opzione " << optname << " richiede un valore\n";
                return false;
            }
            return true;
        };

        if (arg == "-u" || arg == "--user") {
            if (!need_value(arg.c_str())) return false;
            opt.user = argv[++i];

        } else if (arg == "-d" || arg == "--device") {
            if (!need_value(arg.c_str())) return false;
            opt.device = argv[++i];

        } else if (arg == "--config") {
            if (!need_value(arg.c_str())) return false;
            opt.config_path = argv[++i];

        } else if (arg == "--backend") {
            if (!need_value(arg.c_str())) return false;
            opt.backend = argv[++i];

        } else if (arg == "--target") {
            if (!need_value(arg.c_str())) return false;
            opt.target = argv[++i];

        } else if (arg == "--detector") {
            if (!need_value(arg.c_str())) return false;
            opt.detector = argv[++i];

        } else if (arg == "--threshold") {
            if (!need_value(arg.c_str())) return false;
            try {
                opt.threshold = std::stod(argv[++i]);
            } catch (...) {
                cerr << "Valore non valido per --threshold\n";
                return false;
            }

        } else if (arg == "--debug") {
            opt.debug = true;
            opt.debug_override = true;

        } else if (arg == "--no-debug") {
            opt.debug = false;
            opt.debug_override = true;

        } else if (arg == "-h" || arg == "--help") {
            print_usage(argv[0]);
            std::exit(0);

        } else {
            cerr << "Opzione sconosciuta: " << arg << "\n";
            print_usage(argv[0]);
            return false;
        }
    }

    if (opt.user.empty()) {
        cerr << "Errore: devi specificare l'utente con -u / --user\n";
        print_usage(argv[0]);
        return false;
    }

    return true;
}

int facial_test_main(int argc, char *argv[]) {
    TestOptions opt;
    if (!parse_args(argc, argv, opt)) {
        return 1;
    }

    FacialAuthConfig cfg;
    std::string logbuf;

    // Carica configurazione
    if (!fa_load_config(cfg, logbuf, opt.config_path)) {
        cerr << "[ERROR] Impossibile caricare la configurazione da '"
        << opt.config_path << "'\n";
        if (!logbuf.empty())
            cerr << logbuf << "\n";
        return 1;
    }

    // Debug override
    if (opt.debug_override)
        cfg.debug = opt.debug;

    if (cfg.debug && !logbuf.empty()) {
        cerr << "[DEBUG] Config caricata da " << opt.config_path << ":\n"
        << logbuf << "\n";
    }

    // CLI > config
    // ---------------------------------------------

    if (!opt.device.empty())
        cfg.device = opt.device;

    if (!opt.backend.empty()) {
        cfg.dnn_backend   = opt.backend;
        cfg.yunet_backend = opt.backend;
    }

    if (!opt.target.empty())
        cfg.dnn_target = opt.target;

    if (!opt.detector.empty())
        cfg.detector_profile = opt.detector;

    double threshold_override = -1.0;

    if (opt.threshold >= 0.0) {
        threshold_override = opt.threshold;
        cfg.sface_threshold = opt.threshold;
    }

    // Model path
    string modelPath = fa_user_model_path(cfg, opt.user);

    if (cfg.debug) {
        cerr << "[DEBUG] Utente: " << opt.user << "\n"
        << "[DEBUG] Dispositivo: " << cfg.device << "\n"
        << "[DEBUG] Backend DNN: " << cfg.dnn_backend << "\n"
        << "[DEBUG] Target DNN: " << cfg.dnn_target << "\n"
        << "[DEBUG] Detector profile: " << cfg.detector_profile << "\n"
        << "[DEBUG] Modello utente: " << modelPath << "\n"
        << "[DEBUG] Soglia SFace: " << cfg.sface_threshold
        << (threshold_override >= 0.0 ? " (override CLI)" : "") << "\n";
    }

    // Esecuzione test
    double best_conf = 0.0;
    int best_label   = -1;
    std::string logbuf_test;

    bool ok = fa_test_user(
        opt.user,
        cfg,
        modelPath,
        best_conf,
        best_label,
        logbuf_test,
        threshold_override
    );

    if (cfg.debug && !logbuf_test.empty()) {
        cerr << "[DEBUG] Log test:\n" << logbuf_test << "\n";
    }

    double effective_threshold =
    (threshold_override >= 0.0) ? threshold_override : cfg.sface_threshold;

    if (ok) {
        cout << "[SUCCESS] Facial recognition passed for user '" << opt.user << "'\n"
        << "          Similarity = " << best_conf
        << " (threshold = " << effective_threshold << ")\n";
        return 0;
    } else {
        cout << "[FAIL]    Facial recognition failed for user '" << opt.user << "'\n"
        << "          Best similarity = " << best_conf
        << " (threshold = " << effective_threshold << ")\n";
        return 1;
    }
}

int main(int argc, char *argv[]) {
    return facial_test_main(argc, argv);
}
