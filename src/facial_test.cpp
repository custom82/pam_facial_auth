/*
 * Project: pam_facial_auth
 * License: GPL-3.0
 */

#include "libfacialauth.h"
#include <iostream>
#include <vector>
#include <string>

void usage() {
    std::cout << "Usage: facial_test -u <user> -m <path> [options]\n\n"
    << "Options:\n"
    << "  -u, --user <user>        Utente da verificare (obbligatorio)\n"
    << "  -m, --model <path>       File modello XML (obbligatorio)\n"
    << "  -c, --config <file>      File di configurazione (default: /etc/security/pam_facial_auth.conf)\n"
    << "  -d, --device <device>    Dispositivo webcam (es. /dev/video0)\n"
    << "  --threshold <value>      Soglia di confidenza per il match (default: 80.0)\n"
    << "  -v, --verbose            Modalità verbosa\n"
    << "  --nogui                  Disabilita la GUI (solo console)\n"
    << "  -h, --help               Mostra questo messaggio\n";
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

    // Caricamento configurazione base
    fa_load_config(cfg, log, config_path);

    // Parsing parametri CLI (sovrascrivono il config)
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

    if (user.empty() || !model_provided) {
        std::cerr << "Errore: parametri -u (user) e -m (model) sono obbligatori.\n";
        return 1;
    }

    double confidence = 0.0;
    int label = -1;

    if (cfg.verbose) {
        std::cout << "[*] Avvio test di riconoscimento...\n";
        std::cout << "[*] Utente:      " << user << "\n";
        std::cout << "[*] Modello:     " << model_path << "\n";
        std::cout << "[*] Soglia:      " << cfg.threshold << "\n";
        std::cout << "[*] Dispositivo: " << device << "\n";
    }

    cfg.device = device;

    // Esecuzione del test (fa_test_user deve gestire l'apertura del device e il predict)
    if (!fa_test_user(user, cfg, model_path, confidence, label, log)) {
        std::cerr << "ERRORE: " << log << std::endl;
        return 1;
    }

    // Logica di validazione basata sulla soglia (minore è meglio per LBPH/Eigen, maggiore per SFace)
    // Qui assumiamo la logica standard: confidence <= threshold è un match (per i plugin classici)
    bool is_authenticated = (confidence <= cfg.threshold);

    std::cout << "\n-----------------------------------" << std::endl;
    std::cout << " RISULTATO:   " << (is_authenticated ? "AUTENTICATO" : "FALLITO") << std::endl;
    std::cout << " Confidence:  " << confidence << " (Soglia: " << cfg.threshold << ")" << std::endl;
    std::cout << " Label:       " << label << std::endl;
    std::cout << "-----------------------------------" << std::endl;

    return is_authenticated ? 0 : 1;
}
