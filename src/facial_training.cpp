#include "libfacialauth.h"
#include <iostream>

int main(int argc, char **argv) {
    if (argc < 3) { std::cerr << "Usage: facial_training -u <user>\n"; return 1; }
    std::string user = argv[2];

    FacialAuthConfig cfg; std::string log;
    fa_load_config(cfg, log, FACIALAUTH_DEFAULT_CONFIG);

    if (!fa_check_root("facial_training")) return 1;

    std::cout << "[INFO] Training avviato per: " << user << " con metodo: " << cfg.training_method << std::endl;
    if (fa_train_user(user, cfg, log)) {
        std::cout << "[SUCCESS] Modello creato correttamente.\n";
        return 0;
    }
    std::cerr << "[ERROR] " << log << std::endl;
    return 1;
}
