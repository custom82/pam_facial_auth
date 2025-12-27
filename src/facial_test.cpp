#include "libfacialauth.h"
#include <iostream>

int main(int argc, char **argv) {
    if (argc < 3) { std::cerr << "Usage: facial_test -u <user>\n"; return 1; }
    std::string user = argv[2];

    FacialAuthConfig cfg; std::string log;
    fa_load_config(cfg, log, FACIALAUTH_DEFAULT_CONFIG);

    double conf; int label;
    std::string mpath = fa_user_model_path(cfg, user);

    if (fa_test_user(user, cfg, mpath, conf, label, log)) {
        std::cout << "[OK] Utente riconosciuto! Conf: " << conf << std::endl;
    } else {
        std::cout << "[FAIL] Utente non riconosciuto. " << log << std::endl;
    }
    return 0;
}
