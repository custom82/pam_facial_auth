#include "../include/libfacialauth.h"
#include <iostream>

int main(int argc, char** argv) {
    if (!fa_check_root("facial_training")) return 1;
    if (argc < 2) { std::cerr << "Usage: facial_training <user>\n"; return 1; }

    std::string user = argv[1];
    FacialAuthConfig cfg;
    std::string log;

    fa_load_config(cfg, log, FACIALAUTH_DEFAULT_CONFIG);
    if (!fa_train_user(user, cfg, log)) {
        std::cerr << "Error: " << log << "\n";
        return 1;
    }
    std::cout << "Model trained successfully for " << user << "\n";
    return 0;
}
