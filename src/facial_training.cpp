/*
 * Project: pam_facial_auth
 * License: GPL-3.0
 */

#include "libfacialauth.h"
#include <iostream>
#include <vector>

void usage() {
    std::cout << "Usage: facial_training -u <user> [options]\n\n"
    << "Options:\n"
    << "  -u, --user <name>       User\n"
    << "  -m, --method <type>     lbph, eigen, fisher, sface\n"
    << "  -c, --config <file>     Config path\n"
    << "  -f, --force             Overwrite XML\n";
}

int main(int argc, char** argv) {
    if (!fa_check_root("facial_training")) return 1;

    std::string user, log, config_path = "/etc/security/pam_facial_auth.conf";
    FacialAuthConfig cfg;
    bool force = false;

    std::vector<std::string> args(argv + 1, argv + argc);
    for (size_t i = 0; i < args.size(); ++i) {
        if (args[i] == "-u" || args[i] == "--user") user = args[++i];
        else if (args[i] == "-m" || args[i] == "--method") cfg.method = args[++i];
        else if (args[i] == "-f" || args[i] == "--force") force = true;
        else if (args[i] == "-c" || args[i] == "--config") config_path = args[++i];
    }

    if (user.empty()) { usage(); return 1; }
    fa_load_config(cfg, log, config_path);

    if (!fa_train_user(user, cfg, log, force)) {
        std::cerr << "[ERROR] " << log << std::endl;
        return 1;
    }
    std::cout << "[OK] " << log << std::endl;
    return 0;
}
