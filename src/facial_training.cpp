#include "../include/libfacialauth.h"
#include <iostream>
#include <getopt.h>

int main(int argc, char** argv) {
    FacialAuthConfig cfg;
    std::string user, log;

    static struct option long_opts[] = {
        {"user", 1, 0, 'u'}, {"method", 1, 0, 'm'}, {"help", 0, 0, 'h'}, {0,0,0,0}
    };

    int opt;
    while ((opt = getopt_long(argc, argv, "u:m:h", long_opts, NULL)) != -1) {
        switch (opt) {
            case 'u': user = optarg; break;
            case 'm': cfg.training_method = optarg; break;
            case 'h': std::cout << "Usage: facial_training -u <user> [-m method]\n"; return 0;
        }
    }

    if (user.empty() || !fa_check_root("facial_training")) return 1;
    fa_load_config(cfg, log, FACIALAUTH_DEFAULT_CONFIG);

    std::cout << "Training for: " << user << " [Method: " << cfg.training_method << "]\n";

    if (fa_train_user(user, cfg, log)) {
        std::cout << "[SUCCESS] Model: " << fa_user_model_path(cfg, user) << "\n";
    } else {
        std::cerr << "[ERROR] Training failed\n";
        return 1;
    }
    return 0;
}
