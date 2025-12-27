#include "../include/libfacialauth.h"
#include <iostream>
#include <getopt.h>

void usage(const char* p) {
    std::cout << "Usage: " << p << " [options]\n"
    << "  -u, --user <name>      Target user (required)\n"
    << "  -D, --detector <type>  Detection: yunet, none\n"
    << "  -f, --force            Clear old samples\n"
    << "  -h, --help             Show this help\n";
}

int main(int argc, char** argv) {
    FacialAuthConfig cfg;
    std::string user, detector = "none", log;

    static struct option long_options[] = {
        {"user",     required_argument, 0, 'u'},
        {"detector", required_argument, 0, 'D'},
        {"force",    no_argument,       0, 'f'},
        {"help",     no_argument,       0, 'h'},
        {0, 0, 0, 0}
    };

    int opt, option_index = 0;
    while ((opt = getopt_long(argc, argv, "u:D:fh", long_options, &option_index)) != -1) {
        switch (opt) {
            case 'u': user = optarg; break;
            case 'D': detector = optarg; break;
            case 'f': cfg.force = true; break;
            case 'h': usage(argv[0]); return 0;
            default: usage(argv[0]); return 1;
        }
    }

    if (user.empty()) { usage(argv[0]); return 1; }
    fa_load_config(cfg, log, FACIALAUTH_DEFAULT_CONFIG);
    return fa_capture_user(user, cfg, detector, log) ? 0 : 1;
}
