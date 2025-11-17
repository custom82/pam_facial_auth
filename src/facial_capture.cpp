#include "../include/libfacialauth.h"
#include <getopt.h>
#include <iostream>
#include <unistd.h>


static inline bool must_be_root() {
    if (geteuid() != 0) {
        std::cerr << "Error: facial_capture must be run as root.\n";
        return false;
    }
    return true;
}

static void print_usage(const char *prog) {
    std::cerr
    << "Usage: " << prog << " -u <user> [options]\n\n"
    << "Options:\n"
    << "  -u, --user <name>       User name (required)\n"
    << "  -d, --device <dev>      Video device (default: /dev/video0)\n"
    << "  -w, --width <px>        Frame width (default: 640)\n"
    << "  -h, --height <px>       Frame height (default: 480)\n"
    << "  -n, --frames <num>      Number of frames to capture (default: 5)\n"
    << "  -s, --sleep <ms>        Delay between frames (default: 500)\n"
    << "  -f, --force             Overwrite existing images\n"
    << "  -g, --nogui             Disable GUI preview\n"
    << "  -v, --verbose           Enable verbose/debug output\n"
    << "  -c, --config <file>     Config file (default: /etc/security/pam_facial.conf)\n"
    << "  -n, --number <frames>   Number of frames to capture\n"
    << "  --help                  Show this help message\n";
}

int main(int argc, char **argv) {
    if (!must_be_root())
        return 1;

    std::string user;
    std::string config_path = "/etc/security/pam_facial.conf";
    bool force = false;
    bool verbose = false;

    FacialAuthConfig cfg;

    static struct option long_opts[] = {
        {"user", required_argument, 0, 'u'},
        {"device", required_argument, 0, 'd'},
        {"width", required_argument, 0, 'w'},
        {"height", required_argument, 0, 'h'},
        {"frames", required_argument, 0, 'n'},
        {"sleep", required_argument, 0, 's'},
        {"force", no_argument, 0, 'f'},
        {"nogui", no_argument, 0, 'g'},
        {"verbose", no_argument, 0, 'v'},
        {"config", required_argument, 0, 'c'},
        {"help", no_argument, 0, 'H'},
        {0,0,0,0}
    };

    int opt, idx;
    while ((opt = getopt_long(argc, argv, "u:d:w:h:n:s:fgvc:H", long_opts, &idx)) != -1) {
        switch (opt) {
            case 'u': user = optarg; break;
            case 'd': cfg.device = optarg; break;
            case 'w': cfg.width = atoi(optarg); break;
            case 'h': cfg.height = atoi(optarg); break;
            case 'n': cfg.frames = atoi(optarg); break;
            case 's': cfg.sleep_ms = atoi(optarg); break;
            case 'f': force = true; break;
            case 'g': cfg.nogui = true; break;
            case 'v': verbose = true; break;
            case 'c': config_path = optarg; break;
            case 'H': print_usage(argv[0]); return 0;
            default: print_usage(argv[0]); return 1;
        }
    }

    if (user.empty()) {
        std::cerr << "Error: user required.\n";
        print_usage(argv[0]);
        return 1;
    }

    std::string log;
    read_kv_config(config_path, cfg, &log);

    if (verbose) cfg.debug = true;

    // Mostra solo il log iniziale di configurazione
    std::cerr << log;
    log.clear();

    // Esegue la cattura con log runtime
    bool ok = fa_capture_images(user, cfg, force, log);

    // Mostra eventuali messaggi finali, solo se ci sono
    if (!log.empty())
        std::cerr << log;

    return ok ? 0 : 1;
}
