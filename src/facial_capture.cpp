#include "../include/libfacialauth.h"

#include <getopt.h>
#include <iostream>
#include <filesystem>

namespace fs = std::filesystem;

static void usage(const char *prog)
{
    std::cerr <<
    "Usage: " << prog << " [options]\n"
    "  -u, --user USER             User name\n"
    "  -c, --config FILE           Config file (default " FACIALAUTH_CONFIG_DEFAULT ")\n"
    "  -f, --force                 Overwrite existing images\n"
    "      --format EXT            Image format (png,jpg,...) [default: png]\n"
    "      --frames N              Number of frames to capture\n"
    "      --device DEV            Camera device (/dev/video0, etc.)\n"
    "      --debug                 Enable debug logging\n"
    "  -h, --help                  Show this help message\n";
}

int main(int argc, char *argv[])
{
    FacialAuthConfig cfg;

    std::string user;
    std::string config_path = FACIALAUTH_CONFIG_DEFAULT;
    std::string img_format = "png";
    int frames_override = -1;
    bool force = false;

    static struct option long_opts[] = {
        {"user",    required_argument, NULL, 'u'},
        {"config",  required_argument, NULL, 'c'},
        {"force",   no_argument,       NULL, 'f'},
        {"format",  required_argument, NULL,  1 },
        {"frames",  required_argument, NULL,  2 },
        {"device",  required_argument, NULL,  3 },
        {"debug",   no_argument,       NULL,  4 },
        {"help",    no_argument,       NULL, 'h'},
        {NULL, 0, NULL, 0}
    };

    int opt, idx;
    while ((opt = getopt_long(argc, argv, "u:c:fh", long_opts, &idx)) != -1)
    {
        switch (opt) {
            case 'u':
                user = optarg;
                break;

            case 'c':
                config_path = optarg;
                break;

            case 'f':
                force = true;
                cfg.force_overwrite = true;
                break;

            case 'h':
                usage(argv[0]);
                return 0;

            case 1:
                img_format = optarg;
                break;

            case 2:
                frames_override = std::stoi(optarg);
                break;

            case 3:
                cfg.device = optarg;
                break;

            case 4:
                cfg.debug = true;        // <-- debug CLI applicato correttamente
                break;

            default:
                usage(argv[0]);
                return 1;
        }
    }

    if (user.empty()) {
        usage(argv[0]);
        return 1;
    }

    std::string log;

    // carico configurazione (può aggiornare alcuni campi)
    fa_load_config(config_path, cfg, log);

    // ma l’argomento CLI --debug deve SEMPRE sovrascrivere!
    if (cfg.debug)
        std::cerr << "[DEBUG] Debug enabled via CLI or config\n";

    if (frames_override > 0) {
        cfg.frames = frames_override;
    }

    if (!fa_capture_images(user, cfg, force, log, img_format)) {
        std::cerr << "[ERROR] Capture failed for user " << user << "\n";
        return 1;
    }

    std::cout << "[INFO] Capture completed for user " << user << "\n";
    return 0;
}
