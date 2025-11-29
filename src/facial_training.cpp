#include "../include/libfacialauth.h"

#include <iostream>
#include <string>
#include <vector>
#include <cstring>
#include <cstdlib>
#include <unistd.h>
#include <filesystem>

namespace fs = std::filesystem;

// ------------------------------------------------------------
// HELP
// ------------------------------------------------------------

static void print_help()
{
    std::cout <<
    "Usage: facial_training -u USER [options]\n"
    "  -u, --user USER        Username\n"
    "  -d, --device DEV       Override device\n"
    "  -w, --width N          Override width\n"
    "  -h, --height N         Override height\n"
    "  -n, --frames N         Override number of frames\n"
    "  -s, --sleep MS         Delay between frames\n"
    "  -f, --force            Overwrite existing model\n"
    "  -g, --nogui            Disable GUI\n"
    "      --detector NAME    auto|haar|yunet|yunet_int8\n"
    "      --clean            Remove user images only\n"
    "      --reset            Remove user model + images\n"
    "  -v, --debug            Enable debug\n"
    "  -c, --config FILE      Config file path\n"
    "\n";
}

// ------------------------------------------------------------
// MAIN WRAPPER
// ------------------------------------------------------------

int facial_training_cli_main(int argc, char *argv[])
{
    const char *prog = "facial_training";

    if (!fa_check_root(prog))
        return 1;

    std::string user;
    std::string cfg_path;

    bool opt_force  = false;
    bool opt_clean  = false;
    bool opt_reset  = false;
    bool opt_debug  = false;
    bool opt_nogui  = false;

    std::string opt_device;
    std::string opt_detector;

    int opt_width  = -1;
    int opt_height = -1;
    int opt_frames = -1;
    int opt_sleep  = -1;

    // ------------------------------------------------------------
    // Parse CLI options (NO logic here)
    // ------------------------------------------------------------
    for (int i = 1; i < argc; ++i) {
        std::string a = argv[i];

        if ((a == "-u" || a == "--user") && i + 1 < argc)
            user = argv[++i];
        else if ((a == "-c" || a == "--config") && i + 1 < argc)
            cfg_path = argv[++i];
        else if ((a == "-d" || a == "--device") && i + 1 < argc)
            opt_device = argv[++i];
        else if ((a == "-w" || a == "--width") && i + 1 < argc)
            opt_width = atoi(argv[++i]);
        else if ((a == "-h" || a == "--height") && i + 1 < argc)
            opt_height = atoi(argv[++i]);
        else if ((a == "-n" || a == "--frames") && i + 1 < argc)
            opt_frames = atoi(argv[++i]);
        else if ((a == "-s" || a == "--sleep") && i + 1 < argc)
            opt_sleep = atoi(argv[++i]);
        else if (a == "-f" || a == "--force")
            opt_force = true;
        else if (a == "-g" || a == "--nogui")
            opt_nogui = true;
        else if (a == "-v" || a == "--debug")
            opt_debug = true;
        else if (a == "--detector" && i + 1 < argc)
            opt_detector = argv[++i];
        else if (a == "--clean")
            opt_clean = true;
        else if (a == "--reset")
            opt_reset = true;
        else if (a == "--help") {
            print_help();
            return 0;
        }
    }

    if (user.empty()) {
        print_help();
        return 1;
    }

    // ------------------------------------------------------------
    // Load config
    // ------------------------------------------------------------
    FacialAuthConfig cfg;
    std::string logbuf;

    fa_load_config(cfg, logbuf,
                   cfg_path.empty() ? FACIALAUTH_CONFIG_DEFAULT : cfg_path);

    if (!logbuf.empty())
        std::cerr << logbuf;

    logbuf.clear();

    // ------------------------------------------------------------
    // Apply CLI overrides
    // ------------------------------------------------------------
    if (!opt_device.empty())   cfg.device           = opt_device;
    if (!opt_detector.empty()) cfg.detector_profile = opt_detector;
    if (opt_width  > 0)        cfg.width            = opt_width;
    if (opt_height > 0)        cfg.height           = opt_height;
    if (opt_frames > 0)        cfg.frames           = opt_frames;
    if (opt_sleep >= 0)        cfg.sleep_ms         = opt_sleep;
    if (opt_debug)             cfg.debug            = true;
    if (opt_nogui)             cfg.nogui            = true;
    if (opt_force)             cfg.force_overwrite  = true;

    // ------------------------------------------------------------
    // Clean / Reset
    // ------------------------------------------------------------
    std::string img_dir  = fa_user_image_dir(cfg, user);
    std::string mdl_path = fa_user_model_path(cfg, user);

    if (opt_reset) {
        fs::remove_all(img_dir);
        if (fs::exists(mdl_path))
            fs::remove(mdl_path);
        return 0;
    }

    if (opt_clean) {
        fs::remove_all(img_dir);
        return 0;
    }

    // ------------------------------------------------------------
    // Train (delegated fully to library)
    // ------------------------------------------------------------
    bool ok = fa_train_user(user, cfg, logbuf);

    if (!logbuf.empty())
        std::cerr << logbuf;

    return ok ? 0 : 1;
}

int main(int argc, char *argv[])
{
    return facial_training_cli_main(argc, argv);
}
