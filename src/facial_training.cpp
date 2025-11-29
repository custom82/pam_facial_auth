#include "../include/libfacialauth.h"

#include <iostream>
#include <string>
#include <vector>
#include <filesystem>
#include <cstdlib>

namespace fs = std::filesystem;

// ------------------------------------------------------------
// HELP
// ------------------------------------------------------------
static void print_help()
{
    std::cout <<
    "Usage: facial_training -u <user> -m <method> [options]\n"
    "\nOptions:\n"
    "  -u, --user <name>           Username (required)\n"
    "  -m, --method <type>         lbph | eigen | fisher | sface (required)\n"
    "  -i, --input <dir>           Directory with face images\n"
    "  -o, --output <file>         Output XML model path\n"
    "  -f, --force                 Overwrite existing model\n"
    "  -v, --verbose               Verbose output\n"
    "  -c, --config <file>         Configuration file\n"
    "  -h, --help                  Show help\n"
    "\nIf -i or -o are not specified, defaults from configuration are used.\n";
}

// ------------------------------------------------------------
// MAIN
// ------------------------------------------------------------
int facial_training_main(int argc, char *argv[])
{
    if (argc < 2) {
        print_help();
        return 1;
    }

    std::string user;
    std::string method;
    std::string input_dir;
    std::string output_model;
    std::string cfg_path;

    bool opt_force   = false;
    bool opt_verbose = false;

    // --------------------------------------------------------
    // Parse arguments
    // --------------------------------------------------------
    for (int i = 1; i < argc; ++i) {
        std::string a = argv[i];

        if ((a == "-u" || a == "--user") && i + 1 < argc)
            user = argv[++i];
        else if ((a == "-m" || a == "--method") && i + 1 < argc)
            method = argv[++i];
        else if ((a == "-i" || a == "--input") && i + 1 < argc)
            input_dir = argv[++i];
        else if ((a == "-o" || a == "--output") && i + 1 < argc)
            output_model = argv[++i];
        else if (a == "-f" || a == "--force")
            opt_force = true;
        else if (a == "-v" || a == "--verbose")
            opt_verbose = true;
        else if (a == "-c" || a == "--config")
            cfg_path = argv[++i];
        else if (a == "-h" || a == "--help") {
            print_help();
            return 0;
        }
    }

    // --------------------------------------------------------
    // Validate required arguments
    // --------------------------------------------------------
    if (user.empty() || method.empty()) {
        std::cerr << "[ERROR] -u and -m are required.\n";
        print_help();
        return 1;
    }

    // --------------------------------------------------------
    // Load config
    // --------------------------------------------------------
    FacialAuthConfig cfg;
    std::string logbuf;

    fa_load_config(cfg, logbuf,
                   cfg_path.empty() ? FACIALAUTH_CONFIG_DEFAULT : cfg_path);

    if (!logbuf.empty())
        std::cerr << logbuf;
    logbuf.clear();

    // --------------------------------------------------------
    // Auto defaults from config
    // --------------------------------------------------------
    if (input_dir.empty())
        input_dir = fa_user_image_dir(cfg, user);

    if (output_model.empty())
        output_model = fa_user_model_path(cfg, user);

    // --------------------------------------------------------
    // Check input directory
    // --------------------------------------------------------
    if (!fs::exists(input_dir) || !fs::is_directory(input_dir)) {
        std::cerr << "[ERROR] Image directory does not exist: " << input_dir << "\n";
        return 1;
    }

    // --------------------------------------------------------
    // Handle existing model
    // --------------------------------------------------------
    if (fs::exists(output_model) && !opt_force) {
        std::cerr << "[ERROR] Model already exists: " << output_model << "\n";
        std::cerr << "Use --force to overwrite.\n";
        return 1;
    }

    if (opt_force && fs::exists(output_model))
        fs::remove(output_model);

    // --------------------------------------------------------
    // Verbose info
    // --------------------------------------------------------
    if (opt_verbose) {
        std::cout << "[INFO] Training user model\n"
        << "  User:        " << user << "\n"
        << "  Method:      " << method << "\n"
        << "  Input dir:   " << input_dir << "\n"
        << "  Output file: " << output_model << "\n";
    }

    // --------------------------------------------------------
    // Execute training
    // --------------------------------------------------------
    bool ok = fa_train_user(
        user,
        method,
        cfg,
        output_model,
        input_dir,
        logbuf
    );

    if (!logbuf.empty())
        std::cerr << logbuf;

    if (!ok) {
        std::cerr << "[ERROR] Training failed.\n";
        return 1;
    }

    if (opt_verbose)
        std::cout << "[INFO] Training completed successfully.\n";

    return 0;
}

// Actual main()
int main(int argc, char *argv[])
{
    return facial_training_main(argc, argv);
}
