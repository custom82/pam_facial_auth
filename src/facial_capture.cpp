#include "../include/libfacialauth.h"

#include <opencv2/videoio.hpp>

#include <iostream>
#include <string>
#include <vector>
#include <cstring>
#include <cstdlib>
#include <unistd.h>
#include <filesystem>
#include <dirent.h>
#include <regex>
#include <fstream>

namespace fs = std::filesystem;

// ------------------------------------------------------------
// HELP
// ------------------------------------------------------------

static void print_help()
{
    std::cout <<
    "Usage: facial_capture -u USER [options]\n"
    "\n"
    "Core options:\n"
    "  -u, --user USER        Username\n"
    "  -d, --device DEV       Override video device (e.g. /dev/video0)\n"
    "  -w, --width N          Override capture width\n"
    "  -h, --height N         Override capture height\n"
    "  -n, --frames N         Override number of frames to capture\n"
    "  -s, --sleep MS         Delay between frames (milliseconds)\n"
    "  -f, --force            Overwrite existing images\n"
    "  -g, --nogui            Disable GUI (reserved for future use)\n"
    "      --detector NAME    auto|haar|yunet_fp32|yunet_int8\n"
    "      --clean            Remove user images only\n"
    "      --reset            Remove user model + images\n"
    "      --format EXT       jpg|png (image format for saved crops)\n"
    "  -v, --debug            Enable debug output (stderr)\n"
    "  -c, --config FILE      Config file path\n"
    "\n"
    "Info / utility:\n"
    "      --list-devices             List available video devices\n"
    "      --list-resolutions DEV     List supported resolutions for DEV\n"
    "      --help                     Show this help and exit\n"
    "\n";
}

// ------------------------------------------------------------
// /dev/video* helpers (no V4L2, only OpenCV + sysfs)
// ------------------------------------------------------------

// Read a single line from a sysfs file, if it exists.
static std::string read_sysfs_file(const std::string &path)
{
    if (!fs::exists(path))
        return {};

    std::ifstream f(path);
    if (!f.is_open())
        return {};

    std::string s;
    std::getline(f, s);
    return s;
}

// Return a list of /dev/videoX that can be opened by OpenCV.
static std::vector<std::string> list_video_devices()
{
    std::vector<std::string> devs;

    DIR *dir = opendir("/dev");
    if (!dir)
        return devs;

    struct dirent *ent;
    std::regex rx("^video[0-9]+$");

    while ((ent = readdir(dir)) != nullptr) {
        if (std::regex_match(ent->d_name, rx)) {
            std::string path = "/dev/" + std::string(ent->d_name);

            // Try to open with OpenCV to ensure it is a valid video device.
            cv::VideoCapture cap(path, cv::CAP_ANY);
            if (cap.isOpened())
                devs.push_back(path);
        }
    }

    closedir(dir);
    return devs;
}

// Print human-readable info for a given /dev/videoX by reading sysfs.
static void print_device_info(const std::string &dev)
{
    std::string name = fs::path(dev).filename().string();
    std::string base = "/sys/class/video4linux/" + name + "/device/";

    std::string product = read_sysfs_file(base + "product");
    std::string vendor  = read_sysfs_file(base + "idVendor");
    std::string device  = read_sysfs_file(base + "idProduct");
    std::string manuf   = read_sysfs_file(base + "manufacturer");

    std::cout << dev << "\n";

    if (!product.empty())
        std::cout << "  Product:      " << product << "\n";
    if (!manuf.empty())
        std::cout << "  Manufacturer: " << manuf << "\n";
    if (!vendor.empty())
        std::cout << "  Vendor ID:    " << vendor << "\n";
    if (!device.empty())
        std::cout << "  Product ID:   " << device << "\n";

    if (product.empty() && manuf.empty() && vendor.empty() && device.empty())
        std::cout << "  (no sysfs metadata available)\n";

    std::cout << "\n";
}

// Try a list of common resolutions using OpenCV only and report which ones
// are effectively supported (i.e., set + get match the requested size).
static void list_device_resolutions(const std::string &dev)
{
    cv::VideoCapture cap(dev, cv::CAP_ANY);
    if (!cap.isOpened()) {
        std::cerr << "Cannot open device: " << dev << "\n";
        return;
    }

    std::vector<std::pair<int,int>> common_res = {
        {320, 240},
        {640, 480},
        {800, 600},
        {1024, 768},
        {1280, 720},
        {1280, 800},
        {1280, 1024},
        {1600, 900},
        {1920, 1080},
        {2560, 1440},
        {3840, 2160}
    };

    std::cout << "Supported (tested) resolutions for " << dev << ":\n";

    for (auto &r : common_res) {
        cap.set(cv::CAP_PROP_FRAME_WIDTH,  r.first);
        cap.set(cv::CAP_PROP_FRAME_HEIGHT, r.second);

        int w = static_cast<int>(cap.get(cv::CAP_PROP_FRAME_WIDTH));
        int h = static_cast<int>(cap.get(cv::CAP_PROP_FRAME_HEIGHT));

        if (w == r.first && h == r.second) {
            std::cout << "  " << w << "x" << h << "\n";
        }
    }
}

// ------------------------------------------------------------
// MAIN WRAPPER
// ------------------------------------------------------------

int facial_capture_main(int argc, char *argv[])
{
    const char *prog = "facial_capture";
    if (!fa_check_root(prog))
        return 1;

    std::string user;
    std::string cfg_path;
    std::string opt_format;

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

    // First pass: handle "info/utility" options that do not require user/config.
    // We treat them as early exit.
    for (int i = 1; i < argc; ++i) {
        std::string a = argv[i];

        if (a == "--help") {
            print_help();
            return 0;
        } else if (a == "--list-devices") {
            auto devs = list_video_devices();
            if (devs.empty()) {
                std::cout << "No video devices detected.\n";
            } else {
                for (const auto &d : devs)
                    print_device_info(d);
            }
            return 0;
        } else if (a == "--list-resolutions") {
            // Expect a device path as next argument, default to /dev/video0 if not given.
            std::string dev = "/dev/video0";
            if (i + 1 < argc && argv[i+1][0] != '-') {
                dev = argv[++i];
            }
            list_device_resolutions(dev);
            return 0;
        }
    }

    // Second pass: normal capture options
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
        else if (a == "--format" && i + 1 < argc)
            opt_format = argv[++i];
        // --help, --list-* already handled in first pass
    }

    if (user.empty()) {
        std::cerr << "Error: --user is required for capture mode.\n";
        print_help();
        return 1;
    }

    // load config
    FacialAuthConfig cfg;
    std::string logbuf;

    fa_load_config(cfg, logbuf,
                   cfg_path.empty() ? FACIALAUTH_CONFIG_DEFAULT : cfg_path);

    if (!logbuf.empty())
        std::cerr << logbuf;
    logbuf.clear();

    // apply CLI overrides
    if (!opt_device.empty())     cfg.device           = opt_device;
    if (!opt_detector.empty())   cfg.detector_profile = opt_detector;
    if (opt_width  > 0)          cfg.width            = opt_width;
    if (opt_height > 0)          cfg.height           = opt_height;
    if (opt_frames > 0)          cfg.frames           = opt_frames;
    if (opt_sleep >= 0)          cfg.sleep_ms         = opt_sleep;
    if (opt_debug)               cfg.debug            = true;
    if (opt_nogui)               cfg.nogui            = true;
    if (!opt_format.empty())     cfg.image_format     = opt_format;

    std::string user_img_dir = fa_user_image_dir(cfg, user);
    std::string user_model   = fa_user_model_path(cfg, user);

    // --reset: remove images + model
    if (opt_reset) {
        bool removed = false;

        if (fs::exists(user_img_dir)) {
            fs::remove_all(user_img_dir);
            std::cout << "[INFO] Removed all images for user '" << user << "'\n";
            removed = true;
        }

        if (fs::exists(user_model)) {
            fs::remove(user_model);
            std::cout << "[INFO] Removed model for user '" << user << "'\n";
            removed = true;
        }

        if (!removed)
            std::cout << "[INFO] Nothing to reset for user '" << user << "'\n";

        return 0;
    }

    // --clean: remove only images
    if (opt_clean) {
        if (fs::exists(user_img_dir)) {
            fs::remove_all(user_img_dir);
            std::cout << "[INFO] Removed all images for user '" << user << "'\n";
        } else {
            std::cout << "[INFO] No images to remove for user '" << user << "'\n";
        }
        return 0;
    }

    // --force: remove existing images before capture
    if (opt_force) {
        if (fs::exists(user_img_dir)) {
            fs::remove_all(user_img_dir);
            std::cout << "[INFO] Forced removal of existing images for user '" << user << "'\n";
        }
    }

    // perform capture
    bool ok = fa_capture_images(user, cfg, cfg.image_format, logbuf);

    if (!logbuf.empty())
        std::cerr << logbuf;

    return ok ? 0 : 1;
}

int main(int argc, char *argv[])
{
    return facial_capture_main(argc, argv);
}
